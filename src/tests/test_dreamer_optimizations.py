import argparse
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
import yaml

import dreamer.agent as agent_module
import dreamer.models as models_module
from dreamer.agent import DreamerV3Agent, EnvSpec
from dreamer.replay_buffer import EpisodeReplayBuffer
import train_dreamer


class _PrioritySpyBuffer:
    def __init__(self, batch):
        self.batch = batch
        self.updated = None

    def sample_sequences(self, batch_size: int, seq_len: int, device: str = "cpu"):
        return self.batch, [0, 1]

    def update_priorities(self, indices, losses):
        self.updated = (indices, losses)


class DreamerOptimizationTests(unittest.TestCase):
    def test_effective_train_ratio_ramps_to_max(self):
        args = SimpleNamespace(
            train_ratio=1,
            train_ratio_max=4,
            train_ratio_ramp=0,
            learning_starts=100,
            total_timesteps=1_000,
        )

        ratios = [train_dreamer._effective_train_ratio(args, step) for step in (0, 100, 300, 500, 700)]

        self.assertEqual(ratios[0], 1)
        self.assertEqual(ratios[1], 1)
        self.assertGreaterEqual(ratios[2], 2)
        self.assertEqual(ratios[3], 4)
        self.assertEqual(ratios[4], 4)

    def test_build_dreamer_config_applies_yaml_overrides(self):
        with tempfile.NamedTemporaryFile("w", suffix=".yaml") as handle:
            yaml.safe_dump({"dreamer": {"seq_len": 32, "free_bits": 0.5, "no_compile": True}}, handle)
            handle.flush()
            args = argparse.Namespace(
                batch_size=16,
                seq_len=64,
                imagination_horizon=15,
                world_model_lr=1e-4,
                actor_lr=3e-5,
                critic_lr=3e-5,
                discount=0.997,
                dreamer_config=handle.name,
            )

            cfg = train_dreamer.build_dreamer_config(args)

        self.assertEqual(cfg["seq_len"], 32)
        self.assertEqual(cfg["free_bits"], 0.5)
        self.assertTrue(cfg["no_compile"])

    def test_medium_config_reduces_model_scale_to_expected_range(self):
        args = argparse.Namespace(
            batch_size=16,
            seq_len=64,
            imagination_horizon=15,
            world_model_lr=1e-4,
            actor_lr=3e-5,
            critic_lr=3e-5,
            discount=0.997,
            dreamer_config="models/dreamer_medium.yaml",
            no_compile=True,
        )

        cfg = train_dreamer.build_dreamer_config(args)

        self.assertEqual(cfg["feature_dim"], 384)
        self.assertEqual(cfg["hidden_dim"], 384)
        self.assertEqual(cfg["gru_dim"], 384)
        self.assertEqual(cfg["stoch_categories"], 32)
        self.assertEqual(cfg["stoch_classes"], 24)

        env_spec = EnvSpec(
            obs_shape=(2, 64, 64),
            action_dim=2,
            action_low=np.array([0.0, -1.0], dtype=np.float32),
            action_high=np.array([1.0, 1.0], dtype=np.float32),
        )
        agent = DreamerV3Agent(env_spec=env_spec, device="cpu", config=cfg)
        total = sum(p.numel() for module in (
            agent.codec, agent.rssm, agent.heads, agent.actor, agent.critic, agent.critic_target
        ) for p in module.parameters())

        self.assertGreater(total, 8_000_000)
        self.assertLess(total, 9_000_000)

    def test_maybe_compile_falls_back_when_compile_fails(self):
        module = torch.nn.Linear(4, 2)
        with mock.patch.object(agent_module.torch, "compile", side_effect=RuntimeError("boom")):
            compiled = agent_module._maybe_compile(module, "cpu")
        self.assertIs(compiled, module)

    def test_return_normalizer_tracks_percentiles(self):
        normalizer = agent_module.ReturnNormalizer(decay=0.5)
        scaled = normalizer.update_and_scale(torch.tensor([-2.0, -1.0, 0.0, 3.0, 8.0]))

        self.assertTrue(torch.isfinite(scaled).all())
        self.assertIsNotNone(normalizer._low_ema)
        self.assertIsNotNone(normalizer._high_ema)
        self.assertLess(normalizer._low_ema, normalizer._high_ema)

    def test_kl_terms_report_raw_and_clamped_values(self):
        env_spec = EnvSpec(
            obs_shape=(2, 64, 64),
            action_dim=2,
            action_low=np.array([0.0, -1.0], dtype=np.float32),
            action_high=np.array([1.0, 1.0], dtype=np.float32),
        )
        agent = DreamerV3Agent(
            env_spec=env_spec,
            device="cpu",
            config={
                "feature_dim": 16,
                "hidden_dim": 16,
                "gru_dim": 16,
                "stoch_categories": 4,
                "stoch_classes": 4,
                "batch_size": 2,
                "seq_len": 3,
                "imagination_horizon": 2,
                "free_bits": 1.0,
                "no_compile": True,
            },
        )
        logits = torch.zeros(2, 4, 4)

        raw_kl, clamped_kl = agent._kl_terms(logits, logits, free_bits=1.0)

        self.assertLess(raw_kl.item(), 1e-5)
        self.assertAlmostEqual(clamped_kl.item(), 1.0)

    def test_build_mlp_with_norm_includes_layer_norm(self):
        mlp = models_module.build_mlp(4, 8, 2, depth=2, norm=True)
        self.assertTrue(any(isinstance(layer, torch.nn.LayerNorm) for layer in mlp))

    def test_two_hot_critic_outputs_finite_values(self):
        critic = models_module.TwoHotCritic(latent_dim=16, hidden_dim=8)
        features = torch.randn(4, 16)
        targets = torch.randn(4)

        loss = critic.loss(features, targets)
        mean = critic.mean(features)

        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(mean.shape, (4,))
        self.assertTrue(torch.isfinite(mean).all())

    def test_prioritized_replay_returns_indices_and_updates_weights(self):
        buf = EpisodeReplayBuffer(8, (2, 64, 64), 2)
        for episode in range(3):
            for step in range(4):
                obs = np.full((2, 64, 64), episode + step, dtype=np.uint8)
                action = np.zeros(2, dtype=np.float32)
                buf.add_step(obs, action, float(episode), step == 3)
            buf.end_episode()

        batch, indices = buf.sample_sequences(4, 2, "cpu")
        before = list(buf._priorities)
        buf.update_priorities([0], [5.0])

        self.assertEqual(batch["obs"].shape[0], 4)
        self.assertEqual(len(indices), 4)
        self.assertEqual(len(buf._priorities), 3)
        self.assertGreater(buf._priorities[0], before[0])

    def test_replay_buffer_keeps_parallel_streams_separate(self):
        buf = EpisodeReplayBuffer(8, (1,), 1)

        buf.add_step(np.array([1], dtype=np.uint8), np.array([0.1], dtype=np.float32), 1.0, False, stream_id=0)
        buf.add_step(np.array([2], dtype=np.uint8), np.array([0.2], dtype=np.float32), 2.0, False, stream_id=1)
        buf.add_step(np.array([3], dtype=np.uint8), np.array([0.3], dtype=np.float32), 3.0, True, stream_id=0)
        buf.end_episode(stream_id=0)
        buf.add_step(np.array([4], dtype=np.uint8), np.array([0.4], dtype=np.float32), 4.0, True, stream_id=1)
        buf.end_episode(stream_id=1)

        self.assertEqual(len(buf), 2)
        self.assertEqual(buf.total_steps(), 4)
        self.assertEqual(buf._episodes[0]["obs"].reshape(-1).tolist(), [1, 3])
        self.assertEqual(buf._episodes[1]["obs"].reshape(-1).tolist(), [2, 4])

    def test_replay_buffer_can_sample_from_in_progress_streams(self):
        buf = EpisodeReplayBuffer(8, (1,), 1)
        buf.add_step(np.array([7], dtype=np.uint8), np.array([0.1], dtype=np.float32), 1.0, False, stream_id=0)
        buf.add_step(np.array([8], dtype=np.uint8), np.array([0.2], dtype=np.float32), 2.0, False, stream_id=0)
        buf.add_step(np.array([9], dtype=np.uint8), np.array([0.3], dtype=np.float32), 3.0, False, stream_id=0)

        batch, indices = buf.sample_sequences(1, 2, "cpu")

        self.assertEqual(batch["obs"].shape, (1, 2, 1))
        self.assertEqual(indices, [-1])
        self.assertTrue(buf.has_ready_sequences(min_steps=2))

    def test_agent_update_reports_warmup_free_bits_and_priority_feedback(self):
        env_spec = EnvSpec(
            obs_shape=(2, 64, 64),
            action_dim=2,
            action_low=np.array([0.0, -1.0], dtype=np.float32),
            action_high=np.array([1.0, 1.0], dtype=np.float32),
        )
        agent = DreamerV3Agent(
            env_spec=env_spec,
            device="cpu",
            config={
                "feature_dim": 16,
                "hidden_dim": 16,
                "gru_dim": 16,
                "stoch_categories": 4,
                "stoch_classes": 4,
                "batch_size": 2,
                "seq_len": 3,
                "imagination_horizon": 2,
                "free_bits": 1.0,
                "free_bits_warmup_steps": 100,
                "no_compile": True,
            },
        )
        batch = {
            "obs": torch.zeros(2, 3, 2, 64, 64, dtype=torch.uint8),
            "action": torch.zeros(2, 3, 2),
            "reward": torch.zeros(2, 3),
            "done": torch.zeros(2, 3, dtype=torch.bool),
        }
        buffer = _PrioritySpyBuffer(batch)

        with (
            mock.patch.object(
                agent,
                "_train_world_model",
                return_value={
                    "recon_loss": 2.5,
                    "_h_posts": torch.zeros(2, 3, 16),
                    "_z_posts": torch.zeros(2, 3, 16),
                },
            ),
            mock.patch.object(agent, "_train_actor_critic", return_value={"actor_loss": 0.0, "critic_loss": 0.0}),
            mock.patch.object(agent, "_update_critic_target", return_value=None),
        ):
            metrics = agent.update(buffer, global_step=50)

        self.assertAlmostEqual(metrics["free_bits"], 0.5)
        self.assertEqual(buffer.updated, ([0, 1], [2.5, 2.5]))

    def test_policy_batch_tracks_state_per_env_and_supports_targeted_reset(self):
        env_spec = EnvSpec(
            obs_shape=(2, 64, 64),
            action_dim=2,
            action_low=np.array([0.0, -1.0], dtype=np.float32),
            action_high=np.array([1.0, 1.0], dtype=np.float32),
        )
        agent = DreamerV3Agent(
            env_spec=env_spec,
            device="cpu",
            config={
                "feature_dim": 16,
                "hidden_dim": 16,
                "gru_dim": 16,
                "stoch_categories": 4,
                "stoch_classes": 4,
                "batch_size": 2,
                "seq_len": 3,
                "imagination_horizon": 2,
                "no_compile": True,
            },
        )

        obs = np.zeros((2, 2, 64, 64), dtype=np.uint8)
        actions = agent.policy_batch(obs, env_ids=[0, 1], deterministic=True)

        self.assertEqual(actions.shape, (2, 2))
        self.assertIn(0, agent._policy_state_by_env)
        self.assertIn(1, agent._policy_state_by_env)

        agent.reset_policy_state(env_id=1)

        self.assertIn(0, agent._policy_state_by_env)
        self.assertNotIn(1, agent._policy_state_by_env)

    def test_train_actor_critic_reuses_single_target_value_pass(self):
        env_spec = EnvSpec(
            obs_shape=(2, 64, 64),
            action_dim=2,
            action_low=np.array([0.0, -1.0], dtype=np.float32),
            action_high=np.array([1.0, 1.0], dtype=np.float32),
        )
        agent = DreamerV3Agent(
            env_spec=env_spec,
            device="cpu",
            config={
                "feature_dim": 16,
                "hidden_dim": 16,
                "gru_dim": 16,
                "stoch_categories": 4,
                "stoch_classes": 4,
                "batch_size": 2,
                "seq_len": 3,
                "imagination_horizon": 2,
                "no_compile": True,
            },
        )
        h_posts = torch.zeros(2, 3, 16)
        z_posts = torch.zeros(2, 3, 16)

        with mock.patch.object(agent.critic_target, "mean", wraps=agent.critic_target.mean) as mean_spy:
            agent._train_actor_critic(h_posts, z_posts)

        self.assertEqual(mean_spy.call_count, 1)

    def test_train_world_model_batches_non_recurrent_heads_over_time(self):
        env_spec = EnvSpec(
            obs_shape=(2, 64, 64),
            action_dim=2,
            action_low=np.array([0.0, -1.0], dtype=np.float32),
            action_high=np.array([1.0, 1.0], dtype=np.float32),
        )
        agent = DreamerV3Agent(
            env_spec=env_spec,
            device="cpu",
            config={
                "feature_dim": 16,
                "hidden_dim": 16,
                "gru_dim": 16,
                "stoch_categories": 4,
                "stoch_classes": 4,
                "batch_size": 2,
                "seq_len": 3,
                "imagination_horizon": 2,
                "no_compile": True,
            },
        )
        batch = {
            "obs": torch.zeros(2, 3, 2, 64, 64, dtype=torch.uint8),
            "action": torch.zeros(2, 3, 2),
            "reward": torch.zeros(2, 3),
            "done": torch.zeros(2, 3, dtype=torch.bool),
        }

        with (
            mock.patch.object(agent.codec, "encode", wraps=agent.codec.encode) as encode_spy,
            mock.patch.object(agent.codec, "decode", wraps=agent.codec.decode) as decode_spy,
            mock.patch.object(agent.heads, "reward", wraps=agent.heads.reward) as reward_spy,
            mock.patch.object(agent.heads, "continuation", wraps=agent.heads.continuation) as cont_spy,
        ):
            agent._train_world_model(batch, free_bits=0.0)

        self.assertEqual(encode_spy.call_count, 1)
        self.assertEqual(decode_spy.call_count, 1)
        self.assertEqual(reward_spy.call_count, 1)
        self.assertEqual(cont_spy.call_count, 1)


if __name__ == "__main__":
    unittest.main()
