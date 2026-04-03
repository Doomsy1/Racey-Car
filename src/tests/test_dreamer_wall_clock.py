import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import train_dreamer
from dreamer.agent import DreamerV3Agent
import yaml


class FakeWriter:
    def __init__(self):
        self.scalars = []

    def add_scalar(self, tag, value, step):
        self.scalars.append((tag, value, step))

    def close(self):
        pass


class _UpdateManyTarget:
    def __init__(self):
        self.calls = 0

    def update(self, replay_buffer, global_step=0):
        self.calls += 1
        return {"last_step": global_step}


class _FakeLearnerAgent:
    def __init__(self):
        self.update_calls = []
        self.export_calls = 0

    def update_many(self, replay_buffer, num_updates=0, global_step=0):
        self.update_calls.append((replay_buffer, num_updates, global_step))
        return {"wm_loss": 1.0, "actor_loss": 0.0}

    def export_policy_state(self):
        self.export_calls += 1
        return {"actor": "state"}


class _FakeActionSpace:
    def sample(self):
        return np.array([0.0, 0.0], dtype=np.float32)


class _FakeEnvRunner:
    def __init__(self):
        self.num_envs = 2
        self.action_space = _FakeActionSpace()
        self.last_actions = None
        self.reset_calls = 0
        self.attr_updates = []

    def reset(self):
        self.reset_calls += 1
        return np.zeros((2, 2, 64, 64), dtype=np.uint8)

    def step(self, actions):
        self.last_actions = actions
        next_obs = np.ones((2, 2, 64, 64), dtype=np.uint8)
        rewards = np.array([1.0, 2.0], dtype=np.float32)
        dones = np.array([False, True], dtype=bool)
        infos = [{"speed": 1.5, "event": "running"}, {"speed": 3.0, "event": "lap_complete"}]
        return next_obs, rewards, dones, infos

    def set_attr(self, name, value):
        self.attr_updates.append((name, value))

    def close(self):
        pass


class _FakePolicyBatchAgent:
    def __init__(self):
        self.reset_calls = []
        self.policy_calls = []

    def policy_batch(self, obs_batch, env_ids=None, deterministic=False):
        self.policy_calls.append((obs_batch.shape, tuple(env_ids or ()), deterministic))
        return np.array([[0.2, 0.1], [0.4, -0.1]], dtype=np.float32)

    def reset_policy_state(self, env_id=None):
        self.reset_calls.append(env_id)


class _FakeCollectorPolicy(_FakePolicyBatchAgent):
    def __init__(self):
        super().__init__()
        self.load_calls = []

    def load_policy_state(self, state):
        self.load_calls.append(state)


class _FakeParallelBuffer:
    def __init__(self):
        self.steps = []
        self.ended = []
        self.ready = False

    def add_step(self, obs, action, reward, done, stream_id=0):
        self.steps.append((stream_id, float(reward), bool(done)))

    def end_episode(self, stream_id=0):
        self.ended.append(stream_id)

    def has_ready_sequences(self, min_steps=2):
        return self.ready

    def __len__(self):
        return len(self.ended)

    def total_steps(self):
        return len(self.steps)


class DreamerWallClockTests(unittest.TestCase):
    def test_step_timing_accumulates_named_phases(self):
        timer = train_dreamer._StepTimer()
        timer.add("env_step", 0.4)
        timer.add("train", 0.6)

        self.assertEqual(timer.summary()["env_step"], 0.4)
        self.assertEqual(timer.summary()["train"], 0.6)

    def test_maybe_log_timing_writes_scalar_metrics(self):
        writer = FakeWriter()
        timer = train_dreamer._StepTimer()
        timer.add("env_step", 1.0)
        timer.add("train", 2.0)

        train_dreamer._maybe_log_timing(writer, timer, step=100, every=100)

        self.assertIn(("perf/env_step_seconds", 1.0, 100), writer.scalars)
        self.assertIn(("perf/train_seconds", 2.0, 100), writer.scalars)

    def test_env_config_supports_fast_reset_flag(self):
        cfg = train_dreamer.EnvConfig(config_path="x.yaml", fast_reset=True)
        self.assertTrue(cfg.fast_reset)

    def test_parse_args_supports_profile_only_flag(self):
        argv = ["train_dreamer.py", "--profile-only"]
        with mock.patch.object(sys, "argv", argv):
            args = train_dreamer.parse_args()
        self.assertTrue(args.profile_only)

    def test_parse_args_defaults_disable_periodic_checkpoints(self):
        argv = ["train_dreamer.py"]
        with mock.patch.object(sys, "argv", argv):
            args = train_dreamer.parse_args()
        self.assertEqual(args.checkpoint_frequency, 0)
        self.assertEqual(args.actor_checkpoint_frequency, 0)
        self.assertEqual(args.full_checkpoint_frequency, 0)

    def test_training_config_does_not_override_explicit_cli_values(self):
        with tempfile.NamedTemporaryFile("w", suffix=".yaml") as handle:
            yaml.safe_dump({"training": {"actor_checkpoint_frequency": 10000}}, handle)
            handle.flush()
            args = SimpleNamespace(
                dreamer_config=handle.name,
                actor_checkpoint_frequency=20,
                full_checkpoint_frequency=0,
                checkpoint_min_step=50000,
                train_log_frequency=50,
                timing_log_frequency=100,
            )

            updated = train_dreamer._apply_training_overrides(args)

        self.assertEqual(updated.actor_checkpoint_frequency, 20)

    def test_update_many_returns_last_metrics_and_update_count(self):
        target = _UpdateManyTarget()

        metrics = DreamerV3Agent.update_many(target, replay_buffer="buf", num_updates=3, global_step=10)

        self.assertEqual(target.calls, 3)
        self.assertEqual(metrics["num_updates"], 3)
        self.assertEqual(metrics["last_step"], 10)

    def test_scalar_logging_can_be_throttled(self):
        writer = FakeWriter()
        metrics = {"wm_loss": 1.0, "actor_loss": 2.0}

        train_dreamer._maybe_log_train_metrics(writer, metrics, step=9, every=10)

        self.assertEqual(writer.scalars, [])

    def test_should_checkpoint_respects_minimum_interval(self):
        self.assertTrue(train_dreamer._should_checkpoint(step=1000, every=500, min_interval=1000))
        self.assertFalse(train_dreamer._should_checkpoint(step=500, every=500, min_interval=1000))

    def test_checkpoint_kind_distinguishes_full_and_actor_only(self):
        self.assertEqual(train_dreamer._checkpoint_kind(step=1000, actor_every=250, full_every=1000), "full")
        self.assertEqual(train_dreamer._checkpoint_kind(step=250, actor_every=250, full_every=1000), "actor")
        self.assertIsNone(train_dreamer._checkpoint_kind(step=125, actor_every=250, full_every=1000))

    def test_wallclock_config_file_exists(self):
        self.assertTrue(Path("models/dreamer_wallclock.yaml").exists())

    def test_medium_config_file_exists(self):
        self.assertTrue(Path("models/dreamer_medium.yaml").exists())

    def test_medium_config_disables_periodic_checkpoints(self):
        cfg = yaml.safe_load(Path("models/dreamer_medium.yaml").read_text(encoding="utf-8"))
        self.assertEqual(cfg["training"]["actor_checkpoint_frequency"], 0)
        self.assertEqual(cfg["training"]["full_checkpoint_frequency"], 0)
        self.assertEqual(cfg["training"]["checkpoint_min_step"], 0)

    def test_launcher_mentions_wallclock_config(self):
        text = Path("run_dreamer_training.sh").read_text(encoding="utf-8")
        self.assertIn("--dreamer-config", text)

    def test_launcher_defaults_to_medium_config(self):
        text = Path("run_dreamer_training.sh").read_text(encoding="utf-8")
        self.assertIn('DREAMER_CONFIG_PATH="models/dreamer_medium.yaml"', text)

    def test_launcher_default_seq_len_matches_wallclock_preset(self):
        text = Path("run_dreamer_training.sh").read_text(encoding="utf-8")
        self.assertIn("SEQ_LEN=32", text)

    def test_launcher_defaults_disable_periodic_checkpoints(self):
        text = Path("run_dreamer_training.sh").read_text(encoding="utf-8")
        self.assertIn("CHECKPOINT_FREQUENCY=0", text)
        self.assertIn("ACTOR_CHECKPOINT_FREQUENCY=0", text)
        self.assertIn("FULL_CHECKPOINT_FREQUENCY=0", text)

    def test_timing_summary_formats_steps_per_second(self):
        summary = train_dreamer._format_timing_summary({"env_step": 2.0, "train": 3.0}, total_steps=100)
        self.assertIn("steps/sec", summary)

    def test_ingest_collected_batch_records_parallel_steps_and_finalizes_done_envs(self):
        writer = FakeWriter()
        agent = _FakePolicyBatchAgent()
        buffer = _FakeParallelBuffer()
        ctx = train_dreamer._TrainContext(
            agent=agent,
            buffer=buffer,
            writer=writer,
            metadata={},
            ckpt_dir=".",
            tracker=train_dreamer._EpisodeTracker(2),
            episode_count=0,
        )
        batch = train_dreamer._CollectedBatch(
            obs=np.zeros((2, 2, 64, 64), dtype=np.uint8),
            action=np.array([[0.2, 0.1], [0.4, -0.1]], dtype=np.float32),
            reward=np.array([1.0, 2.0], dtype=np.float32),
            done=np.array([False, True], dtype=bool),
            info=[{"speed": 1.5, "event": "running"}, {"speed": 3.0, "event": "lap_complete"}],
            env_ids=[0, 1],
            count=2,
            start_step=10,
        )

        train_dreamer._ingest_collected_batch(ctx, batch)
        self.assertEqual(buffer.steps, [(0, 1.0, False), (1, 2.0, True)])
        self.assertEqual(buffer.ended, [1])
        self.assertEqual(agent.reset_calls, [])

    def test_async_collector_emits_batches_until_total_step_limit(self):
        collector = train_dreamer._AsyncCollector(
            env_runner=_FakeEnvRunner(),
            policy=_FakeCollectorPolicy(),
            total_timesteps=3,
            learning_starts=0,
            queue_size=2,
        )
        collector.start(start_step=0)
        try:
            batch1 = collector.get()
            batch2 = collector.get()
        finally:
            collector.close()

        self.assertEqual(batch1.count, 2)
        self.assertEqual(batch1.env_ids, [0, 1])
        self.assertEqual(batch2.count, 1)
        self.assertEqual(batch2.env_ids, [0])
        self.assertEqual(batch1.start_step, 1)
        self.assertEqual(batch2.start_step, 3)
        self.assertIn(1, collector.policy.reset_calls)

    def test_async_collector_applies_pending_policy_sync(self):
        policy = _FakeCollectorPolicy()
        collector = train_dreamer._AsyncCollector(
            env_runner=_FakeEnvRunner(),
            policy=policy,
            total_timesteps=2,
            learning_starts=0,
            queue_size=1,
        )
        collector.submit_policy_state({"actor": "state"})
        collector.start(start_step=0)
        try:
            collector.get()
        finally:
            collector.close()

        self.assertEqual(policy.load_calls, [{"actor": "state"}])

    def test_async_collector_applies_pending_env_attr_updates(self):
        env_runner = _FakeEnvRunner()
        collector = train_dreamer._AsyncCollector(
            env_runner=env_runner,
            policy=_FakeCollectorPolicy(),
            total_timesteps=2,
            learning_starts=0,
            queue_size=1,
        )
        collector.submit_env_attrs({"gate_reward": 0.0, "reward_speed_weight": 0.0})
        collector.start(start_step=0)
        try:
            collector.get()
        finally:
            collector.close()

        self.assertIn(("gate_reward", 0.0), env_runner.attr_updates)
        self.assertIn(("reward_speed_weight", 0.0), env_runner.attr_updates)

    def test_reward_stage_controller_applies_stage_settings(self):
        env_runner = _FakeEnvRunner()
        controller = train_dreamer._RewardStageController(
            total_timesteps=100,
            stage_fracs=[0.5, 0.5],
            progress_weights=[0.0, 1.0],
            speed_weights=[0.0, 0.2],
            steer_penalties=[0.05, 0.02],
            gate_rewards=[0.0, 1.0],
            off_track_penalties=[-2.0, -1.0],
            start_zone_stall_penalties=[0.4, 0.1],
            start_zone_spin_penalties=[0.6, 0.2],
        )

        changed0 = controller.apply_for_step(env_runner, step=0)
        changed1 = controller.apply_for_step(env_runner, step=60)

        self.assertTrue(changed0)
        self.assertTrue(changed1)
        self.assertIn(("reward_progress_weight", 0.0), env_runner.attr_updates)
        self.assertIn(("gate_reward", 1.0), env_runner.attr_updates)
        self.assertIn(("start_zone_spin_penalty", 0.6), env_runner.attr_updates)
        self.assertEqual(controller.current_stage, 1)

    def test_run_step_trains_from_ready_in_progress_sequences(self):
        writer = FakeWriter()
        agent = mock.Mock()
        buffer = _FakeParallelBuffer()
        buffer.ready = True
        collector = mock.Mock()
        learner = mock.Mock()
        learner.drain_results.return_value = []
        collector.get.return_value = train_dreamer._CollectedBatch(
            obs=np.zeros((1, 2, 64, 64), dtype=np.uint8),
            action=np.array([[0.2, 0.1]], dtype=np.float32),
            reward=np.array([1.0], dtype=np.float32),
            done=np.array([False], dtype=bool),
            info=[{"speed": 1.0}],
            env_ids=[0],
            count=1,
            start_step=12,
        )
        ctx = train_dreamer._TrainContext(
            agent=agent,
            buffer=buffer,
            writer=writer,
            metadata={},
            ckpt_dir=".",
            tracker=train_dreamer._EpisodeTracker(1),
            episode_count=0,
            collector=collector,
            learner=learner,
        )
        args = SimpleNamespace(
            learning_starts=12,
            train_freq=1,
            train_ratio=1,
            train_ratio_max=1,
            train_ratio_ramp=0,
            total_timesteps=100,
            collector_sync_updates=64,
            train_log_frequency=100,
            timing_log_frequency=100,
            full_checkpoint_frequency=0,
            checkpoint_frequency=0,
            actor_checkpoint_frequency=0,
            checkpoint_min_step=0,
        )
        state = {
            "step": 11,
            "updates": 0,
            "pending_train_steps": 0,
            "metrics": {},
            "timer": train_dreamer._StepTimer(),
            "last_sync_updates": 0,
        }
        pbar = mock.Mock()

        train_dreamer._run_step(args, ctx, step=12, pbar=pbar, state=state)

        learner.submit_updates.assert_called_once_with(1, global_step=12)
        agent.update_many.assert_not_called()

    def test_async_learner_processes_updates_and_syncs_policy(self):
        agent = _FakeLearnerAgent()
        buffer = object()
        collector = mock.Mock()
        learner = train_dreamer._AsyncLearner(
            agent=agent,
            replay_buffer=buffer,
            collector=collector,
            collector_sync_updates=2,
            max_updates_per_batch=4,
        )
        learner.start()
        try:
            learner.submit_updates(2, global_step=20)
            self.assertTrue(learner.wait_idle(timeout=2.0))
            results = learner.drain_results()
        finally:
            learner.close(drain=True)

        self.assertEqual(agent.update_calls, [(buffer, 2, 20)])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].num_updates, 2)
        collector.submit_policy_state.assert_called_once_with({"actor": "state"})


if __name__ == "__main__":
    unittest.main()
