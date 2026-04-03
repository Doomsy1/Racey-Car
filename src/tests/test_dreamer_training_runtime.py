import argparse
import contextlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
from types import SimpleNamespace

import numpy as np

import dreamer
import dreamer.agent as agent_module
import train_dreamer


class _FakeLoadedAgent:
    device = "cpu"


class _FakeBuffer:
    def __init__(self):
        self.reset_called = False

    def __len__(self):
        return 4

    def total_steps(self):
        return 99

    def reset_current_episode(self):
        self.reset_called = True


class _FakeAgent:
    device = "cpu"
    loaded_from = None

    @classmethod
    def load(cls, path: str, device: str | None = None):
        cls.loaded_from = (path, device)
        return _FakeLoadedAgent()


class _FakeExpertEnv:
    def __init__(self):
        self.steps = 0
        self.closed = False

    def reset(self):
        return np.array([0], dtype=np.uint8), {"event": "reset"}

    def step(self, action):
        self.steps += 1
        obs = np.array([self.steps], dtype=np.uint8)
        return obs, 1.0, False, False, {"event": "running"}

    def close(self):
        self.closed = True


class _FakeCenterlineController:
    def __init__(self, *args, **kwargs):
        self.reset_calls = []

    def reset(self, env_id=0):
        self.reset_calls.append(env_id)

    def act(self, env, env_id=0):
        return np.array([0.5, 0.0], dtype=np.float32)


class DreamerTrainingRuntimeTests(unittest.TestCase):
    def test_resume_checkpoint_restores_agent_buffer_and_step(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            meta_path = os.path.join(tmpdir, "metadata.json")
            with open(meta_path, "w", encoding="utf-8") as handle:
                json.dump({"checkpoint_timesteps": 123}, handle)
            replay_path = os.path.join(tmpdir, "replay_buffer.pkl")
            with open(replay_path, "wb") as handle:
                handle.write(b"buffer")

            loaded_buffer = _FakeBuffer()
            ctx = SimpleNamespace(
                agent=_FakeAgent(),
                buffer="old-buffer",
                metadata={"run_name": "demo"},
            )
            args = argparse.Namespace(resume_from=tmpdir)

            original_load = dreamer.EpisodeReplayBuffer.load
            dreamer.EpisodeReplayBuffer.load = classmethod(lambda cls, path: loaded_buffer)
            try:
                resumed_step = train_dreamer.resume_checkpoint(args, ctx)
            finally:
                dreamer.EpisodeReplayBuffer.load = original_load

        self.assertEqual(resumed_step, 123)
        self.assertIs(ctx.agent.__class__, _FakeLoadedAgent)
        self.assertIs(ctx.buffer, loaded_buffer)
        self.assertTrue(loaded_buffer.reset_called)
        self.assertEqual(_FakeAgent.loaded_from, (tmpdir, "cpu"))

    def test_autocast_ctx_uses_nullcontext_on_cpu(self):
        self.assertIsInstance(agent_module._autocast_ctx("cpu"), contextlib.nullcontext)

    def test_configure_device_runtime_sets_mps_fallback(self):
        previous = os.environ.pop("PYTORCH_ENABLE_MPS_FALLBACK", None)
        try:
            agent_module._configure_device_runtime("mps")
            self.assertEqual(os.environ["PYTORCH_ENABLE_MPS_FALLBACK"], "1")
        finally:
            if previous is None:
                os.environ.pop("PYTORCH_ENABLE_MPS_FALLBACK", None)
            else:
                os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = previous

    def test_configure_device_runtime_enables_cudnn_benchmark(self):
        original = agent_module.torch.backends.cudnn.benchmark
        try:
            agent_module.torch.backends.cudnn.benchmark = False
            agent_module._configure_device_runtime("cuda")
            self.assertTrue(agent_module.torch.backends.cudnn.benchmark)
        finally:
            agent_module.torch.backends.cudnn.benchmark = original

    def test_expert_prefill_stops_at_requested_step_budget(self):
        buffer = dreamer.EpisodeReplayBuffer(capacity=16, obs_shape=(1,), action_dim=2)
        ctx = SimpleNamespace(buffer=buffer)
        args = argparse.Namespace(
            expert_prefill_steps=5,
            expert_kp=2.0,
            expert_kd=0.5,
            expert_throttle=0.5,
            config="models/track_config.yaml",
            num_envs=2,
            obs_scale=0.25,
        )
        envs = [_FakeExpertEnv(), _FakeExpertEnv()]

        with mock.patch.object(train_dreamer, "CenterlineController", _FakeCenterlineController):
            with mock.patch.object(train_dreamer, "build_dreamer_env", side_effect=envs):
                with mock.patch.object(train_dreamer, "load_track_seed", return_value=0):
                    train_dreamer._expert_prefill(args, ctx)

        self.assertEqual(buffer.total_steps(), 5)
        self.assertEqual(len(buffer), 0)
        self.assertEqual(sum(env.steps for env in envs), 5)
        self.assertTrue(all(env.closed for env in envs))

    @unittest.skipUnless(
        hasattr(agent_module.torch.backends, "mps") and agent_module.torch.backends.mps.is_available(),
        "MPS runtime required",
    )
    def test_mps_update_path_survives_first_training_step(self):
        repo_root = os.path.dirname(os.path.dirname(__file__))
        script = """
import numpy as np
import torch
from dreamer.agent import DreamerV3Agent, EnvSpec
from dreamer.replay_buffer import EpisodeReplayBuffer

spec = EnvSpec(
    obs_shape=(2, 64, 64),
    action_dim=2,
    action_low=np.array([0.0, -1.0], dtype=np.float32),
    action_high=np.array([1.0, 1.0], dtype=np.float32),
)
agent = DreamerV3Agent(
    spec,
    device="mps",
    config={
        "feature_dim": 32,
        "hidden_dim": 32,
        "gru_dim": 32,
        "stoch_categories": 4,
        "stoch_classes": 4,
        "batch_size": 2,
        "seq_len": 4,
        "imagination_horizon": 3,
        "no_compile": True,
    },
)
buffer = EpisodeReplayBuffer(capacity=8, obs_shape=(2, 64, 64), action_dim=2)
for _ in range(3):
    for step in range(6):
        obs = np.random.randint(0, 256, size=(2, 64, 64), dtype=np.uint8)
        action = np.random.uniform(low=[0.0, -1.0], high=[1.0, 1.0], size=(2,)).astype(np.float32)
        buffer.add_step(obs, action, float(np.random.randn()), step == 5)
    buffer.end_episode()
metrics = agent.update(buffer, global_step=50)
assert "actor_loss" in metrics
print("ok", flush=True)
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(
            result.returncode,
            0,
            msg=f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}",
        )
        self.assertIn("ok", result.stdout)


if __name__ == "__main__":
    unittest.main()
