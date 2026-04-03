import argparse
import json
import os
from dataclasses import dataclass, field
from datetime import datetime
import queue
import threading
import time
from typing import Any

import numpy as np
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
import yaml
from tqdm import tqdm

from dreamer.agent import DreamerPolicy, EnvSpec
from dreamer.centerline_controller import CenterlineController
from environment.race_env import RaceCarEnv

@dataclass
class EnvConfig:
    """Bundles environment construction parameters."""
    config_path: str
    obs_scale: float = 0.25
    gui: bool = False
    cache_track: bool = True
    fast_reset: bool = True
    track_seed: int | None = None
    random_spawn: bool = True
    throttle_bias: float = 0.0

def build_dreamer_env(cfg: EnvConfig) -> RaceCarEnv:
    """Create a bare RaceCarEnv (no VecEnv, no frame stacking)."""
    return RaceCarEnv(
        config_path=cfg.config_path,
        gui=cfg.gui,
        observation_scale=cfg.obs_scale,
        # RaceCarEnv already reuses the world when cache_track=True; fast_reset just
        # makes that intent explicit at the training entrypoint.
        cache_track=(cfg.cache_track or cfg.fast_reset),
        track_seed=cfg.track_seed,
        random_spawn=cfg.random_spawn,
        terminate_off_track=False,
        max_consecutive_off_track_steps=200,
        throttle_bias=cfg.throttle_bias,
    )

def load_track_seed(config_path: str) -> int | None:
    try:
        with open(config_path, "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
    except OSError:
        return None
    seed = config.get("track", {}).get("seed")
    try:
        return int(seed) if seed is not None else None
    except (TypeError, ValueError):
        return None


def build_env_fn(cfg: EnvConfig):
    """Factory used by subprocess vector envs."""
    def _init() -> RaceCarEnv:
        return build_dreamer_env(cfg)
    return _init


class _LocalEnvRunner:
    """Serial runner that still exposes a batched reset/step interface."""

    def __init__(self, envs: list[RaceCarEnv]):
        self.envs = envs
        self.num_envs = len(envs)
        self.observation_space = envs[0].observation_space
        self.action_space = envs[0].action_space

    def reset(self) -> np.ndarray:
        obs = [env.reset()[0] for env in self.envs]
        return np.stack(obs, axis=0)

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
        next_obs, rewards, dones, infos = [], [], [], []
        for env, action in zip(self.envs, actions):
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if done:
                obs = env.reset()[0]
            next_obs.append(obs)
            rewards.append(float(reward))
            dones.append(bool(done))
            infos.append(info)
        return (
            np.stack(next_obs, axis=0),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(dones, dtype=bool),
            infos,
        )

    def close(self) -> None:
        for env in self.envs:
            env.close()

    def set_attr(self, name: str, value: Any) -> None:
        for env in self.envs:
            setattr(env, name, value)


class _VectorEnvRunner:
    """Subprocess vector env runner for multi-env collection throughput."""

    def __init__(self, vec_env):
        self.vec_env = vec_env
        self.num_envs = int(vec_env.num_envs)
        self.observation_space = vec_env.observation_space
        self.action_space = vec_env.action_space

    def reset(self) -> np.ndarray:
        return np.asarray(self.vec_env.reset())

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
        next_obs, rewards, dones, infos = self.vec_env.step(actions)
        return (
            np.asarray(next_obs),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(dones, dtype=bool),
            list(infos),
        )

    def close(self) -> None:
        self.vec_env.close()

    def set_attr(self, name: str, value: Any) -> None:
        self.vec_env.set_attr(name, value)


@dataclass
class _CollectedBatch:
    """One collected batch of parallel environment transitions."""

    obs: np.ndarray
    action: np.ndarray
    reward: np.ndarray
    done: np.ndarray
    info: list[dict]
    env_ids: list[int]
    count: int
    start_step: int


@dataclass
class _LearnerResult:
    """One completed background learner chunk."""

    metrics: dict
    num_updates: int
    global_step: int
    duration: float


class _AsyncCollector:
    """Continuously collect experience in the background while the learner trains."""

    def __init__(
        self,
        env_runner,
        policy: DreamerPolicy,
        total_timesteps: int,
        learning_starts: int,
        queue_size: int = 2,
    ):
        self.env_runner = env_runner
        self.policy = policy
        self.total_timesteps = int(total_timesteps)
        self.learning_starts = int(learning_starts)
        self.queue_size = max(1, int(queue_size))
        self._queue: queue.Queue[_CollectedBatch | BaseException | None] = queue.Queue(maxsize=self.queue_size)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._pending_policy_state: dict | None = None
        self._pending_env_attrs: dict[str, Any] | None = None
        self._policy_lock = threading.Lock()
        self._env_attr_lock = threading.Lock()
        self._start_step = 0

    def start(self, start_step: int = 0) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._start_step = int(start_step)
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="dreamer-collector", daemon=True)
        self._thread.start()

    def submit_policy_state(self, state: dict) -> None:
        with self._policy_lock:
            self._pending_policy_state = state

    def submit_env_attrs(self, attrs: dict[str, Any]) -> None:
        with self._env_attr_lock:
            merged = dict(self._pending_env_attrs or {})
            merged.update(attrs)
            self._pending_env_attrs = merged

    def set_attr(self, name: str, value: Any) -> None:
        self.submit_env_attrs({name: value})

    def _maybe_apply_pending_policy(self) -> None:
        with self._policy_lock:
            state = self._pending_policy_state
            self._pending_policy_state = None
        if state:
            self.policy.load_policy_state(state)

    def _maybe_apply_pending_env_attrs(self) -> None:
        with self._env_attr_lock:
            attrs = self._pending_env_attrs
            self._pending_env_attrs = None
        if not attrs:
            return
        for name, value in attrs.items():
            self.env_runner.set_attr(name, value)

    def _queue_put(self, item: _CollectedBatch | BaseException | None) -> None:
        while not self._stop.is_set():
            try:
                self._queue.put(item, timeout=0.1)
                return
            except queue.Full:
                continue

    def _build_actions(self, obs: np.ndarray, current_step: int) -> np.ndarray:
        num_envs = self.env_runner.num_envs
        actions = np.stack([self.env_runner.action_space.sample() for _ in range(num_envs)], axis=0)
        policy_envs = [
            env_idx
            for env_idx in range(num_envs)
            if (current_step + env_idx + 1) >= self.learning_starts
        ]
        if not policy_envs:
            return actions
        policy_actions = self.policy.policy_batch(obs[policy_envs], env_ids=policy_envs, deterministic=False)
        actions[policy_envs] = policy_actions
        return actions

    def _run(self) -> None:
        try:
            obs = self.env_runner.reset()
            self.policy.reset_policy_state()
            current_step = self._start_step
            while not self._stop.is_set() and current_step < self.total_timesteps:
                self._maybe_apply_pending_policy()
                self._maybe_apply_pending_env_attrs()
                remaining = self.total_timesteps - current_step
                actions = self._build_actions(obs, current_step)
                next_obs, rewards, dones, infos = self.env_runner.step(actions)
                for env_idx, done in enumerate(dones):
                    if bool(done):
                        self.policy.reset_policy_state(env_id=env_idx)
                count = min(remaining, self.env_runner.num_envs)
                self._queue_put(
                    _CollectedBatch(
                        obs=np.asarray(obs[:count]).copy(),
                        action=np.asarray(actions[:count]).copy(),
                        reward=np.asarray(rewards[:count], dtype=np.float32).copy(),
                        done=np.asarray(dones[:count], dtype=bool).copy(),
                        info=list(infos[:count]),
                        env_ids=list(range(count)),
                        count=count,
                        start_step=current_step + 1,
                    )
                )
                obs = next_obs
                current_step += count
            self._queue_put(None)
        except BaseException as exc:
            self._queue_put(exc)
        finally:
            self.env_runner.close()

    def get(self, timeout: float | None = None) -> _CollectedBatch:
        item = self._queue.get(timeout=timeout)
        if item is None:
            raise StopIteration
        if isinstance(item, BaseException):
            raise item
        return item

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)


class _AsyncLearner:
    """Run Dreamer updates on a background thread while collection continues."""

    def __init__(
        self,
        agent: Any,
        replay_buffer: Any,
        collector: _AsyncCollector | None = None,
        collector_sync_updates: int = 64,
        max_updates_per_batch: int = 8,
        max_pending_updates: int = 64,
    ):
        self.agent = agent
        self.replay_buffer = replay_buffer
        self.collector = collector
        self.collector_sync_updates = max(1, int(collector_sync_updates))
        self.max_updates_per_batch = max(1, int(max_updates_per_batch))
        self.max_pending_updates = max(1, int(max_pending_updates))
        self._results: queue.Queue[_LearnerResult | BaseException] = queue.Queue()
        self._condition = threading.Condition()
        self._pending_updates = 0
        self._latest_step = 0
        self._busy = False
        self._stop = False
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None
        self._agent_lock = threading.Lock()
        self._completed_updates = 0
        self._last_synced_updates = 0

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop = False
        self._error = None
        self._thread = threading.Thread(target=self._run, name="dreamer-learner", daemon=True)
        self._thread.start()

    def submit_updates(self, num_updates: int, global_step: int) -> None:
        if num_updates <= 0:
            return
        with self._condition:
            # Backpressure: wait until pending updates drop below cap
            while self._pending_updates >= self.max_pending_updates and not self._stop:
                self._condition.wait(timeout=0.5)
            self._pending_updates += int(num_updates)
            self._latest_step = max(self._latest_step, int(global_step))
            self._condition.notify_all()

    def drain_results(self) -> list[_LearnerResult]:
        results: list[_LearnerResult] = []
        while True:
            try:
                item = self._results.get_nowait()
            except queue.Empty:
                break
            if isinstance(item, BaseException):
                raise item
            results.append(item)
        if self._error is not None:
            raise self._error
        return results

    def wait_idle(self, timeout: float | None = None) -> bool:
        deadline = None if timeout is None else (time.perf_counter() + timeout)
        with self._condition:
            while (self._pending_updates > 0 or self._busy) and self._error is None:
                remaining = None if deadline is None else max(0.0, deadline - time.perf_counter())
                if remaining == 0.0:
                    return False
                self._condition.wait(timeout=remaining)
            if self._error is not None:
                raise self._error
            return True

    def run_with_agent_lock(self, fn, *args, **kwargs):
        with self._agent_lock:
            return fn(*args, **kwargs)

    def _run(self) -> None:
        try:
            while True:
                with self._condition:
                    while self._pending_updates <= 0 and not self._stop:
                        self._condition.wait(timeout=0.1)
                    if self._pending_updates <= 0 and self._stop:
                        return
                    num_updates = min(self.max_updates_per_batch, self._pending_updates)
                    global_step = self._latest_step
                    self._pending_updates -= num_updates
                    self._busy = True
                started = time.perf_counter()
                with self._agent_lock:
                    metrics = self.agent.update_many(
                        self.replay_buffer, num_updates=num_updates, global_step=global_step,
                    )
                    self._completed_updates += num_updates
                    if (
                        self.collector is not None
                        and (self._completed_updates - self._last_synced_updates) >= self.collector_sync_updates
                    ):
                        self.collector.submit_policy_state(self.agent.export_policy_state())
                        self._last_synced_updates = self._completed_updates
                self._results.put(
                    _LearnerResult(
                        metrics=metrics,
                        num_updates=num_updates,
                        global_step=global_step,
                        duration=time.perf_counter() - started,
                    )
                )
                with self._condition:
                    self._busy = False
                    self._condition.notify_all()
        except BaseException as exc:
            with self._condition:
                self._error = exc
                self._busy = False
                self._condition.notify_all()
            self._results.put(exc)

    def close(self, drain: bool = True) -> None:
        with self._condition:
            if not drain:
                self._pending_updates = 0
            self._stop = True
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=30.0 if drain else 5.0)
        if self._error is not None:
            raise self._error


def create_env_runner(args: argparse.Namespace):
    """Instantiate the Dreamer collector backend."""
    num_envs = max(1, args.num_envs)
    base_seed = load_track_seed(args.config)
    cfgs: list[EnvConfig] = []
    for idx in range(num_envs):
        track_seed = (base_seed + idx) if base_seed is not None else idx
        cfgs.append(
            EnvConfig(
                config_path=args.config,
                obs_scale=args.obs_scale,
                track_seed=track_seed,
                fast_reset=getattr(args, "fast_reset", True),
                throttle_bias=getattr(args, "throttle_bias", 0.0),
            )
        )
    if num_envs == 1:
        return _LocalEnvRunner([build_dreamer_env(cfgs[0])])
    from stable_baselines3.common.vec_env import SubprocVecEnv

    env_fns = [build_env_fn(cfg) for cfg in cfgs]
    return _VectorEnvRunner(SubprocVecEnv(env_fns, start_method="spawn"))

_DREAMER_CFG_KEYS = (
    "batch_size", "seq_len", "imagination_horizon",
    "world_model_lr", "actor_lr", "critic_lr", "discount",
)

_TRAINING_CONFIG_DEFAULTS = {
    "actor_checkpoint_frequency": 0,
    "full_checkpoint_frequency": 0,
    "checkpoint_min_step": 0,
    "train_log_frequency": 50,
    "timing_log_frequency": 100,
    "reward_stage_fracs": "0.35,0.35,0.30",
    "stage_progress_weights": "1.0,1.0,1.0",
    "stage_speed_weights": "0.0,0.05,0.2",
    "stage_steer_penalties": "0.03,0.02,0.02",
    "stage_gate_rewards": "0.0,0.0,1.0",
    "stage_off_track_penalties": "-0.5,-0.5,-0.5",
    "stage_start_zone_stall_penalties": "0.4,0.2,0.1",
    "stage_start_zone_spin_penalties": "0.6,0.3,0.15",
}

def build_dreamer_config(args: argparse.Namespace) -> dict:
    """Build the config dict passed to DreamerV3Agent from CLI args."""
    base = {k: getattr(args, k) for k in _DREAMER_CFG_KEYS}
    dreamer_config_path = getattr(args, "dreamer_config", None)
    if dreamer_config_path and os.path.exists(dreamer_config_path):
        with open(dreamer_config_path, "r", encoding="utf-8") as handle:
            overrides = (yaml.safe_load(handle) or {}).get("dreamer", {})
        base.update(overrides)
    if getattr(args, "no_compile", False):
        base["no_compile"] = True
    return base


def _apply_training_overrides(args: argparse.Namespace) -> argparse.Namespace:
    dreamer_config_path = getattr(args, "dreamer_config", None)
    if not dreamer_config_path or not os.path.exists(dreamer_config_path):
        return args
    with open(dreamer_config_path, "r", encoding="utf-8") as handle:
        training_cfg = (yaml.safe_load(handle) or {}).get("training", {})
    for key, value in training_cfg.items():
        if hasattr(args, key) and getattr(args, key) == _TRAINING_CONFIG_DEFAULTS.get(key, getattr(args, key)):
            setattr(args, key, value)
    return args

def _utc_stamp() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


class _NullWriter:
    def add_scalar(self, *args, **kwargs) -> None:
        return None

    def close(self) -> None:
        return None


@dataclass
class _StepTimer:
    totals: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)

    def add(self, name: str, seconds: float) -> None:
        self.totals[name] = self.totals.get(name, 0.0) + float(seconds)
        self.counts[name] = self.counts.get(name, 0) + 1

    def summary(self) -> dict[str, float]:
        return dict(self.totals)


def _maybe_log_timing(writer, timer: _StepTimer, step: int, every: int) -> None:
    if every <= 0 or step % every != 0:
        return
    for name, total in timer.summary().items():
        writer.add_scalar(f"perf/{name}_seconds", total, step)


def _maybe_log_train_metrics(writer, metrics: dict, step: int, every: int) -> None:
    if every <= 0 or step % every != 0:
        return
    for tag, val in metrics.items():
        writer.add_scalar(f"train/{tag}", val, step)


def _should_checkpoint(step: int, every: int, min_interval: int) -> bool:
    return every > 0 and step % every == 0 and step >= min_interval


def _checkpoint_kind(step: int, actor_every: int, full_every: int) -> str | None:
    if full_every > 0 and step % full_every == 0:
        return "full"
    if actor_every > 0 and step % actor_every == 0:
        return "actor"
    return None


def _format_timing_summary(
    totals: dict[str, float], total_steps: int, wall_seconds: float | None = None,
) -> str:
    total = max(wall_seconds if wall_seconds is not None else sum(totals.values()), 1e-9)
    steps_per_second = total_steps / total
    return f"total={total:.2f}s  steps/sec={steps_per_second:.2f}  breakdown={totals}"


def resume_checkpoint(args: argparse.Namespace, ctx: "_TrainContext") -> int:
    """Load agent and replay buffer from a checkpoint directory."""
    ckpt = args.resume_from
    meta_path = os.path.join(ckpt, "metadata.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"No metadata.json in {ckpt}")
    with open(meta_path, "r", encoding="utf-8") as handle:
        meta = json.load(handle)
    resumed_step = int(meta.get("checkpoint_timesteps", 0))
    ctx.agent = ctx.agent.__class__.load(ckpt, ctx.agent.device)
    buf_path = os.path.join(ckpt, "replay_buffer.pkl")
    if os.path.exists(buf_path):
        from dreamer import EpisodeReplayBuffer

        ctx.buffer = EpisodeReplayBuffer.load(buf_path)
        ctx.buffer.reset_current_episode()
    tqdm.write(
        f"\n  [resume] loaded from {ckpt}\n"
        f"  resumed_step={resumed_step}  buffer={len(ctx.buffer)}ep/{ctx.buffer.total_steps()}st"
    )
    return resumed_step

def build_metadata(args: argparse.Namespace, run_name: str, device: str) -> dict:
    return {
        "algorithm": "DreamerV3",
        "config_path": args.config,
        "obs_scale": args.obs_scale,
        "num_envs": args.num_envs,
        "total_timesteps": args.total_timesteps,
        "run_name": run_name,
        "device": device,
        "created_utc": _utc_stamp(),
        "dreamer_config": build_dreamer_config(args),
        "buffer_capacity": args.buffer_capacity,
        "learning_starts": args.learning_starts,
        "train_freq": args.train_freq,
        "train_ratio": args.train_ratio,
        "train_ratio_max": args.train_ratio_max,
        "train_ratio_ramp": args.train_ratio_ramp,
        "train_log_frequency": args.train_log_frequency,
        "timing_log_frequency": args.timing_log_frequency,
        "actor_checkpoint_frequency": args.actor_checkpoint_frequency,
        "full_checkpoint_frequency": args.full_checkpoint_frequency or args.checkpoint_frequency,
        "checkpoint_min_step": args.checkpoint_min_step,
        "collector_queue_size": args.collector_queue_size,
        "collector_sync_updates": args.collector_sync_updates,
        "learner_max_updates_per_batch": args.learner_max_updates_per_batch,
        "learner_max_pending_updates": args.learner_max_pending_updates,
        "disable_reward_stages": bool(args.disable_reward_stages),
        "reward_stage_fracs": args.reward_stage_fracs,
        "stage_progress_weights": args.stage_progress_weights,
        "stage_speed_weights": args.stage_speed_weights,
        "stage_steer_penalties": args.stage_steer_penalties,
        "stage_gate_rewards": args.stage_gate_rewards,
        "stage_off_track_penalties": args.stage_off_track_penalties,
        "stage_start_zone_stall_penalties": args.stage_start_zone_stall_penalties,
        "stage_start_zone_spin_penalties": args.stage_start_zone_spin_penalties,
        "throttle_bias": args.throttle_bias,
        "throttle_bias_anneal_frac": args.throttle_bias_anneal_frac,
    }

def save_checkpoint(
    agent: Any, buffer: Any, save_dir: str, metadata: dict, step: int,
) -> None:
    """Save agent, actor-only weights, buffer, and metadata."""
    os.makedirs(save_dir, exist_ok=True)
    agent.save(save_dir)
    agent.save_actor_only(save_dir)
    buffer.save(os.path.join(save_dir, "replay_buffer.pkl"))
    # Merge training metadata with env_spec fields so DreamerV3Agent.load() works
    env_meta = agent.env_spec.to_json()
    meta = {**env_meta, **metadata, "checkpoint_timesteps": step,
            "saved_utc": _utc_stamp(), "config": agent.config}
    with open(os.path.join(save_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)


def save_actor_checkpoint(agent: Any, save_dir: str, metadata: dict, step: int) -> None:
    """Save lightweight actor-only checkpoint for cheaper periodic snapshots."""
    os.makedirs(save_dir, exist_ok=True)
    agent.save_actor_only(save_dir)
    env_meta = agent.env_spec.to_json()
    meta = {
        **env_meta,
        **metadata,
        "checkpoint_timesteps": step,
        "saved_utc": _utc_stamp(),
        "config": agent.config,
        "checkpoint_kind": "actor",
    }
    with open(os.path.join(save_dir, "metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, sort_keys=True)

_ARG_DEFS: list[tuple[str, type, Any]] = [
    ("--config",               str,   None),
    ("--obs-scale",            float, 0.25),
    ("--fast-reset",           int,   1),
    ("--num-envs",             int,   1),
    ("--total-timesteps",      int,   500_000),
    ("--buffer-capacity",      int,   1000),
    ("--learning-starts",      int,   5000),
    ("--train-freq",           int,   2),
    ("--train-ratio",          int,   1),
    ("--train-ratio-max",      int,   1),
    ("--train-ratio-ramp",     int,   0),
    ("--batch-size",           int,   16),
    ("--seq-len",              int,   16),
    ("--imagination-horizon",  int,   8),
    ("--world-model-lr",       float, 1e-4),
    ("--actor-lr",             float, 3e-5),
    ("--critic-lr",            float, 3e-5),
    ("--discount",             float, 0.997),
    ("--run-name",             str,   None),
    ("--checkpoint-dir",       str,   "dreamerv3-checkpoints"),
    ("--tensorboard-log",      str,   "dreamerv3-runs"),
    ("--checkpoint-frequency", int,   0),
    ("--actor-checkpoint-frequency", int, 0),
    ("--full-checkpoint-frequency", int, 0),
    ("--checkpoint-min-step",  int,   0),
    ("--train-log-frequency",  int,   50),
    ("--timing-log-frequency", int,   100),
    ("--collector-queue-size", int,   2),
    ("--collector-sync-updates", int, 64),
    ("--learner-max-updates-per-batch", int, 8),
    ("--learner-max-pending-updates", int, 64),
    ("--output-model",         str,   None),
    ("--resume-from",          str,   None),
    ("--dreamer-config",       str,   None),
    ("--reward-stage-fracs",   str,   "0.35,0.35,0.30"),
    ("--stage-progress-weights", str, "1.0,1.0,1.0"),
    ("--stage-speed-weights",  str,   "0.0,0.05,0.2"),
    ("--stage-steer-penalties", str,  "0.03,0.02,0.02"),
    ("--stage-gate-rewards",   str,   "0.0,0.0,1.0"),
    ("--stage-off-track-penalties", str, "-2.0,-1.5,-1.0"),
    ("--stage-start-zone-stall-penalties", str, "0.4,0.2,0.1"),
    ("--stage-start-zone-spin-penalties", str, "0.6,0.3,0.15"),
    ("--throttle-bias",        float, 0.3),
    ("--throttle-bias-anneal-frac", float, 0.3),
    ("--expert-prefill-steps", int, 0),
    ("--expert-kp",            float, 2.0),
    ("--expert-kd",            float, 0.5),
    ("--expert-throttle",      float, 0.5),
    ("--wm-pretrain-updates",  int,   0),
]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DreamerV3 on Racey-Car.")
    default_config = os.path.join(os.path.dirname(__file__), "models", "track_config.yaml")
    for name, typ, default in _ARG_DEFS:
        parser.add_argument(name, type=typ, default=(default_config if name == "--config" else default))
    parser.add_argument("--no-compile", action="store_true", default=False)
    parser.add_argument("--disable-reward-stages", action="store_true", default=False)
    parser.add_argument("--profile-only", action="store_true", default=False)
    parser.add_argument("--profile-steps", type=int, default=500)
    return parser.parse_args()


def _effective_train_ratio(args: argparse.Namespace, step: int) -> int:
    """Linearly ramp train_ratio from the base value to train_ratio_max."""
    ramp_end = args.train_ratio_ramp or (args.total_timesteps // 2)
    if step >= ramp_end or args.train_ratio_max <= args.train_ratio:
        return args.train_ratio_max
    frac = max(0.0, step - args.learning_starts) / max(1, ramp_end - args.learning_starts)
    scheduled = args.train_ratio + frac * (args.train_ratio_max - args.train_ratio)
    return max(args.train_ratio, round(scheduled))


def _parse_float_csv(text: str) -> list[float]:
    return [float(part.strip()) for part in text.split(",") if part.strip()]


class _RewardStageController:
    """Apply a staged reward curriculum to the Dreamer collector environments."""

    def __init__(
        self,
        total_timesteps: int,
        stage_fracs: list[float],
        progress_weights: list[float],
        speed_weights: list[float],
        steer_penalties: list[float],
        gate_rewards: list[float],
        off_track_penalties: list[float],
        start_zone_stall_penalties: list[float],
        start_zone_spin_penalties: list[float],
    ):
        stage_count = len(stage_fracs)
        if stage_count == 0:
            raise ValueError("reward stages cannot be empty")
        for values in (
            progress_weights,
            speed_weights,
            steer_penalties,
            gate_rewards,
            off_track_penalties,
            start_zone_stall_penalties,
            start_zone_spin_penalties,
        ):
            if len(values) != stage_count:
                raise ValueError("All reward-stage lists must match stage_fracs length.")
        frac_sum = float(sum(stage_fracs))
        if frac_sum <= 0.0:
            raise ValueError("reward-stage-fracs must sum to a positive value.")
        self.total_timesteps = max(1, int(total_timesteps))
        self.stage_fracs = [float(max(0.0, frac)) / frac_sum for frac in stage_fracs]
        self.progress_weights = [float(v) for v in progress_weights]
        self.speed_weights = [float(v) for v in speed_weights]
        self.steer_penalties = [float(v) for v in steer_penalties]
        self.gate_rewards = [float(v) for v in gate_rewards]
        self.off_track_penalties = [float(v) for v in off_track_penalties]
        self.start_zone_stall_penalties = [float(v) for v in start_zone_stall_penalties]
        self.start_zone_spin_penalties = [float(v) for v in start_zone_spin_penalties]
        self.stage_starts = self._build_stage_starts()
        self.current_stage = -1

    def _build_stage_starts(self) -> list[int]:
        starts: list[int] = [0]
        cumulative = 0.0
        for idx, frac in enumerate(self.stage_fracs[:-1]):
            cumulative += frac
            step = int(round(cumulative * self.total_timesteps))
            starts.append(min(max(step, starts[idx]), self.total_timesteps))
        return starts

    def _stage_for_timestep(self, step: int) -> int:
        stage = 0
        for idx, start in enumerate(self.stage_starts):
            if step >= start:
                stage = idx
            else:
                break
        return stage

    def _stage_attrs(self, stage_idx: int) -> dict[str, float]:
        return {
            "reward_progress_weight": self.progress_weights[stage_idx],
            "reward_speed_weight": self.speed_weights[stage_idx],
            "steer_rate_penalty": self.steer_penalties[stage_idx],
            "gate_reward": self.gate_rewards[stage_idx],
            "off_track_penalty": self.off_track_penalties[stage_idx],
            "start_zone_stall_penalty": self.start_zone_stall_penalties[stage_idx],
            "start_zone_spin_penalty": self.start_zone_spin_penalties[stage_idx],
        }

    def apply_for_step(self, env_target: Any, step: int) -> bool:
        stage = self._stage_for_timestep(step)
        if stage == self.current_stage:
            return False
        self.current_stage = stage
        for name, value in self._stage_attrs(stage).items():
            env_target.set_attr(name, value)
        return True


def _build_reward_stage_controller(args: argparse.Namespace) -> _RewardStageController | None:
    if getattr(args, "disable_reward_stages", False):
        return None
    return _RewardStageController(
        total_timesteps=args.total_timesteps,
        stage_fracs=_parse_float_csv(args.reward_stage_fracs),
        progress_weights=_parse_float_csv(args.stage_progress_weights),
        speed_weights=_parse_float_csv(args.stage_speed_weights),
        steer_penalties=_parse_float_csv(args.stage_steer_penalties),
        gate_rewards=_parse_float_csv(args.stage_gate_rewards),
        off_track_penalties=_parse_float_csv(args.stage_off_track_penalties),
        start_zone_stall_penalties=_parse_float_csv(args.stage_start_zone_stall_penalties),
        start_zone_spin_penalties=_parse_float_csv(args.stage_start_zone_spin_penalties),
    )

class _EpisodeTracker:
    """Lightweight per-env episode statistics."""
    def __init__(self, n: int):
        self.reward = [0.0] * n
        self.steps = [0] * n
        self.speed_sum = [0.0] * n

    def update(self, idx: int, reward: float, speed: float) -> None:
        self.reward[idx] += reward
        self.steps[idx] += 1
        self.speed_sum[idx] += speed

    def finish(self, idx: int) -> tuple[float, int, float]:
        """Finalise episode; returns (total_reward, length, mean_speed)."""
        r, length, s = self.reward[idx], self.steps[idx], self.speed_sum[idx]
        self.reward[idx] = self.steps[idx] = self.speed_sum[idx] = 0
        return r, length, s / max(1, length)

def _log_episode(writer, step, count, reward, length, speed, event, buf) -> None:
    scalars = {
        "episode/reward": reward, "episode/length": length,
        "episode/count": count, "episode/speed": speed,
        f"episode/event_{event}": 1.0,
        "diagnostics/buffer_episodes": len(buf),
        "diagnostics/buffer_steps": buf.total_steps(),
    }
    for tag, val in scalars.items():
        writer.add_scalar(tag, val, step)
    tqdm.write(
        f"  ep {count:>4d} | step {step:>7d} | "
        f"R={reward:+7.2f}  len={length:>4d}  spd={speed:.2f}  "
        f"buf={len(buf)}ep/{buf.total_steps()}st  [{event}]"
    )

def _select_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

@dataclass
class _TrainContext:
    """Holds all mutable training state to avoid long parameter lists."""
    agent: Any
    buffer: Any
    writer: Any | None = None
    metadata: dict = field(default_factory=dict)
    ckpt_dir: str = ""
    env_runner: Any | None = None
    collector: _AsyncCollector | None = None
    learner: _AsyncLearner | None = None
    reward_stage_controller: _RewardStageController | None = None
    tracker: _EpisodeTracker | None = None
    episode_count: int = 0
    throttle_bias_start: float = 0.0
    throttle_bias_anneal_steps: int = 1

    def init_episodes(self, start_step: int = 0) -> None:
        if self.collector is None:
            raise RuntimeError("collector not configured")
        if self.reward_stage_controller is not None:
            self.reward_stage_controller.apply_for_step(self.collector.env_runner, start_step)
        self.tracker = _EpisodeTracker(self.collector.env_runner.num_envs)
        self.episode_count = 0
        if self.learner is not None:
            self.learner.start()
        self.collector.start(start_step=start_step)

    def handle_done(self, env_idx: int, info: dict, step: int) -> None:
        """Process end-of-episode: finalize replay episode and log stats."""
        self.buffer.end_episode(stream_id=env_idx)
        self.episode_count += 1
        ep_r, ep_l, ep_s = self.tracker.finish(env_idx)
        _log_episode(self.writer, step, self.episode_count, ep_r, ep_l,
                     ep_s, info.get("event", "unknown"), self.buffer)

def _setup(args) -> _TrainContext:
    """Create agent, async collector, buffer, writer, and metadata."""
    from dreamer import DreamerV3Agent, EpisodeReplayBuffer
    from torch.utils.tensorboard import SummaryWriter

    run_name = args.run_name or datetime.utcnow().strftime("dreamerv3_%Y%m%d_%H%M%S")
    ckpt_dir = os.path.join(args.checkpoint_dir, run_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    device = _select_device()
    env_runner = create_env_runner(args)
    obs_shape = env_runner.observation_space.shape
    action_dim = env_runner.action_space.shape[0]
    env_spec = EnvSpec(
        obs_shape=obs_shape, action_dim=action_dim,
        action_low=env_runner.action_space.low,
        action_high=env_runner.action_space.high,
    )
    agent = DreamerV3Agent(
        env_spec=env_spec, device=device, config=build_dreamer_config(args),
    )
    buffer = EpisodeReplayBuffer(
        capacity=args.buffer_capacity, obs_shape=obs_shape, action_dim=action_dim,
    )
    collector_policy = DreamerPolicy(env_spec=env_spec, device="cpu", config=build_dreamer_config(args))
    collector_policy.load_policy_state(agent.export_policy_state())
    collector = _AsyncCollector(
        env_runner=env_runner,
        policy=collector_policy,
        total_timesteps=args.total_timesteps,
        learning_starts=args.learning_starts,
        queue_size=args.collector_queue_size,
    )
    learner = _AsyncLearner(
        agent=agent,
        replay_buffer=buffer,
        collector=collector,
        collector_sync_updates=args.collector_sync_updates,
        max_updates_per_batch=args.learner_max_updates_per_batch,
        max_pending_updates=args.learner_max_pending_updates,
    )
    reward_stage_controller = _build_reward_stage_controller(args)
    if args.profile_only:
        writer = _NullWriter()
    else:
        tb_dir = os.path.join(args.tensorboard_log, run_name)
        os.makedirs(tb_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=tb_dir)
    bias_anneal_steps = max(1, int(args.total_timesteps * args.throttle_bias_anneal_frac))
    return _TrainContext(
        agent=agent, buffer=buffer, env_runner=env_runner, collector=collector, learner=learner,
        writer=writer, reward_stage_controller=reward_stage_controller,
        metadata=build_metadata(args, run_name, device),
        ckpt_dir=ckpt_dir,
        throttle_bias_start=args.throttle_bias,
        throttle_bias_anneal_steps=bias_anneal_steps,
    )

def _crossed_interval(prev_step: int, step: int, every: int) -> bool:
    return every > 0 and step >= every and (step // every) > (prev_step // every)


def _ingest_collected_batch(ctx: _TrainContext, batch: _CollectedBatch) -> None:
    """Merge one collected batch into replay/logging state on the learner thread."""
    for offset, env_idx in enumerate(batch.env_ids):
        info = batch.info[offset] if offset < len(batch.info) else {}
        reward = float(batch.reward[offset])
        done = bool(batch.done[offset])
        ctx.buffer.add_step(batch.obs[offset], batch.action[offset], reward, done, stream_id=env_idx)
        ctx.tracker.update(env_idx, reward, info.get("speed", 0.0))
        if done:
            ctx.handle_done(env_idx, info, batch.start_step + offset)


def _drain_learner_results(ctx: _TrainContext, state: dict) -> int:
    """Merge completed learner work back into main-loop state."""
    if ctx.learner is None:
        return 0
    drained = 0
    for result in ctx.learner.drain_results():
        state["metrics"] = result.metrics
        state["updates"] += result.num_updates
        state["timer"].add("train", result.duration)
        drained += 1
    return drained

_DIAG_FREQ = 500

def _log_progress(pbar, state: dict, ep: int, active: bool) -> None:
    """Update tqdm postfix and optionally print a full diagnostics line."""
    m = state["metrics"]
    if not m:
        return
    pbar.set_postfix(
        wm=f"{m.get('wm_loss', 0):.3f}",
        kl=f"{m.get('kl_loss', 0):.3f}",
        klr=f"{m.get('raw_kl_loss', m.get('kl_loss', 0)):.3f}",
        act=f"{m.get('actor_loss', 0):.3f}",
        H=f"{m.get('entropy', 0):.2f}",
        ep=ep,
        refresh=False,
    )
    step, updates = state["step"], state["updates"]
    if active and step % _DIAG_FREQ == 0:
        tqdm.write(
            f"  diag | step {step:>7d} | updates={updates:>6d} | "
            f"wm={m['wm_loss']:.4f}  kl_raw={m.get('raw_kl_loss', m['kl_loss']):.4f}  "
            f"kl={m.get('clamped_kl_loss', m['kl_loss']):.4f}  recon={m['recon_loss']:.4f}  "
            f"actor={m['actor_loss']:.4f}  critic={m['critic_loss']:.4f}  "
            f"H={m['entropy']:.3f}  imag_r={m['imagined_reward']:.3f}"
        )

def _run_step(args, ctx: _TrainContext, step: int, pbar, state: dict) -> None:
    """Ingest one collected batch and optionally train."""
    step_timer = state["timer"]
    prev_step = state["step"]
    if ctx.collector is None:
        raise RuntimeError("collector not configured")
    drained_results = _drain_learner_results(ctx, state)
    try:
        started = time.perf_counter()
        batch = ctx.collector.get(timeout=30.0)
        step_timer.add("env_step", time.perf_counter() - started)
    except StopIteration:
        return
    started = time.perf_counter()
    _ingest_collected_batch(ctx, batch)
    step_timer.add("ingest", time.perf_counter() - started)
    collected = batch.count
    state["step"] = prev_step + collected
    if ctx.reward_stage_controller is not None and ctx.collector is not None:
        ctx.reward_stage_controller.apply_for_step(ctx.collector, state["step"])
    if ctx.throttle_bias_start > 0.0 and ctx.collector is not None:
        ratio = min(1.0, state["step"] / max(1, ctx.throttle_bias_anneal_steps))
        current_bias = ctx.throttle_bias_start * (1.0 - ratio)
        ctx.collector.set_attr("throttle_bias", current_bias)
    pbar.update(collected)
    active = state["step"] >= args.learning_starts and ctx.buffer.has_ready_sequences(min_steps=2)
    if active and prev_step < args.learning_starts <= state["step"]:
        tqdm.write(
            f"\n{'─'*70}\n  prefill complete — {ctx.buffer.total_steps()} buffered steps "
            f"({len(ctx.buffer)} complete episodes)\n  starting world-model training\n{'─'*70}"
        )
        pbar.set_description("collect+train")
    if active:
        state["pending_train_steps"] += collected
    train_windows = state["pending_train_steps"] // max(1, args.train_freq) if active else 0
    if train_windows > 0:
        ratio = _effective_train_ratio(args, state["step"])
        total_updates = train_windows * ratio
        if ctx.learner is None:
            raise RuntimeError("learner not configured")
        ctx.learner.submit_updates(total_updates, global_step=state["step"])
        state["pending_train_steps"] -= train_windows * max(1, args.train_freq)
        drained_results += _drain_learner_results(ctx, state)
        if drained_results > 0 and _crossed_interval(prev_step, state["step"], args.train_log_frequency):
            _maybe_log_train_metrics(ctx.writer, state["metrics"], state["step"], 1)
            ctx.writer.add_scalar("train/effective_ratio", ratio, state["step"])
    if active and _crossed_interval(prev_step, state["step"], max(1, args.train_freq * 50)):
        prios = list(getattr(ctx.buffer, "_priorities", []))
        if prios:
            ctx.writer.add_scalar("diagnostics/priority_max", max(prios), state["step"])
            ctx.writer.add_scalar("diagnostics/priority_min", min(prios), state["step"])
    if _crossed_interval(prev_step, state["step"], args.timing_log_frequency):
        _maybe_log_timing(ctx.writer, step_timer, state["step"], 1)
    _log_progress(pbar, state, ctx.episode_count, active)
    full_every = args.full_checkpoint_frequency or args.checkpoint_frequency
    full_crossed = _crossed_interval(prev_step, state["step"], full_every)
    actor_crossed = _crossed_interval(prev_step, state["step"], args.actor_checkpoint_frequency)
    kind = "full" if full_crossed else ("actor" if actor_crossed else None)
    if kind and state["step"] >= args.checkpoint_min_step:
        if ctx.learner is not None:
            ctx.learner.wait_idle(timeout=300.0)
            _drain_learner_results(ctx, state)
        started = time.perf_counter()
        if kind == "full":
            d = os.path.join(ctx.ckpt_dir, f"step_{state['step']}")
            if ctx.learner is None:
                save_checkpoint(ctx.agent, ctx.buffer, d, ctx.metadata, state["step"])
            else:
                ctx.learner.run_with_agent_lock(
                    save_checkpoint, ctx.agent, ctx.buffer, d, ctx.metadata, state["step"],
                )
        else:
            d = os.path.join(ctx.ckpt_dir, f"actor_step_{state['step']}")
            if ctx.learner is None:
                save_actor_checkpoint(ctx.agent, d, ctx.metadata, state["step"])
            else:
                ctx.learner.run_with_agent_lock(
                    save_actor_checkpoint, ctx.agent, d, ctx.metadata, state["step"],
                )
        step_timer.add("checkpoint", time.perf_counter() - started)
        tqdm.write(f"  [ckpt:{kind}] step {state['step']} → {d}")


def _collect_and_train(args, ctx: _TrainContext, start_step: int = 0) -> dict[str, float]:
    ctx.init_episodes(start_step=start_step)
    state = {
        "metrics": {},
        "updates": 0,
        "step": start_step,
        "pending_train_steps": 0,
        "timer": _StepTimer(),
    }
    tqdm.write(
        f"\n  run={ctx.metadata.get('run_name','?')}  device={ctx.agent.device}"
        f"  envs={ctx.collector.env_runner.num_envs if ctx.collector else 0}  steps={args.total_timesteps:,}"
        + (f"  (resuming from step {start_step:,})" if start_step else "")
    )
    pbar = tqdm(
        total=args.total_timesteps,
        initial=start_step,
        desc="prefill",
        ncols=110,
        unit="step",
    )
    while state["step"] < args.total_timesteps:
        _run_step(args, ctx, state["step"] + 1, pbar, state)
    if ctx.learner is not None:
        ctx.learner.close(drain=True)
        _drain_learner_results(ctx, state)
    pbar.close()
    return state["timer"].summary()

def _expert_prefill(args, ctx: _TrainContext) -> None:
    """Phase 1: Fill the replay buffer with expert centerline-following demonstrations."""
    target_steps = args.expert_prefill_steps
    if target_steps <= 0:
        return
    controller = CenterlineController(
        kp=args.expert_kp, kd=args.expert_kd, base_throttle=args.expert_throttle,
    )
    base_seed = load_track_seed(args.config)

    # Use fewer envs than requested if there are fewer steps to collect.
    num_envs = min(args.num_envs, target_steps)

    # Build one env per worker, each with a distinct track seed.
    envs = []
    for idx in range(num_envs):
        seed = (base_seed + idx) if base_seed is not None else idx
        env_cfg = EnvConfig(
            config_path=args.config, obs_scale=args.obs_scale,
            track_seed=seed, fast_reset=True, throttle_bias=0.0,
        )
        envs.append(build_dreamer_env(env_cfg))

    tqdm.write(
        f"\n{'─'*70}\n  Phase 1: Expert prefill — collecting {target_steps} steps "
        f"across {num_envs} parallel envs\n{'─'*70}"
    )

    # Reset all envs and initialise per-env state.
    obs_list = []
    for env_idx, env in enumerate(envs):
        obs, _info = env.reset()
        controller.reset(env_id=env_idx)
        obs_list.append(obs)

    ep_rewards = [0.0] * num_envs
    ep_steps = [0] * num_envs
    collected_steps = 0

    pbar = tqdm(total=target_steps, desc="expert prefill", ncols=110, unit="step")
    log_interval = max(1, target_steps // 10)

    while collected_steps < target_steps:
        for env_idx, env in enumerate(envs):
            action = controller.act(env, env_id=env_idx)
            next_obs, reward, terminated, truncated, _info = env.step(action)
            done = terminated or truncated

            ctx.buffer.add_step(obs_list[env_idx], action, float(reward), done, stream_id=env_idx)
            collected_steps += 1
            pbar.update(1)
            ep_rewards[env_idx] += reward
            ep_steps[env_idx] += 1

            if collected_steps % log_interval == 0 or collected_steps >= target_steps:
                tqdm.write(
                    f"  [expert] {collected_steps}/{target_steps} steps "
                    f"(env {env_idx}) | active_ep_len={ep_steps[env_idx]} "
                    f"buf={len(ctx.buffer)}ep/{ctx.buffer.total_steps()}st"
                )

            if collected_steps >= target_steps:
                obs_list[env_idx] = next_obs
                break

            if done:
                ctx.buffer.end_episode(stream_id=env_idx)
                ep_rewards[env_idx] = 0.0
                ep_steps[env_idx] = 0

                if collected_steps < target_steps:
                    obs_list[env_idx], _info = env.reset()
                    controller.reset(env_id=env_idx)
                else:
                    obs_list[env_idx] = next_obs
            else:
                obs_list[env_idx] = next_obs
        if collected_steps >= target_steps:
            break

    pbar.close()

    for env in envs:
        env.close()

    tqdm.write(
        f"  [expert] prefill complete: {len(ctx.buffer)} episodes, "
        f"{ctx.buffer.total_steps()} steps in buffer"
    )


def _wm_pretrain(args, ctx: _TrainContext) -> None:
    """Phase 2: Pretrain world model on expert buffer without actor/critic training."""
    num_updates = args.wm_pretrain_updates
    if num_updates <= 0:
        return
    if not ctx.buffer.has_ready_sequences(min_steps=2):
        tqdm.write("  [wm-pretrain] skipping: buffer has no ready sequences")
        return
    tqdm.write(f"\n{'─'*70}\n  Phase 2: World model pretraining — {num_updates} updates\n{'─'*70}")
    log_every = max(1, num_updates // 20)
    for i in tqdm(range(num_updates), desc="wm pretrain", ncols=110):
        metrics = ctx.agent.update_world_model_only(ctx.buffer, global_step=i)
        if (i + 1) % log_every == 0:
            tqdm.write(
                f"  [wm-pretrain] update {i+1}/{num_updates} | "
                f"wm={metrics.get('wm_loss', 0):.4f}  recon={metrics.get('recon_loss', 0):.4f}  "
                f"kl={metrics.get('kl_loss', 0):.4f}"
            )
    tqdm.write("  [wm-pretrain] complete")
    # Sync pretrained weights to collector policy
    if ctx.collector is not None:
        ctx.collector.submit_policy_state(ctx.agent.export_policy_state())


def main() -> None:
    args = parse_args()
    args = _apply_training_overrides(args)
    args.fast_reset = bool(args.fast_reset)
    if args.profile_only:
        args.total_timesteps = min(args.total_timesteps, args.profile_steps)
        args.checkpoint_frequency = 0
        args.actor_checkpoint_frequency = 0
        args.full_checkpoint_frequency = 0
    ctx = _setup(args)
    start_step = 0
    if args.resume_from:
        start_step = resume_checkpoint(args, ctx)
        if ctx.learner is not None:
            ctx.learner.agent = ctx.agent
        if ctx.collector is not None:
            ctx.collector.policy = DreamerPolicy(env_spec=ctx.agent.env_spec, device="cpu", config=ctx.agent.config)
            ctx.collector.policy.load_policy_state(ctx.agent.export_policy_state())
    # Phase 1 & 2: Expert prefill + world model pretraining (skipped if episodes=0)
    if start_step == 0:
        _expert_prefill(args, ctx)
        _wm_pretrain(args, ctx)
    run_started = time.perf_counter()
    # Phase 3 & 4: Actor training in imagination + online fine-tuning
    timing_totals = _collect_and_train(args, ctx, start_step=start_step)
    wall_seconds = time.perf_counter() - run_started
    if ctx.collector is not None:
        ctx.collector.close()
    if args.profile_only:
        print(_format_timing_summary(timing_totals, args.total_timesteps - start_step, wall_seconds=wall_seconds))
        ctx.writer.close()
        return
    final = args.output_model or os.path.join(ctx.ckpt_dir, "dreamerv3_final")
    if ctx.learner is None:
        save_checkpoint(ctx.agent, ctx.buffer, final, ctx.metadata, args.total_timesteps)
    else:
        ctx.learner.run_with_agent_lock(
            save_checkpoint, ctx.agent, ctx.buffer, final, ctx.metadata, args.total_timesteps,
        )
    ctx.writer.close()
    print(f"Training complete. Model saved to {final}")

if __name__ == "__main__":
    main()
