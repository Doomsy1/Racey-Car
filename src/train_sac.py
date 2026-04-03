import argparse
import json
import os
from datetime import datetime
from typing import Callable, List, Optional

import torch
import torch.nn as nn
import yaml
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecEnv,
    VecFrameStack,
)
from tqdm import tqdm

from environment.race_env import RaceCarEnv


def _parse_float_csv(value: str) -> list[float]:
    parts = [item.strip() for item in str(value).split(",") if item.strip()]
    if not parts:
        raise ValueError("Expected a comma-separated list of floats.")
    return [float(item) for item in parts]


def build_env_fn(
    config_path: str,
    gui: bool = False,
    obs_scale: float = 0.25,
    throttle_bias: float = 0.0,
    track_seed: int | None = None,
    cache_track: bool = False,
    random_spawn: bool = False,
    terminate_off_track: bool = False,
) -> Callable[[], RaceCarEnv]:
    """Factory for (Monitor-wrapped) RaceCarEnv instances."""

    def _init() -> RaceCarEnv:
        env = RaceCarEnv(
            config_path=config_path,
            gui=gui,
            observation_scale=obs_scale,
            throttle_bias=throttle_bias,
            cache_track=cache_track,
            track_seed=track_seed,
            random_spawn=random_spawn,
            terminate_off_track=terminate_off_track,
        )
        return Monitor(env)

    return _init


def load_track_seed(config_path: str) -> int | None:
    try:
        with open(config_path, "r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
    except OSError:
        return None
    track_cfg = config.get("track", {})
    seed = track_cfg.get("seed")
    if seed is None:
        return None
    try:
        return int(seed)
    except (TypeError, ValueError):
        return None


def make_vector_env(
    config_path: str,
    num_envs: int,
    obs_scale: float,
    throttle_bias: float,
    frame_stack: int = 1,
) -> VecEnv:
    """Create either a dummy or subprocess vectorized env, depending on n."""
    base_seed = load_track_seed(config_path)
    env_fns: List[Callable[[], RaceCarEnv]] = []
    for idx in range(num_envs):
        track_seed = (base_seed + idx) if base_seed is not None else idx
        env_fns.append(
            build_env_fn(
                config_path=config_path,
                gui=False,
                obs_scale=obs_scale,
                throttle_bias=throttle_bias,
                track_seed=track_seed,
                cache_track=True,
                random_spawn=True,
                terminate_off_track=False,  # non-terminal during training
            )
        )
    if num_envs <= 1:
        vec_env = DummyVecEnv(env_fns)
    else:
        vec_env = SubprocVecEnv(env_fns)
    if frame_stack > 1:
        vec_env = VecFrameStack(vec_env, n_stack=frame_stack, channels_order="first")
    return vec_env


def build_metadata(
    args: argparse.Namespace, run_name: str, num_envs: int, device: str
) -> dict:
    return {
        "algorithm": "SAC",
        "policy": "CnnPolicy",
        "config_path": args.config,
        "obs_scale": args.obs_scale,
        "frame_stack": args.frame_stack,
        "num_envs": num_envs,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "buffer_size": args.buffer_size,
        "learning_starts": args.learning_starts,
        "train_freq": args.train_freq,
        "gradient_steps": args.gradient_steps,
        "total_timesteps": args.total_timesteps,
        "run_name": run_name,
        "device": device,
        "created_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def save_readable_model(model: SAC, save_dir: str, metadata: dict) -> None:
    os.makedirs(save_dir, exist_ok=True)
    policy_state = {
        key: value.detach().cpu() for key, value in model.policy.state_dict().items()
    }
    torch.save(policy_state, os.path.join(save_dir, "policy_state.pt"))
    metadata = dict(metadata)
    metadata["observation_space"] = str(model.observation_space)
    metadata["action_space"] = str(model.action_space)
    metadata["saved_utc"] = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    with open(os.path.join(save_dir, "metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    # Save full model zip + replay buffer so inter-stage loading preserves
    # optimizer state and experience (avoids cold-start warmup in each stage).
    model.save(os.path.join(save_dir, "model"))


def apply_he_init(module: nn.Module) -> None:
    # Only apply to conv layers; actor/critic output Linear layers use tanh/identity
    # and should keep SB3's default orthogonal initialisation.
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class ReadableCheckpointCallback(BaseCallback):
    """Checkpoint callback that writes policy weights + JSON metadata."""

    def __init__(
        self, save_freq: int, save_path: str, name_prefix: str, base_metadata: dict
    ):
        super().__init__()
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix
        self.base_metadata = base_metadata

    def _on_training_start(self) -> None:
        os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            step_dir = os.path.join(
                self.save_path, f"{self.name_prefix}_step_{self.num_timesteps}"
            )
            metadata = dict(self.base_metadata)
            metadata["checkpoint_timesteps"] = int(self.num_timesteps)
            save_readable_model(self.model, step_dir, metadata)
        return True


class ProgressBarCallback(BaseCallback):
    """tqdm-based callback that mirrors SB3's internal timestep counter."""

    def __init__(self, total_timesteps: int):
        super().__init__()
        self.total_timesteps = total_timesteps
        self.progress_bar: Optional[tqdm] = None
        self._last_num = 0

    def _on_training_start(self) -> None:
        self.progress_bar = tqdm(total=self.total_timesteps, desc="Training", ncols=100)
        self._last_num = self.num_timesteps  # anchor to current offset (non-zero when reset_num_timesteps=False)

    def _on_step(self) -> bool:
        if self.progress_bar is None:
            return True
        delta = self.num_timesteps - self._last_num
        if delta > 0:
            remaining = self.total_timesteps - self.progress_bar.n
            if remaining > 0:
                self.progress_bar.update(min(delta, remaining))
            self._last_num = self.num_timesteps
        return True

    def _on_training_end(self) -> None:
        if self.progress_bar is not None:
            self.progress_bar.n = min(self.progress_bar.n, self.total_timesteps)
            self.progress_bar.close()
            self.progress_bar = None


class DiagnosticsCallback(BaseCallback):
    """Emit lightweight training diagnostics to stdout + TensorBoard."""

    def __init__(self, log_freq: int = 200):
        super().__init__()
        self.log_freq = log_freq
        self._reward_sum = 0.0
        self._speed_sum = 0.0
        self._count = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        rewards = self.locals.get("rewards", None)
        if rewards is not None:
            self._reward_sum += float(rewards.mean())
            self._count += 1
        if infos:
            speeds = [
                info.get("speed")
                for info in infos
                if isinstance(info, dict) and "speed" in info
            ]
            if speeds:
                self._speed_sum += float(sum(speeds) / len(speeds))
        if self.n_calls % self.log_freq == 0 and self._count:
            mean_reward = self._reward_sum / self._count
            mean_speed = self._speed_sum / max(1, self._count)
            if self.logger is not None:
                self.logger.record("diagnostics/mean_reward", mean_reward)
                self.logger.record("diagnostics/mean_speed", mean_speed)
                self.logger.dump(self.num_timesteps)
            print(
                f"[diagnostics] steps={self.num_timesteps} "
                f"mean_reward={mean_reward:.4f} mean_speed={mean_speed:.3f}"
            )
            self._reward_sum = 0.0
            self._speed_sum = 0.0
            self._count = 0
        return True


class BiasAndEntropyAnnealCallback(BaseCallback):
    """Linearly anneal throttle bias on envs and target entropy on the SAC model."""

    def __init__(
        self,
        bias_start: float,
        bias_anneal_steps: int,
        ent_start_mult: float,
        ent_anneal_steps: int,
    ):
        super().__init__()
        self.bias_start = bias_start
        self.bias_anneal_steps = max(1, bias_anneal_steps)
        self.ent_start_mult = ent_start_mult
        self.ent_anneal_steps = max(1, ent_anneal_steps)
        self.base_target_entropy: float | None = None

    def _on_training_start(self) -> None:
        if not hasattr(self.model, "target_entropy"):
            return
        self.base_target_entropy = float(self.model.target_entropy)

        if self.ent_start_mult <= 1.0:
            return

        # Reset log_ent_coef to match the exploratory start_target immediately,
        # rather than waiting for the optimizer to slowly climb from the previous
        # stage's converged (low) value.  We set log_alpha = log(|start_target|)
        # as a reasonable warm-start: it makes alpha proportional to the target
        # magnitude, which is a common SB3 initialisation heuristic.
        import math
        import torch

        start_target = self.base_target_entropy / max(self.ent_start_mult, 1e-6)
        init_log_ent = math.log(max(abs(start_target), 1e-8))
        if hasattr(self.model, "log_ent_coef") and self.model.log_ent_coef is not None:
            with torch.no_grad():
                self.model.log_ent_coef.fill_(init_log_ent)
            # Clear Adam momentum so old Stage-N gradients don't fight the reset.
            if getattr(self.model, "ent_coef_optimizer", None) is not None:
                self.model.ent_coef_optimizer.state.clear()

    def _on_step(self) -> bool:
        # Anneal bias
        step = self.num_timesteps
        bias_ratio = min(1.0, step / self.bias_anneal_steps)
        current_bias = self.bias_start * (1.0 - bias_ratio)
        try:
            self.training_env.set_attr("throttle_bias", current_bias)
        except Exception:
            pass

        # Anneal entropy target: start less-negative (high entropy / exploratory),
        # anneal toward the standard target (lower entropy / exploitative).
        # Dividing base_target_entropy by ent_start_mult makes it less negative,
        # which drives the entropy coefficient higher → more exploration early on.
        if self.base_target_entropy is not None:
            ent_ratio = min(1.0, step / self.ent_anneal_steps)
            start_target = self.base_target_entropy / max(self.ent_start_mult, 1e-6)
            current_target = start_target + ent_ratio * (
                self.base_target_entropy - start_target
            )
            self.model.target_entropy = current_target
            if self.logger is not None and self.n_calls % 200 == 0:
                self.logger.record("diagnostics/target_entropy", current_target)
        return True


class RewardStagesCallback(BaseCallback):
    """Apply staged reward-term weights during training."""

    def __init__(
        self,
        total_timesteps: int,
        stage_fracs: list[float],
        progress_weights: list[float],
        speed_weights: list[float],
        steer_penalties: list[float],
        gate_rewards: list[float],
        off_track_penalties: list[float],
    ):
        super().__init__()
        stage_count = len(stage_fracs)
        if stage_count == 0:
            raise ValueError("reward stages cannot be empty")
        for values in (
            progress_weights,
            speed_weights,
            steer_penalties,
            gate_rewards,
            off_track_penalties,
        ):
            if len(values) != stage_count:
                raise ValueError("All stage-weight lists must match stage_fracs length.")
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

    def _apply_stage(self, stage_idx: int) -> None:
        self.training_env.set_attr(
            "reward_progress_weight", self.progress_weights[stage_idx]
        )
        self.training_env.set_attr("reward_speed_weight", self.speed_weights[stage_idx])
        self.training_env.set_attr("steer_rate_penalty", self.steer_penalties[stage_idx])
        self.training_env.set_attr("gate_reward", self.gate_rewards[stage_idx])
        self.training_env.set_attr(
            "off_track_penalty", self.off_track_penalties[stage_idx]
        )
        print(
            "[reward-stage] "
            f"stage={stage_idx + 1}/{len(self.stage_fracs)} "
            f"start_step={self.stage_starts[stage_idx]} "
            f"progress_w={self.progress_weights[stage_idx]:.3f} "
            f"speed_w={self.speed_weights[stage_idx]:.3f} "
            f"steer_pen={self.steer_penalties[stage_idx]:.4f} "
            f"gate_reward={self.gate_rewards[stage_idx]:.3f} "
            f"off_track_pen={self.off_track_penalties[stage_idx]:.3f}"
        )
        if self.logger is not None:
            self.logger.record("curriculum/reward_stage", float(stage_idx + 1))
            self.logger.record(
                "curriculum/reward_progress_weight", self.progress_weights[stage_idx]
            )
            self.logger.record(
                "curriculum/reward_speed_weight", self.speed_weights[stage_idx]
            )
            self.logger.record(
                "curriculum/steer_rate_penalty", self.steer_penalties[stage_idx]
            )
            self.logger.record("curriculum/gate_reward", self.gate_rewards[stage_idx])
            self.logger.record(
                "curriculum/off_track_penalty", self.off_track_penalties[stage_idx]
            )

    def _on_training_start(self) -> None:
        self.current_stage = self._stage_for_timestep(self.num_timesteps)
        self._apply_stage(self.current_stage)

    def _on_step(self) -> bool:
        stage = self._stage_for_timestep(self.num_timesteps)
        if stage != self.current_stage:
            self.current_stage = stage
            self._apply_stage(stage)
        return True


def build_reward_stages_callback(
    args: argparse.Namespace,
) -> RewardStagesCallback | None:
    if args.disable_reward_stages:
        return None
    stage_fracs = _parse_float_csv(args.reward_stage_fracs)
    progress_weights = _parse_float_csv(args.stage_progress_weights)
    speed_weights = _parse_float_csv(args.stage_speed_weights)
    steer_penalties = _parse_float_csv(args.stage_steer_penalties)
    gate_rewards = _parse_float_csv(args.stage_gate_rewards)
    off_track_penalties = _parse_float_csv(args.stage_off_track_penalties)
    return RewardStagesCallback(
        total_timesteps=args.total_timesteps,
        stage_fracs=stage_fracs,
        progress_weights=progress_weights,
        speed_weights=speed_weights,
        steer_penalties=steer_penalties,
        gate_rewards=gate_rewards,
        off_track_penalties=off_track_penalties,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train SAC on the Racey-Car simulator."
    )
    default_config = os.path.join(
        os.path.dirname(__file__), "models", "track_config.yaml"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=default_config,
        help="Path to the simulator configuration file.",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=500_000,
        help="Total number of environment steps for training.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-2,
        help="Optimizer learning rate.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional name for logs/checkpoints; defaults to timestamp.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory where periodic checkpoints are saved.",
    )
    parser.add_argument(
        "--tensorboard-log",
        type=str,
        default="runs",
        help="Directory for TensorBoard summaries.",
    )
    parser.add_argument(
        "--checkpoint-frequency",
        type=int,
        default=100_000,
        help="Number of steps between checkpoint saves.",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=1,
        help="How many parallel environments to launch for data collection.",
    )
    parser.add_argument(
        "--obs-scale",
        type=float,
        default=0.25,
        help="Scale factor (0 < scale <= 1) controlling driver FOV occupancy map resolution.",
    )
    parser.add_argument(
        "--frame-stack",
        type=int,
        default=4,
        help="Number of consecutive observations stacked for temporal context.",
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=200_000,
        help="Replay buffer size for SAC.",
    )
    parser.add_argument(
        "--learning-starts",
        type=int,
        default=25_000,
        help="Number of steps before learning starts.",
    )
    parser.add_argument(
        "--train-freq",
        type=int,
        default=1,
        help="How often (in steps) to train the policy.",
    )
    parser.add_argument(
        "--gradient-steps",
        type=int,
        default=1,
        help="Gradient steps to run after each rollout step.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Mini-batch size for SAC updates.",
    )
    parser.add_argument(
        "--throttle-bias",
        type=float,
        default=0.0,
        help="Initial positive throttle bias added to actions (clamped) to encourage early movement.",
    )
    parser.add_argument(
        "--throttle-bias-anneal-frac",
        type=float,
        default=0.5,
        help="Fraction of total timesteps over which throttle bias is linearly annealed to zero.",
    )
    parser.add_argument(
        "--ent-start-mult",
        type=float,
        default=4.0,
        help="Multiplier for initial target entropy (higher = more exploration early).",
    )
    parser.add_argument(
        "--ent-anneal-frac",
        type=float,
        default=0.9,
        help="Fraction of total timesteps over which target entropy is annealed back to default.",
    )
    parser.add_argument(
        "--progress-anneal-frac",
        type=float,
        default=0.0,
        help="(Deprecated; ignored) Formerly controlled progress reward annealing.",
    )
    parser.add_argument(
        "--disable-reward-stages",
        action="store_true",
        help="Disable staged reward curriculum and keep reward term weights fixed.",
    )
    parser.add_argument(
        "--reward-stage-fracs",
        type=str,
        default="0.35,0.35,0.30",
        help="Comma-separated stage fractions that split total timesteps.",
    )
    parser.add_argument(
        "--stage-progress-weights",
        type=str,
        default="1.0,1.0,1.0",
        help="Per-stage weights for normalized progress reward.",
    )
    parser.add_argument(
        "--stage-speed-weights",
        type=str,
        default="0.0,0.05,0.1",
        help="Per-stage weights for speed bonus.",
    )
    parser.add_argument(
        "--stage-steer-penalties",
        type=str,
        default="0.0,0.01,0.02",
        help="Per-stage steer-rate penalties.",
    )
    parser.add_argument(
        "--stage-gate-rewards",
        type=str,
        default="0.0,0.5,1.0",
        help="Per-stage gate rewards.",
    )
    parser.add_argument(
        "--stage-off-track-penalties",
        type=str,
        default="-1.0,-1.0,-1.0",
        help="Per-stage off-track penalties.",
    )
    parser.add_argument(
        "--output-model",
        type=str,
        default=None,
        help="Override path for the final saved model directory (skips checkpoint_dir subdirectory).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run_name = args.run_name or datetime.utcnow().strftime("sac_racey_%Y%m%d_%H%M%S")
    checkpoint_dir = os.path.join(args.checkpoint_dir, run_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    num_envs = max(1, args.num_envs)
    env = make_vector_env(
        args.config, num_envs, args.obs_scale, args.throttle_bias, args.frame_stack
    )

    if torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "auto"

    base_metadata = build_metadata(args, run_name, num_envs, device)
    base_metadata["reward_stage_fracs"] = args.reward_stage_fracs
    base_metadata["stage_progress_weights"] = args.stage_progress_weights
    base_metadata["stage_speed_weights"] = args.stage_speed_weights
    base_metadata["stage_steer_penalties"] = args.stage_steer_penalties
    base_metadata["stage_gate_rewards"] = args.stage_gate_rewards
    base_metadata["stage_off_track_penalties"] = args.stage_off_track_penalties
    base_metadata["disable_reward_stages"] = bool(args.disable_reward_stages)
    progress_callback = ProgressBarCallback(args.total_timesteps)
    diagnostics_callback = DiagnosticsCallback()
    bias_anneal_steps = max(
        1, int(args.total_timesteps * args.throttle_bias_anneal_frac)
    )
    ent_anneal_steps = max(1, int(args.total_timesteps * args.ent_anneal_frac))
    bias_entropy_callback = BiasAndEntropyAnnealCallback(
        bias_start=args.throttle_bias,
        bias_anneal_steps=bias_anneal_steps,
        ent_start_mult=args.ent_start_mult,
        ent_anneal_steps=ent_anneal_steps,
    )
    reward_stages_callback = build_reward_stages_callback(args)
    callback_items: list[BaseCallback] = [
        progress_callback,
        diagnostics_callback,
        bias_entropy_callback,
    ]
    if args.checkpoint_frequency > 0:
        checkpoint_callback = ReadableCheckpointCallback(
            save_freq=max(1, args.checkpoint_frequency // env.num_envs),
            save_path=checkpoint_dir,
            name_prefix="sac_racey",
            base_metadata=base_metadata,
        )
        callback_items.insert(0, checkpoint_callback)
    if reward_stages_callback is not None:
        callback_items.append(reward_stages_callback)
    callbacks: CallbackList = CallbackList(callback_items)

    model = SAC(
        "CnnPolicy",
        env,
        verbose=1,
        learning_rate=args.learning_rate,
        tensorboard_log=os.path.join(args.tensorboard_log, run_name),
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        train_freq=args.train_freq,
        gradient_steps=args.gradient_steps,
        batch_size=args.batch_size,
        device=device,
        policy_kwargs={
            "optimizer_class": torch.optim.AdamW,
            "optimizer_kwargs": {"weight_decay": 1e-4},
        },
    )
    model.policy.apply(apply_he_init)

    model.learn(total_timesteps=args.total_timesteps, callback=callbacks)

    env.close()
    final_model_path = args.output_model if args.output_model else os.path.join(checkpoint_dir, "sac_racey_final")
    final_metadata = dict(base_metadata)
    final_metadata["checkpoint_timesteps"] = int(model.num_timesteps)
    final_metadata["final"] = True
    save_readable_model(model, final_model_path, final_metadata)
    print(f"Training complete. Model saved to {final_model_path}")


if __name__ == "__main__":
    main()
