import argparse
import os
from datetime import datetime
from typing import Callable, List

import torch
import yaml
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv, VecFrameStack

from environment.race_env import RaceCarEnv
from train_sac import (
    BiasAndEntropyAnnealCallback,
    DiagnosticsCallback,
    ProgressBarCallback,
    ReadableCheckpointCallback,
    apply_he_init,
    build_metadata,
    build_env_fn,
    build_reward_stages_callback,
    load_track_seed,
    save_readable_model,
)


def build_pretrain_config(
    config_path: str,
    output_dir: str,
    shared_track_seed: int,
) -> str:
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    track_cfg = config.setdefault("track", {})
    track_cfg["track_mode"] = "random"
    track_cfg["seed"] = int(shared_track_seed)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "pretrain_oval_config.yaml")
    with open(out_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    return out_path


def make_vector_env_shared_track(
    config_path: str,
    num_envs: int,
    obs_scale: float,
    throttle_bias: float,
    shared_track_seed: int,
    frame_stack: int = 1,
) -> VecEnv:
    env_fns: List[Callable[[], RaceCarEnv]] = []
    for _ in range(num_envs):
        env_fns.append(
            build_env_fn(
                config_path=config_path,
                gui=False,
                obs_scale=obs_scale,
                throttle_bias=throttle_bias,
                track_seed=shared_track_seed,
                cache_track=True,
                random_spawn=True,
                terminate_off_track=False,  # non-terminal during training
            )
        )
    if num_envs <= 1:
        vec_env: VecEnv = DummyVecEnv(env_fns)
    else:
        vec_env = SubprocVecEnv(env_fns)
    if frame_stack > 1:
        vec_env = VecFrameStack(vec_env, n_stack=frame_stack, channels_order="first")
    return vec_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pre-train SAC on a shared oval track."
    )
    default_config = os.path.join(
        os.path.dirname(__file__), "models", "track_config.yaml"
    )
    parser.add_argument("--config", type=str, default=default_config)
    parser.add_argument("--total-timesteps", type=int, default=500_000)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--tensorboard-log", type=str, default="runs")
    parser.add_argument("--checkpoint-frequency", type=int, default=100_000)
    parser.add_argument("--num-envs", type=int, default=1)
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
    parser.add_argument("--buffer-size", type=int, default=200_000)
    parser.add_argument("--learning-starts", type=int, default=25_000)
    parser.add_argument("--train-freq", type=int, default=1)
    parser.add_argument("--gradient-steps", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--throttle-bias", type=float, default=0.0)
    parser.add_argument("--throttle-bias-anneal-frac", type=float, default=0.5)
    parser.add_argument("--ent-start-mult", type=float, default=4.0)
    parser.add_argument("--ent-anneal-frac", type=float, default=0.9)
    parser.add_argument("--progress-anneal-frac", type=float, default=0.0)
    parser.add_argument("--disable-reward-stages", action="store_true")
    parser.add_argument("--reward-stage-fracs", type=str, default="0.35,0.35,0.30")
    parser.add_argument("--stage-progress-weights", type=str, default="1.0,1.0,1.0")
    parser.add_argument("--stage-speed-weights", type=str, default="0.0,0.05,0.1")
    parser.add_argument("--stage-steer-penalties", type=str, default="0.0,0.01,0.02")
    parser.add_argument("--stage-gate-rewards", type=str, default="0.0,0.5,1.0")
    parser.add_argument(
        "--stage-off-track-penalties", type=str, default="-1.0,-1.0,-1.0"
    )
    parser.add_argument(
        "--shared-track-seed",
        type=int,
        default=None,
        help="Seed used for every env so all workers train on the same track.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run_name = args.run_name or datetime.utcnow().strftime("pre_sac_racey_%Y%m%d_%H%M%S")
    checkpoint_dir = os.path.join(args.checkpoint_dir, run_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    loaded_seed = load_track_seed(args.config)
    base_seed = (
        int(args.shared_track_seed)
        if args.shared_track_seed is not None
        else (loaded_seed if loaded_seed is not None else 0)
    )
    pretrain_config = build_pretrain_config(args.config, checkpoint_dir, base_seed)
    args.config = pretrain_config

    num_envs = max(1, args.num_envs)
    env = make_vector_env_shared_track(
        pretrain_config,
        num_envs,
        args.obs_scale,
        args.throttle_bias,
        shared_track_seed=base_seed,
        frame_stack=args.frame_stack,
    )

    device = "mps" if torch.backends.mps.is_available() else "auto"

    base_metadata = build_metadata(args, run_name, num_envs, device)
    base_metadata["training_stage"] = "pre"
    base_metadata["shared_track_seed"] = int(base_seed)
    base_metadata["track_mode"] = "oval"
    base_metadata["reward_stage_fracs"] = args.reward_stage_fracs
    base_metadata["stage_progress_weights"] = args.stage_progress_weights
    base_metadata["stage_speed_weights"] = args.stage_speed_weights
    base_metadata["stage_steer_penalties"] = args.stage_steer_penalties
    base_metadata["stage_gate_rewards"] = args.stage_gate_rewards
    base_metadata["stage_off_track_penalties"] = args.stage_off_track_penalties
    base_metadata["disable_reward_stages"] = bool(args.disable_reward_stages)

    checkpoint_callback = ReadableCheckpointCallback(
        save_freq=max(1, args.checkpoint_frequency // env.num_envs),
        save_path=checkpoint_dir,
        name_prefix="sac_racey_pre",
        base_metadata=base_metadata,
    )
    progress_callback = ProgressBarCallback(args.total_timesteps)
    diagnostics_callback = DiagnosticsCallback()
    bias_anneal_steps = max(1, int(args.total_timesteps * args.throttle_bias_anneal_frac))
    ent_anneal_steps = max(1, int(args.total_timesteps * args.ent_anneal_frac))
    bias_entropy_callback = BiasAndEntropyAnnealCallback(
        bias_start=args.throttle_bias,
        bias_anneal_steps=bias_anneal_steps,
        ent_start_mult=args.ent_start_mult,
        ent_anneal_steps=ent_anneal_steps,
    )
    reward_stages_callback = build_reward_stages_callback(args)
    callback_items = [
        checkpoint_callback,
        progress_callback,
        diagnostics_callback,
        bias_entropy_callback,
    ]
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
    final_model_path = os.path.join(checkpoint_dir, "sac_racey_pre_final")
    final_metadata = dict(base_metadata)
    final_metadata["checkpoint_timesteps"] = int(model.num_timesteps)
    final_metadata["final"] = True
    save_readable_model(model, final_model_path, final_metadata)
    print(f"Pre-training complete. Model saved to {final_model_path}")


if __name__ == "__main__":
    main()
