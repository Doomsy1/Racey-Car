import argparse
import json
import os
from datetime import datetime

import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.utils import get_schedule_fn

from train_sac import (
    BiasAndEntropyAnnealCallback,
    DiagnosticsCallback,
    ProgressBarCallback,
    ReadableCheckpointCallback,
    build_metadata,
    build_reward_stages_callback,
    make_vector_env,
    save_readable_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Post-train SAC by resuming from a saved model."
    )
    default_config = os.path.join(
        os.path.dirname(__file__), "models", "track_config.yaml"
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--config", type=str, default=default_config)
    parser.add_argument("--total-timesteps", type=int, default=500_000)
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-3,
        help="Fine-tuning LR (defaults lower than base training script).",
    )
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
        "--output-model",
        type=str,
        default=None,
        help="Override path for the final saved model directory (skips checkpoint_dir subdirectory).",
    )
    return parser.parse_args()


def _apply_learning_rate(model: SAC, learning_rate: float) -> None:
    model.learning_rate = learning_rate
    model.lr_schedule = get_schedule_fn(learning_rate)
    for optimizer_name in ("actor", "critic"):
        module = getattr(model, optimizer_name, None)
        optimizer = getattr(module, "optimizer", None)
        if optimizer is None:
            continue
        for param_group in optimizer.param_groups:
            param_group["lr"] = learning_rate
    if getattr(model, "ent_coef_optimizer", None) is not None:
        for param_group in model.ent_coef_optimizer.param_groups:
            param_group["lr"] = learning_rate


def load_post_model(model_path: str, env, device: str, args: argparse.Namespace) -> SAC:
    if os.path.isdir(model_path):
        zip_path = os.path.join(model_path, "model.zip")
        buffer_path = os.path.join(model_path, "replay_buffer.pkl")

        if os.path.exists(zip_path):
            # Full restore: preserves optimizer momentum and replay experience.
            model = SAC.load(
                zip_path,
                env=env,
                device=device,
                custom_objects={
                    "learning_rate": args.learning_rate,
                    "lr_schedule": get_schedule_fn(args.learning_rate),
                },
            )
            model.verbose = 1
            model.tensorboard_log = os.path.join(args.tensorboard_log, args.run_name)
            if os.path.exists(buffer_path):
                try:
                    model.load_replay_buffer(buffer_path)
                    # Buffer is pre-populated; skip random-action warmup.
                    model.learning_starts = 0
                except Exception as exc:
                    print(f"[load_post_model] Could not load replay buffer: {exc}")
            _apply_learning_rate(model, args.learning_rate)
            return model

        # Fallback: policy weights only (no optimizer state or replay buffer).
        metadata_path = os.path.join(model_path, "metadata.json")
        weights_path = os.path.join(model_path, "policy_state.pt")
        if not (os.path.exists(metadata_path) and os.path.exists(weights_path)):
            raise FileNotFoundError(
                f"Readable model directory missing metadata.json/policy_state.pt: {model_path}"
            )
        with open(metadata_path, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        policy_id = metadata.get("policy", "CnnPolicy")
        model = SAC(
            policy_id,
            env,
            verbose=1,
            learning_rate=args.learning_rate,
            tensorboard_log=os.path.join(args.tensorboard_log, args.run_name),
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
        state_dict = torch.load(weights_path, map_location=device)
        model.policy.load_state_dict(state_dict)
        _apply_learning_rate(model, args.learning_rate)
        return model

    model = SAC.load(model_path, env=env, device=device)
    _apply_learning_rate(model, args.learning_rate)
    return model


def main() -> None:
    args = parse_args()
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model path not found: {args.model}")

    run_name = args.run_name or datetime.utcnow().strftime("post_sac_racey_%Y%m%d_%H%M%S")
    args.run_name = run_name
    checkpoint_dir = os.path.join(args.checkpoint_dir, run_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    num_envs = max(1, args.num_envs)
    env = make_vector_env(
        args.config, num_envs, args.obs_scale, args.throttle_bias, args.frame_stack
    )

    device = "mps" if torch.backends.mps.is_available() else "auto"

    base_metadata = build_metadata(args, run_name, num_envs, device)
    base_metadata["training_stage"] = "post"
    base_metadata["source_model"] = args.model
    base_metadata["reward_stage_fracs"] = args.reward_stage_fracs
    base_metadata["stage_progress_weights"] = args.stage_progress_weights
    base_metadata["stage_speed_weights"] = args.stage_speed_weights
    base_metadata["stage_steer_penalties"] = args.stage_steer_penalties
    base_metadata["stage_gate_rewards"] = args.stage_gate_rewards
    base_metadata["stage_off_track_penalties"] = args.stage_off_track_penalties
    base_metadata["disable_reward_stages"] = bool(args.disable_reward_stages)

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
        progress_callback,
        diagnostics_callback,
        bias_entropy_callback,
    ]
    if args.checkpoint_frequency > 0:
        checkpoint_callback = ReadableCheckpointCallback(
            save_freq=max(1, args.checkpoint_frequency // env.num_envs),
            save_path=checkpoint_dir,
            name_prefix="sac_racey_post",
            base_metadata=base_metadata,
        )
        callback_items.insert(0, checkpoint_callback)
    if reward_stages_callback is not None:
        callback_items.append(reward_stages_callback)
    callbacks: CallbackList = CallbackList(callback_items)

    model = load_post_model(args.model, env, device, args)

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callbacks,
        reset_num_timesteps=False,
    )

    env.close()
    final_model_path = args.output_model if args.output_model else os.path.join(checkpoint_dir, "sac_racey_post_final")
    final_metadata = dict(base_metadata)
    final_metadata["checkpoint_timesteps"] = int(model.num_timesteps)
    final_metadata["final"] = True
    save_readable_model(model, final_model_path, final_metadata)
    print(f"Post-training complete. Model saved to {final_model_path}")


if __name__ == "__main__":
    main()
