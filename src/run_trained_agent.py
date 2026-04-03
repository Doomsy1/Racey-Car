import argparse
import json
import os
import time

import numpy as np
import torch
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecFrameStack

from environment.race_env import RaceCarEnv


def _load_readable_metadata(model_path: str) -> dict | None:
    if not os.path.isdir(model_path):
        return None
    metadata_path = os.path.join(model_path, "metadata.json")
    if not os.path.exists(metadata_path):
        return None
    with open(metadata_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def build_eval_env(
    config_path: str,
    obs_scale: float,
    max_episode_duration: float | None,
    terminate_off_track: bool,
    frame_stack: int,
) -> VecEnv:
    vec_env: VecEnv = DummyVecEnv(
        [
            lambda: RaceCarEnv(
                config_path=config_path,
                gui=True,
                observation_scale=obs_scale,
                max_episode_duration=max_episode_duration,
                terminate_off_track=terminate_off_track,
                cache_track=True,
            )
        ]
    )
    if frame_stack > 1:
        vec_env = VecFrameStack(vec_env, n_stack=frame_stack, channels_order="first")
    return vec_env


def load_model(model_path: str, env: VecEnv, device: str, algo: str) -> SAC | PPO:
    algo = algo.lower()
    if os.path.isdir(model_path):
        metadata_path = os.path.join(model_path, "metadata.json")
        weights_path = os.path.join(model_path, "policy_state.pt")
        if not (os.path.exists(metadata_path) and os.path.exists(weights_path)):
            raise FileNotFoundError(
                f"Readable model directory missing metadata.json/policy_state.pt: {model_path}"
            )
        with open(metadata_path, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        policy_id = metadata.get("policy", "CnnPolicy")
        algo = metadata.get("algorithm", algo).lower()
        cls = PPO if algo == "ppo" else SAC
        model = cls(policy_id, env, device=device, verbose=0)
        state_dict = torch.load(weights_path, map_location=device)
        model.policy.load_state_dict(state_dict)
        return model
    if algo == "ppo":
        return PPO.load(model_path, env=env, device=device)
    return SAC.load(model_path, env=env, device=device)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a trained PPO/SAC policy in the simulator GUI.")
    default_config = os.path.join(
        os.path.dirname(__file__), "models", "track_config_eval_figure8.yaml"
    )
    parser.add_argument(
        "--model",
        required=True,
        type=str,
        help="Path to the saved PPO/SAC model directory (metadata.json + policy_state.pt) or SB3 .zip.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=default_config,
        help="Configuration file used during training.",
    )
    parser.add_argument(
        "--obs-scale",
        type=float,
        default=0.25,
        help="Scale factor (0 < scale <= 1) for the driver FOV occupancy map. Should match training.",
    )
    parser.add_argument(
        "--frame-stack",
        type=int,
        default=4,
        help="Number of consecutive observations stacked for temporal context.",
    )
    parser.add_argument(
        "--max-episode-duration",
        type=float,
        default=None,
        help="Episode time limit in seconds; set to None to disable timeout so the agent drives until you quit.",
    )
    parser.add_argument(
        "--terminate-off-track",
        action="store_true",
        help="If set, terminate the episode when the car leaves the track (default: keep driving in eval).",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic actions instead of sampling from the policy distribution.",
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="sac",
        choices=["ppo", "sac"],
        help="Algorithm used to train the model.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model file not found: {args.model}")

    metadata = _load_readable_metadata(args.model)
    if metadata is not None and "obs_scale" in metadata:
        trained_obs_scale = float(metadata["obs_scale"])
        if abs(trained_obs_scale - float(args.obs_scale)) > 1e-9:
            print(
                "Overriding --obs-scale with trained value from metadata: "
                f"{trained_obs_scale}"
            )
            args.obs_scale = trained_obs_scale
    if metadata is not None and "frame_stack" in metadata:
        trained_frame_stack = int(metadata["frame_stack"])
        if trained_frame_stack != int(args.frame_stack):
            print(
                "Overriding --frame-stack with trained value from metadata: "
                f"{trained_frame_stack}"
            )
            args.frame_stack = trained_frame_stack

    vec_env = build_eval_env(
        args.config,
        args.obs_scale,
        args.max_episode_duration,
        args.terminate_off_track,
        args.frame_stack,
    )
    device = "mps" if torch.backends.mps.is_available() else "auto"
    model = load_model(args.model, vec_env, device, args.algo)
    obs = vec_env.reset()
    episode_idx = 0
    deterministic_eval = args.deterministic
    stalled_steps = 0

    while True:
        action, _ = model.predict(obs, deterministic=deterministic_eval)
        if deterministic_eval and np.all(np.abs(action) < 1e-3):
            stalled_steps += 1
            if stalled_steps >= 200:
                print("Deterministic actions are near zero; switching to stochastic sampling.")
                deterministic_eval = False
                stalled_steps = 0
        else:
            stalled_steps = 0
        obs, rewards, dones, infos = vec_env.step(action)

        if dones[0]:
            lap_time = infos[0].get("lap_time")
            event = infos[0].get("event")
            print(f"Episode {episode_idx} finished: event={event}, lap_time={lap_time}")
            episode_idx += 1
            time.sleep(0.5)
            obs = vec_env.reset()


if __name__ == "__main__":
    main()
