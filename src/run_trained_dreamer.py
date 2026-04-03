import argparse
import json
import os
import time

import numpy as np
import torch

from environment.race_env import RaceCarEnv


def _default_eval_config_path() -> str:
    return os.path.join(
        os.path.dirname(__file__), "models", "track_config_eval_figure8.yaml",
    )

def _load_metadata(model_path: str) -> dict | None:
    if not os.path.isdir(model_path):
        return None
    meta_file = os.path.join(model_path, "metadata.json")
    if not os.path.exists(meta_file):
        return None
    with open(meta_file, "r", encoding="utf-8") as f:
        return json.load(f)

def _detect_load_mode(model_path: str) -> str:
    """Return 'actor_only' or 'full' depending on available files."""
    if os.path.exists(os.path.join(model_path, "actor_only.pt")):
        return "actor_only"
    if os.path.exists(os.path.join(model_path, "full_agent.pt")):
        return "full"
    raise FileNotFoundError(
        f"No actor_only.pt or full_agent.pt found in {model_path}"
    )

def _select_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

def _load_agent(model_path: str, device: str):
    """Load a DreamerV3 agent, preferring actor-only when available."""
    from dreamer import DreamerV3Agent
    from dreamer.agent import EnvSpec

    mode = _detect_load_mode(model_path)
    if mode == "actor_only":
        env_spec = EnvSpec(
            obs_shape=(2, 64, 64), action_dim=2,
            action_low=np.array([0.0, -1.0], dtype=np.float32),
            action_high=np.array([1.0, 1.0], dtype=np.float32),
        )
        return DreamerV3Agent.load_actor_only(model_path, env_spec, device)
    return DreamerV3Agent.load(model_path, device)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a trained DreamerV3 agent.")
    parser.add_argument("--model", required=True, type=str, help="Path to saved model directory.")
    parser.add_argument("--config", type=str, default=_default_eval_config_path())
    parser.add_argument("--obs-scale", type=float, default=0.25)
    parser.add_argument("--max-episode-duration", type=float, default=None)
    parser.add_argument("--terminate-off-track", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    return parser.parse_args()

def _apply_metadata_overrides(args: argparse.Namespace) -> None:
    """Override CLI defaults with values from metadata.json when present."""
    metadata = _load_metadata(args.model)
    if metadata is None:
        return
    if "config_path" in metadata:
        trained_config = str(metadata["config_path"])
        if args.config == _default_eval_config_path():
            print(f"Overriding --config with trained value: {trained_config}")
            args.config = trained_config
    if "obs_scale" in metadata:
        trained_scale = float(metadata["obs_scale"])
        if abs(trained_scale - args.obs_scale) > 1e-9:
            print(f"Overriding --obs-scale with trained value: {trained_scale}")
            args.obs_scale = trained_scale

def _run_eval_loop(agent, env, deterministic: bool) -> None:
    """Infinite eval loop: step, render, reset on done."""
    obs, _ = env.reset()
    agent.reset_policy_state()
    episode_idx = 0
    while True:
        action = agent.policy(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            event = info.get("event", "unknown")
            lap_time = info.get("lap_time")
            print(f"Episode {episode_idx} finished: event={event}, lap_time={lap_time}")
            episode_idx += 1
            time.sleep(0.5)
            obs, _ = env.reset()
            agent.reset_policy_state()

def main() -> None:
    args = parse_args()
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model path not found: {args.model}")
    _apply_metadata_overrides(args)
    device = _select_device()
    agent = _load_agent(args.model, device)
    env = RaceCarEnv(
        config_path=args.config,
        gui=True,
        observation_scale=args.obs_scale,
        max_episode_duration=args.max_episode_duration,
        terminate_off_track=args.terminate_off_track,
        cache_track=True,
    )
    _run_eval_loop(agent, env, args.deterministic)

if __name__ == "__main__":
    main()
