from __future__ import annotations
import contextlib
import copy
import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical, kl_divergence
from .models import ImageCodec, RSSM, PredictionHeads, Actor, TwoHotCritic, symlog, symexp
from .replay_buffer import EpisodeReplayBuffer

DEFAULT_CONFIG = {
    "feature_dim": 256,
    "hidden_dim": 256,
    "gru_dim": 256,
    "stoch_categories": 16,
    "stoch_classes": 32,
    "imagination_horizon": 8,
    "batch_size": 16,
    "seq_len": 16,
    "discount": 0.997,
    "lambda_": 0.95,
    "world_model_lr": 1e-4,
    "actor_lr": 3e-5,
    "critic_lr": 3e-5,
    "max_grad_norm": 100.0,
    "free_bits": 1.0,
    "kl_balance": 0.8,
    "critic_ema_tau": 0.02,
    "entropy_scale": 3e-3,
    "weight_decay": 1e-6,
    "free_bits_warmup_steps": 50_000,
    "no_compile": False,
}

@dataclass
class EnvSpec:
    """Bundles environment geometry so it can be passed as a single argument."""
    obs_shape: tuple
    action_dim: int
    action_low: np.ndarray
    action_high: np.ndarray
    def to_json(self) -> dict:
        return {"obs_shape": list(self.obs_shape), "action_dim": self.action_dim,
                "action_low": self.action_low.tolist(), "action_high": self.action_high.tolist()}
    @classmethod
    def from_json(cls, d: dict) -> EnvSpec:
        return cls(obs_shape=tuple(d["obs_shape"]), action_dim=d["action_dim"],
                   action_low=np.array(d["action_low"], dtype=np.float32),
                   action_high=np.array(d["action_high"], dtype=np.float32))

def _select_device() -> str:
    """Pick the best available torch device (MPS > CUDA > CPU)."""
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _configure_device_runtime(device: str) -> None:
    """Enable safe runtime flags for supported accelerator backends."""
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
    if device == "mps":
        import os as _os

        _os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def _autocast_ctx(device: str):
    """Return the appropriate autocast context for the device."""
    if device == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    if device == "mps":
        return torch.autocast("mps", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def _maybe_compile(module: torch.nn.Module, device: str) -> torch.nn.Module:
    """Attempt torch.compile; fall back silently on unsupported platforms."""
    try:
        return torch.compile(module, mode="reduce-overhead", fullgraph=False)
    except Exception:
        return module


def _module_for_io(module: torch.nn.Module) -> torch.nn.Module:
    """Unwrap compiled modules for stable save/load state dict handling."""
    return getattr(module, "_orig_mod", module)


def _reset_policy_state_map(
    state_by_env: dict[int, tuple[torch.Tensor, torch.Tensor]], env_id: int | None = None,
) -> None:
    if env_id is None:
        state_by_env.clear()
        return
    state_by_env.pop(env_id, None)


@torch.no_grad()
def _policy_batch_from_modules(
    codec: torch.nn.Module,
    rssm: RSSM,
    actor: Actor,
    state_by_env: dict[int, tuple[torch.Tensor, torch.Tensor]],
    obs_batch: np.ndarray,
    device: str,
    env_ids: list[int] | None = None,
    deterministic: bool = False,
) -> np.ndarray:
    obs_t = torch.as_tensor(obs_batch, device=device)
    if obs_t.ndim == 3:
        obs_t = obs_t.unsqueeze(0)
    batch_size = int(obs_t.shape[0])
    if env_ids is None:
        env_ids = list(range(batch_size))
    if len(env_ids) != batch_size:
        raise ValueError("env_ids length must match observation batch size")

    states = [state_by_env.get(env_id) for env_id in env_ids]
    if any(state is None for state in states):
        missing = [env_id for env_id, state in zip(env_ids, states) if state is None]
        init_h, init_z = rssm.initial_state(len(missing), device)
        for idx, env_id in enumerate(missing):
            state_by_env[env_id] = (init_h[idx : idx + 1], init_z[idx : idx + 1])
        states = [state_by_env[env_id] for env_id in env_ids]

    h = torch.cat([state[0] for state in states], dim=0)
    z = torch.cat([state[1] for state in states], dim=0)
    enc = codec.encode(obs_t)
    post_logits = rssm.posterior(h, enc)
    z_post = RSSM.straight_through_sample(post_logits)
    feat = rssm.get_features(h, z_post)
    action, _, _ = actor(feat, deterministic=deterministic)
    next_h = rssm.img_step(h, z_post, action)
    for idx, env_id in enumerate(env_ids):
        state_by_env[env_id] = (
            next_h[idx : idx + 1].detach(),
            z_post[idx : idx + 1].detach(),
        )
    return action.cpu().numpy()


def _compute_lambda_returns(rewards: torch.Tensor, values: torch.Tensor,
                            continues: torch.Tensor, discount: float, lambda_: float) -> torch.Tensor:
    """GAE-style lambda-returns. All inputs shape (B, T)."""
    B, T = rewards.shape
    returns = torch.zeros_like(rewards)
    last_val = values[:, -1]
    for t in reversed(range(T)):
        next_val = last_val if t == T - 1 else (1 - lambda_) * values[:, t + 1] + lambda_ * returns[:, t + 1]
        returns[:, t] = rewards[:, t] + discount * continues[:, t] * next_val
    return returns


class ReturnNormalizer:
    """Running percentile normalizer for Dreamer-style advantage scaling."""

    def __init__(self, decay: float = 0.99, low: float = 0.05, high: float = 0.95):
        self.decay = decay
        self.low = low
        self.high = high
        self._low_ema: float | None = None
        self._high_ema: float | None = None

    def state_dict(self) -> dict:
        return {"low_ema": self._low_ema, "high_ema": self._high_ema}

    def load_state_dict(self, state: dict | None) -> None:
        if not state:
            return
        self._low_ema = state.get("low_ema")
        self._high_ema = state.get("high_ema")

    def update_and_scale(self, returns: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            p_low = torch.quantile(returns.float(), self.low).item()
            p_high = torch.quantile(returns.float(), self.high).item()
        if self._low_ema is None or self._high_ema is None:
            self._low_ema, self._high_ema = p_low, p_high
        else:
            self._low_ema = self.decay * self._low_ema + (1.0 - self.decay) * p_low
            self._high_ema = self.decay * self._high_ema + (1.0 - self.decay) * p_high
        scale = max(abs(self._high_ema - self._low_ema), 1.0)
        return (returns - self._low_ema) / scale


class DreamerPolicy:
    """Inference-only Dreamer policy used by async collectors."""

    def __init__(self, env_spec: EnvSpec, device: str = "cpu", config: dict | None = None):
        self.env_spec = env_spec
        self.obs_shape = env_spec.obs_shape
        self.action_dim = env_spec.action_dim
        self.device = device
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        c = self.config
        latent_dim = c["gru_dim"] + c["stoch_categories"] * c["stoch_classes"]
        self.codec = ImageCodec(
            in_channels=self.obs_shape[0], feature_dim=c["feature_dim"], latent_dim=latent_dim,
        ).to(self.device)
        self.rssm = RSSM(
            action_dim=self.action_dim, gru_dim=c["gru_dim"],
            stoch_dims=(c["stoch_categories"], c["stoch_classes"]),
            feature_dim=c["feature_dim"], hidden_dim=c["hidden_dim"],
        ).to(self.device)
        self.actor = Actor(latent_dim=latent_dim, hidden_dim=c["hidden_dim"], action_dim=self.action_dim).to(self.device)
        self._policy_state_by_env: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def load_policy_state(self, state: dict[str, dict[str, torch.Tensor]]) -> None:
        self.codec.load_state_dict(state["codec"])
        self.rssm.load_state_dict(state["rssm"])
        self.actor.load_state_dict(state["actor"])
        self.reset_policy_state()

    def policy_batch(
        self, obs_batch: np.ndarray, env_ids: list[int] | None = None, deterministic: bool = False,
    ) -> np.ndarray:
        return _policy_batch_from_modules(
            self.codec, self.rssm, self.actor, self._policy_state_by_env,
            obs_batch, self.device, env_ids=env_ids, deterministic=deterministic,
        )

    def reset_policy_state(self, env_id: int | None = None):
        _reset_policy_state_map(self._policy_state_by_env, env_id=env_id)


class DreamerV3Agent:
    """Self-contained DreamerV3 agent for continuous-action environments."""

    def __init__(self, env_spec: EnvSpec, device: str | None = None, config: dict | None = None):
        self.env_spec = env_spec
        self.obs_shape = env_spec.obs_shape
        self.action_dim = env_spec.action_dim
        self.action_low = np.asarray(env_spec.action_low, dtype=np.float32)
        self.action_high = np.asarray(env_spec.action_high, dtype=np.float32)
        self.device = device or _select_device()
        _configure_device_runtime(self.device)
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        c = self.config
        latent_dim = c["gru_dim"] + c["stoch_categories"] * c["stoch_classes"]
        self.codec = ImageCodec(
            in_channels=self.obs_shape[0], feature_dim=c["feature_dim"], latent_dim=latent_dim,
        ).to(self.device)
        self.rssm = RSSM(
            action_dim=self.action_dim, gru_dim=c["gru_dim"],
            stoch_dims=(c["stoch_categories"], c["stoch_classes"]),
            feature_dim=c["feature_dim"], hidden_dim=c["hidden_dim"],
        ).to(self.device)
        self.heads = PredictionHeads(latent_dim=latent_dim, hidden_dim=c["hidden_dim"]).to(self.device)
        self.actor = Actor(latent_dim=latent_dim, hidden_dim=c["hidden_dim"], action_dim=self.action_dim).to(self.device)
        self.critic = TwoHotCritic(latent_dim=latent_dim, hidden_dim=c["hidden_dim"]).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self._return_norm = ReturnNormalizer()
        if not c.get("no_compile", False):
            self.codec = _maybe_compile(self.codec, self.device)
            self.rssm = _maybe_compile(self.rssm, self.device)
            self.heads = _maybe_compile(self.heads, self.device)
            self.actor = _maybe_compile(self.actor, self.device)
            self.critic = _maybe_compile(self.critic, self.device)
            self.critic_target = _maybe_compile(self.critic_target, self.device)
        for p in self.critic_target.parameters():
            p.requires_grad_(False)
        self._build_optimizers()
        self._policy_state_by_env: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def _build_optimizers(self):
        c = self.config
        wm_params = list(self.codec.parameters()) + list(self.rssm.parameters()) + list(self.heads.parameters())
        self.wm_opt = torch.optim.AdamW(wm_params, lr=c["world_model_lr"], weight_decay=c["weight_decay"])
        self.actor_opt = torch.optim.AdamW(self.actor.parameters(), lr=c["actor_lr"], weight_decay=c["weight_decay"])
        self.critic_opt = torch.optim.AdamW(self.critic.parameters(), lr=c["critic_lr"], weight_decay=c["weight_decay"])

    # ------ inference ------
    @torch.no_grad()
    def policy(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Select an action given a single (C,H,W) uint8 observation."""
        return self.policy_batch(np.expand_dims(obs, axis=0), env_ids=[0], deterministic=deterministic)[0]

    @torch.no_grad()
    def policy_batch(
        self, obs_batch: np.ndarray, env_ids: list[int] | None = None, deterministic: bool = False,
    ) -> np.ndarray:
        """Select actions for a batch of observations while keeping per-env RSSM state."""
        return _policy_batch_from_modules(
            self.codec, self.rssm, self.actor, self._policy_state_by_env,
            obs_batch, self.device, env_ids=env_ids, deterministic=deterministic,
        )

    def reset_policy_state(self, env_id: int | None = None):
        _reset_policy_state_map(self._policy_state_by_env, env_id=env_id)

    def export_policy_state(self) -> dict[str, dict[str, torch.Tensor]]:
        """Return CPU policy weights for async collector snapshots."""
        return {
            "codec": {k: v.detach().cpu() for k, v in _module_for_io(self.codec).state_dict().items()},
            "rssm": {k: v.detach().cpu() for k, v in _module_for_io(self.rssm).state_dict().items()},
            "actor": {k: v.detach().cpu() for k, v in _module_for_io(self.actor).state_dict().items()},
        }

    # ------ training ------
    def update(self, replay_buffer: EpisodeReplayBuffer, global_step: int = 0) -> dict:
        c = self.config
        warmup = c.get("free_bits_warmup_steps", 0)
        if warmup > 0 and global_step < warmup:
            free_bits = c["free_bits"] * (global_step / warmup)
        else:
            free_bits = c["free_bits"]
        batch, ep_indices = replay_buffer.sample_sequences(c["batch_size"], c["seq_len"], self.device)
        wm_metrics = self._train_world_model(batch, free_bits=free_bits)
        recon_loss = wm_metrics.get("recon_loss", 1.0)
        if hasattr(replay_buffer, "update_priorities"):
            replay_buffer.update_priorities(ep_indices, [recon_loss] * len(ep_indices))
        h_posts, z_posts = wm_metrics.pop("_h_posts"), wm_metrics.pop("_z_posts")
        ac_metrics = self._train_actor_critic(h_posts.detach(), z_posts.detach())
        self._update_critic_target()
        return {**wm_metrics, **ac_metrics, "free_bits": free_bits}

    def update_world_model_only(self, replay_buffer: EpisodeReplayBuffer, global_step: int = 0) -> dict:
        """Train only the world model (encoder, RSSM, prediction heads). No actor/critic."""
        c = self.config
        warmup = c.get("free_bits_warmup_steps", 0)
        free_bits = c["free_bits"] * (global_step / warmup) if warmup > 0 and global_step < warmup else c["free_bits"]
        batch, ep_indices = replay_buffer.sample_sequences(c["batch_size"], c["seq_len"], self.device)
        wm_metrics = self._train_world_model(batch, free_bits=free_bits)
        recon_loss = wm_metrics.get("recon_loss", 1.0)
        if hasattr(replay_buffer, "update_priorities"):
            replay_buffer.update_priorities(ep_indices, [recon_loss] * len(ep_indices))
        wm_metrics.pop("_h_posts", None)
        wm_metrics.pop("_z_posts", None)
        return {**wm_metrics, "free_bits": free_bits}

    def update_many(self, replay_buffer: EpisodeReplayBuffer, num_updates: int, global_step: int = 0,
                    world_model_only: bool = False) -> dict:
        last = {}
        for _ in range(num_updates):
            if world_model_only:
                last = self.update_world_model_only(replay_buffer, global_step=global_step)
            else:
                last = self.update(replay_buffer, global_step=global_step)
        return {**last, "num_updates": num_updates}

    def _train_world_model(self, batch: dict, free_bits: float | None = None) -> dict:
        c = self.config
        obs, actions = batch["obs"], batch["action"]       # (B,T,C,H,W), (B,T,A)
        rewards, dones = batch["reward"], batch["done"]    # (B,T), (B,T)
        B, T = obs.shape[:2]
        h, _ = self.rssm.initial_state(B, self.device)
        with _autocast_ctx(self.device):
            encodings = self.codec.encode(obs.reshape(B * T, *obs.shape[2:])).reshape(B, T, -1)
            seq_out = self.rssm.forward_sequence(
                encodings, actions, h,
                kl_fn=self._kl_terms, free_bits=free_bits,
            )
            features = seq_out["features"]
            flat_feat = features.reshape(B * T, -1)
            # Keep loss numerics in fp32; MPS rejects mixed bf16/fp32 BCE operands.
            recon_pred = self.codec.decode(flat_feat).reshape(B, T, *obs.shape[2:])
            reward_pred = self.heads.reward(flat_feat).reshape(B, T)
            cont_pred = self.heads.continuation(flat_feat).reshape(B, T)
        loss_recon = F.mse_loss(recon_pred.float(), symlog(obs.float() / 255.0))
        loss_reward = F.mse_loss(reward_pred.float(), symlog(rewards).float())
        loss_continue = F.binary_cross_entropy(cont_pred.float(), 1.0 - dones.float())
        raw_kl = torch.stack(seq_out["raw_kl_losses"]).mean()
        loss_kl = torch.stack(seq_out["clamped_kl_losses"]).mean()
        total = loss_recon + loss_reward + loss_continue + loss_kl
        self.wm_opt.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.codec.parameters()) + list(self.rssm.parameters()) + list(self.heads.parameters()),
            c["max_grad_norm"],
        )
        self.wm_opt.step()
        return {
            "wm_loss": total.item(), "recon_loss": loss_recon.item(),
            "reward_loss": loss_reward.item(), "continue_loss": loss_continue.item(),
            "kl_loss": loss_kl.item(),
            "raw_kl_loss": raw_kl.item(),
            "clamped_kl_loss": loss_kl.item(),
            "_h_posts": seq_out["h_posts"],
            "_z_posts": seq_out["z_posts"],
        }

    def _kl_terms(
        self, post_logits: torch.Tensor, prior_logits: torch.Tensor, free_bits: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return both the raw and free-bits-clamped balanced KL losses."""
        c = self.config
        fb = c["free_bits"] if free_bits is None else free_bits
        post_dist = Categorical(logits=post_logits)
        prior_dist = Categorical(logits=prior_logits)
        sg_prior = Categorical(logits=prior_logits.detach())
        kl_fwd_raw = kl_divergence(post_dist, sg_prior).sum(-1).mean()
        sg_post = Categorical(logits=post_logits.detach())
        kl_rev_raw = kl_divergence(sg_post, prior_dist).sum(-1).mean()
        raw_kl = c["kl_balance"] * kl_fwd_raw + (1.0 - c["kl_balance"]) * kl_rev_raw
        kl_fwd = torch.clamp(kl_fwd_raw, min=fb)
        kl_rev = torch.clamp(kl_rev_raw, min=fb)
        clamped_kl = c["kl_balance"] * kl_fwd + (1.0 - c["kl_balance"]) * kl_rev
        return raw_kl, clamped_kl

    def _kl_loss(
        self, post_logits: torch.Tensor, prior_logits: torch.Tensor, free_bits: float | None = None,
    ) -> torch.Tensor:
        """KL divergence with free-bits and balancing across 32 category groups."""
        _, clamped_kl = self._kl_terms(post_logits, prior_logits, free_bits=free_bits)
        return clamped_kl

    def _train_actor_critic(self, h_posts: torch.Tensor, z_posts: torch.Tensor) -> dict:
        """Imagine trajectories from posterior states and train actor + critic."""
        c = self.config
        B, T_wm = h_posts.shape[:2]
        flat_idx = torch.randint(T_wm, (B,), device=self.device)
        batch_idx = torch.arange(B, device=self.device)
        h, z = h_posts[batch_idx, flat_idx], z_posts[batch_idx, flat_idx]
        with _autocast_ctx(self.device):
            imag = self.rssm.imagine_trajectory(
                h, z, self.actor, self.heads, c["imagination_horizon"],
            )
        feats_t = imag["features"]                                     # (B, H, latent)
        rewards_t = symexp(imag["rewards"]).float()                    # (B, H)
        continues_t = imag["continues"].float()                        # (B, H)
        log_probs_t = imag["log_probs"].float()                        # (B, H)
        base_entropy_t = imag["base_entropy"].float()                    # (B, H)
        with torch.no_grad():
            value_logits = self.critic_target.mean(feats_t)
            values_target = symexp(value_logits)
            baseline = value_logits.detach()
        lambda_ret = _compute_lambda_returns(rewards_t, values_target, continues_t, c["discount"], c["lambda_"])
        critic_loss = self.critic.loss(feats_t.detach(), symlog(lambda_ret.detach()))
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), c["max_grad_norm"])
        self.critic_opt.step()
        advantage = self._return_norm.update_and_scale((symlog(lambda_ret) - baseline).detach())
        # Policy gradient uses Jacobian-corrected log_probs; entropy uses
        # analytic Gaussian entropy (well-behaved, no Jacobian explosion).
        actor_loss = -(advantage * log_probs_t).mean() - c["entropy_scale"] * base_entropy_t.mean()
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), c["max_grad_norm"])
        self.actor_opt.step()
        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "entropy": base_entropy_t.mean().item(),
            "imagined_reward": rewards_t.mean().item(),
            "ret_norm_low": self._return_norm._low_ema or 0.0,
            "ret_norm_high": self._return_norm._high_ema or 0.0,
        }

    def _update_critic_target(self):
        tau = self.config["critic_ema_tau"]
        for tp, sp in zip(self.critic_target.parameters(), self.critic.parameters()):
            tp.data.lerp_(sp.data, tau)

    # ------ save / load ------
    def _metadata(self, deployment: bool = False) -> dict:
        meta = self.env_spec.to_json()
        meta["config"] = self.config
        meta["deployment"] = deployment
        return meta

    def save(self, path: str):
        """Save full agent state (all networks + optimizers) to a directory."""
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        torch.save({
            "codec": _module_for_io(self.codec).state_dict(),
            "rssm": _module_for_io(self.rssm).state_dict(),
            "heads": _module_for_io(self.heads).state_dict(),
            "actor": _module_for_io(self.actor).state_dict(),
            "critic": _module_for_io(self.critic).state_dict(),
            "critic_target": _module_for_io(self.critic_target).state_dict(),
            "wm_opt": self.wm_opt.state_dict(), "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(), "config": self.config,
            "return_norm": self._return_norm.state_dict(),
        }, p / "full_agent.pt")
        (p / "metadata.json").write_text(json.dumps(self._metadata(), indent=2))

    def save_actor_only(self, path: str):
        """Save lightweight deployment bundle (encoder + RSSM + actor only)."""
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        torch.save({
            "codec": _module_for_io(self.codec).state_dict(),
            "rssm": _module_for_io(self.rssm).state_dict(),
            "actor": _module_for_io(self.actor).state_dict(),
            "config": self.config,
        }, p / "actor_only.pt")
        (p / "metadata.json").write_text(json.dumps(self._metadata(deployment=True), indent=2))

    @classmethod
    def _from_metadata(cls, meta: dict, device: str | None) -> DreamerV3Agent:
        """Construct an agent from saved metadata."""
        return cls(env_spec=EnvSpec.from_json(meta), device=device or _select_device(), config=meta["config"])

    @classmethod
    def load(cls, path: str, device: str | None = None) -> DreamerV3Agent:
        """Load full agent from a saved directory."""
        p = Path(path)
        meta = json.loads((p / "metadata.json").read_text())
        agent = cls._from_metadata(meta, device)
        dev = agent.device
        ckpt = torch.load(p / "full_agent.pt", map_location=dev, weights_only=False)
        _module_for_io(agent.codec).load_state_dict(ckpt["codec"])
        try:
            _module_for_io(agent.rssm).load_state_dict(ckpt["rssm"])
        except RuntimeError as exc:
            print(f"[WARNING] RSSM state_dict mismatch (old checkpoint?): {exc}")
            _module_for_io(agent.rssm).load_state_dict(ckpt["rssm"], strict=False)
        _module_for_io(agent.heads).load_state_dict(ckpt["heads"])
        _module_for_io(agent.actor).load_state_dict(ckpt["actor"])
        try:
            _module_for_io(agent.critic).load_state_dict(ckpt["critic"])
            _module_for_io(agent.critic_target).load_state_dict(ckpt["critic_target"])
        except RuntimeError as exc:
            print(f"[WARNING] critic state_dict mismatch (old checkpoint?): {exc}")
            _module_for_io(agent.critic).load_state_dict(ckpt["critic"], strict=False)
            _module_for_io(agent.critic_target).load_state_dict(ckpt["critic_target"], strict=False)
        agent.wm_opt.load_state_dict(ckpt["wm_opt"])
        agent.actor_opt.load_state_dict(ckpt["actor_opt"])
        agent.critic_opt.load_state_dict(ckpt["critic_opt"])
        agent._return_norm.load_state_dict(ckpt.get("return_norm"))
        return agent

    @classmethod
    def load_actor_only(cls, path: str, env_spec: EnvSpec, device: str | None = None) -> DreamerV3Agent:
        """Load lightweight inference-only agent (encoder + RSSM + actor)."""
        p = Path(path)
        dev = device or _select_device()
        ckpt = torch.load(p / "actor_only.pt", map_location=dev, weights_only=False)
        agent = cls(env_spec=env_spec, device=dev, config=ckpt["config"])
        _module_for_io(agent.codec).load_state_dict(ckpt["codec"])
        _module_for_io(agent.rssm).load_state_dict(ckpt["rssm"])
        _module_for_io(agent.actor).load_state_dict(ckpt["actor"])
        return agent
