import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal


class SymlogTransform:
    """Magnitude-compressing transform: sign(x) * log(1 + |x|) and its inverse."""
    @staticmethod
    def forward(x: torch.Tensor) -> torch.Tensor:
        return torch.sign(x) * torch.log1p(torch.abs(x))
    @staticmethod
    def inverse(x: torch.Tensor) -> torch.Tensor:
        return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)

# Module-level aliases used throughout the codebase
symlog = SymlogTransform.forward
symexp = SymlogTransform.inverse

def build_mlp(
    in_dim: int, hidden_dim: int, out_dim: int, depth: int = 2, norm: bool = False,
) -> nn.Sequential:
    """Build an MLP with `depth` hidden layers (Linear + optional LayerNorm + SiLU) and a final Linear."""
    layers: list[nn.Module] = []
    prev = in_dim
    for _ in range(depth):
        layers.append(nn.Linear(prev, hidden_dim))
        if norm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.SiLU())
        prev = hidden_dim
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)


class ImageCodec(nn.Module):
    """Encodes (C,64,64) uint8 images to features and decodes features back to images."""
    _ENC_CHANNELS = [32, 64, 128, 256]
    _FLAT = 256 * 2 * 2
    def __init__(self, in_channels: int = 2, feature_dim: int = 512, latent_dim: int = 1536):
        super().__init__()
        enc_layers: list[nn.Module] = []
        prev_c = in_channels
        for c in self._ENC_CHANNELS:
            enc_layers += [nn.Conv2d(prev_c, c, kernel_size=4, stride=2), nn.SiLU()]
            prev_c = c
        self.enc_convs = nn.Sequential(*enc_layers)
        self.enc_fc = nn.Linear(self._FLAT, feature_dim)
        self.dec_fc = nn.Linear(latent_dim, self._FLAT)
        self.dec_convs = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2), nn.SiLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2), nn.SiLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2), nn.SiLU(),
            nn.ConvTranspose2d(32, in_channels, kernel_size=6, stride=2),
        )
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode (B, C, 64, 64) uint8 observation to (B, feature_dim) float features."""
        h = self.enc_convs(x.float() / 255.0)
        return self.enc_fc(h.reshape(h.shape[0], -1))
    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """Decode latent features to (B, C, 64, 64) reconstruction in symlog space."""
        return self.dec_convs(self.dec_fc(features).reshape(features.shape[0], 256, 2, 2))


class RSSM(nn.Module):
    """Recurrent State-Space Model with categorical stochastic state."""
    def __init__(self, action_dim: int = 2, gru_dim: int = 512,
                 stoch_dims: tuple = (32, 32), feature_dim: int = 512, hidden_dim: int = 512):
        super().__init__()
        self.gru_dim = gru_dim
        self.stoch_categories, self.stoch_classes = stoch_dims
        self.stoch_dim = self.stoch_categories * self.stoch_classes
        self.img_in = nn.Linear(self.stoch_dim + action_dim, hidden_dim)
        self.gru = nn.GRUCell(hidden_dim, gru_dim)
        self.gru_norm = nn.LayerNorm(gru_dim)
        self.prior_net = build_mlp(gru_dim, hidden_dim, self.stoch_dim, depth=1, norm=True)
        self.posterior_net = build_mlp(gru_dim + feature_dim, hidden_dim, self.stoch_dim, depth=1, norm=True)
    def img_step(self, h: torch.Tensor, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        next_h = self.gru(F.silu(self.img_in(torch.cat([z, action], dim=-1))), h)
        return self.gru_norm(next_h)
    def _to_logits(self, raw: torch.Tensor) -> torch.Tensor:
        return raw.reshape(-1, self.stoch_categories, self.stoch_classes)
    def prior(self, h: torch.Tensor) -> torch.Tensor:
        return self._to_logits(self.prior_net(h))
    def posterior(self, h: torch.Tensor, enc_feat: torch.Tensor) -> torch.Tensor:
        return self._to_logits(self.posterior_net(torch.cat([h, enc_feat], dim=-1)))
    def get_features(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return torch.cat([h, z], dim=-1)
    def initial_state(self, batch_size: int, device: str = "cpu") -> tuple:
        return (torch.zeros(batch_size, self.gru_dim, device=device),
                torch.zeros(batch_size, self.stoch_dim, device=device))
    @staticmethod
    def straight_through_sample(logits: torch.Tensor) -> torch.Tensor:
        """Sample one-hot from categorical but pass gradients through soft probabilities."""
        B, num_cat, num_cls = logits.shape
        dist = Categorical(logits=logits)
        hard = F.one_hot(dist.sample(), num_cls).float()
        soft = dist.probs
        return (hard + soft - soft.detach()).reshape(B, -1)

    def forward_sequence(
        self, encodings: torch.Tensor, actions: torch.Tensor,
        h: torch.Tensor, kl_fn=None, free_bits: float | None = None,
    ) -> dict:
        """Process full observation sequence, returning all posterior states and KL losses.

        Consolidates the per-timestep RSSM loop into a single method so that
        torch.compile can optimize kernel dispatch across iterations.

        Args:
            encodings: (B, T, feat_dim) encoded observations
            actions: (B, T, action_dim)
            h: (B, gru_dim) initial hidden state
            kl_fn: optional callable(post_logits, prior_logits, free_bits) -> (raw_kl, clamped_kl)
            free_bits: free-bits value for KL clamping
        Returns:
            dict with keys: h_posts (B,T,gru_dim), z_posts (B,T,stoch_dim),
            features (B,T,latent_dim), raw_kl_losses list, clamped_kl_losses list
        """
        T = encodings.shape[1]
        all_h, all_z, all_feat = [], [], []
        raw_kls, clamped_kls = [], []
        for t in range(T):
            enc = encodings[:, t]
            post_logits = self.posterior(h, enc)
            prior_logits = self.prior(h)
            z_post = self.straight_through_sample(post_logits)
            all_h.append(h)
            all_z.append(z_post)
            all_feat.append(self.get_features(h, z_post))
            if kl_fn is not None:
                raw_kl, clamped_kl = kl_fn(post_logits, prior_logits, free_bits)
                raw_kls.append(raw_kl)
                clamped_kls.append(clamped_kl)
            h = self.img_step(h, z_post, actions[:, t])
        return {
            "h_posts": torch.stack(all_h, dim=1),
            "z_posts": torch.stack(all_z, dim=1),
            "features": torch.stack(all_feat, dim=1),
            "raw_kl_losses": raw_kls,
            "clamped_kl_losses": clamped_kls,
        }

    def imagine_trajectory(
        self, h: torch.Tensor, z: torch.Tensor,
        actor: "Actor", heads: "PredictionHeads", horizon: int,
    ) -> dict:
        """Roll out imagined trajectory using actor and prior dynamics.

        Consolidates the imagination loop into a single method for
        torch.compile optimization.

        Returns:
            dict with keys: features (B,H,latent), rewards (B,H),
            continues (B,H), log_probs (B,H)
        """
        feats, rewards, continues, log_probs, entropies = [], [], [], [], []
        for _ in range(horizon):
            feat = self.get_features(h, z)
            action, log_prob, base_ent = actor(feat)
            h = self.img_step(h, z, action)
            z = self.straight_through_sample(self.prior(h))
            feat = self.get_features(h, z)
            feats.append(feat)
            rewards.append(heads.reward(feat))
            continues.append(heads.continuation(feat))
            log_probs.append(log_prob)
            entropies.append(base_ent)
        return {
            "features": torch.stack(feats, dim=1),
            "rewards": torch.stack(rewards, dim=1),
            "continues": torch.stack(continues, dim=1),
            "log_probs": torch.stack(log_probs, dim=1),
            "base_entropy": torch.stack(entropies, dim=1),
        }


class PredictionHeads(nn.Module):
    """Bundles reward (symlog), continue (Bernoulli), and value (symlog) prediction heads."""
    def __init__(self, latent_dim: int = 1536, hidden_dim: int = 512):
        super().__init__()
        self.reward_net = build_mlp(latent_dim, hidden_dim, 1)
        self.continue_net = build_mlp(latent_dim, hidden_dim, 1, depth=1)
        self.value_net = build_mlp(latent_dim, hidden_dim, 1)
    def reward(self, features: torch.Tensor) -> torch.Tensor:
        return self.reward_net(features).squeeze(-1)
    def continuation(self, features: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.continue_net(features)).squeeze(-1)
    def value(self, features: torch.Tensor) -> torch.Tensor:
        return self.value_net(features).squeeze(-1)


class TwoHotCritic(nn.Module):
    """Value head with two-hot encoded cross-entropy loss."""

    NUM_BINS = 255
    LOW = -20.0
    HIGH = 20.0

    def __init__(self, latent_dim: int = 1536, hidden_dim: int = 512):
        super().__init__()
        self.net = build_mlp(latent_dim, hidden_dim, self.NUM_BINS, norm=True)
        self.register_buffer("bins", torch.linspace(self.LOW, self.HIGH, self.NUM_BINS))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)

    def mean(self, features: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(self.forward(features), dim=-1)
        return (probs * self.bins).sum(dim=-1)

    def loss(self, features: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = self.forward(features)
        targets_c = targets.clamp(self.LOW, self.HIGH)
        flat_targets = targets_c.reshape(-1)
        flat_logits = logits.reshape(-1, self.NUM_BINS)
        lower_idx = torch.searchsorted(self.bins.contiguous(), flat_targets.contiguous(), right=True) - 1
        lower_idx = lower_idx.clamp(0, self.NUM_BINS - 2)
        upper_idx = lower_idx + 1
        lower_val = self.bins[lower_idx]
        upper_val = self.bins[upper_idx]
        upper_weight = ((flat_targets - lower_val) / (upper_val - lower_val + 1e-8)).clamp(0.0, 1.0)
        lower_weight = 1.0 - upper_weight
        two_hot = torch.zeros_like(flat_logits)
        two_hot.scatter_(1, lower_idx.unsqueeze(-1), lower_weight.unsqueeze(-1))
        two_hot.scatter_(1, upper_idx.unsqueeze(-1), upper_weight.unsqueeze(-1))
        return -(two_hot * torch.log_softmax(flat_logits, dim=-1)).sum(dim=-1).mean()


class Actor(nn.Module):
    """Produces continuous actions with action-specific squashing (sigmoid/tanh)."""
    def __init__(self, latent_dim: int = 1536, hidden_dim: int = 512, action_dim: int = 2):
        super().__init__()
        self.action_dim = action_dim
        self.net = build_mlp(latent_dim, hidden_dim, 2 * action_dim)
    @staticmethod
    def _squash(raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        throttle = torch.sigmoid(raw[..., 0:1])
        steer = torch.tanh(raw[..., 1:2])
        return torch.cat([throttle, steer], dim=-1), throttle, steer
    def forward(self, features: torch.Tensor, deterministic: bool = False) -> tuple:
        """Returns (action, log_prob, base_entropy).

        log_prob includes Jacobian corrections (for policy gradient).
        base_entropy is the analytic Gaussian entropy (for regularization),
        which stays well-behaved even when actions are near squashing boundaries.
        """
        mean, log_std = self.net(features).chunk(2, dim=-1)
        log_std = log_std.clamp(-2.0, 2.0)
        if deterministic:
            action, _, _ = self._squash(mean)
            zeros = torch.zeros(features.shape[0], device=features.device)
            return action, zeros, zeros
        dist = Normal(mean, log_std.exp())
        raw = dist.rsample()
        action, throttle, steer = self._squash(raw)
        lp = dist.log_prob(raw)
        # Jacobian corrections: sigmoid for throttle, tanh for steer
        log_prob = (
            lp[..., 0:1] - torch.log(throttle * (1.0 - throttle) + 1e-6)
            + lp[..., 1:2] - torch.log(1.0 - steer.pow(2) + 1e-6)
        ).squeeze(-1)
        # Analytic Gaussian entropy (sum over action dims), no Jacobian
        base_entropy = dist.entropy().sum(-1)
        return action, log_prob, base_entropy
