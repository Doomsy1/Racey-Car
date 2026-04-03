import pickle
from collections import deque
from pathlib import Path
import threading
import numpy as np
import torch


class EpisodeReplayBuffer:
    """Episode-based replay buffer that stores complete episodes and samples subsequences."""

    def __init__(self, capacity: int, obs_shape: tuple, action_dim: int):
        self.capacity = capacity
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self._lock = threading.RLock()
        self._episodes: deque[dict] = deque(maxlen=capacity)
        self._current: dict = self._new_episode()
        self._current_by_stream: dict[int, dict] = {0: self._current}
        self._priorities: deque[float] = deque(maxlen=capacity)
        self._default_priority: float = 1.0
        self._priority_alpha: float = 0.6

    def __getstate__(self) -> dict:
        state = dict(self.__dict__)
        state.pop("_lock", None)
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._lock = threading.RLock()

    def _ensure_priority_state(self) -> None:
        if not hasattr(self, "_default_priority"):
            self._default_priority = 1.0
        if not hasattr(self, "_priority_alpha"):
            self._priority_alpha = 0.6
        if not hasattr(self, "_priorities"):
            self._priorities = deque(maxlen=self.capacity)
        while len(self._priorities) < len(self._episodes):
            seed = float(max(self._priorities)) if self._priorities else self._default_priority
            self._priorities.append(seed)

    def _new_episode(self) -> dict:
        return {"obs": [], "action": [], "reward": [], "done": []}

    def _ensure_current_state(self) -> None:
        if not hasattr(self, "_current_by_stream"):
            self._current = getattr(self, "_current", self._new_episode())
            self._current_by_stream = {0: self._current}
        if not hasattr(self, "_current"):
            self._current = self._current_by_stream.setdefault(0, self._new_episode())

    def _get_current(self, stream_id: int) -> dict:
        self._ensure_current_state()
        episode = self._current_by_stream.get(stream_id)
        if episode is None:
            episode = self._new_episode()
            self._current_by_stream[stream_id] = episode
        if stream_id == 0:
            self._current = episode
        return episode

    def _reset_stream(self, stream_id: int) -> None:
        self._ensure_current_state()
        self._current_by_stream[stream_id] = self._new_episode()
        if stream_id == 0:
            self._current = self._current_by_stream[stream_id]

    def add_step(
        self, obs: np.ndarray, action: np.ndarray, reward: float, done: bool, stream_id: int = 0,
    ) -> None:
        """Add a single transition to the current in-progress episode."""
        with self._lock:
            current = self._get_current(stream_id)
            current["obs"].append(obs)
            current["action"].append(action)
            current["reward"].append(reward)
            current["done"].append(done)

    def end_episode(self, stream_id: int = 0) -> None:
        """Finalize current episode. Discard if fewer than 2 steps."""
        with self._lock:
            current = self._get_current(stream_id)
            if len(current["obs"]) >= 2:
                self._ensure_priority_state()
                finalized = {
                    "obs": np.stack(current["obs"]),
                    "action": np.stack(current["action"]),
                    "reward": np.array(current["reward"], dtype=np.float32),
                    "done": np.array(current["done"], dtype=bool),
                }
                self._episodes.append(finalized)
                seed = float(max(self._priorities)) if self._priorities else self._default_priority
                self._priorities.append(seed)
            self._reset_stream(stream_id)

    def reset_current_episode(self, stream_id: int | None = None) -> None:
        """Discard any in-progress episode, optionally for a single collection stream."""
        with self._lock:
            self._ensure_current_state()
            if stream_id is None:
                self._current_by_stream = {sid: self._new_episode() for sid in self._current_by_stream}
                self._current = self._current_by_stream.setdefault(0, self._new_episode())
                return
            self._reset_stream(stream_id)

    def update_priorities(self, indices: list[int], losses: list[float]) -> None:
        """Update priorities for sampled episodes."""
        with self._lock:
            self._ensure_priority_state()
            for idx, loss in zip(indices, losses):
                if 0 <= idx < len(self._priorities):
                    self._priorities[idx] = max(float(loss), 1e-6) ** self._priority_alpha

    def _source_len(self, episode: dict) -> int:
        return len(episode["reward"])

    def _candidate_sources(self, min_steps: int = 2) -> list[tuple[int, dict, float]]:
        with self._lock:
            self._ensure_priority_state()
            self._ensure_current_state()
            candidates: list[tuple[int, dict, float]] = []
            for ep_idx, episode in enumerate(self._episodes):
                if self._source_len(episode) >= min_steps:
                    candidates.append((ep_idx, episode, float(self._priorities[ep_idx])))
            live_priority = float(max(self._priorities)) if self._priorities else self._default_priority
            for episode in self._current_by_stream.values():
                if self._source_len(episode) >= min_steps:
                    candidates.append((-1, episode, live_priority))
            return candidates

    def has_ready_sequences(self, min_steps: int = 2) -> bool:
        """Return whether any complete or in-progress stream can provide a sequence."""
        with self._lock:
            return bool(self._candidate_sources(min_steps=min_steps))

    def _copy_sequence_into_batch(
        self,
        episode: dict,
        start: int,
        usable_len: int,
        obs_batch: np.ndarray,
        act_batch: np.ndarray,
        rew_batch: np.ndarray,
        done_batch: np.ndarray,
        row: int,
    ) -> None:
        obs_slice = episode["obs"][start : start + usable_len]
        act_slice = episode["action"][start : start + usable_len]
        rew_slice = episode["reward"][start : start + usable_len]
        done_slice = episode["done"][start : start + usable_len]
        if isinstance(obs_slice, list):
            obs_batch[row, :usable_len] = np.stack(obs_slice)
            act_batch[row, :usable_len] = np.stack(act_slice)
            rew_batch[row, :usable_len] = np.asarray(rew_slice, dtype=np.float32)
            done_batch[row, :usable_len] = np.asarray(done_slice, dtype=bool)
            return
        obs_batch[row, :usable_len] = obs_slice
        act_batch[row, :usable_len] = act_slice
        rew_batch[row, :usable_len] = rew_slice
        done_batch[row, :usable_len] = done_slice

    def sample_sequences(self, batch_size: int, seq_len: int, device: str = "cpu") -> tuple[dict, list[int]]:
        """Sample random subsequences from stored episodes.
        Returns (batch_dict, episode_indices_sampled).
        """
        with self._lock:
            candidates = self._candidate_sources(min_steps=2)
            if not candidates:
                raise RuntimeError("Cannot sample from empty replay buffer")
            raw_weights = np.array([weight for _, _, weight in candidates], dtype=np.float64)
            if raw_weights.sum() <= 0:
                raw_weights = np.ones(len(candidates), dtype=np.float64)
            weights = raw_weights / raw_weights.sum()
            chosen_slots = np.random.choice(len(candidates), size=batch_size, replace=True, p=weights)
            obs_batch = np.zeros((batch_size, seq_len, *self.obs_shape), dtype=np.uint8)
            act_batch = np.zeros((batch_size, seq_len, self.action_dim), dtype=np.float32)
            rew_batch = np.zeros((batch_size, seq_len), dtype=np.float32)
            done_batch = np.zeros((batch_size, seq_len), dtype=bool)
            sampled_indices: list[int] = []
            for i, slot in enumerate(chosen_slots):
                ep_idx, episode, _ = candidates[int(slot)]
                ep_len = self._source_len(episode)
                usable_len = min(ep_len, seq_len)
                max_start = max(0, ep_len - usable_len)
                start = np.random.randint(0, max_start + 1)
                self._copy_sequence_into_batch(
                    episode, start, usable_len, obs_batch, act_batch, rew_batch, done_batch, i,
                )
                sampled_indices.append(ep_idx)
        return {
            "obs": torch.tensor(obs_batch, device=device),
            "action": torch.tensor(act_batch, device=device),
            "reward": torch.tensor(rew_batch, device=device),
            "done": torch.tensor(done_batch, device=device),
        }, sampled_indices

    def total_steps(self) -> int:
        """Total transitions across complete episodes and current in-progress streams."""
        with self._lock:
            self._ensure_current_state()
            complete_steps = sum(len(ep["reward"]) for ep in self._episodes)
            live_steps = sum(len(ep["reward"]) for ep in self._current_by_stream.values())
            return complete_steps + live_steps

    def __len__(self) -> int:
        """Number of complete episodes in the buffer."""
        with self._lock:
            return len(self._episodes)

    def save(self, path: str) -> None:
        """Pickle the buffer to disk."""
        with self._lock:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, "wb") as f:
                pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "EpisodeReplayBuffer":
        """Load a previously saved buffer from pickle."""
        with open(path, "rb") as f:
            obj = pickle.load(f)
        obj._ensure_priority_state()
        obj._ensure_current_state()
        return obj
