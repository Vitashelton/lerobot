"""Offline replay buffer for multimodal navigation dataset."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch


class OfflineReplayBuffer:
    """Fixed offline replay buffer that stores all transitions.

    Parameters
    ----------
    data : dict
        Unified format dataset (training split).
    normalizer : ObservationNormalizer or None
        Fitted normalizer for observations and actions.
    device : str
        Torch device.
    """

    def __init__(
        self,
        data: Dict[str, any],
        normalizer=None,
        device: str = "cpu",
    ) -> None:
        self.device = device
        self.normalizer = normalizer

        obs = data["observations"]

        # Store all data as tensors
        self.rgb = self._to_tensor(obs.get("rgb"), dtype=torch.float32)
        self.scan64 = self._to_tensor(obs.get("scan64"), dtype=torch.float32)
        self.state = self._to_tensor(obs.get("state"), dtype=torch.float32)
        self.goal = self._to_tensor(obs.get("goal"), dtype=torch.float32)

        self.actions = self._to_tensor(data["actions"], dtype=torch.float32)
        self.rewards = self._to_tensor(data["rewards"], dtype=torch.float32)
        self.dones = self._to_tensor(data["dones"], dtype=torch.bool)

        self.size = len(self.actions)

        # Build next-step indices
        episode_ids = data.get("episode_ids")
        if episode_ids is not None:
            self._next_idx = self._build_next_idx(episode_ids)
        else:
            self._next_idx = torch.arange(1, self.size + 1) % self.size

    def _to_tensor(
        self,
        arr: Optional[np.ndarray],
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor | None:
        if arr is None:
            return None
        t = torch.from_numpy(arr).to(dtype=dtype)
        return t

    def _build_next_idx(self, episode_ids: np.ndarray) -> torch.Tensor:
        """Build next-step indices respecting episode boundaries."""
        next_idx = np.zeros(self.size, dtype=np.int64)
        for i in range(self.size):
            if i + 1 < self.size and episode_ids[i + 1] == episode_ids[i]:
                next_idx[i] = i + 1
            else:
                next_idx[i] = i  # last frame of episode → self
        return torch.from_numpy(next_idx)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample a random batch.

        Returns
        -------
        dict with keys:
            rgb, scan64, state, goal, action, reward, next_rgb,
            next_scan64, next_state, next_goal, done
        """
        indices = torch.randint(0, self.size, (batch_size,), device="cpu")
        next_indices = self._next_idx[indices]

        batch: Dict[str, torch.Tensor] = {}

        def _gather(t: torch.Tensor | None, idx: torch.Tensor) -> torch.Tensor | None:
            if t is None:
                return None
            return t[idx].to(self.device)

        batch["action"] = _gather(self.actions, indices)
        batch["reward"] = _gather(self.rewards, indices)
        batch["done"] = _gather(self.dones, indices)

        # Current observations
        batch["rgb"] = _gather(self.rgb, indices)
        batch["scan64"] = _gather(self.scan64, indices)
        batch["state"] = _gather(self.state, indices)
        batch["goal"] = _gather(self.goal, indices)

        # Next observations
        batch["next_rgb"] = _gather(self.rgb, next_indices)
        batch["next_scan64"] = _gather(self.scan64, next_indices)
        batch["next_state"] = _gather(self.state, next_indices)
        batch["next_goal"] = _gather(self.goal, next_indices)

        return batch

    def __len__(self) -> int:
        return self.size
