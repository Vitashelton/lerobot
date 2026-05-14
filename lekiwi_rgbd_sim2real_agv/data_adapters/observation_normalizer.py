"""Fit and apply observation normalization statistics.

Computes mean/std for RGB, Scan64, state, goal, and actions from the
training set, then provides methods to normalize/unnormalize.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


class ObservationNormalizer:
    """Compute and apply per-modality normalization.

    Parameters
    ----------
    eps : float
        Small constant to avoid division by zero.
    """

    def __init__(self, eps: float = 1e-8) -> None:
        self.eps = eps
        self._stats: Dict[str, Dict[str, np.ndarray]] = {}

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, dataset: Dict[str, Any]) -> "ObservationNormalizer":
        """Compute normalization statistics from the dataset.

        Parameters
        ----------
        dataset : dict
            Unified format dataset (training split).

        Returns
        -------
        self
        """
        obs = dataset.get("observations", {})

        # RGB: channel-wise mean/std over all frames
        if obs.get("rgb") is not None and obs["rgb"].size > 0:
            rgb = obs["rgb"].astype(np.float32) / 255.0
            # (T, H, W, C) → per-channel stats
            self._stats["rgb"] = {
                "mean": rgb.mean(axis=(0, 1, 2)).astype(np.float32),
                "std": rgb.std(axis=(0, 1, 2)).astype(np.float32) + self.eps,
            }

        # Scan64: per-beam stats (only over valid/non-NaN values)
        if obs.get("scan64") is not None and obs["scan64"].size > 0:
            s = obs["scan64"]
            valid_mask = ~np.isnan(s)
            beam_mean = np.zeros(s.shape[1], dtype=np.float32)
            beam_std = np.ones(s.shape[1], dtype=np.float32)
            for i in range(s.shape[1]):
                valid = s[:, i][valid_mask[:, i]]
                if len(valid) > 0:
                    beam_mean[i] = valid.mean()
                    beam_std[i] = valid.std() + self.eps
            self._stats["scan64"] = {"mean": beam_mean, "std": beam_std}

        # State
        if obs.get("state") is not None and obs["state"].size > 0:
            st = obs["state"]
            self._stats["state"] = {
                "mean": st.mean(axis=0).astype(np.float32),
                "std": st.std(axis=0).astype(np.float32) + self.eps,
            }

        # Goal
        if obs.get("goal") is not None and obs["goal"].size > 0:
            g = obs["goal"]
            self._stats["goal"] = {
                "mean": g.mean(axis=0).astype(np.float32),
                "std": g.std(axis=0).astype(np.float32) + self.eps,
            }

        # Actions
        if dataset.get("actions") is not None:
            a = dataset["actions"]
            self._stats["action"] = {
                "mean": a.mean(axis=0).astype(np.float32),
                "std": a.std(axis=0).astype(np.float32) + self.eps,
            }

        return self

    # ------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------

    def normalize_observation(
        self, obs: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Normalize a single observation dict.

        Parameters
        ----------
        obs : dict
            Observation dict with keys "rgb", "scan64", "state", "goal".

        Returns
        -------
        dict
            Normalized observations.
        """
        out: Dict[str, np.ndarray] = {}

        if "rgb" in obs and obs["rgb"] is not None and "rgb" in self._stats:
            rgb = obs["rgb"].astype(np.float32) / 255.0
            out["rgb"] = (rgb - self._stats["rgb"]["mean"]) / self._stats["rgb"]["std"]

        if "scan64" in obs and obs["scan64"] is not None and "scan64" in self._stats:
            s = obs["scan64"].copy()
            stats = self._stats["scan64"]
            out["scan64"] = (s - stats["mean"]) / stats["std"]
            out["scan64"] = np.nan_to_num(out["scan64"], nan=0.0)

        if "state" in obs and obs["state"] is not None and "state" in self._stats:
            out["state"] = (obs["state"] - self._stats["state"]["mean"]) / self._stats["state"]["std"]

        if "goal" in obs and obs["goal"] is not None and "goal" in self._stats:
            out["goal"] = (obs["goal"] - self._stats["goal"]["mean"]) / self._stats["goal"]["std"]

        return out

    def normalize_action(self, action: np.ndarray) -> np.ndarray:
        """Normalize actions."""
        if "action" in self._stats:
            return (action - self._stats["action"]["mean"]) / self._stats["action"]["std"]
        return action

    def unnormalize_action(self, action: np.ndarray) -> np.ndarray:
        """Unnormalize actions back to original scale."""
        if "action" in self._stats:
            return action * self._stats["action"]["std"] + self._stats["action"]["mean"]
        return action

    def get_stats(self) -> Dict[str, Any]:
        """Return the fitted statistics dict (serializable)."""
        return {k: {k2: v2.tolist() for k2, v2 in v.items()} for k, v in self._stats.items()}
