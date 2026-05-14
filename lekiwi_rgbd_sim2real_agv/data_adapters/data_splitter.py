"""Trajectory-aware train/val/test splitter.

Splits episodes (not individual transitions) to prevent data leakage
across trajectory boundaries.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np


def split_trajectories(
    unified_data: Dict[str, Any],
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Split unified dataset into train/val/test by episode.

    Parameters
    ----------
    unified_data : dict
        Output of ``BaseDatasetAdapter.convert_all()``.
    train_frac, val_frac, test_frac : float
        Split fractions, must sum to 1.0.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    train_data, val_data, test_data : dict
        Each with the same structure as ``unified_data``.
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6

    episode_ids = unified_data["episode_ids"]
    unique_eps = np.unique(episode_ids)
    n_eps = len(unique_eps)

    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_eps)

    n_train = max(1, int(n_eps * train_frac))
    n_val = max(1, int(n_eps * val_frac))

    train_eps = set(unique_eps[perm[:n_train]])
    val_eps = set(unique_eps[perm[n_train : n_train + n_val]])
    test_eps = set(unique_eps[perm[n_train + n_val :]])

    def _mask_split(eps_set: set) -> Dict[str, Any]:
        mask = np.array([eid in eps_set for eid in episode_ids], dtype=bool)
        result: Dict[str, Any] = {"observations": {}, "info": {}}

        for key in unified_data.get("observations", {}):
            val = unified_data["observations"][key]
            result["observations"][key] = val[mask] if val is not None else None

        for key in unified_data.get("info", {}):
            val = unified_data["info"][key]
            result["info"][key] = val[mask] if val is not None else None

        for key in ["actions", "rewards", "dones", "episode_ids"]:
            if key in unified_data:
                result[key] = unified_data[key][mask]

        return result

    return _mask_split(train_eps), _mask_split(val_eps), _mask_split(test_eps)
