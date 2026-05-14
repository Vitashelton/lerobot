"""Abstract base class for all public dataset adapters.

Each adapter must produce a unified dict format:
    {
        "observations": {
            "rgb": np.ndarray (T, H, W, 3) uint8,
            "depth": np.ndarray (T, H, W) float32 or None,
            "scan64": np.ndarray (T, 64) float32 or None,
            "state": np.ndarray (T, 3) float32 [vx, vy, omega],
            "goal": np.ndarray (T, 3) float32 [dx, dy, dtheta],
            "prev_action": np.ndarray (T, 3) float32,
        },
        "actions": np.ndarray (T, 3) float32 [vx, vy, omega],
        "rewards": np.ndarray (T,) float32 (optional, may be zeros),
        "dones": np.ndarray (T,) bool,
        "episode_ids": np.ndarray (T,) int32,
        "info": {
            "collision": np.ndarray (T,) bool,
            "intervention": np.ndarray (T,) bool,
            "goal_reached": np.ndarray (T,) bool,
            "robot_position": np.ndarray (T, 2) float32 or None,
            "goal_position": np.ndarray (T, 2) float32 or None,
        }
    }
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np


class BaseDatasetAdapter(ABC):
    """Abstract adapter for public navigation datasets.

    Parameters
    ----------
    data_dir : str
        Path to the raw dataset directory.
    scan_dim : int
        Number of beams in the virtual Scan64.
    image_size : tuple[int, int]
        Target RGB image size (H, W).
    """

    def __init__(
        self,
        data_dir: str,
        scan_dim: int = 64,
        image_size: tuple[int, int] = (224, 224),
    ) -> None:
        self.data_dir = data_dir
        self.scan_dim = scan_dim
        self.image_size = image_size

    @abstractmethod
    def load_episodes(self) -> List[Dict[str, Any]]:
        """Load all episodes from the raw dataset.

        Returns
        -------
        list of dict
            Each dict is one episode with raw data fields.
        """

    @abstractmethod
    def convert_episode(self, raw_episode: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a single raw episode to the unified format.

        Parameters
        ----------
        raw_episode : dict
            Raw episode data from the dataset.

        Returns
        -------
        dict
            Unified format episode dict.
        """

    def convert_all(self) -> Dict[str, Any]:
        """Load all episodes and convert to unified format.

        Returns
        -------
        dict
            Unified format dict with all episodes concatenated.
        """
        raw_episodes = self.load_episodes()
        converted: List[Dict[str, Any]] = []
        for i, ep in enumerate(raw_episodes):
            unified = self.convert_episode(ep)
            unified["episode_ids"] = np.full(
                len(unified["actions"]), i, dtype=np.int32
            )
            converted.append(unified)

        return self._concatenate_episodes(converted)

    @staticmethod
    def _concatenate_episodes(
        episodes: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Concatenate multiple episode dicts along the time axis.

        Parameters
        ----------
        episodes : list of dict
            List of unified-format episode dicts.

        Returns
        -------
        dict
            Single unified dict with all data concatenated.
        """
        result: Dict[str, Any] = {"observations": {}, "info": {}}

        obs_keys = set(episodes[0]["observations"].keys())
        info_keys = set(episodes[0].get("info", {}).keys())

        for key in obs_keys:
            arrays = [ep["observations"][key] for ep in episodes]
            if arrays[0] is not None:
                result["observations"][key] = np.concatenate(arrays, axis=0)
            else:
                result["observations"][key] = None

        for key in info_keys:
            arrays = [ep.get("info", {}).get(key) for ep in episodes]
            if arrays and arrays[0] is not None:
                result["info"][key] = np.concatenate(arrays, axis=0)
            else:
                result["info"][key] = None

        scalar_keys = ["actions", "rewards", "dones", "episode_ids"]
        for key in scalar_keys:
            if key in episodes[0]:
                result[key] = np.concatenate(
                    [ep[key] for ep in episodes], axis=0
                )

        return result

    @staticmethod
    def compute_scan64_from_depth(
        depth_m: np.ndarray,
        scan_dim: int = 64,
        fov_deg: float = 87.0,
        slice_fraction: float = 0.3,
        percentile: float = 10.0,
    ) -> np.ndarray:
        """Convert a depth image to Scan64 representation.

        Parameters
        ----------
        depth_m : np.ndarray, shape (H, W)
            Depth image in meters.
        scan_dim : int
            Number of angular bins.
        fov_deg : float
            Horizontal field of view in degrees.
        slice_fraction : float
            Fraction of image height to use (centered).
        percentile : float
            Percentile for pooling within each bin.

        Returns
        -------
        np.ndarray, shape (scan_dim,)
            Scan64 distances in meters.
        """
        h, w = depth_m.shape
        band_half = max(1, int(h * slice_fraction / 2.0))
        row_center = h // 2
        band = depth_m[row_center - band_half : row_center + band_half, :]

        bin_edges = np.linspace(0, w, scan_dim + 1, dtype=np.int32)
        meter_scan = np.full(scan_dim, np.nan, dtype=np.float32)

        for i in range(scan_dim):
            col_start, col_end = bin_edges[i], bin_edges[i + 1]
            if col_end <= col_start:
                continue
            bin_pixels = band[:, col_start:col_end].ravel()
            valid = bin_pixels[~np.isnan(bin_pixels)]
            if len(valid) > 0:
                meter_scan[i] = float(
                    np.percentile(valid, percentile)
                )

        return meter_scan

    @staticmethod
    def _resize_rgb(rgb: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
        """Resize RGB image to target size. Simple center-crop + resize."""
        import cv2

        h, w = rgb.shape[:2]
        th, tw = target_size

        # Center crop to target aspect ratio
        aspect = tw / th
        if w / h > aspect:
            new_w = int(h * aspect)
            start = (w - new_w) // 2
            rgb = rgb[:, start : start + new_w]
        else:
            new_h = int(w / aspect)
            start = (h - new_h) // 2
            rgb = rgb[start : start + new_h]

        return cv2.resize(rgb, (tw, th), interpolation=cv2.INTER_LINEAR)
