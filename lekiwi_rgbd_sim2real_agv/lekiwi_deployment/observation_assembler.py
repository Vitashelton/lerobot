"""Assemble multimodal observations from real LeKiwi sensors.

Integrates:
    - D435i RGB camera → RGB image
    - D435i depth → depth image → Scan64
    - Robot odometry → state [vx, vy, omega]
    - Planned goal → goal [dx, dy, dtheta]
    - Previous action tracking
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

import numpy as np
import cv2


class ObservationAssembler:
    """Build multimodal observation dict from real sensor streams.

    Parameters
    ----------
    image_size : tuple[int, int]
        Target (H, W) for RGB images.
    scan_dim : int
        Number of Scan64 beams.
    fov_deg : float
        Camera horizontal FOV for Scan64 computation.
    max_range : float
        Maximum depth range (m).
    """

    def __init__(
        self,
        image_size: tuple[int, int] = (224, 224),
        scan_dim: int = 64,
        fov_deg: float = 87.0,
        max_range: float = 5.0,
    ) -> None:
        self.image_size = image_size
        self.scan_dim = scan_dim
        self.fov_deg = fov_deg
        self.max_range = max_range
        self.prev_action = np.zeros(3, dtype=np.float32)

    def assemble(
        self,
        rgb_frame: Optional[np.ndarray],
        depth_frame: Optional[np.ndarray],
        velocity: np.ndarray,
        goal_vector: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Build full observation dict.

        Parameters
        ----------
        rgb_frame : np.ndarray, shape (H, W, 3) uint8
            Raw RGB frame from D435i.
        depth_frame : np.ndarray, shape (H, W) float32
            Depth frame in meters.
        velocity : np.ndarray, shape (3,)
            Current velocity [vx, vy, omega].
        goal_vector : np.ndarray, shape (3,)
            Goal in robot frame [dx, dy, dtheta].

        Returns
        -------
        dict with keys: rgb, depth, scan64, state, goal, prev_action
        """
        obs: Dict[str, np.ndarray] = {}

        # RGB
        if rgb_frame is not None:
            obs["rgb"] = self._process_rgb(rgb_frame)
        else:
            obs["rgb"] = np.zeros((*self.image_size, 3), dtype=np.uint8)

        # Depth → Scan64
        if depth_frame is not None:
            obs["depth"] = cv2.resize(
                depth_frame, (self.image_size[1], self.image_size[0])
            )
            obs["scan64"] = self._depth_to_scan64(depth_frame)
        else:
            obs["depth"] = None
            obs["scan64"] = np.full(self.scan_dim, np.nan, dtype=np.float32)

        # State
        obs["state"] = np.asarray(velocity, dtype=np.float32)

        # Goal
        obs["goal"] = np.asarray(goal_vector, dtype=np.float32)

        # Previous action
        obs["prev_action"] = self.prev_action.copy()

        return obs

    def update_prev_action(self, action: np.ndarray) -> None:
        """Update previous action buffer."""
        self.prev_action = np.asarray(action, dtype=np.float32).copy()

    def _process_rgb(self, frame: np.ndarray) -> np.ndarray:
        """Resize and normalize RGB frame."""
        h, w = frame.shape[:2]
        th, tw = self.image_size
        # Center crop to aspect ratio
        aspect = tw / th
        if w / h > aspect:
            new_w = int(h * aspect)
            start = (w - new_w) // 2
            frame = frame[:, start : start + new_w]
        else:
            new_h = int(w / aspect)
            start = (h - new_h) // 2
            frame = frame[start : start + new_h]
        return cv2.resize(frame, (tw, th), interpolation=cv2.INTER_LINEAR)

    def _depth_to_scan64(self, depth_m: np.ndarray) -> np.ndarray:
        """Convert depth image to Scan64."""
        from data_adapters.base_adapter import BaseDatasetAdapter

        return BaseDatasetAdapter.compute_scan64_from_depth(
            depth_m,
            scan_dim=self.scan_dim,
            fov_deg=self.fov_deg,
            slice_fraction=0.3,
            percentile=10.0,
        )
