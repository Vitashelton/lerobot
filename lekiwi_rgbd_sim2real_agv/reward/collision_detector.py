"""Detect collisions from Scan64 and other signals."""

from __future__ import annotations

import numpy as np


class CollisionDetector:
    """Detect collision events from Scan64 and velocity patterns.

    Parameters
    ----------
    scan_threshold : float
        Minimum distance (m) that counts as collision.
    velocity_threshold : float
        Sudden velocity drop threshold (fraction of previous speed).
    """

    def __init__(
        self,
        scan_threshold: float = 0.1,
        velocity_threshold: float = 0.3,
    ) -> None:
        self.scan_threshold = scan_threshold
        self.velocity_threshold = velocity_threshold

    def detect(
        self,
        scan64: np.ndarray,
        velocities: np.ndarray,
        collision_labels: np.ndarray | None = None,
    ) -> np.ndarray:
        """Detect collisions per timestep.

        Parameters
        ----------
        scan64 : np.ndarray, shape (T, 64)
            Scan64 distances.
        velocities : np.ndarray, shape (T, 3)
            Robot velocities [vx, vy, omega].
        collision_labels : np.ndarray or None
            Ground-truth collision labels if available.

        Returns
        -------
        np.ndarray, shape (T,)
            Boolean collision indicators.
        """
        T = len(velocities)
        collisions = np.zeros(T, dtype=bool)

        for t in range(T):
            # Method 1: Scan64 minimum below threshold
            s = scan64[t]
            valid = s[~np.isnan(s)]
            if len(valid) > 0 and np.min(valid) < self.scan_threshold:
                collisions[t] = True
                continue

            # Method 2: Sudden velocity drop
            if t > 0:
                prev_speed = np.linalg.norm(velocities[t - 1, :2])
                curr_speed = np.linalg.norm(velocities[t, :2])
                if prev_speed > 0.05 and curr_speed < self.velocity_threshold * prev_speed:
                    collisions[t] = True
                    continue

            # Method 3: Use ground truth if available
            if collision_labels is not None and t < len(collision_labels):
                collisions[t] = bool(collision_labels[t])

        return collisions
