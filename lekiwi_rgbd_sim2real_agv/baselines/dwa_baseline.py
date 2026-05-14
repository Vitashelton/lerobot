"""DWA (Dynamic Window Approach) traditional navigation baseline.

Wraps the existing DWA implementation for use in evaluation.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from control.dwa_policy import DWAPolicy


class DWABaseline:
    """DWA traditional planner as an evaluation baseline.

    Parameters
    ----------
    dwa_config : dict or None
        DWA parameter overrides.
    """

    def __init__(self, dwa_config: Dict | None = None) -> None:
        kwargs = dwa_config or {}
        self.dwa = DWAPolicy(**kwargs)

    def compute_action(
        self,
        scan64: np.ndarray,
        goal_position: np.ndarray,
        current_velocity: np.ndarray,
    ) -> Dict:
        """Compute DWA action.

        Parameters
        ----------
        scan64 : np.ndarray, shape (64,)
            Current Scan64.
        goal_position : np.ndarray, shape (3,)
            Goal in robot frame [dx, dy, dtheta].
        current_velocity : np.ndarray, shape (3,)
            Current velocity [vx, vy, omega].

        Returns
        -------
        dict with keys "x.vel", "y.vel", "theta.vel", "score".
        """
        result = self.dwa.compute_action(scan64, goal_position, current_velocity)
        return result

    def get_action_array(
        self,
        scan64: np.ndarray,
        goal_position: np.ndarray,
        current_velocity: np.ndarray,
    ) -> np.ndarray:
        """Return action as numpy array [vx, vy, omega]."""
        result = self.compute_action(scan64, goal_position, current_velocity)
        return np.array(
            [result["x.vel"], result["y.vel"], result["theta.vel"]],
            dtype=np.float32,
        )
