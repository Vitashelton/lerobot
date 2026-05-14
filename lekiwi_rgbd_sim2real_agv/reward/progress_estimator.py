"""Estimate goal progress from trajectory data.

Supports multiple fallback strategies when explicit goal position is unavailable.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


class ProgressEstimator:
    """Estimate per-timestep progress toward the navigation goal.

    Parameters
    ----------
    method : str
        Estimation method:
        - "position": use explicit robot + goal positions
        - "goal_vector": use goal vector magnitude change
        - "temporal": use fraction of episode completed (weak proxy)
    """

    def __init__(self, method: str = "position") -> None:
        self.method = method

    def estimate(
        self,
        robot_positions: Optional[np.ndarray] = None,
        goal_position: Optional[np.ndarray] = None,
        goal_vectors: Optional[np.ndarray] = None,
        episode_length: Optional[int] = None,
    ) -> np.ndarray:
        """Estimate progress for each timestep.

        Returns
        -------
        np.ndarray
            Progress values, positive = approaching goal.
        """
        if self.method == "position" and robot_positions is not None and goal_position is not None:
            return self._position_progress(robot_positions, goal_position)
        elif self.method == "goal_vector" and goal_vectors is not None:
            return self._goal_vector_progress(goal_vectors)
        elif self.method == "temporal" and episode_length is not None:
            return self._temporal_progress(episode_length)
        else:
            raise ValueError(f"No valid data for progress method: {self.method}")

    @staticmethod
    def _position_progress(
        robot_positions: np.ndarray,
        goal_position: np.ndarray,
    ) -> np.ndarray:
        """Progress from distance-to-goal change."""
        T = len(robot_positions)
        progress = np.zeros(T, dtype=np.float32)
        for t in range(1, T):
            d_prev = np.linalg.norm(robot_positions[t - 1, :2] - goal_position[:2])
            d_curr = np.linalg.norm(robot_positions[t, :2] - goal_position[:2])
            progress[t] = d_prev - d_curr
        return progress

    @staticmethod
    def _goal_vector_progress(goal_vectors: np.ndarray) -> np.ndarray:
        """Progress from goal vector magnitude change."""
        T = len(goal_vectors)
        progress = np.zeros(T, dtype=np.float32)
        for t in range(1, T):
            d_prev = np.linalg.norm(goal_vectors[t - 1, :2])
            d_curr = np.linalg.norm(goal_vectors[t, :2])
            progress[t] = d_prev - d_curr
        return progress

    @staticmethod
    def _temporal_progress(episode_length: int) -> np.ndarray:
        """Weak proxy: uniform progress assuming episode ends at goal."""
        return np.ones(episode_length, dtype=np.float32) / episode_length
