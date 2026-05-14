"""Core reward calculator for offline navigation data.

Computes:
    r_t = w_p * progress
          - w_o * obstacle_penalty
          - w_a * action_smoothness_penalty
          - w_i * intervention_penalty
          - w_c * collision_penalty
          + w_g * success_reward
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class RewardConfig:
    """Reward function weights and thresholds."""

    w_progress: float = 1.0
    w_obstacle: float = 0.5
    w_smoothness: float = 0.1
    w_intervention: float = 1.0
    w_collision: float = 5.0
    w_success: float = 10.0

    d_critical: float = 0.3   # m - emergency zone
    d_warning: float = 0.8    # m - caution zone
    d_collision: float = 0.1  # m - contact threshold
    d_success: float = 0.5    # m - goal reached threshold

    gamma: float = 0.99       # for return computation


class RewardCalculator:
    """Compute per-timestep reward from trajectory data.

    Parameters
    ----------
    config : RewardConfig
        Reward weights and thresholds.
    """

    def __init__(self, config: RewardConfig | None = None) -> None:
        self.config = config or RewardConfig()

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------

    def compute_rewards(
        self,
        unified_data: Dict[str, Any],
    ) -> np.ndarray:
        """Compute rewards for all transitions in a unified dataset.

        Parameters
        ----------
        unified_data : dict
            Unified format dataset with observations, actions, info.

        Returns
        -------
        np.ndarray, shape (T,)
            Reward value for each timestep.
        """
        obs = unified_data["observations"]
        actions = unified_data.get("actions")
        info = unified_data.get("info", {})

        T = len(actions) if actions is not None else len(obs["rgb"])

        r_progress = self._compute_progress(obs, info, T)
        r_obstacle = self._compute_obstacle(obs, T)
        r_smoothness = self._compute_smoothness(actions, T)
        r_intervention = self._compute_intervention(info, T)
        r_collision = self._compute_collision(obs, info, T)
        r_success = self._compute_success(obs, info, T)

        rewards = (
            self.config.w_progress * r_progress
            + self.config.w_obstacle * r_obstacle
            + self.config.w_smoothness * r_smoothness
            + self.config.w_intervention * r_intervention
            + self.config.w_collision * r_collision
            + self.config.w_success * r_success
        )

        return rewards.astype(np.float32)

    def compute_returns(self, rewards: np.ndarray, dones: np.ndarray) -> np.ndarray:
        """Compute discounted returns for each timestep."""
        T = len(rewards)
        returns = np.zeros(T, dtype=np.float32)
        running = 0.0
        for t in range(T - 1, -1, -1):
            if dones[t]:
                running = 0.0
            running = rewards[t] + self.config.gamma * running
            returns[t] = running
        return returns

    # ------------------------------------------------------------------
    # Individual reward components
    # ------------------------------------------------------------------

    def _compute_progress(
        self,
        obs: Dict[str, np.ndarray],
        info: Dict[str, np.ndarray],
        T: int,
    ) -> np.ndarray:
        """Progress toward goal: positive when getting closer.

        Uses robot_position and goal_position from info.
        Falls back to goal vector from observations.
        """
        r = np.zeros(T, dtype=np.float32)

        robot_pos = info.get("robot_position")
        goal_pos = info.get("goal_position")
        goal_vec = obs.get("goal")

        for t in range(1, T):
            if robot_pos is not None and goal_pos is not None:
                d_prev = np.linalg.norm(robot_pos[t - 1] - goal_pos[t])
                d_curr = np.linalg.norm(robot_pos[t] - goal_pos[t])
                r[t] = d_prev - d_curr  # positive = getting closer
            elif goal_vec is not None:
                # Use goal vector magnitude change
                d_prev = np.linalg.norm(goal_vec[t - 1, :2])
                d_curr = np.linalg.norm(goal_vec[t, :2])
                r[t] = d_prev - d_curr

        return r

    def _compute_obstacle(
        self,
        obs: Dict[str, np.ndarray],
        T: int,
    ) -> np.ndarray:
        """Obstacle penalty based on Scan64 forward sector."""
        r = np.zeros(T, dtype=np.float32)
        scan = obs.get("scan64")
        if scan is None:
            return r

        for t in range(T):
            s = scan[t]
            valid = s[~np.isnan(s)]
            if len(valid) == 0:
                continue

            # Forward-facing beams (middle 16 of 64, index 24..40)
            n = len(s)
            front = s[max(0, n // 2 - 8) : min(n, n // 2 + 8)]
            front_valid = front[~np.isnan(front)]
            if len(front_valid) == 0:
                continue

            min_scan = float(np.min(front_valid))

            if min_scan < self.config.d_critical:
                r[t] = -(self.config.d_critical - min_scan) / self.config.d_critical
            elif min_scan < self.config.d_warning:
                r[t] = -0.5 * (self.config.d_warning - min_scan) / self.config.d_warning

        return r

    def _compute_smoothness(
        self,
        actions: Optional[np.ndarray],
        T: int,
    ) -> np.ndarray:
        """Penalize large action changes between consecutive steps."""
        r = np.zeros(T, dtype=np.float32)
        if actions is None or T < 2:
            return r

        for t in range(1, T):
            diff = np.linalg.norm(actions[t] - actions[t - 1])
            r[t] = -diff

        return r

    def _compute_intervention(
        self,
        info: Dict[str, np.ndarray],
        T: int,
    ) -> np.ndarray:
        """Penalty for human intervention timesteps."""
        r = np.zeros(T, dtype=np.float32)
        intervention = info.get("intervention")
        if intervention is not None:
            for t in range(T):
                if bool(intervention[t]):
                    r[t] = -1.0
        return r

    def _compute_collision(
        self,
        obs: Dict[str, np.ndarray],
        info: Dict[str, np.ndarray],
        T: int,
    ) -> np.ndarray:
        """Collision penalty: large negative reward."""
        r = np.zeros(T, dtype=np.float32)

        collision_flag = info.get("collision")
        scan = obs.get("scan64")

        for t in range(T):
            is_collision = False
            if collision_flag is not None and t < len(collision_flag):
                is_collision = bool(collision_flag[t])
            if not is_collision and scan is not None and t < len(scan):
                s = scan[t]
                valid = s[~np.isnan(s)]
                if len(valid) > 0 and np.min(valid) < self.config.d_collision:
                    is_collision = True
            if is_collision:
                r[t] = -1.0

        return r

    def _compute_success(
        self,
        obs: Dict[str, np.ndarray],
        info: Dict[str, np.ndarray],
        T: int,
    ) -> np.ndarray:
        """Success reward: large positive when near goal."""
        r = np.zeros(T, dtype=np.float32)

        goal_reached = info.get("goal_reached")
        goal_vec = obs.get("goal")

        for t in range(T):
            is_success = False
            if goal_reached is not None and t < len(goal_reached):
                is_success = bool(goal_reached[t])
            if not is_success and goal_vec is not None and t < len(goal_vec):
                dist = np.linalg.norm(goal_vec[t, :2])
                if dist < self.config.d_success:
                    is_success = True
            if is_success:
                r[t] = 1.0

        return r
