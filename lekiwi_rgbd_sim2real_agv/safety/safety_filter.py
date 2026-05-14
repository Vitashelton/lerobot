"""Main safety filter orchestrator.

Applies a sequence of safety checks to RL policy output before
sending commands to the physical robot.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass
class SafetyConfig:
    """Safety filter parameters."""

    stop_distance: float = 0.15          # m
    slow_distance: float = 0.5           # m
    lateral_inhibit_distance: float = 0.3  # m
    rotation_stop_multiplier: float = 1.5
    max_velocity: Tuple[float, float, float] = (0.3, 0.3, 90.0)
    max_acceleration: Tuple[float, float, float] = (0.5, 0.5, 180.0)
    smoothing_alpha: float = 0.3
    dt: float = 0.1
    timeout_threshold: float = 0.5       # s
    depth_invalid_timeout: float = 0.3   # s


class SafetyFilter:
    """Multi-layer safety filter for navigation actions.

    Parameters
    ----------
    config : SafetyConfig
    """

    def __init__(self, config: SafetyConfig | None = None) -> None:
        self.config = config or SafetyConfig()
        self.last_action = np.zeros(3, dtype=np.float32)
        self.last_comm_time: Optional[float] = None
        self.last_depth_valid_time: Optional[float] = None
        self._trigger_stats = {
            "depth_invalid": 0,
            "timeout": 0,
            "emergency_stop": 0,
            "lateral_inhibit": 0,
            "velocity_scaled": 0,
            "rotation_stopped": 0,
        }

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------

    def filter(
        self,
        rl_action: np.ndarray,
        scan64: np.ndarray,
        depth_valid: bool = True,
        comm_healthy: bool = True,
        timestamp: Optional[float] = None,
    ) -> Tuple[np.ndarray, Dict[str, any]]:
        """Apply full safety filtering pipeline.

        Parameters
        ----------
        rl_action : np.ndarray, shape (3,)
            Raw RL policy output [vx, vy, omega].
        scan64 : np.ndarray, shape (64,)
            Current Scan64 distances in meters.
        depth_valid : bool
            Whether the current depth image is valid.
        comm_healthy : bool
            Whether ZMQ communication is healthy.
        timestamp : float or None
            Current timestamp for timeout tracking.

        Returns
        -------
        safe_action : np.ndarray, shape (3,)
        info : dict
            Trigger flags and diagnostic info.
        """
        vx, vy, omega = float(rl_action[0]), float(rl_action[1]), float(rl_action[2])
        info = {
            "shielded": False,
            "trigger_reason": None,
            "original_action": rl_action.copy(),
            "depth_invalid": False,
            "timeout": False,
            "emergency_stop": False,
            "lateral_inhibit": False,
            "velocity_scaled": False,
            "rotation_stopped": False,
        }

        # ---- Layer 0: Invalid depth fallback ----
        if not depth_valid:
            self._trigger_stats["depth_invalid"] += 1
            info["depth_invalid"] = True
            info["shielded"] = True
            info["trigger_reason"] = "depth_invalid"
            return np.zeros(3, dtype=np.float32), info

        # ---- Layer 1: Communication timeout fallback ----
        if not comm_healthy:
            self._trigger_stats["timeout"] += 1
            info["timeout"] = True
            info["shielded"] = True
            info["trigger_reason"] = "comm_timeout"
            return np.zeros(3, dtype=np.float32), info

        # ---- Compute sector minimums ----
        n = len(scan64)
        scan_valid = scan64.copy()
        scan_valid[np.isnan(scan_valid)] = 5.0

        front_beams = slice(max(0, n // 2 - 8), min(n, n // 2 + 8))
        left_beams = slice(0, n // 3)
        right_beams = slice(2 * n // 3, n)

        front_min = float(np.min(scan_valid[front_beams]))
        left_min = float(np.min(scan_valid[left_beams]))
        right_min = float(np.min(scan_valid[right_beams]))
        all_min = float(np.min(scan_valid))

        # ---- Layer 2: Emergency stop ----
        if front_min < self.config.stop_distance:
            vx = 0.0
            self._trigger_stats["emergency_stop"] += 1
            info["emergency_stop"] = True
            info["shielded"] = True
            info["trigger_reason"] = (
                f"emergency_stop: front={front_min:.3f}m < {self.config.stop_distance}m"
            )

        # ---- Layer 3: Lateral inhibit ----
        if left_min < self.config.lateral_inhibit_distance and vy > 0:
            vy = 0.0
            self._trigger_stats["lateral_inhibit"] += 1
            info["lateral_inhibit"] = True
            info["shielded"] = True
            if info["trigger_reason"] is None:
                info["trigger_reason"] = f"lateral_inhibit: left={left_min:.3f}m"

        if right_min < self.config.lateral_inhibit_distance and vy < 0:
            vy = 0.0
            self._trigger_stats["lateral_inhibit"] += 1
            info["lateral_inhibit"] = True
            info["shielded"] = True
            if info["trigger_reason"] is None:
                info["trigger_reason"] = f"lateral_inhibit: right={right_min:.3f}m"

        # ---- Layer 4: Velocity scaling ----
        if vx > 0 and self.config.stop_distance <= front_min < self.config.slow_distance:
            scale = (front_min - self.config.stop_distance) / (
                self.config.slow_distance - self.config.stop_distance
            )
            scale = max(0.0, min(1.0, scale))
            vx *= scale
            self._trigger_stats["velocity_scaled"] += 1
            info["velocity_scaled"] = True
            info["shielded"] = True

        # ---- Layer 5: Rotation stop ----
        rotation_stop_dist = self.config.stop_distance * self.config.rotation_stop_multiplier
        if all_min < rotation_stop_dist:
            omega = 0.0
            self._trigger_stats["rotation_stopped"] += 1
            info["rotation_stopped"] = True
            info["shielded"] = True

        # ---- Layer 6: Action clipping ----
        vx = np.clip(vx, -self.config.max_velocity[0], self.config.max_velocity[0])
        vy = np.clip(vy, -self.config.max_velocity[1], self.config.max_velocity[1])
        omega = np.clip(omega, -self.config.max_velocity[2], self.config.max_velocity[2])

        action = np.array([vx, vy, omega], dtype=np.float32)

        # ---- Layer 7: Acceleration limit ----
        max_delta = np.array(self.config.max_acceleration, dtype=np.float32) * self.config.dt
        delta = np.clip(action - self.last_action, -max_delta, max_delta)
        action = self.last_action + delta

        # ---- Layer 8: Low-pass EMA smoothing ----
        alpha = self.config.smoothing_alpha
        action = alpha * action + (1.0 - alpha) * self.last_action

        self.last_action = action.copy()
        return action, info

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, int]:
        """Return cumulative trigger statistics."""
        return dict(self._trigger_stats)

    def reset_stats(self) -> None:
        """Reset trigger statistics to zero."""
        for k in self._trigger_stats:
            self._trigger_stats[k] = 0

    def reset_last_action(self) -> None:
        """Reset last action to zeros (e.g., at episode start)."""
        self.last_action = np.zeros(3, dtype=np.float32)

    @staticmethod
    def compute_sector_mins(scan64: np.ndarray) -> Dict[str, float]:
        """Compute minimum distances in front, left, right sectors."""
        n = len(scan64)
        valid = scan64.copy()
        valid[np.isnan(valid)] = 5.0

        front = slice(max(0, n // 2 - 8), min(n, n // 2 + 8))
        left = slice(0, n // 3)
        right = slice(2 * n // 3, n)

        return {
            "front_min": float(np.min(valid[front])),
            "left_min": float(np.min(valid[left])),
            "right_min": float(np.min(valid[right])),
        }
