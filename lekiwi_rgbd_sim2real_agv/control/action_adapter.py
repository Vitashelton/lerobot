"""
Post-process actions: clipping, acceleration limiting, and low-pass filtering.

The :class:`ActionAdapter` is a lightweight deterministic wrapper that
ensures the velocity commands sent to the robot respect physical limits
(velocity bounds, acceleration limits) and are temporally smooth.
"""

from __future__ import annotations

import copy
from typing import Optional

import numpy as np


class ActionAdapter:
    """Clip, limit acceleration, and smooth actions.

    Parameters
    ----------
    vx_limits : tuple[float, float]
        Allowed range for forward velocity (m/s).
    vy_limits : tuple[float, float]
        Allowed range for lateral velocity (m/s).
    omega_limits : tuple[float, float]
        Allowed range for angular velocity (deg/s).
    max_accel_v : float
        Maximum linear acceleration (m/s^2).  Used to clamp the
        change from ``last_action``.
    max_accel_omega : float
        Maximum angular acceleration (deg/s^2).
    smoothing_alpha : float
        EMA smoothing factor in [0, 1].  Lower = more smoothing.
        ``alpha=1`` disables smoothing.
    dt : float
        Control time step (s).  Used for acceleration clamping.
    """

    def __init__(
        self,
        vx_limits: tuple[float, float] = (-0.3, 0.3),
        vy_limits: tuple[float, float] = (-0.3, 0.3),
        omega_limits: tuple[float, float] = (-90.0, 90.0),
        max_accel_v: float = 0.5,
        max_accel_omega: float = 180.0,
        smoothing_alpha: float = 0.3,
        dt: float = 0.1,
    ) -> None:
        self.vx_limits = vx_limits
        self.vy_limits = vy_limits
        self.omega_limits = omega_limits
        self.max_accel_v = max_accel_v
        self.max_accel_omega = max_accel_omega
        self.smoothing_alpha = float(np.clip(smoothing_alpha, 0.0, 1.0))
        self.dt = dt

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------

    def adapt(
        self,
        raw_action: dict,
        last_action: Optional[dict] = None,
    ) -> dict:
        """Apply the full post-processing pipeline.

        Pipeline order:
          1. Clip to velocity limits.
          2. Limit acceleration relative to last_action.
          3. EMA smoothing.

        Parameters
        ----------
        raw_action : dict
            Proposed action with keys ``"x.vel"``, ``"y.vel"``, ``"theta.vel"``.
        last_action : dict, optional
            Previous action for acceleration limiting and smoothing.
            If ``None``, steps 2 and 3 are skipped.

        Returns
        -------
        dict
            Adapted action in the same format.
        """
        action = copy.deepcopy(raw_action)

        # Step 1: Clip
        action = self.clip_action(action)

        # Step 2: Acceleration limit
        if last_action is not None:
            action = self.limit_acceleration(action, last_action, self.dt)

        # Step 3: Smooth
        if last_action is not None:
            action = self.smooth(action, last_action, self.smoothing_alpha)

        return action

    # ------------------------------------------------------------------
    # Individual steps
    # ------------------------------------------------------------------

    def clip_action(self, action: dict) -> dict:
        """Clip each axis to its configured limits.

        Returns a new dict (shallow copy).
        """
        out = copy.deepcopy(action)
        out["x.vel"] = float(np.clip(out.get("x.vel", 0.0), *self.vx_limits))
        out["y.vel"] = float(np.clip(out.get("y.vel", 0.0), *self.vy_limits))
        out["theta.vel"] = float(np.clip(out.get("theta.vel", 0.0), *self.omega_limits))
        return out

    def limit_acceleration(
        self,
        action: dict,
        last_action: dict,
        dt: float,
    ) -> dict:
        """Limit the change from *last_action* based on max acceleration.

        ``|action_i - last_i| <= max_accel * dt``
        """
        out = copy.deepcopy(action)
        dt = max(dt, 1e-3)

        for key, prev_key, max_accel in [
            ("x.vel", "x.vel", self.max_accel_v),
            ("y.vel", "y.vel", self.max_accel_v),
            ("theta.vel", "theta.vel", self.max_accel_omega),
        ]:
            current = float(out.get(key, 0.0))
            previous = float(last_action.get(prev_key, 0.0))
            max_delta = max_accel * dt
            delta = np.clip(current - previous, -max_delta, max_delta)
            out[key] = previous + delta

        return out

    def smooth(
        self,
        action: dict,
        last_action: dict,
        alpha: float,
    ) -> dict:
        """Exponential moving average smoothing.

        ``out = alpha * action + (1 - alpha) * last_action``
        """
        if alpha >= 1.0:
            return copy.deepcopy(action)
        if alpha <= 0.0:
            return copy.deepcopy(last_action)

        out = copy.deepcopy(action)
        for key in ("x.vel", "y.vel", "theta.vel"):
            current = float(out.get(key, 0.0))
            previous = float(last_action.get(key, 0.0))
            out[key] = alpha * current + (1.0 - alpha) * previous

        return out

    # ------------------------------------------------------------------
    # Convenience: numpy I/O
    # ------------------------------------------------------------------

    @staticmethod
    def action_to_array(action: dict) -> np.ndarray:
        """Convert dict action to numpy array [vx, vy, omega]."""
        return np.array(
            [action.get("x.vel", 0.0), action.get("y.vel", 0.0), action.get("theta.vel", 0.0)],
            dtype=np.float32,
        )

    @staticmethod
    def array_to_action(arr: np.ndarray) -> dict:
        """Convert numpy array [vx, vy, omega] to dict action."""
        return {
            "x.vel": float(arr[0]),
            "y.vel": float(arr[1]),
            "theta.vel": float(arr[2]),
        }

    def adapt_from_arrays(
        self,
        raw: np.ndarray,
        last: Optional[np.ndarray] = None,
    ) -> dict:
        """Like :meth:`adapt` but accepts / returns numpy arrays internally.

        Returns
        -------
        dict
            Adapted action dict.
        """
        raw_dict = self.array_to_action(raw)
        last_dict = self.array_to_action(last) if last is not None else None
        return self.adapt(raw_dict, last_dict)
