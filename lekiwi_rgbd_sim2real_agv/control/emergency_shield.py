"""
Emergency stop and motion inhibition based on safety zones.

This is a **hard** safety layer -- it overrides all other controllers
when imminent collision is detected.  It operates on a sector-based
model of the LiDAR scan: front, left, and right.
"""

from __future__ import annotations

import copy
from typing import Optional

import numpy as np


class EmergencyShield:
    """Hard safety override based on proximity zones.

    The shield applies a series of rules, in priority order, to scale
    down or zero out velocity commands when obstacles are too close.

    Parameters
    ----------
    stop_distance_m : float
        Distance (m) below which forward motion is fully stopped.
    slow_distance_m : float
        Distance (m) below which forward speed is proportionally scaled.
    lateral_inhibit_distance_m : float
        Distance (m) below which lateral motion toward the obstacle is blocked.
    rotation_stop_multiplier : float
        Multiplier on *stop_distance_m* that triggers a rotation stop.
    """

    def __init__(
        self,
        stop_distance_m: float = 0.15,
        slow_distance_m: float = 0.5,
        lateral_inhibit_distance_m: float = 0.3,
        rotation_stop_multiplier: float = 1.5,
    ) -> None:
        self.stop_distance_m = stop_distance_m
        self.slow_distance_m = slow_distance_m
        self.lateral_inhibit_distance_m = lateral_inhibit_distance_m
        self.rotation_stop_multiplier = rotation_stop_multiplier

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------

    def apply(
        self,
        action: dict,
        scan_m: np.ndarray,
        front_min: Optional[float] = None,
        left_min: Optional[float] = None,
        right_min: Optional[float] = None,
    ) -> dict:
        """Apply safety shielding to an action.

        Parameters
        ----------
        action : dict
            Raw action with keys ``"x.vel"``, ``"y.vel"``, ``"theta.vel"``.
        scan_m : np.ndarray
            Full LiDAR scan in metres (used to auto-compute sector mins
            when not explicitly provided).
        front_min : float, optional
            Pre-computed minimum range in the front sector.
        left_min : float, optional
            Pre-computed minimum range in the left sector.
        right_min : float, optional
            Pre-computed minimum range in the right sector.

        Returns
        -------
        dict
            ``{
                "action": {"x.vel": ..., "y.vel": ..., "theta.vel": ...},
                "shielded": bool,
                "trigger_reason": str or None,
                "original_action": dict
            }``
        """
        original = copy.deepcopy(action)
        shielded = action.copy()

        # Auto-compute sector mins if not provided
        if front_min is None or left_min is None or right_min is None:
            sector_mins = self._compute_sector_mins(np.asarray(scan_m, dtype=np.float32))
            if front_min is None:
                front_min = sector_mins["front"]
            if left_min is None:
                left_min = sector_mins["left"]
            if right_min is None:
                right_min = sector_mins["right"]

        vx = float(shielded.get("x.vel", 0.0))
        vy = float(shielded.get("y.vel", 0.0))
        omega = float(shielded.get("theta.vel", 0.0))

        triggered = False
        reason: Optional[str] = None

        # --------------------------------------------------------------
        # Rule 1: Emergency stop forward if front < stop_distance
        # --------------------------------------------------------------
        if front_min < self.stop_distance_m:
            vx = 0.0
            triggered = True
            reason = f"front_min={front_min:.3f}m < stop={self.stop_distance_m}m (vx=0)"

        # --------------------------------------------------------------
        # Rule 2: Inhibit lateral motion toward obstacles
        # --------------------------------------------------------------
        if left_min < self.lateral_inhibit_distance_m and vy > 0:
            vy = 0.0
            triggered = True
            reason = reason or f"left_min={left_min:.3f}m < lateral={self.lateral_inhibit_distance_m}m (vy=0)"

        if right_min < self.lateral_inhibit_distance_m and vy < 0:
            vy = 0.0
            triggered = True
            reason = reason or f"right_min={right_min:.3f}m < lateral={self.lateral_inhibit_distance_m}m (vy=0)"

        # --------------------------------------------------------------
        # Rule 3: Proportional slowdown when front < slow_distance
        # --------------------------------------------------------------
        if (
            self.stop_distance_m <= front_min < self.slow_distance_m
            and vx > 0
        ):
            # Scale vx: 0 at stop_distance, 1 at slow_distance
            scale = (front_min - self.stop_distance_m) / (
                self.slow_distance_m - self.stop_distance_m
            )
            scale = max(0.0, min(1.0, scale))
            vx *= scale
            triggered = True
            reason = reason or f"front_min={front_min:.3f}m < slow={self.slow_distance_m}m (vx scaled={scale:.2f})"

        # --------------------------------------------------------------
        # Rule 4: Stop rotation when any sector is critically close
        # --------------------------------------------------------------
        rotation_stop_dist = self.stop_distance_m * self.rotation_stop_multiplier
        if front_min < rotation_stop_dist or left_min < rotation_stop_dist or right_min < rotation_stop_dist:
            omega = 0.0
            triggered = True
            reason = reason or f"rotation stop (min sector < {rotation_stop_dist:.3f}m)"

        shielded["x.vel"] = vx
        shielded["y.vel"] = vy
        shielded["theta.vel"] = omega

        return {
            "action": shielded,
            "shielded": triggered,
            "trigger_reason": reason,
            "original_action": original,
        }

    # ------------------------------------------------------------------
    # Sector helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_sector_mins(scan_m: np.ndarray) -> dict[str, float]:
        """Compute minimum range for front, left, right sectors.

        Assumes 64-beam 360-degree scan:
            front : beams 56..8   (centred on 0 deg)
            right : beams 8..24   (centred on 90 deg)
            left  : beams 40..56  (centred on 270 deg)
        """
        n = len(scan_m)
        h = n // 4  # 16

        def _sector(start: int, end: int) -> np.ndarray:
            start = start % n
            end = end % n
            if start <= end:
                return scan_m[start:end]
            return np.concatenate([scan_m[start:], scan_m[:end]])

        front = _sector(n - h // 2, h // 2)
        right = _sector(h // 2, h + h // 2)
        left = _sector(2 * h + h // 2, 3 * h + h // 2)

        return {
            "front": float(front.min()),
            "right": float(right.min()),
            "left": float(left.min()),
        }

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def quick_check(self, scan_m: np.ndarray) -> bool:
        """Return ``True`` if a stop is required right now."""
        mins = self._compute_sector_mins(np.asarray(scan_m, dtype=np.float32))
        return mins["front"] < self.stop_distance_m
