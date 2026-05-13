"""
Safety zone evaluator for AGV navigation.

Defines configurable safety regions (stop, slow-down, lateral-inhibit) and
evaluates depth-scan data against them to produce per-axis inhibition flags
and a speed scaling factor.  Designed to be the final safety-gating layer
before motion commands are issued to the robot.

Typical usage:

    zone = SafetyZone(stop_distance=0.15, slow_down_distance=0.5)
    state = zone.evaluate(scan_m, front_min=0.3, left_min=0.8, right_min=1.2)
    if state["stop"]:
        ...  # halt immediately
    else:
        scale = zone.speed_scale(state["min_clearance_m"])
        cmd_vel *= scale
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class SafetyState:
    """Structured result of a safety evaluation."""

    stop: bool
    slow_down: bool
    inhibit_left: bool
    inhibit_right: bool
    inhibit_forward: bool
    min_clearance_m: float
    trigger_reason: str

    def to_dict(self) -> dict:
        return {
            "stop": self.stop,
            "slow_down": self.slow_down,
            "inhibit_left": self.inhibit_left,
            "inhibit_right": self.inhibit_right,
            "inhibit_forward": self.inhibit_forward,
            "min_clearance_m": round(self.min_clearance_m, 4),
            "trigger_reason": self.trigger_reason,
        }

    @property
    def any_inhibited(self) -> bool:
        """True if any motion direction is inhibited."""
        return (
            self.stop
            or self.slow_down
            or self.inhibit_left
            or self.inhibit_right
            or self.inhibit_forward
        )


class SafetyZone:
    """
    Defines and evaluates safety zones around the AGV.

    Three concentric distance thresholds determine the response:

    * **stop_distance**: Trigger an emergency stop (hard limit).
    * **slow_down_distance**: Reduce forward speed proportionally.
    * **lateral_inhibit**: Block lateral (sideways) motion on that side.

    Safety evaluation can be performed directly from a 1D depth scan (e.g. 64
    pseudo-LiDAR bins) or from pre-computed per-sector minimum distances.

    Parameters
    ----------
    stop_distance : float
        Distance (meters) below which a full stop is triggered.
    slow_down_distance : float
        Distance (meters) below which speed is reduced.
    lateral_inhibit : float
        Distance (meters) below which lateral motion on the blocked side is inhibited.
    scan_sector_config : dict or None
        Optionally override the default sector split configuration.
        Default splits scan_dim into left/front/right thirds.
    """

    def __init__(
        self,
        stop_distance: float = 0.15,
        slow_down_distance: float = 0.50,
        lateral_inhibit: float = 0.30,
        scan_sector_config: Optional[dict] = None,
    ):
        if not (0 < stop_distance <= slow_down_distance):
            raise ValueError(
                f"Require 0 < stop_distance ({stop_distance}) "
                f"<= slow_down_distance ({slow_down_distance})"
            )
        self.stop_distance = stop_distance
        self.slow_down_distance = slow_down_distance
        self.lateral_inhibit = lateral_inhibit
        self._sector_config = scan_sector_config

    # ------------------------------------------------------------------
    # Public API: scan-based evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        scan_m: np.ndarray,
        front_min: Optional[float] = None,
        left_min: Optional[float] = None,
        right_min: Optional[float] = None,
    ) -> dict:
        """
        Evaluate safety state from scan data and/or pre-computed sector mins.

        Parameters
        ----------
        scan_m : np.ndarray
            1D depth scan in meters.  If the array is empty or all values
            are NaN/Inf, per-sector minimums must be provided explicitly.
        front_min : float or None
            Pre-computed minimum distance in the front sector (overrides
            scan-derived value if provided).
        left_min : float or None
            Pre-computed minimum distance in the left sector.
        right_min : float or None
            Pre-computed minimum distance in the right sector.

        Returns
        -------
        dict
            Keys: ``stop``, ``slow_down``, ``inhibit_left``,
            ``inhibit_right``, ``inhibit_forward``, ``min_clearance_m``,
            ``trigger_reason``.
        """
        # Compute per-sector minima from scan if not explicitly provided
        if front_min is None or left_min is None or right_min is None:
            sector_mins = self._compute_sector_minima(scan_m)
        else:
            sector_mins = {
                "front": front_min,
                "left": left_min,
                "right": right_min,
            }

        # Override any explicitly provided values
        if front_min is not None:
            sector_mins["front"] = front_min
        if left_min is not None:
            sector_mins["left"] = left_min
        if right_min is not None:
            sector_mins["right"] = right_min

        f_min = sector_mins["front"]
        l_min = sector_mins["left"]
        r_min = sector_mins["right"]

        # Global minimum clearance
        min_clearance = float(min(f_min, l_min, r_min))
        # Cap at a reasonable upper bound
        min_clearance = min(min_clearance, self.slow_down_distance * 2.0)

        # ----- Stop -----
        if f_min <= self.stop_distance:
            return SafetyState(
                stop=True,
                slow_down=True,
                inhibit_left=True,
                inhibit_right=True,
                inhibit_forward=True,
                min_clearance_m=f_min,
                trigger_reason=f"stop: front_min={f_min:.3f}m <= stop={self.stop_distance:.3f}m",
            ).to_dict()

        if l_min <= self.stop_distance:
            return SafetyState(
                stop=True,
                slow_down=True,
                inhibit_left=True,
                inhibit_right=True,
                inhibit_forward=True,
                min_clearance_m=l_min,
                trigger_reason=f"stop: left_min={l_min:.3f}m <= stop={self.stop_distance:.3f}m",
            ).to_dict()

        if r_min <= self.stop_distance:
            return SafetyState(
                stop=True,
                slow_down=True,
                inhibit_left=True,
                inhibit_right=True,
                inhibit_forward=True,
                min_clearance_m=r_min,
                trigger_reason=f"stop: right_min={r_min:.3f}m <= stop={self.stop_distance:.3f}m",
            ).to_dict()

        # ----- Slow-down -----
        slow_down = f_min <= self.slow_down_distance
        slow_reason = ""
        if slow_down:
            slow_reason = f"slow_down: front_min={f_min:.3f}m <= slow={self.slow_down_distance:.3f}m"

        # ----- Lateral inhibit -----
        inhibit_left = l_min <= self.lateral_inhibit
        inhibit_right = r_min <= self.lateral_inhibit
        inhibit_forward = f_min <= self.stop_distance  # already checked above, always False here

        lateral_reasons: List[str] = []
        if inhibit_left:
            lateral_reasons.append(
                f"inhibit_left: left_min={l_min:.3f}m <= lateral={self.lateral_inhibit:.3f}m"
            )
        if inhibit_right:
            lateral_reasons.append(
                f"inhibit_right: right_min={r_min:.3f}m <= lateral={self.lateral_inhibit:.3f}m"
            )

        # Compose reason string
        reasons: List[str] = []
        if slow_reason:
            reasons.append(slow_reason)
        reasons.extend(lateral_reasons)
        if not reasons:
            reasons.append("safe: all sectors clear")

        return SafetyState(
            stop=False,
            slow_down=slow_down,
            inhibit_left=inhibit_left,
            inhibit_right=inhibit_right,
            inhibit_forward=inhibit_forward,
            min_clearance_m=min_clearance,
            trigger_reason="; ".join(reasons),
        ).to_dict()

    def evaluate_from_obstacle_detector(self, detector_result: dict) -> dict:
        """
        Evaluate safety state directly from the output of
        :class:`ObstacleDetector.detect()`.

        Parameters
        ----------
        detector_result : dict
            The dictionary returned by ``ObstacleDetector.detect()``.

        Returns
        -------
        dict
            Same as :meth:`evaluate()`.
        """
        sectors = detector_result.get("sectors", {})
        front_min = sectors.get("front", {}).get("min_dist", float("inf"))
        left_min = sectors.get("left", {}).get("min_dist", float("inf"))
        right_min = sectors.get("right", {}).get("min_dist", float("inf"))

        return self.evaluate(
            scan_m=np.array([]),
            front_min=front_min,
            left_min=left_min,
            right_min=right_min,
        )

    # ------------------------------------------------------------------
    # Public API: speed scaling
    # ------------------------------------------------------------------

    def speed_scale(self, distance_m: float) -> float:
        """
        Compute a speed scaling factor (0.0 to 1.0) based on clearance distance.

        The scaling ramps linearly from 0 at ``stop_distance`` to 1 at
        ``slow_down_distance``.  Distances beyond ``slow_down_distance``
        return 1.0; distances below ``stop_distance`` return 0.0.

        Parameters
        ----------
        distance_m : float
            Clearance distance in meters (e.g. ``state["min_clearance_m"]``).

        Returns
        -------
        float
            Speed scale factor clamped to [0, 1].
        """
        if distance_m <= self.stop_distance:
            return 0.0
        if distance_m >= self.slow_down_distance:
            return 1.0

        span = self.slow_down_distance - self.stop_distance
        if span <= 1e-6:
            return 0.0
        scale = (distance_m - self.stop_distance) / span
        return float(np.clip(scale, 0.0, 1.0))

    def speed_scale_smooth(self, distance_m: float) -> float:
        """
        Smooth speed scaling using a sigmoid-like curve (smoother at edges).

        Uses a cubic hermite (smoothstep) between ``stop_distance`` and
        ``slow_down_distance``.
        """
        if distance_m <= self.stop_distance:
            return 0.0
        if distance_m >= self.slow_down_distance:
            return 1.0

        t = (distance_m - self.stop_distance) / (
            self.slow_down_distance - self.stop_distance
        )
        # Smoothstep: 3t^2 - 2t^3
        scale = t * t * (3.0 - 2.0 * t)
        return float(np.clip(scale, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Public API: sector helpers
    # ------------------------------------------------------------------

    def sector_distances(self, scan_m: np.ndarray) -> Dict[str, float]:
        """
        Return ``{left, front, right}`` minimum distances from a scan.

        Parameters
        ----------
        scan_m : np.ndarray
            1D depth scan in meters.

        Returns
        -------
        dict[str, float]
        """
        return self._compute_sector_minima(scan_m)

    def sector_safety(self, scan_m: np.ndarray) -> Dict[str, str]:
        """
        Return per-sector risk level from a scan.

        Returns
        -------
        dict[str, str]
            ``{left: "safe"|"warning"|"danger", ...}``
        """
        mins = self._compute_sector_minima(scan_m)
        safety: Dict[str, str] = {}
        for sector, dist in mins.items():
            if dist <= self.stop_distance:
                safety[sector] = "danger"
            elif dist <= self.slow_down_distance:
                safety[sector] = "warning"
            else:
                safety[sector] = "safe"
        return safety

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_sector_minima(
        self, scan_m: np.ndarray
    ) -> Dict[str, float]:
        """Split scan into left/front/right and return per-sector minimums.

        Returns ``float("inf")`` for sectors with no valid data.
        """
        n = len(scan_m)
        if n == 0:
            return {"left": float("inf"), "front": float("inf"), "right": float("inf")}

        # Use default or custom sector split
        if self._sector_config is not None:
            left_range = self._sector_config.get("left", (0, n // 3))
            front_range = self._sector_config.get("front", (n // 3, 2 * n // 3))
            right_range = self._sector_config.get("right", (2 * n // 3, n))
        else:
            third = n // 3
            remainder = n - 3 * third
            left_range = (0, third)
            front_range = (third, third + third + remainder)
            right_range = (third + third + remainder, n)

        def _safe_min(scan_slice: np.ndarray) -> float:
            valid = scan_slice[np.isfinite(scan_slice) & (scan_slice > 0)]
            if len(valid) == 0:
                return float("inf")
            return float(np.min(valid))

        return {
            "left": _safe_min(scan_m[left_range[0] : left_range[1]]),
            "front": _safe_min(scan_m[front_range[0] : front_range[1]]),
            "right": _safe_min(scan_m[right_range[0] : right_range[1]]),
        }
