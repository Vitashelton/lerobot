"""
Risk scorer that computes collision risk from LiDAR scan data.

Used for:
  - Generating weak / safety labels during dataset construction.
  - Computing risk-weighted loss terms during training.
  - Real-time risk assessment in the control loop.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


class RiskScorer:
    """Compute collision risk from a 64-beam LiDAR scan (metres).

    The scanner is assumed to cover 360 degrees, centred on the robot,
    with beam 0 pointing forward (0 deg) and beams increasing
    counter-clockwise.  The helper :meth:`sector_ranges` extracts the
    four cardinal sectors (front / back / left / right), each spanning
    a 90-degree wedge.
    """

    # Number of beams per sector (64 beams / 4 sectors = 16).
    SECTOR_SIZE = 16

    def __init__(
        self,
        collision_threshold: float = 0.15,
        danger_threshold: float = 0.3,
        warning_threshold: float = 0.5,
        emergency_multiplier: float = 1.5,
    ) -> None:
        """
        Parameters
        ----------
        collision_threshold : float
            Distance (m) below which a collision is considered imminent.
        danger_threshold : float
            Distance (m) below which the situation is dangerous.
        warning_threshold : float
            Distance (m) below which the system is warned.
        emergency_multiplier : float
            Multiplier on *collision_threshold* used to trigger a full
            rotation stop in the emergency shield.
        """
        self.collision_threshold = collision_threshold
        self.danger_threshold = danger_threshold
        self.warning_threshold = warning_threshold
        self.emergency_multiplier = emergency_multiplier

    # ------------------------------------------------------------------
    # Sector helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sector_range(scan: np.ndarray, start: int, end: int) -> np.ndarray:
        """Extract a contiguous angular sector from the scan."""
        n = len(scan)
        start = start % n
        end = end % n
        if start <= end:
            return scan[start:end]
        # Wrap-around case
        return np.concatenate([scan[start:], scan[:end]])

    @staticmethod
    def sector_ranges(scan: np.ndarray) -> dict[str, np.ndarray]:
        """Return front, back, left, right sector arrays.

        Sectors (64-beam, 0 = forward, CCW):
            front   : beams 56..8   (centred on 0)
            right   : beams 8..24   (centred on 90)
            back    : beams 24..40  (centred on 180)
            left    : beams 40..56  (centred on 270)
        """
        n = len(scan)
        h = n // 4  # 16
        return {
            "front": RiskScorer._sector_range(scan, n - h // 2, h // 2),
            "right": RiskScorer._sector_range(scan, h // 2, h + h // 2),
            "back": RiskScorer._sector_range(scan, h + h // 2, 2 * h + h // 2),
            "left": RiskScorer._sector_range(scan, 2 * h + h // 2, 3 * h + h // 2),
        }

    # ------------------------------------------------------------------
    # Risk computation
    # ------------------------------------------------------------------

    def compute_risk(self, scan_m: np.ndarray) -> dict:
        """Compute collision risk metrics from a full scan.

        Parameters
        ----------
        scan_m : np.ndarray, shape (N,)
            LiDAR ranges in metres.  Typically N = 64.

        Returns
        -------
        dict
            Keys:
            - ``collision_risk`` : float [0, 1]
            - ``min_distance`` : float
            - ``danger_ratio`` : float
                Fraction of beams below *danger_threshold*.
            - ``warning_ratio`` : float
                Fraction of beams below *warning_threshold*.
            - ``closest_sector`` : str
                One of ``"front"``, ``"back"``, ``"left"``, ``"right"``.
            - ``is_emergency`` : bool
                True if any beam is below ``collision_threshold * emergency_multiplier``.
            - ``per_sector_min`` : dict[str, float]
                Minimum distance in each sector.
        """
        scan_m = np.asarray(scan_m, dtype=np.float32)
        min_dist = float(scan_m.min())

        sectors = self.sector_ranges(scan_m)
        per_sector_min = {k: float(v.min()) for k, v in sectors.items()}
        closest_sector = min(per_sector_min, key=lambda k: per_sector_min[k])  # type: ignore[type-var]

        danger_beams = (scan_m < self.danger_threshold).sum()
        warning_beams = (scan_m < self.warning_threshold).sum()
        n = len(scan_m)

        danger_ratio = float(danger_beams / n) if n > 0 else 0.0
        warning_ratio = float(warning_beams / n) if n > 0 else 0.0

        # Collision risk: 1 when min_dist <= collision_threshold, decays linearly to 0 at danger_threshold.
        if min_dist <= self.collision_threshold:
            collision_risk = 1.0
        elif min_dist >= self.danger_threshold:
            collision_risk = 0.0
        else:
            collision_risk = float(
                1.0 - (min_dist - self.collision_threshold) / (self.danger_threshold - self.collision_threshold)
            )

        is_emergency = min_dist < self.collision_threshold * self.emergency_multiplier

        return {
            "collision_risk": collision_risk,
            "min_distance": min_dist,
            "danger_ratio": danger_ratio,
            "warning_ratio": warning_ratio,
            "closest_sector": closest_sector,
            "is_emergency": is_emergency,
            "per_sector_min": per_sector_min,
        }

    # ------------------------------------------------------------------
    # Label generation
    # ------------------------------------------------------------------

    def compute_safety_label(
        self,
        scan_m: np.ndarray,
        action: np.ndarray,
        safe_action: np.ndarray,
    ) -> dict:
        """Compare raw action vs safe action to generate a training label.

        Parameters
        ----------
        scan_m : np.ndarray
            LiDAR scan in metres.
        action : np.ndarray, shape (3,)
            Raw proposed action [vx, vy, omega].
        safe_action : np.ndarray, shape (3,)
            Expert / safe action [vx, vy, omega] (e.g. from DWA).

        Returns
        -------
        dict
            Keys:
            - ``delta_action`` : np.ndarray, shape (3,)
                safe_action - action, clipped to [-max_delta, max_delta].
            - ``weight`` : float
                Sample weight proportional to collision risk severity.
            - ``risk`` : dict
                The full risk dict for the scan.
        """
        delta = np.asarray(safe_action, dtype=np.float32) - np.asarray(action, dtype=np.float32)
        # Clamp delta so the network is not asked to learn extreme corrections.
        max_delta = np.array([0.5, 0.5, 90.0], dtype=np.float32)
        delta = np.clip(delta, -max_delta, max_delta)

        risk = self.compute_risk(scan_m)
        # Weight: high-risk samples contribute more to the loss.
        weight = float(
            risk["collision_risk"] * 0.7
            + risk["danger_ratio"] * 0.2
            + max(0.0, 1.0 - risk["min_distance"] / self.danger_threshold) * 0.1
        )
        weight = max(weight, 0.05)  # floor so every sample contributes a little

        return {
            "delta_action": delta,
            "weight": weight,
            "risk": risk,
        }

    # ------------------------------------------------------------------
    # Real-time utilities
    # ------------------------------------------------------------------

    def is_safe(self, scan_m: np.ndarray) -> bool:
        """Quick check: is the current state safe to move in?  Returns
        ``True`` when no beam is below *danger_threshold*."""
        return bool(np.all(np.asarray(scan_m) > self.danger_threshold))

    def get_front_min(self, scan_m: np.ndarray) -> float:
        """Return the minimum range in the front sector."""
        sectors = self.sector_ranges(scan_m)
        return float(sectors["front"].min())

    def get_sector_mins(self, scan_m: np.ndarray) -> dict[str, float]:
        """Return ``{sector: min_range}`` for all four sectors."""
        sectors = self.sector_ranges(scan_m)
        return {k: float(v.min()) for k, v in sectors.items()}


# ------------------------------------------------------------------
# Standalone convenience function
# ------------------------------------------------------------------

def compute_sample_weight(
    scan_m: np.ndarray,
    collision_threshold: float = 0.15,
    danger_threshold: float = 0.3,
) -> float:
    """Compute a scalar sample weight from a scan for use in a weighted loss.

    Returns a float in [0.05, 1.0].
    """
    scorer = RiskScorer(
        collision_threshold=collision_threshold,
        danger_threshold=danger_threshold,
    )
    risk = scorer.compute_risk(scan_m)
    w = float(
        risk["collision_risk"] * 0.7
        + risk["danger_ratio"] * 0.2
        + max(0.0, 1.0 - risk["min_distance"] / danger_threshold) * 0.1
    )
    return max(w, 0.05)
