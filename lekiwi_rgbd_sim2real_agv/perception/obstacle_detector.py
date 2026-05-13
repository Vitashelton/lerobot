"""
Obstacle detector using depth scan data.

Splits a 1D depth scan (e.g., 64-bin pseudo-LiDAR) into left, front, and
right sectors. Computes per-sector minimum distance, risk level, and clusters
contiguous near-range pixels into discrete obstacles with bounding-box and
bearing information.

Typical usage:

    detector = ObstacleDetector(scan_dim=64, safe_threshold=1.0, danger_threshold=0.2)
    result = detector.detect(scan_meters)
    if result["risk_level"] == "danger":
        ...  # trigger emergency stop
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class SectorInfo:
    """Per-sector analysis result."""

    indices: np.ndarray
    min_dist: float
    risk: str  # "safe" | "warning" | "danger"
    median_dist: float = 0.0
    obstacle_count: int = 0


@dataclass
class Obstacle:
    """A single detected obstacle."""

    bbox: Tuple[int, int, int, int]  # [x1, y1, x2, y2] in scan-index / distance space
    distance: float  # minimum distance in meters
    bearing_deg: float  # bearing angle in degrees
    sector: str  # "left" | "front" | "right"
    width_indices: int = 0  # number of scan bins occupied
    mean_distance: float = 0.0


class ObstacleDetector:
    """
    Detects obstacles from a 1D depth scan array.

    The scan is assumed to cover a horizontal field of view (default 87 deg) and
    is divided evenly into left, front, and right thirds.  Each sector receives a
    risk label ("safe", "warning", "danger") based on the closest pixel within it.

    Consecutive pixels that fall below *grouping_threshold* are clustered into
    discrete obstacles and reported alongside sector summaries.

    Parameters
    ----------
    scan_dim : int
        Number of bins in the 1D scan.
    safe_threshold : float
        Distance (meters) above which the sector is considered safe.
    warning_threshold : float
        Distance (meters) below which the sector triggers a warning.
    danger_threshold : float
        Distance (meters) below which the sector triggers danger / emergency stop.
    fov_horizontal_deg : float
        Horizontal field of view of the scan in degrees (used for bearing).
    grouping_threshold : float
        Maximum gap in meters between consecutive near pixels for them to be
        considered part of the same obstacle cluster.
    """

    def __init__(
        self,
        scan_dim: int = 64,
        safe_threshold: float = 1.0,
        warning_threshold: float = 0.5,
        danger_threshold: float = 0.2,
        fov_horizontal_deg: float = 87.0,
        grouping_threshold: float = 0.3,
    ):
        self.scan_dim = scan_dim
        self.safe_threshold = safe_threshold
        self.warning_threshold = warning_threshold
        self.danger_threshold = danger_threshold
        self.fov_horizontal_deg = fov_horizontal_deg
        self.grouping_threshold = grouping_threshold

        # Pre-compute sector index splits
        self._sector_bounds = self._split_sectors(scan_dim)
        # Pre-compute per-index bearing angles
        self._bearings = self._compute_bearings(scan_dim, fov_horizontal_deg)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, scan_m: np.ndarray) -> dict:
        """
        Detect obstacles in each sector and return a structured result.

        Parameters
        ----------
        scan_m : np.ndarray
            1D float array of depth values in meters.  Length must equal
            ``self.scan_dim``.  Values may be ``np.inf`` or large for
            out-of-range readings.

        Returns
        -------
        dict
            Dictionary with keys ``risk_level``, ``sectors``, and ``obstacles``.
        """
        if scan_m.ndim != 1 or len(scan_m) != self.scan_dim:
            raise ValueError(
                f"scan_m must be 1D with length {self.scan_dim}, "
                f"got shape {getattr(scan_m, 'shape', ())}"
            )

        # Clamp infinite / NaN values to a large sentinel
        scan = np.where(
            np.isfinite(scan_m) & (scan_m > 0),
            scan_m,
            np.finfo(np.float32).max,
        )

        # Per-sector analysis
        sectors: dict[str, dict] = {}
        for name, (start, end) in self._sector_bounds.items():
            sector_scan = scan[start:end]
            min_dist = float(np.min(sector_scan))
            risk = self._classify_risk(min_dist)
            sectors[name] = {
                "min_dist": round(min_dist, 4),
                "risk": risk,
                "indices": list(range(start, end)),
                "median_dist": round(float(np.median(sector_scan[sector_scan < 999])), 4),
                "obstacle_count": int(np.sum(sector_scan < self.warning_threshold)),
            }

        # Global risk level (worst across sectors)
        risk_levels = [s["risk"] for s in sectors.values()]
        if "danger" in risk_levels:
            risk_level = "danger"
        elif "warning" in risk_levels:
            risk_level = "warning"
        else:
            risk_level = "safe"

        # Obstacle clustering
        obstacles = self._find_obstacle_clusters(scan)

        return {
            "risk_level": risk_level,
            "sectors": sectors,
            "obstacles": obstacles,
        }

    def sector_analysis(self, scan_m: np.ndarray) -> dict[str, SectorInfo]:
        """Return per-sector :class:`SectorInfo` dataclass objects."""
        result = self.detect(scan_m)
        out: dict[str, SectorInfo] = {}
        for name, s in result["sectors"].items():
            out[name] = SectorInfo(
                indices=np.array(s["indices"]),
                min_dist=s["min_dist"],
                risk=s["risk"],
                median_dist=s["median_dist"],
                obstacle_count=s["obstacle_count"],
            )
        return out

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _split_sectors(self, scan_dim: int) -> dict[str, Tuple[int, int]]:
        """Split scan indices into three ranges: left, front, right."""
        third = scan_dim // 3
        remainder = scan_dim - 3 * third
        # Distribute remainder to front sector
        left_end = third
        front_end = third + third + remainder
        return {
            "left": (0, left_end),
            "front": (left_end, front_end),
            "right": (front_end, scan_dim),
        }

    def _compute_bearings(
        self, scan_dim: int, fov_deg: float
    ) -> np.ndarray:
        """Pre-compute bearing angle (deg) for each scan index.

        Index 0 = left edge of FOV, last index = right edge.
        Bearing 0 = straight ahead; negative = left; positive = right.
        """
        half_fov = fov_deg / 2.0
        return np.linspace(-half_fov, half_fov, scan_dim)

    def _classify_risk(self, min_dist_m: float) -> str:
        if min_dist_m <= self.danger_threshold:
            return "danger"
        elif min_dist_m <= self.warning_threshold:
            return "warning"
        return "safe"

    def _find_obstacle_clusters(self, scan: np.ndarray) -> List[dict]:
        """Cluster consecutive near-range pixels into discrete obstacles.

        A pixel is "near" if its distance is below ``warning_threshold``.
        Pixels separated by more than ``grouping_threshold`` meters in depth
        (or with gaps that are not near-range) start a new cluster.

        Returns
        -------
        list[dict]
            Each dict with ``bbox``, ``distance``, ``bearing_deg``, ``sector``,
            ``width_indices``, ``mean_distance``.
        """
        near_mask = scan < self.warning_threshold
        if not np.any(near_mask):
            return []

        # Identify runs of consecutive near-range indices
        clusters = self._extract_runs(near_mask, scan)
        return clusters

    def _extract_runs(
        self, near_mask: np.ndarray, scan: np.ndarray
    ) -> List[dict]:
        """Extract contiguous runs of near_mask and split by depth gaps."""
        # Find where near_mask transitions
        padded = np.concatenate([[False], near_mask, [False]])
        starts = np.where(~padded[:-1] & padded[1:])[0]
        ends = np.where(padded[:-1] & ~padded[1:])[0]

        obstacles: List[dict] = []
        for s, e in zip(starts, ends):
            # Optionally split this run further if depth gap exceeds threshold
            sub_runs = self._split_by_depth_gap(s, e, scan)
            for sub_start, sub_end in sub_runs:
                indices = np.arange(sub_start, sub_end)
                segment = scan[sub_start:sub_end]
                min_dist = float(np.min(segment))
                mean_dist = float(np.mean(segment))
                width = sub_end - sub_start
                center_idx = (sub_start + sub_end) / 2.0
                bearing = self._index_to_bearing(center_idx)

                # Determine sector
                sector = self._index_to_sector(sub_start)

                obstacles.append(
                    {
                        "bbox": [sub_start, 0, sub_end, 1],
                        "distance": round(min_dist, 4),
                        "bearing_deg": round(float(bearing), 2),
                        "sector": sector,
                        "width_indices": width,
                        "mean_distance": round(mean_dist, 4),
                    }
                )
        return obstacles

    def _split_by_depth_gap(
        self, start: int, end: int, scan: np.ndarray
    ) -> List[Tuple[int, int]]:
        """Split an index range further where depth gaps exceed threshold."""
        segment = scan[start:end]
        diffs = np.abs(np.diff(segment))
        split_points = np.where(diffs > self.grouping_threshold)[0] + 1  # offset

        if len(split_points) == 0:
            return [(start, end)]

        runs: List[Tuple[int, int]] = []
        prev = start
        for sp in split_points:
            runs.append((prev, start + sp))
            prev = start + sp
        runs.append((prev, end))
        return runs

    def _index_to_bearing(self, index: float) -> float:
        """Convert a (possibly fractional) scan index to bearing degrees."""
        if index < 0:
            index = 0
        elif index >= self.scan_dim:
            index = self.scan_dim - 1

        lo = int(np.floor(index))
        hi = int(np.ceil(index))
        if lo == hi:
            return float(self._bearings[lo])
        frac = index - lo
        return float((1 - frac) * self._bearings[lo] + frac * self._bearings[hi])

    def _index_to_sector(self, index: int) -> str:
        for name, (start, end) in self._sector_bounds.items():
            if start <= index < end:
                return name
        return "unknown"
