"""
Depth localizer: converts image coordinates + depth values to 3D positions.

Provides utility functions to back-project a pixel into a 3D camera-frame
point given intrinsics and depth, to extract a representative depth value
from a bounding-box region, and to compute bearing angles and Euclidean
distances.

When intrinsics are not provided, they are estimated from image dimensions
using a default horizontal field of view.

Typical usage:

    localizer = DepthLocalizer(fx=615.0, fy=615.0, cx=320.0, cy=240.0)
    x, y, z = localizer.pixel_to_3d(u=160, v=120, depth_m=1.5)
    d = localizer.roi_depth(depth_map, bbox=(100, 80, 200, 160))
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


class DepthLocalizer:
    """
    Converts image coordinates and depth maps into 3D camera-frame positions.

    Supports multiple depth-aggregation methods for bounding-box regions and
    automatically estimates camera intrinsics from image size when calibration
    data is unavailable.

    Parameters
    ----------
    fx : float or None
        Focal length in x (pixels).  Estimated from FOV if None.
    fy : float or None
        Focal length in y (pixels).  Defaults to ``fx`` if None.
    cx : float or None
        Principal point x (pixels).  Defaults to ``width / 2`` if None.
    cy : float or None
        Principal point y (pixels).  Defaults to ``height / 2`` if None.
    aggregation : str
        Default depth aggregation method for ``roi_depth``.
        One of ``"median"``, ``"percentile_10"``, ``"percentile_20"``,
        ``"percentile_30"``, ``"mean"``, ``"min"``, ``"center"``.
    default_fov_horizontal_deg : float
        Horizontal FOV used to estimate intrinsics when ``fx`` is not provided.
    default_fov_vertical_deg : float or None
        Vertical FOV.  If None, derived from aspect ratio and horizontal FOV.
    """

    _AGGREGATION_FUNCTIONS = {
        "median": lambda valid: float(np.median(valid)),
        "percentile_10": lambda valid: float(np.percentile(valid, 10)),
        "percentile_20": lambda valid: float(np.percentile(valid, 20)),
        "percentile_30": lambda valid: float(np.percentile(valid, 30)),
        "mean": lambda valid: float(np.mean(valid)),
        "min": lambda valid: float(np.min(valid)),
        "center": None,  # handled separately
    }

    def __init__(
        self,
        fx: Optional[float] = None,
        fy: Optional[float] = None,
        cx: Optional[float] = None,
        cy: Optional[float] = None,
        aggregation: str = "median",
        default_fov_horizontal_deg: float = 87.0,
        default_fov_vertical_deg: Optional[float] = None,
    ):
        self._fx = fx
        self._fy = fy
        self._cx = cx
        self._cy = cy
        self.default_fov_horizontal_deg = default_fov_horizontal_deg
        self.default_fov_vertical_deg = default_fov_vertical_deg

        if aggregation not in self._AGGREGATION_FUNCTIONS:
            raise ValueError(
                f"Unknown aggregation '{aggregation}'. "
                f"Choose from: {list(self._AGGREGATION_FUNCTIONS.keys())}"
            )
        self.aggregation = aggregation

        # Resolved intrinsics, cached per image size
        self._fx_resolved: Optional[float] = fx
        self._fy_resolved: Optional[float] = fy if fy is not None else fx
        self._cx_resolved: Optional[float] = cx
        self._cy_resolved: Optional[float] = cy
        self._cached_img_size: Optional[Tuple[int, int]] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def pixel_to_3d(
        self,
        u: float,
        v: float,
        depth_m: float,
        img_width: Optional[int] = None,
        img_height: Optional[int] = None,
    ) -> Tuple[float, float, float]:
        """
        Convert a pixel coordinate plus depth to a 3D camera-frame point.

        Camera frame: x = right, y = down, z = forward.

        Parameters
        ----------
        u : float
            Horizontal pixel coordinate (column).
        v : float
            Vertical pixel coordinate (row).
        depth_m : float
            Depth in meters.
        img_width : int or None
            Image width in pixels (required if intrinsics were not provided).
        img_height : int or None
            Image height in pixels (required if intrinsics were not provided).

        Returns
        -------
        tuple (x, y, z)
            3D position in meters (camera frame).
        """
        if depth_m <= 0 or not np.isfinite(depth_m):
            return (float("nan"), float("nan"), float("nan"))

        if img_width is not None and img_height is not None:
            self._ensure_intrinsics(img_width, img_height)

        fx = self._fx_resolved
        fy = self._fy_resolved
        cx = self._cx_resolved
        cy = self._cy_resolved

        if fx is None or fy is None or cx is None or cy is None:
            raise ValueError(
                "Camera intrinsics not available. Provide fx/fy/cx/cy in "
                "constructor or pass img_width/img_height to pixel_to_3d()."
            )

        x = (u - cx) * depth_m / fx
        y = (v - cy) * depth_m / fy
        z = depth_m

        return (float(x), float(y), float(z))

    def roi_depth(
        self,
        depth_m: np.ndarray,
        bbox: Tuple[int, int, int, int],
        method: Optional[str] = None,
    ) -> float:
        """
        Extract a representative depth value for a bounding-box region.

        Parameters
        ----------
        depth_m : np.ndarray
            Depth map of shape ``(H, W)``, values in meters.
        bbox : tuple[int, int, int, int]
            ``(x1, y1, x2, y2)`` image coordinates (inclusive).
        method : str or None
            Aggregation method.  Defaults to ``self.aggregation``.

        Returns
        -------
        float
            Representative depth in meters, or NaN if no valid data.
        """
        x1, y1, x2, y2 = bbox
        # Clamp to image bounds
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(depth_m.shape[1], int(x2))
        y2 = min(depth_m.shape[0], int(y2))

        if x2 <= x1 or y2 <= y1:
            return float("nan")

        method = method if method is not None else self.aggregation

        if method == "center":
            cy = (y1 + y2) // 2
            cx = (x1 + x2) // 2
            val = depth_m[cy, cx]
            if np.isfinite(val) and val > 0.01:
                return float(val)
            return float("nan")

        roi = depth_m[y1:y2, x1:x2]
        valid = roi[np.isfinite(roi) & (roi > 0.01)]

        if len(valid) < 2:
            return float("nan")

        agg_fn = self._AGGREGATION_FUNCTIONS.get(method)
        if agg_fn is None:
            raise ValueError(f"Unknown aggregation method: '{method}'")
        return agg_fn(valid)

    def pixel_to_3d_from_bbox(
        self,
        bbox: Tuple[int, int, int, int],
        depth_m: np.ndarray,
        depth_method: Optional[str] = None,
        img_width: Optional[int] = None,
        img_height: Optional[int] = None,
    ) -> Tuple[float, float, float]:
        """
        Convenience: get depth from bbox ROI then back-project the bbox center.

        Parameters
        ----------
        bbox : tuple
            ``(x1, y1, x2, y2)``.
        depth_m : np.ndarray
            Depth map.
        depth_method : str or None
            Depth aggregation method.
        img_width, img_height : int or None
            Required if intrinsics were not provided.

        Returns
        -------
        tuple (x, y, z)
        """
        x1, y1, x2, y2 = bbox
        cx_u = (x1 + x2) / 2.0
        cy_v = (y1 + y2) / 2.0
        d = self.roi_depth(depth_m, bbox, method=depth_method)
        if not np.isfinite(d) or d <= 0:
            return (float("nan"), float("nan"), float("nan"))
        return self.pixel_to_3d(cx_u, cy_v, d, img_width, img_height)

    def compute_bearing(
        self,
        u: float,
        image_width: int,
        fov_horizontal_deg: float = 87.0,
    ) -> float:
        """
        Compute bearing angle (degrees) from pixel x-coordinate.

        Center of image = 0 deg.  Negative = left; positive = right.

        Parameters
        ----------
        u : float
            Horizontal pixel coordinate.
        image_width : int
            Image width in pixels.
        fov_horizontal_deg : float
            Horizontal field of view in degrees.

        Returns
        -------
        float
            Bearing angle in degrees.
        """
        half_fov = fov_horizontal_deg / 2.0
        center_x = image_width / 2.0
        bearing = (u - center_x) / center_x * half_fov
        return float(bearing)

    def compute_distance(self, position_3d: Tuple[float, float, float]) -> float:
        """
        Euclidean distance from camera origin to a 3D point.

        Parameters
        ----------
        position_3d : tuple[float, float, float]
            ``(x, y, z)`` in camera frame.

        Returns
        -------
        float
            Distance in meters.
        """
        x, y, z = position_3d
        if not all(np.isfinite(v) for v in (x, y, z)):
            return float("nan")
        return float(np.sqrt(x * x + y * y + z * z))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_intrinsics(self, width: int, height: int) -> None:
        """Resolve or estimate intrinsics, caching by image size."""
        current_size = (width, height)
        if self._cached_img_size == current_size and self._fx_resolved is not None:
            return

        self._cached_img_size = current_size

        # Use provided values or estimate from FOV
        if self._fx is not None:
            self._fx_resolved = self._fx
        else:
            self._fx_resolved = width / (
                2.0 * np.tan(np.radians(self.default_fov_horizontal_deg / 2.0))
            )

        if self._fy is not None:
            self._fy_resolved = self._fy
        elif self._fx is not None:
            # Assume square pixels if only fx given
            self._fy_resolved = self._fx_resolved
        elif self.default_fov_vertical_deg is not None:
            self._fy_resolved = height / (
                2.0 * np.tan(np.radians(self.default_fov_vertical_deg / 2.0))
            )
        else:
            self._fy_resolved = self._fx_resolved  # square pixels

        self._cx_resolved = self._cx if self._cx is not None else width / 2.0
        self._cy_resolved = self._cy if self._cy is not None else height / 2.0

    @property
    def camera_matrix(self) -> np.ndarray:
        """Return the 3x3 intrinsic matrix (requires prior call to pixel_to_3d or manual setup)."""
        if self._fx_resolved is None:
            raise RuntimeError(
                "Intrinsics not yet resolved. Call pixel_to_3d() first "
                "or provide img_width/img_height."
            )
        return np.array(
            [
                [self._fx_resolved, 0, self._cx_resolved],
                [0, self._fy_resolved, self._cy_resolved],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )

    @property
    def fx(self) -> Optional[float]:
        return self._fx_resolved

    @property
    def fy(self) -> Optional[float]:
        return self._fy_resolved

    @property
    def cx(self) -> Optional[float]:
        return self._cx_resolved

    @property
    def cy(self) -> Optional[float]:
        return self._cy_resolved
