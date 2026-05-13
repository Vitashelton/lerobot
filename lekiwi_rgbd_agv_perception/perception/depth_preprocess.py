"""
Depth image preprocessing: cleaning, filtering, hole filling, temporal smoothing.

Operations (applied in order):
    1. Remove invalid pixels (0, NaN, inf)
    2. Apply depth scale (raw uint16 -> meters)
    3. Clip to [min_depth, max_depth]
    4. Median filter (optional)
    5. Temporal smoothing (optional)
    6. Hole filling (optional)
"""
import logging
from typing import Optional

import numpy as np
import cv2
from scipy import ndimage

logger = logging.getLogger(__name__)


class DepthPreprocessor:
    """Clean and filter depth images from RealSense D435i.

    Args:
        min_depth: minimum valid depth in meters.
        max_depth: maximum valid depth in meters.
        depth_scale: multiplier to convert raw depth to meters (D435i: 0.001).
        use_median_filter: apply cv2.medianBlur.
        median_kernel: kernel size (must be odd).
        use_temporal_smoothing: smooth across frames.
        temporal_alpha: 0-1, weight for new frame (1 = no smoothing).
        use_hole_filling: fill small zero-regions.
        hole_filling_radius: max radius of holes to fill.
    """

    def __init__(
        self,
        min_depth: float = 0.15,
        max_depth: float = 5.0,
        depth_scale: float = 0.001,
        use_median_filter: bool = True,
        median_kernel: int = 5,
        use_temporal_smoothing: bool = True,
        temporal_alpha: float = 0.5,
        use_hole_filling: bool = True,
        hole_filling_radius: int = 2,
    ):
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.depth_scale = depth_scale
        self.use_median_filter = use_median_filter
        self.median_kernel = median_kernel
        self.use_temporal_smoothing = use_temporal_smoothing
        self.temporal_alpha = temporal_alpha
        self.use_hole_filling = use_hole_filling
        self.hole_filling_radius = hole_filling_radius

        self._prev_depth: Optional[np.ndarray] = None

    def process(self, depth_raw: np.ndarray, is_raw_uint16: bool = False) -> np.ndarray:
        """Process raw depth image and return clean depth in meters.

        Args:
            depth_raw: input depth image (uint16 raw or float32 meters).
            is_raw_uint16: if True, apply depth_scale conversion first.

        Returns:
            clean_depth: HxW float32 array in meters. Invalid pixels set to 0.
        """
        depth = depth_raw.astype(np.float32, copy=False)

        if is_raw_uint16:
            depth *= self.depth_scale

        # Step 1: remove invalid values
        invalid = ~np.isfinite(depth) | (depth <= 0)
        depth[invalid] = 0.0

        # Step 2: clip to valid range
        depth[(depth < self.min_depth) | (depth > self.max_depth)] = 0.0

        # Step 3: median filter
        if self.use_median_filter and self.median_kernel > 1:
            k = self.median_kernel
            if k % 2 == 0:
                k += 1
            depth = cv2.medianBlur(depth, k)

        # Step 4: temporal smoothing
        if self.use_temporal_smoothing and self._prev_depth is not None:
            mask = (depth > 0) & (self._prev_depth > 0)
            depth[mask] = (self.temporal_alpha * depth[mask] +
                           (1 - self.temporal_alpha) * self._prev_depth[mask])
            # For pixels where new depth is 0, keep previous
            keep_prev = (depth <= 0) & (self._prev_depth > 0)
            depth[keep_prev] = self._prev_depth[keep_prev]

        self._prev_depth = depth.copy()

        # Step 5: hole filling
        if self.use_hole_filling and self.hole_filling_radius > 0:
            mask = (depth <= 0).astype(np.uint8)
            # Dilate the valid regions to fill small holes
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (self.hole_filling_radius * 2 + 1, self.hole_filling_radius * 2 + 1)
            )
            filled = cv2.dilate(depth, kernel)
            depth[mask > 0] = filled[mask > 0]

        # Final: ensure invalid is 0
        depth[~np.isfinite(depth)] = 0.0

        return depth

    def reset_temporal(self):
        """Reset the temporal smoothing buffer (e.g. after scene change)."""
        self._prev_depth = None

    @staticmethod
    def colored_depth_map(
        depth: np.ndarray,
        max_display: float = 3.0,
        colormap: int = cv2.COLORMAP_JET,
        invalid_color: tuple = (0, 0, 0),
    ) -> np.ndarray:
        """Convert depth (meters) to a BGR colormap for visualization.

        Args:
            depth: HxW float32 depth in meters.
            max_display: depth value mapped to max color.
            colormap: OpenCV colormap enum.
            invalid_color: color for 0-depth pixels.

        Returns:
            HxWx3 uint8 BGR image.
        """
        valid = depth > 0
        depth_clamped = np.clip(depth, 0, max_display)
        depth_norm = (depth_clamped / max_display * 255).astype(np.uint8)
        colored = cv2.applyColorMap(depth_norm, colormap)
        colored[~valid] = invalid_color
        return colored
