"""
Depth-based 3D localization module.

Given a bounding box (or center pixel) and a depth image,
estimate the target's 3D position in camera coordinates and
compute bearing angle for directional guidance.
"""
import logging
from typing import Optional, Tuple

import numpy as np

from .pointcloud_utils import pixel_to_camera, bearing_angle

logger = logging.getLogger(__name__)


class DepthLocalizer:
    """Estimate 3D position of detected objects from depth.

    Args:
        method: depth sampling method — 'median', 'percentile_20', or 'mode'.
        depth_sample_ratio: fraction of bbox center region to sample.
        min_valid_depth_ratio: minimum fraction of valid depth pixels required.
    """

    def __init__(
        self,
        method: str = "median",
        depth_sample_ratio: float = 0.3,
        min_valid_depth_ratio: float = 0.1,
    ):
        self.method = method
        self.depth_sample_ratio = depth_sample_ratio
        self.min_valid_depth_ratio = min_valid_depth_ratio

    def localize(
        self,
        bbox: list[int],
        depth_image: np.ndarray,
        intrinsics: dict,
    ) -> dict:
        """Estimate 3D position from a 2D bounding box.

        Args:
            bbox: [x1, y1, x2, y2] in pixel coordinates.
            depth_image: HxW float32 depth in meters.
            intrinsics: camera intrinsics dict.

        Returns:
            {
                "center_pixel": [u, v],
                "depth_m": float,
                "position_camera_xyz": [x, y, z],
                "bearing_angle_deg": float,
                "bearing_direction": "left" | "front" | "right",
                "valid_depth_ratio": float,
            }
        """
        return self._localize_bbox(bbox, depth_image, intrinsics)

    def localize_center(
        self,
        center_pixel: tuple[int, int],
        depth_image: np.ndarray,
        intrinsics: dict,
    ) -> dict:
        """Estimate 3D position from a center pixel.

        Args:
            center_pixel: (u, v).
            depth_image: HxW float32 depth in meters.
            intrinsics: camera intrinsics dict.

        Returns:
            same format as localize().
        """
        u, v = center_pixel
        bbox = [u - 5, v - 5, u + 5, v + 5]
        return self._localize_bbox(bbox, depth_image, intrinsics)

    def _localize_bbox(
        self,
        bbox: list[int],
        depth_image: np.ndarray,
        intrinsics: dict,
    ) -> dict:
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        bw = x2 - x1
        bh = y2 - y1
        h, w = depth_image.shape

        # Sample center portion
        sx1 = max(0, int(cx - bw * self.depth_sample_ratio / 2))
        sx2 = min(w, int(cx + bw * self.depth_sample_ratio / 2))
        sy1 = max(0, int(cy - bh * self.depth_sample_ratio / 2))
        sy2 = min(h, int(cy + bh * self.depth_sample_ratio / 2))

        region = depth_image[sy1:sy2, sx1:sx2]
        valid = region[region > 0]
        valid_ratio = len(valid) / region.size if region.size > 0 else 0

        result: dict = {
            "center_pixel": [cx, cy],
            "depth_m": None,
            "position_camera_xyz": None,
            "bearing_angle_deg": None,
            "bearing_direction": "unknown",
            "valid_depth_ratio": round(valid_ratio, 3),
        }

        if len(valid) < 5 or valid_ratio < self.min_valid_depth_ratio:
            return result

        z = self._estimate_depth(valid)
        x, y = pixel_to_camera(cx, cy, z, intrinsics)
        bearing = float(np.degrees(bearing_angle(x, z)))

        result["depth_m"] = round(z, 3)
        result["position_camera_xyz"] = [round(x, 3), round(y, 3), round(z, 3)]
        result["bearing_angle_deg"] = round(bearing, 1)
        result["bearing_direction"] = self._direction_from_bearing(bearing)

        return result

    def _estimate_depth(self, valid_depths: np.ndarray) -> float:
        if self.method == "median":
            return float(np.median(valid_depths))
        elif self.method == "percentile_20":
            return float(np.percentile(valid_depths, 20))
        elif self.method == "mode":
            # Simple histogram mode
            hist, edges = np.histogram(valid_depths, bins=20)
            idx = np.argmax(hist)
            return float((edges[idx] + edges[idx + 1]) / 2)
        return float(np.median(valid_depths))

    @staticmethod
    def _direction_from_bearing(bearing_deg: float) -> str:
        if bearing_deg < -15:
            return "left"
        elif bearing_deg > 15:
            return "right"
        return "front"
