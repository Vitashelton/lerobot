"""
Color / shape based pallet detection (fallback when no ArUco/YOLO).

Modes:
    - color: detect regions matching specified HSV color ranges.
    - shape: detect rectangular contours as potential pallets.
"""
import logging
from typing import Optional

import numpy as np
import cv2

from .pointcloud_utils import pixel_to_camera, bearing_angle

logger = logging.getLogger(__name__)


class PalletDetector:
    """Detect pallet-like objects by color or shape.

    Args:
        mode: 'color' or 'shape'.
        color_ranges: dict of {name: {lower: [H,S,V], upper: [H,S,V]}}.
        min_area: minimum contour area in pixels.
        shape_min_area: minimum area for shape detection.
        shape_approx_epsilon: fraction of contour perimeter for polygon approx.
    """

    def __init__(
        self,
        mode: str = "color",
        color_ranges: Optional[dict] = None,
        min_area: int = 500,
        shape_min_area: int = 1000,
        shape_approx_epsilon: float = 0.02,
    ):
        self.mode = mode
        self.color_ranges = color_ranges or {
            "yellow": {"lower": [20, 100, 100], "upper": [35, 255, 255]},
            "blue": {"lower": [100, 100, 100], "upper": [130, 255, 255]},
        }
        self.min_area = min_area
        self.shape_min_area = shape_min_area
        self.shape_approx_epsilon = shape_approx_epsilon

    def detect(
        self,
        color_image: np.ndarray,
        depth_image: Optional[np.ndarray] = None,
        intrinsics: Optional[dict] = None,
    ) -> list[dict]:
        """Detect pallets by color or shape.

        Returns:
            list of detection dicts with bbox, center, distance, 3D position.
        """
        if self.mode == "color":
            return self._detect_color(color_image, depth_image, intrinsics)
        elif self.mode == "shape":
            return self._detect_shape(color_image, depth_image, intrinsics)
        else:
            logger.warning("Unknown pallet detection mode: %s", self.mode)
            return []

    def _detect_color(
        self,
        color_image: np.ndarray,
        depth_image: Optional[np.ndarray],
        intrinsics: Optional[dict],
    ) -> list[dict]:
        """Detect regions by HSV color thresholding."""
        hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
        results = []

        for color_name, ranges in self.color_ranges.items():
            lower = np.array(ranges["lower"], dtype=np.uint8)
            upper = np.array(ranges["upper"], dtype=np.uint8)
            mask = cv2.inRange(hsv, lower, upper)

            # Morphological cleanup
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < self.min_area:
                    continue

                x, y, w, h = cv2.boundingRect(cnt)
                cx, cy = x + w // 2, y + h // 2

                det = {
                    "type": "pallet_color",
                    "color": color_name,
                    "bbox": [x, y, x + w, y + h],
                    "center_pixel": [cx, cy],
                    "distance": None,
                    "position_camera": None,
                    "bearing_deg": None,
                }

                if depth_image is not None and intrinsics is not None:
                    dist, pos, bearing = self._localize(cx, cy, depth_image, intrinsics)
                    det["distance"] = dist
                    det["position_camera"] = pos
                    det["bearing_deg"] = bearing

                results.append(det)

        return results

    def _detect_shape(
        self,
        color_image: np.ndarray,
        depth_image: Optional[np.ndarray],
        intrinsics: Optional[dict],
    ) -> list[dict]:
        """Detect rectangular shapes as potential pallets."""
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        results = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.shape_min_area:
                continue

            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, self.shape_approx_epsilon * peri, True)

            # Look for roughly rectangular shapes (4-6 vertices after approx)
            if not (4 <= len(approx) <= 6):
                continue

            # Check aspect ratio is not extreme
            x, y, w, h = cv2.boundingRect(cnt)
            aspect = max(w, h) / (min(w, h) + 1e-6)
            if aspect > 4:
                continue

            cx, cy = x + w // 2, y + h // 2

            det = {
                "type": "pallet_shape",
                "bbox": [x, y, x + w, y + h],
                "center_pixel": [cx, cy],
                "distance": None,
                "position_camera": None,
                "bearing_deg": None,
            }

            if depth_image is not None and intrinsics is not None:
                dist, pos, bearing = self._localize(cx, cy, depth_image, intrinsics)
                det["distance"] = dist
                det["position_camera"] = pos
                det["bearing_deg"] = bearing

            results.append(det)

        return results

    @staticmethod
    def _localize(
        cx: int, cy: int,
        depth_image: np.ndarray,
        intrinsics: dict,
    ) -> tuple[Optional[float], Optional[list], Optional[float]]:
        r = 5
        h, w = depth_image.shape
        y1, y2 = max(0, cy - r), min(h, cy + r + 1)
        x1, x2 = max(0, cx - r), min(w, cx + r + 1)
        patch = depth_image[y1:y2, x1:x2]
        valid = patch[patch > 0]
        if len(valid) < 3:
            return None, None, None

        z = float(np.median(valid))
        x, y = pixel_to_camera(cx, cy, z, intrinsics)
        bearing = round(np.degrees(bearing_angle(x, z)), 1)
        return round(z, 3), [round(x, 3), round(y, 3), round(z, 3)], bearing
