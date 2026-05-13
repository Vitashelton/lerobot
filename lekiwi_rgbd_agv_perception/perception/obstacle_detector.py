"""
Depth-based near-field obstacle detection.

Divides the depth image into left/front/right zones and finds obstacle regions
using connected components on thresholded depth.

Output format per obstacle:
    {
        "bbox": [x1, y1, x2, y2],
        "center_pixel": [u, v],
        "distance": z,          # average depth
        "min_distance": z_min,  # closest depth in region
        "region": "front" | "left" | "right",
        "risk_level": "safe" | "warning" | "danger",
        "area_pixels": int,
    }
"""
import logging
from typing import Optional

import numpy as np
import cv2
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)


class ObstacleDetector:
    """Detect obstacles from a depth image using depth thresholds and
    connected components.

    Args:
        danger_distance: distances below this are 'danger' (meters).
        warning_distance: distances below this are 'warning' (meters).
        min_obstacle_area: minimum connected component size in pixels.
        use_percentile: which percentile to use for zone distance (0=min, 10=p10).
        connectivity: 4 or 8 for connected components.
    """

    def __init__(
        self,
        danger_distance: float = 0.25,
        warning_distance: float = 0.5,
        min_obstacle_area: int = 200,
        use_percentile: int = 10,
        connectivity: int = 8,
        zone_left: tuple = (0.0, 0.33),
        zone_front: tuple = (0.33, 0.66),
        zone_right: tuple = (0.66, 1.0),
    ):
        self.danger_distance = danger_distance
        self.warning_distance = warning_distance
        self.min_obstacle_area = min_obstacle_area
        self.use_percentile = use_percentile
        self.connectivity = connectivity
        self.zone_left = zone_left
        self.zone_front = zone_front
        self.zone_right = zone_right

    def detect(self, depth: np.ndarray) -> list[dict]:
        """Detect obstacles in a depth image.

        Args:
            depth: HxW float32 depth image in meters (0 = invalid).

        Returns:
            list of obstacle dicts sorted by distance (nearest first).
        """
        h, w = depth.shape
        valid_mask = depth > 0

        # Binary obstacle mask: pixels closer than warning_distance
        close_mask = valid_mask & (depth < self.warning_distance)
        close_mask_u8 = close_mask.astype(np.uint8) * 255

        # Connected components
        connectivity = 4 if self.connectivity == 4 else 8
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            close_mask_u8, connectivity, cv2.CV_32S
        )

        obstacles = []
        for label_id in range(1, num_labels):
            area = stats[label_id, cv2.CC_STAT_AREA]
            if area < self.min_obstacle_area:
                continue

            x1 = stats[label_id, cv2.CC_STAT_LEFT]
            y1 = stats[label_id, cv2.CC_STAT_TOP]
            x2 = x1 + stats[label_id, cv2.CC_STAT_WIDTH]
            y2 = y1 + stats[label_id, cv2.CC_STAT_HEIGHT]
            cx = int(centroids[label_id][0])
            cy = int(centroids[label_id][1])

            region_mask = labels == label_id
            region_depths = depth[region_mask]
            valid_depths = region_depths[region_depths > 0]

            if len(valid_depths) == 0:
                continue

            avg_depth = float(np.median(valid_depths))
            min_depth = float(np.min(valid_depths))

            # Determine which zone this obstacle falls in
            region = self._get_region(cx, w)

            # Risk level
            risk = self._risk_level(min_depth)

            obstacles.append({
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "center_pixel": [cx, cy],
                "distance": round(avg_depth, 3),
                "min_distance": round(min_depth, 3),
                "region": region,
                "risk_level": risk,
                "area_pixels": int(area),
            })

        # Sort nearest first
        obstacles.sort(key=lambda o: o["distance"])
        return obstacles

    def zone_distances(self, depth: np.ndarray) -> dict:
        """Compute per-zone distance statistics.

        Args:
            depth: HxW float32 depth in meters.

        Returns:
            dict with keys 'left', 'front', 'right', each containing
            'min', 'p10', 'median', 'risk_level'.
        """
        h, w = depth.shape
        valid = depth > 0

        zones = {
            "left": (int(w * self.zone_left[0]), int(w * self.zone_left[1])),
            "front": (int(w * self.zone_front[0]), int(w * self.zone_front[1])),
            "right": (int(w * self.zone_right[0]), int(w * self.zone_right[1])),
        }

        result = {}
        for name, (x1, x2) in zones.items():
            x1, x2 = max(0, x1), min(w, x2)
            zone_depth = depth[:, x1:x2]
            zone_valid = zone_depth[zone_depth > 0]

            if len(zone_valid) == 0:
                result[name] = {"min": 5.0, "p10": 5.0, "median": 5.0, "risk_level": "safe"}
                continue

            p10 = float(np.percentile(zone_valid, self.use_percentile))
            zmin = float(np.min(zone_valid))
            zmed = float(np.median(zone_valid))
            result[name] = {
                "min": round(zmin, 3),
                "p10": round(p10, 3),
                "median": round(zmed, 3),
                "risk_level": self._risk_level(p10),
            }

        return result

    def _get_region(self, cx: int, width: int) -> str:
        frac = cx / width
        if frac < self.zone_front[0]:
            return "left"
        elif frac < self.zone_front[1]:
            return "front"
        else:
            return "right"

    def _risk_level(self, distance: float) -> str:
        if distance < self.danger_distance:
            return "danger"
        elif distance < self.warning_distance:
            return "warning"
        return "safe"


def draw_obstacle_overlay(
    image: np.ndarray,
    obstacles: list[dict],
    zone_distances: dict,
) -> np.ndarray:
    """Draw obstacle bounding boxes, zone info, and risk overlay on RGB image.

    Args:
        image: HxWx3 BGR image.
        obstacles: list of obstacle dicts from ObstacleDetector.detect().
        zone_distances: dict from ObstacleDetector.zone_distances().

    Returns:
        annotated BGR image.
    """
    img = image.copy()
    h, w = img.shape[:2]

    color_map = {"danger": (0, 0, 255), "warning": (0, 200, 255), "safe": (0, 255, 0)}

    for obs in obstacles:
        risk = obs["risk_level"]
        color = color_map[risk]
        x1, y1, x2, y2 = obs["bbox"]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cx, cy = obs["center_pixel"]
        cv2.circle(img, (cx, cy), 4, color, -1)
        label = f"{obs['region']} {obs['distance']:.2f}m {risk}"
        cv2.putText(img, label, (x1, max(y1 - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    # Draw zone overlays (semi-transparent bands at top)
    overlay = img.copy()
    zone_colors = {"left": (255, 200, 100), "front": (100, 200, 255), "right": (100, 255, 100)}
    zone_rects = {
        "left": (0, int(w * 0.33)),
        "front": (int(w * 0.33), int(w * 0.66)),
        "right": (int(w * 0.66), w),
    }
    for name, (x1, x2) in zone_rects.items():
        cv2.rectangle(overlay, (x1, 0), (x2, 30), zone_colors[name], -1)
        zd = zone_distances.get(name, {})
        text = f"{name[0].upper()}: {zd.get('p10', 5.0):.2f}m"
        cv2.putText(overlay, text, (x1 + 4, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)

    return img
