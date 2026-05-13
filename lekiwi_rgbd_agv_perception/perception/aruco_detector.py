"""
ArUco / AprilTag marker detection for pallet/target simulation.

Detects ArUco markers in the RGB image, estimates their 3D pose
using depth information, and returns marker ID, bbox, center, and
camera-frame XYZ position.
"""
import logging
from typing import Optional

import numpy as np
import cv2

from .pointcloud_utils import pixel_to_camera, bearing_angle

logger = logging.getLogger(__name__)

# ArUco dictionary name -> OpenCV constant
ARUCO_DICT_MAP = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
}


class ArUcoDetector:
    """Detect ArUco markers and estimate 3D position using depth.

    Args:
        dictionary_name: ArUco dictionary name (see ARUCO_DICT_MAP).
        marker_size: physical marker side length in meters.
    """

    def __init__(
        self,
        dictionary_name: str = "DICT_4X4_50",
        marker_size: float = 0.08,
    ):
        dict_id = ARUCO_DICT_MAP.get(dictionary_name, cv2.aruco.DICT_4X4_50)
        self.dictionary = cv2.aruco.getPredefinedDictionary(dict_id)
        self.marker_size = marker_size

        params = cv2.aruco.DetectorParameters()
        params.adaptiveThreshWinSizeMin = 3
        params.adaptiveThreshWinSizeMax = 23
        params.adaptiveThreshWinSizeStep = 10
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector = cv2.aruco.ArucoDetector(self.dictionary, params)

    def detect(
        self,
        color_image: np.ndarray,
        depth_image: Optional[np.ndarray] = None,
        intrinsics: Optional[dict] = None,
    ) -> list[dict]:
        """Detect ArUco markers and optionally estimate 3D pose.

        Args:
            color_image: HxWx3 BGR image.
            depth_image: HxW float32 depth in meters (optional, for 3D localization).
            intrinsics: camera intrinsics dict (required if depth provided).

        Returns:
            list of marker dicts:
                {
                    "id": int,
                    "bbox": [x1, y1, x2, y2],
                    "corners": [[u,v], ...],  # 4 corners
                    "center_pixel": [u, v],
                    "position_camera": [x, y, z] | None,
                    "bearing_deg": float | None,
                    "distance": float | None,
                }
        """
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)

        results = []
        if ids is None:
            return results

        for i, marker_id in enumerate(ids.flatten()):
            corner = corners[i][0]  # 4x2
            x1 = int(np.min(corner[:, 0]))
            y1 = int(np.min(corner[:, 1]))
            x2 = int(np.max(corner[:, 0]))
            y2 = int(np.max(corner[:, 1]))
            cx = int(np.mean(corner[:, 0]))
            cy = int(np.mean(corner[:, 1]))

            result = {
                "id": int(marker_id),
                "bbox": [x1, y1, x2, y2],
                "corners": corner.tolist(),
                "center_pixel": [cx, cy],
                "position_camera": None,
                "bearing_deg": None,
                "distance": None,
            }

            # 3D localization via depth
            if depth_image is not None and intrinsics is not None:
                pos, dist, bearing = self._estimate_3d(
                    cx, cy, depth_image, intrinsics
                )
                result["position_camera"] = pos
                result["distance"] = dist
                result["bearing_deg"] = bearing

            results.append(result)

        return results

    def _estimate_3d(
        self, cx: int, cy: int,
        depth_image: np.ndarray,
        intrinsics: dict,
    ) -> tuple[Optional[list], Optional[float], Optional[float]]:
        """Estimate 3D position for a marker center using depth."""
        # Sample a small patch around center
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
        bearing = np.degrees(bearing_angle(x, z))

        return [round(x, 3), round(y, 3), round(z, 3)], round(z, 3), round(bearing, 1)


def draw_aruco_overlay(
    image: np.ndarray,
    markers: list[dict],
) -> np.ndarray:
    """Draw detected ArUco markers on the image.

    Args:
        image: HxWx3 BGR image.
        markers: list of marker dicts from ArUcoDetector.detect().

    Returns:
        annotated BGR image.
    """
    img = image.copy()
    for m in markers:
        x1, y1, x2, y2 = m["bbox"]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)

        cx, cy = m["center_pixel"]
        cv2.circle(img, (cx, cy), 5, (0, 255, 255), -1)

        label_parts = [f"ID:{m['id']}"]
        if m["distance"] is not None:
            label_parts.append(f"{m['distance']:.2f}m")
        if m["bearing_deg"] is not None:
            label_parts.append(f"{m['bearing_deg']:.0f}deg")
        label = " ".join(label_parts)
        cv2.putText(img, label, (x1, max(y1 - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    return img
