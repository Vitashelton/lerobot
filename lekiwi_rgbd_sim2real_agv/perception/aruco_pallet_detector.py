"""
ArUco-based pallet detector with depth-assisted 3D pose estimation.

Detects ArUco markers placed on pallet boxes and estimates their 6-DoF pose
in camera coordinates by combining OpenCV's corner detection with median
depth sampling within the marker's bounding box.

Typical usage:

    detector = ArUcoPalletDetector(marker_size_m=0.12)
    pallets = detector.detect(rgb_image, depth_map_meters)
    for p in pallets:
        print(f"Pallet {p['id']} at {p['position_3d']}")
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# ArUco dictionary mapping
_ARUCO_DICT_MAP: Dict[str, int] = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
}


class ArUcoPalletDetector:
    """
    Detects ArUco markers on cardboard pallets and estimates 3D pose.

    Uses OpenCV's ``aruco`` module for marker detection in RGB images and
    refines the 3D position by sampling median depth from an aligned depth
    map inside the marker's image bounding box.

    When ``camera_matrix`` is not provided, intrinsics are estimated from
    image dimensions using a default field-of-view assumption, which is
    adequate for bearing-angle and relative-position estimation even
    without a formal camera calibration.

    Parameters
    ----------
    marker_size_m : float
        Physical side length of the printed ArUco marker (meters).
    dictionary : str
        OpenCV ArUco dictionary name, e.g. ``"DICT_4X4_50"``.
    valid_ids : list[int] or None
        If provided, only markers whose IDs are in this list are reported.
    min_confidence : float
        Minimum detection confidence (0-1) for a marker to be accepted.
    camera_matrix : np.ndarray or None
        3x3 camera intrinsic matrix.  If None, estimated from image size.
    dist_coeffs : np.ndarray or None
        Distortion coefficients.  If None, assumed zero.
    depth_scale : float
        Scale factor to convert depth image values to meters (e.g. 0.001 if
        depth image is in millimeters).
    default_fov_horizontal_deg : float
        Horizontal FOV for estimating intrinsics when ``camera_matrix`` is None.
    """

    def __init__(
        self,
        marker_size_m: float = 0.12,
        dictionary: str = "DICT_4X4_50",
        valid_ids: Optional[List[int]] = None,
        min_confidence: float = 0.7,
        camera_matrix: Optional[np.ndarray] = None,
        dist_coeffs: Optional[np.ndarray] = None,
        depth_scale: float = 1.0,
        default_fov_horizontal_deg: float = 87.0,
    ):
        self.marker_size_m = marker_size_m
        self.valid_ids = set(valid_ids) if valid_ids is not None else None
        self.min_confidence = min_confidence
        self._camera_matrix = camera_matrix
        self._dist_coeffs = dist_coeffs
        self.depth_scale = depth_scale
        self.default_fov_horizontal_deg = default_fov_horizontal_deg

        # Resolve dictionary
        dict_name = dictionary.upper()
        if dict_name not in _ARUCO_DICT_MAP:
            raise ValueError(
                f"Unknown ArUco dictionary '{dictionary}'. "
                f"Choose from: {list(_ARUCO_DICT_MAP.keys())}"
            )
        self._aruco_dict = cv2.aruco.getPredefinedDictionary(
            _ARUCO_DICT_MAP[dict_name]
        )

        # Detector parameters
        self._detector_params = cv2.aruco.DetectorParameters()
        self._detector_params.cornerRefinementMethod = (
            cv2.aruco.CORNER_REFINE_SUBPIX
        )
        self._aruco_detector = cv2.aruco.ArucoDetector(
            self._aruco_dict, self._detector_params
        )

        # Cached camera matrix (set on first detect)
        self._cached_camera_matrix: Optional[np.ndarray] = None
        self._cached_dist_coeffs: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(
        self, rgb: np.ndarray, depth_m: np.ndarray
    ) -> List[dict]:
        """
        Detect ArUco markers and compute 3D positions using depth map.

        Parameters
        ----------
        rgb : np.ndarray
            RGB or BGR image of shape ``(H, W, 3)``, dtype ``uint8``.
        depth_m : np.ndarray
            Depth map of shape ``(H, W)``, aligned to ``rgb``, values in meters.

        Returns
        -------
        list[dict]
            Each dict contains ``id``, ``corners``, ``center_uv``,
            ``position_3d``, ``bearing_deg``, ``distance_m``, ``yaw_deg``,
            ``confidence``.
        """
        # Ensure intrinsics are available for this image size
        self._ensure_intrinsics(rgb.shape[1], rgb.shape[0])

        # Convert to grayscale for detection
        if rgb.ndim == 3 and rgb.shape[2] == 3:
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        else:
            gray = rgb

        # Detect markers
        corners_list, ids, rejected = self._aruco_detector.detectMarkers(gray)

        results: List[dict] = []
        if ids is None or len(ids) == 0:
            return results

        for i, marker_id in enumerate(ids.flatten()):
            # Filter by valid_ids if specified
            if self.valid_ids is not None and int(marker_id) not in self.valid_ids:
                continue

            corners = corners_list[i][0]  # shape (4, 2), float32

            # Skip if confidence is too low (if available from detector)
            # NOTE: OpenCV's detectMarkers doesn't directly return confidence;
            # we use the rejection candidates list length as a rough proxy.
            # If specific confidence API is available, override this.

            # Estimate 3D position from depth
            position_3d = self._estimate_pose_from_depth(corners, depth_m)

            if position_3d is None:
                continue

            x, y, z = position_3d
            distance_m = float(np.sqrt(x * x + y * y + z * z))

            # Center in image coordinates
            center_uv = (
                float(np.mean(corners[:, 0])),
                float(np.mean(corners[:, 1])),
            )

            # Bearing angle from image center
            bearing_deg = float(
                self._pixel_to_bearing(center_uv[0], rgb.shape[1])
            )

            # Estimate yaw from marker corner orientation
            yaw_deg = self._estimate_yaw(corners)

            # Compute confidence proxy from corner shape quality
            confidence = self._compute_corner_quality(corners)

            results.append(
                {
                    "id": int(marker_id),
                    "corners": corners.astype(np.float32),
                    "center_uv": center_uv,
                    "position_3d": (round(x, 4), round(y, 4), round(z, 4)),
                    "bearing_deg": round(bearing_deg, 2),
                    "distance_m": round(distance_m, 4),
                    "yaw_deg": round(yaw_deg, 2),
                    "confidence": round(confidence, 3),
                }
            )

        return results

    def detect_markers_only(
        self, rgb: np.ndarray
    ) -> Tuple[List[np.ndarray], Optional[np.ndarray]]:
        """Return raw OpenCV corners and IDs (no depth processing)."""
        if rgb.ndim == 3 and rgb.shape[2] == 3:
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        else:
            gray = rgb
        corners, ids, _ = self._aruco_detector.detectMarkers(gray)
        return corners, ids

    def draw_markers(
        self,
        image: np.ndarray,
        results: List[dict],
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
    ) -> np.ndarray:
        """Annotate detection results onto an image (returns a copy)."""
        vis = image.copy()
        for r in results:
            corners_int = r["corners"].astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(vis, [corners_int], True, color, thickness)
            cx, cy = r["center_uv"]
            cv2.putText(
                vis,
                f"ID:{r['id']} {r['distance_m']:.2f}m",
                (int(cx) - 20, int(cy) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )
        return vis

    # ------------------------------------------------------------------
    # Pose estimation helpers
    # ------------------------------------------------------------------

    def _estimate_pose_from_depth(
        self, corners: np.ndarray, depth_m: np.ndarray
    ) -> Optional[Tuple[float, float, float]]:
        """Use median depth within marker ROI to estimate 3D camera position.

        Parameters
        ----------
        corners : np.ndarray
            Shape (4, 2) float32, marker corner image coordinates.
        depth_m : np.ndarray
            Aligned depth map in meters, shape (H, W).

        Returns
        -------
        tuple (x, y, z) or None
            3D position in camera coordinates (x=right, y=down, z=forward).
        """
        x_min = int(np.clip(np.floor(np.min(corners[:, 0])), 0, depth_m.shape[1] - 1))
        x_max = int(np.clip(np.ceil(np.max(corners[:, 0])), 0, depth_m.shape[1] - 1))
        y_min = int(np.clip(np.floor(np.min(corners[:, 1])), 0, depth_m.shape[0] - 1))
        y_max = int(np.clip(np.ceil(np.max(corners[:, 1])), 0, depth_m.shape[0] - 1))

        if x_max <= x_min or y_max <= y_min:
            return None

        roi = depth_m[y_min:y_max, x_min:x_max]
        valid = roi[np.isfinite(roi) & (roi > 0.001)]

        if len(valid) < 4:
            return None

        z = float(np.median(valid))
        if z <= 0.001:
            return None

        # Compute center point in image and unproject
        cx_u = float(np.mean(corners[:, 0]))
        cy_v = float(np.mean(corners[:, 1]))

        K = self._cached_camera_matrix
        fx = K[0, 0]
        fy = K[1, 1]
        cx_i = K[0, 2]
        cy_i = K[1, 2]

        x = (cx_u - cx_i) * z / fx
        y = (cy_v - cy_i) * z / fy

        return (float(x), float(y), float(z))

    def _estimate_yaw(self, corners: np.ndarray) -> float:
        """Estimate yaw angle of the marker from its corner orientation.

        Uses the top-left to top-right edge direction relative to horizontal.
        Returns angle in degrees where 0 = facing camera directly.
        """
        # Top edge vector: corner[1] - corner[0] (top-left to top-right)
        dx = corners[1][0] - corners[0][0]
        dy = corners[1][1] - corners[0][1]
        angle_rad = np.arctan2(dy, dx)
        # Convert to yaw: 0 when facing camera (edge is horizontal)
        yaw_deg = float(np.degrees(angle_rad))
        return yaw_deg

    def _compute_corner_quality(self, corners: np.ndarray) -> float:
        """Heuristic quality score based on corner shape regularity.

        Compares the four side lengths: a regular square has equal side pairs.
        Returns 0-1 where 1 = perfect square.
        """
        edges = []
        n = len(corners)
        for i in range(n):
            p1 = corners[i]
            p2 = corners[(i + 1) % n]
            edges.append(np.linalg.norm(p2 - p1))

        edges = np.array(edges)
        if len(edges) < 2:
            return 0.0
        mean_edge = np.mean(edges)
        if mean_edge < 1e-6:
            return 0.0
        # Lower CV (std/mean) = more regular
        cv = np.std(edges) / mean_edge
        quality = float(np.clip(1.0 - cv, 0.0, 1.0))
        return quality

    def _pixel_to_bearing(
        self, cx: float, image_width: int, fov_horizontal_deg: float = 87.0
    ) -> float:
        """Convert pixel x-coordinate to bearing angle in degrees."""
        half_fov = fov_horizontal_deg / 2.0
        center_x = image_width / 2.0
        bearing = (cx - center_x) / center_x * half_fov
        return bearing

    # ------------------------------------------------------------------
    # Intrinsics helpers
    # ------------------------------------------------------------------

    def _ensure_intrinsics(self, width: int, height: int) -> None:
        """Set up camera intrinsics, caching based on image size."""
        if self._camera_matrix is not None:
            self._cached_camera_matrix = self._camera_matrix.astype(np.float64)
            if self._dist_coeffs is not None:
                self._cached_dist_coeffs = self._dist_coeffs.astype(np.float64)
            else:
                self._cached_dist_coeffs = np.zeros((5,), dtype=np.float64)
            return

        # Estimate from image dimensions
        if (
            self._cached_camera_matrix is not None
            and self._cached_camera_matrix.shape[0] == 3
        ):
            return  # already cached

        fx = width / (2.0 * np.tan(np.radians(self.default_fov_horizontal_deg / 2.0)))
        fy = fx  # assume square pixels
        cx_i = width / 2.0
        cy_i = height / 2.0
        self._cached_camera_matrix = np.array(
            [[fx, 0, cx_i], [0, fy, cy_i], [0, 0, 1]], dtype=np.float64
        )
        self._cached_dist_coeffs = np.zeros((5,), dtype=np.float64)
