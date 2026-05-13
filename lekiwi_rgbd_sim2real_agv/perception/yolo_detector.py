"""
YOLO-based object detector with depth-assisted 3D localization.

Wraps ultralytics YOLO for 2D bounding-box detection and augments each
detection with a 3D position estimate by sampling the aligned depth map
inside the bounding box.

Typical usage:

    detector = YOLODetector(model_path="yolov8n.pt", classes=[0, 39, 56])
    detections = detector.detect(rgb_image, depth_map_meters)
    for d in detections:
        print(f"{d['class_name']} at {d['position_3d']}")
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# COCO 80-class names (subset for reference; can be expanded)
# ---------------------------------------------------------------------------
COCO_CLASS_NAMES: dict[int, str] = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
    5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light",
    10: "fire hydrant", 11: "stop sign", 12: "parking meter", 13: "bench",
    14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow",
    20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe", 24: "backpack",
    25: "umbrella", 26: "handbag", 27: "tie", 28: "suitcase", 29: "frisbee",
    30: "skis", 31: "snowboard", 32: "sports ball", 33: "kite",
    34: "baseball bat", 35: "baseball glove", 36: "skateboard",
    37: "surfboard", 38: "tennis racket", 39: "bottle",
    40: "wine glass", 41: "cup", 42: "fork", 43: "knife", 44: "spoon",
    45: "bowl", 46: "banana", 47: "apple", 48: "sandwich", 49: "orange",
    50: "broccoli", 51: "carrot", 52: "hot dog", 53: "pizza", 54: "donut",
    55: "cake", 56: "chair", 57: "couch", 58: "potted plant", 59: "bed",
    60: "dining table", 61: "toilet", 62: "tv", 63: "laptop", 64: "mouse",
    65: "remote", 66: "keyboard", 67: "cell phone", 68: "microwave",
    69: "oven", 70: "toaster", 71: "sink", 72: "refrigerator", 73: "book",
    74: "clock", 75: "vase", 76: "scissors", 77: "teddy bear",
    78: "hair drier", 79: "toothbrush",
}


class YOLODetector:
    """
    YOLO-based object detection with depth-assisted 3D localization.

    Detects objects using an ultralytics YOLO model and estimates each
    object's 3D camera-frame position by sampling the depth map within the
    detection's bounding-box region using a specified percentile.

    Parameters
    ----------
    model_path : str
        Path to a YOLO weights file (e.g. ``"yolov8n.pt"``).
    classes : list[int] or None
        COCO class IDs to detect.  If None, all 80 classes are reported.
    confidence : float
        Minimum confidence threshold (0-1).
    depth_percentile : int
        Percentile (0-100) used when extracting depth within the bounding
        box.  Lower values (e.g. 20) are more robust against background
        depth bleeding through the object edges.
    nms_iou : float
        IoU threshold for Non-Maximum Suppression.
    fov_horizontal_deg : float
        Horizontal field of view for bearing angle calculation.
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        classes: Optional[List[int]] = None,
        confidence: float = 0.5,
        depth_percentile: int = 20,
        nms_iou: float = 0.45,
        fov_horizontal_deg: float = 87.0,
    ):
        self.model_path = model_path
        self._classes = set(classes) if classes is not None else None
        self.confidence = confidence
        self.depth_percentile = depth_percentile
        self.nms_iou = nms_iou
        self.fov_horizontal_deg = fov_horizontal_deg

        self._model = None  # Lazy-loaded

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(
        self, rgb: np.ndarray, depth_m: np.ndarray
    ) -> List[dict]:
        """
        Run YOLO detection and estimate per-object 3D position.

        Parameters
        ----------
        rgb : np.ndarray
            RGB (or BGR) image of shape ``(H, W, 3)``, dtype ``uint8``.
        depth_m : np.ndarray
            Depth map of shape ``(H, W)``, aligned to ``rgb``, values in meters.

        Returns
        -------
        list[dict]
            Each dict contains ``class_id``, ``class_name``, ``confidence``,
            ``bbox``, ``center_uv``, ``distance_m``, ``bearing_deg``,
            ``position_3d``.
        """
        self._ensure_model_loaded()

        # Convert to BGR if needed (ultralytics expects BGR)
        if rgb.ndim == 3 and rgb.shape[2] == 3:
            img_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = rgb

        # Run YOLO inference
        results = self._model(
            img_bgr,
            conf=self.confidence,
            iou=self.nms_iou,
            verbose=False,
        )

        detections: List[dict] = []
        if len(results) == 0 or results[0].boxes is None:
            return detections

        boxes = results[0].boxes
        if boxes.xyxy is None or len(boxes.xyxy) == 0:
            return detections

        img_h, img_w = depth_m.shape[:2]
        cls_ids = boxes.cls.cpu().numpy() if boxes.cls is not None else []
        confs = boxes.conf.cpu().numpy() if boxes.conf is not None else []
        xyxy = boxes.xyxy.cpu().numpy()

        for i in range(len(xyxy)):
            class_id = int(cls_ids[i])
            if self._classes is not None and class_id not in self._classes:
                continue

            x1, y1, x2, y2 = xyxy[i].astype(np.float64)
            # Clamp to image bounds
            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = min(img_w, int(x2))
            y2 = min(img_h, int(y2))

            if x2 <= x1 or y2 <= y1:
                continue

            bbox = (x1, y1, x2, y2)
            center_uv = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            confidence = float(confs[i])

            distance_m = self._estimate_distance(
                bbox, depth_m, self.depth_percentile
            )

            bearing_deg = self._pixel_to_bearing(
                center_uv[0], img_w, self.fov_horizontal_deg
            )

            # Approximate 3D position using camera model
            position_3d = self._bbox_center_to_3d(center_uv, distance_m, img_w, img_h)

            class_name = COCO_CLASS_NAMES.get(class_id, f"cls_{class_id}")

            detections.append(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": round(confidence, 3),
                    "bbox": bbox,
                    "center_uv": center_uv,
                    "distance_m": round(float(distance_m), 4),
                    "bearing_deg": round(bearing_deg, 2),
                    "position_3d": position_3d,
                }
            )

        return detections

    def detect_2d_only(self, rgb: np.ndarray) -> List[dict]:
        """Run YOLO inference without depth processing (returns 2D data only)."""
        self._ensure_model_loaded()

        if rgb.ndim == 3 and rgb.shape[2] == 3:
            img_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = rgb

        results = self._model(
            img_bgr,
            conf=self.confidence,
            iou=self.nms_iou,
            verbose=False,
        )

        detections: List[dict] = []
        if len(results) == 0 or results[0].boxes is None:
            return detections

        boxes = results[0].boxes
        if boxes.xyxy is None or len(boxes.xyxy) == 0:
            return detections

        cls_ids = boxes.cls.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        xyxy = boxes.xyxy.cpu().numpy()

        for i in range(len(xyxy)):
            class_id = int(cls_ids[i])
            if self._classes is not None and class_id not in self._classes:
                continue
            x1, y1, x2, y2 = xyxy[i]
            class_name = COCO_CLASS_NAMES.get(class_id, f"cls_{class_id}")
            detections.append(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": round(float(confs[i]), 3),
                    "bbox": (int(x1), int(y1), int(x2), int(y2)),
                    "center_uv": ((x1 + x2) / 2.0, (y1 + y2) / 2.0),
                }
            )
        return detections

    def draw_detections(
        self,
        image: np.ndarray,
        detections: List[dict],
        color: Tuple[int, int, int] = (0, 255, 255),
        thickness: int = 2,
    ) -> np.ndarray:
        """Annotate detection results onto an RGB image (returns copy)."""
        vis = image.copy()
        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
            label = f"{d['class_name']} {d.get('distance_m', '?')}m"
            cv2.putText(
                vis,
                label,
                (x1, max(y1 - 5, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )
        return vis

    # ------------------------------------------------------------------
    # Depth estimation
    # ------------------------------------------------------------------

    def _estimate_distance(
        self,
        bbox: Tuple[int, int, int, int],
        depth_m: np.ndarray,
        percentile: int = 20,
    ) -> float:
        """Estimate distance to object using depth percentile within bbox.

        Uses a low percentile (e.g., 20th) to pick pixels on the front
        surface of the object, avoiding background depth bleeding through
        gaps or object edges.
        """
        x1, y1, x2, y2 = bbox
        roi = depth_m[y1:y2, x1:x2]
        valid = roi[np.isfinite(roi) & (roi > 0.01)]

        if len(valid) < 4:
            # Fallback: use center pixel
            cy = (y1 + y2) // 2
            cx = (x1 + x2) // 2
            cy = np.clip(cy, 0, depth_m.shape[0] - 1)
            cx = np.clip(cx, 0, depth_m.shape[1] - 1)
            center_val = depth_m[cy, cx]
            if np.isfinite(center_val) and center_val > 0.01:
                return float(center_val)
            return float("nan")

        return float(np.percentile(valid, percentile))

    def _bbox_center_to_3d(
        self,
        center_uv: Tuple[float, float],
        distance_m: float,
        img_w: int,
        img_h: int,
    ) -> Tuple[float, float, float]:
        """Convert bbox center + distance to 3D camera coordinates.

        Uses a pinhole model with default FOV assumptions (fx = fy).
        """
        if not np.isfinite(distance_m) or distance_m <= 0:
            return (float("nan"), float("nan"), float("nan"))

        cx_u, cy_v = center_uv
        # Estimate fx from FOV
        fx = img_w / (2.0 * np.tan(np.radians(self.fov_horizontal_deg / 2.0)))
        fy = fx
        img_cx = img_w / 2.0
        img_cy = img_h / 2.0

        x = (cx_u - img_cx) * distance_m / fx
        y = (cy_v - img_cy) * distance_m / fy
        z = distance_m

        return (round(float(x), 4), round(float(y), 4), round(float(z), 4))

    def _pixel_to_bearing(
        self, cx: float, image_width: int, fov_horizontal_deg: float = 87.0
    ) -> float:
        """Convert pixel x-coordinate to bearing angle in degrees.

        Center pixel = 0 deg, left = negative, right = positive.
        """
        half_fov = fov_horizontal_deg / 2.0
        center_x = image_width / 2.0
        bearing = (cx - center_x) / center_x * half_fov
        return float(bearing)

    # ------------------------------------------------------------------
    # Model management
    # ------------------------------------------------------------------

    def _ensure_model_loaded(self) -> None:
        """Lazy-load the YOLO model on first use."""
        if self._model is not None:
            return
        try:
            from ultralytics import YOLO

            self._model = YOLO(self.model_path)
        except ImportError:
            raise ImportError(
                "ultralytics is required for YOLODetector. "
                "Install with: pip install ultralytics"
            )

    def load_model(self, model_path: Optional[str] = None) -> None:
        """Explicitly load (or reload) the YOLO model."""
        if model_path is not None:
            self.model_path = model_path
        self._model = None
        self._ensure_model_loaded()


# Delayed import so the module can be imported even without OpenCV at top level
import cv2  # noqa: E402
