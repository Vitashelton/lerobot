"""
YOLO-based object detection wrapper using ultralytics.

Detects common objects (box, person, chair, etc.) and optionally
estimates their 3D position using depth information.
"""
import logging
from typing import Optional

import numpy as np
import cv2

logger = logging.getLogger(__name__)


class YOLODetector:
    """YOLO object detector with optional depth-based 3D localization.

    Args:
        model_path: path to YOLO weights (.pt file).
        confidence_threshold: minimum confidence for detections.
        iou_threshold: NMS IoU threshold.
        target_classes: list of class names to report (empty = all).
        device: 'auto', 'cpu', or 'cuda:0'.
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        target_classes: Optional[list[str]] = None,
        device: str = "auto",
    ):
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.target_classes = target_classes or []
        self.device = device

        self._model = None
        self._model_path = model_path
        self._class_names: dict[int, str] = {}

    def load(self):
        """Load the YOLO model."""
        from ultralytics import YOLO

        self._model = YOLO(self._model_path)
        self._class_names = self._model.names or {}
        logger.info("YOLO model loaded: %s (%d classes) on %s",
                     self._model_path, len(self._class_names), self.device)

    def detect(
        self,
        color_image: np.ndarray,
        depth_image: Optional[np.ndarray] = None,
        intrinsics: Optional[dict] = None,
    ) -> list[dict]:
        """Run YOLO detection and optionally estimate 3D positions.

        Args:
            color_image: HxWx3 BGR image.
            depth_image: HxW float32 depth in meters.
            intrinsics: camera intrinsics dict.

        Returns:
            list of detection dicts:
                {
                    "class_name": str,
                    "class_id": int,
                    "confidence": float,
                    "bbox": [x1, y1, x2, y2],
                    "center_pixel": [u, v],
                    "distance": float | None,
                    "position_camera": [x, y, z] | None,
                    "bearing_deg": float | None,
                }
        """
        if self._model is None:
            self.load()

        results = self._model(
            color_image,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                cls_id = int(box.cls[0])
                cls_name = self._class_names.get(cls_id, str(cls_id))

                # Filter by target classes
                if self.target_classes and cls_name not in self.target_classes:
                    continue

                xyxy = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, xyxy)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                det = {
                    "class_name": cls_name,
                    "class_id": cls_id,
                    "confidence": round(conf, 3),
                    "bbox": [x1, y1, x2, y2],
                    "center_pixel": [cx, cy],
                    "distance": None,
                    "position_camera": None,
                    "bearing_deg": None,
                }

                # 3D localization
                if depth_image is not None and intrinsics is not None:
                    dist, pos, bearing = self._localize_3d(
                        x1, y1, x2, y2, depth_image, intrinsics
                    )
                    det["distance"] = dist
                    det["position_camera"] = pos
                    det["bearing_deg"] = bearing

                detections.append(det)

        return detections

    @staticmethod
    def _localize_3d(
        x1: int, y1: int, x2: int, y2: int,
        depth_image: np.ndarray,
        intrinsics: dict,
    ) -> tuple[Optional[float], Optional[list], Optional[float]]:
        """Estimate 3D position from a bounding box using depth."""
        from .pointcloud_utils import pixel_to_camera, bearing_angle

        h, w = depth_image.shape
        # Sample center 30% of bbox for robustness
        cx_pix = (x1 + x2) // 2
        cy_pix = (y1 + y2) // 2
        bw, bh = x2 - x1, y2 - y1
        sx1 = max(0, int(cx_pix - bw * 0.15))
        sx2 = min(w, int(cx_pix + bw * 0.15))
        sy1 = max(0, int(cy_pix - bh * 0.15))
        sy2 = min(h, int(cy_pix + bh * 0.15))

        region = depth_image[sy1:sy2, sx1:sx2]
        valid = region[region > 0]

        if len(valid) < 5:
            return None, None, None

        z = float(np.median(valid))
        x, y = pixel_to_camera(cx_pix, cy_pix, z, intrinsics)
        bearing = round(np.degrees(bearing_angle(x, z)), 1)

        return round(z, 3), [round(x, 3), round(y, 3), round(z, 3)], bearing

    @property
    def class_names(self) -> dict[int, str]:
        if self._model is None:
            self.load()
        return dict(self._class_names)


def draw_yolo_overlay(image: np.ndarray, detections: list[dict]) -> np.ndarray:
    """Draw YOLO detections on the image."""
    img = image.copy()
    class_colors = {}

    for det in detections:
        cls = det["class_name"]
        if cls not in class_colors:
            class_colors[cls] = tuple(int(c) for c in np.random.randint(60, 255, 3))

        color = class_colors[cls]
        x1, y1, x2, y2 = det["bbox"]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        label = f"{cls} {det['confidence']:.2f}"
        if det["distance"] is not None:
            label += f" {det['distance']:.2f}m"
        cv2.putText(img, label, (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    return img
