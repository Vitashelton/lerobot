"""
Ground-truth label generation for synthetic RGB-D scenes.

Tracks object instances across a scene and produces:
  - Semantic label maps (pixel-level class IDs)
  - Per-object annotations (class name, 2-D bounding box, 3-D centre)
  - Free-space mask (traversable area in front of the camera)
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple


class LabelGenerator:
    """Generates ground-truth annotations for synthetic scenes.

    Maintains an object registry so that each rendered object receives a
    unique instance ID.  After rendering, methods are provided to extract
    2-D bounding boxes and traversable free-space from the label map.

    Class map (consistent across synthetic data and downstream models):
        0  background
        1  wall
        2  box
        3  pallet
        4  shelf
        5  chair
        6  person
    """

    CLASS_MAP: Dict[str, int] = {
        "background": 0,
        "wall": 1,
        "box": 2,
        "pallet": 3,
        "shelf": 4,
        "chair": 5,
        "person": 6,
    }

    CLASS_NAMES: Dict[int, str] = {v: k for k, v in CLASS_MAP.items()}

    def __init__(self):
        self.instance_counter: int = 0
        self._objects: List[dict] = []  # registered objects

    # ------------------------------------------------------------------
    #  Registration
    # ------------------------------------------------------------------

    def register_object(self, class_name: str,
                        center_3d: Tuple[float, float, float],
                        size_3d: Tuple[float, float, float],
                        **kwargs) -> int:
        """Register a new object and return its instance ID.

        Parameters
        ----------
        class_name : one of the CLASS_MAP keys.
        center_3d  : (x, y, z) metres, world frame.
        size_3d    : (sx, sy, sz) metres.

        Returns
        -------
        instance_id : int  globally unique.
        """
        if class_name not in self.CLASS_MAP:
            raise ValueError(f"Unknown class '{class_name}'. "
                             f"Known: {list(self.CLASS_MAP.keys())}")

        obj_id = self.instance_counter
        self.instance_counter += 1

        obj = {
            "id": obj_id,
            "class": class_name,
            "class_id": self.CLASS_MAP[class_name],
            "center_3d": list(center_3d),
            "size_3d": list(size_3d),
            **kwargs,
        }
        self._objects.append(obj)
        return obj_id

    def reset(self) -> None:
        """Clear all registered objects for a new scene."""
        self._objects.clear()
        self.instance_counter = 0

    # ------------------------------------------------------------------
    #  Annotation extraction
    # ------------------------------------------------------------------

    def generate_annotation(self, label_map: np.ndarray,
                            objects: Optional[List[dict]] = None,
                            ) -> Dict:
        """Produce a structured annotation dict from the label map.

        Parameters
        ----------
        label_map : ndarray (H, W) int32
            Pixel-level semantic label map (class IDs 0-6).
        objects : list of dict, optional
            Pre-registered objects.  If None, uses internal registry.

        Returns
        -------
        annotation : dict with keys:
            "objects"    : list of per-object dicts:
                id, class, class_id, center_3d, size_3d, bbox_2d [x1,y1,x2,y2]
            "free_space" : ndarray (H, W) bool  True where traversable.
        """
        if objects is None:
            objects = list(self._objects)

        obj_list = []
        for obj in objects:
            cls_id = obj.get("class_id", self.CLASS_MAP.get(
                obj.get("class", "background"), 0))
            bbox = self._compute_2d_bbox(label_map, cls_id, obj.get("instance_id"))
            obj_list.append({
                "id": obj.get("id", -1),
                "class": obj.get("class", "unknown"),
                "class_id": int(cls_id),
                "center_3d": obj.get("center_3d", [0, 0, 0]),
                "size_3d": obj.get("size_3d", [0, 0, 0]),
                "bbox_2d": bbox,
            })

        free_space = self.compute_free_space(label_map)

        return {
            "objects": obj_list,
            "free_space": free_space,
        }

    # ------------------------------------------------------------------
    #  Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_2d_bbox(label_map: np.ndarray,
                         class_id: int,
                         instance_id: Optional[int] = None) -> List[int]:
        """Compute axis-aligned 2-D bounding box [x1, y1, x2, y2] for a class.

        If instance_id is provided, only that instance is considered
        (requires instance-level label maps; not yet implemented in
        this simple rasterizer -- falls back to class-level bbox).
        """
        mask = (label_map == class_id)
        if not np.any(mask):
            return [0, 0, 0, 0]

        rows, cols = np.where(mask)
        y1, y2 = int(rows.min()), int(rows.max())
        x1, x2 = int(cols.min()), int(cols.max())
        return [x1, y1, x2, y2]

    @staticmethod
    def compute_free_space(label_map: np.ndarray,
                           obstacle_ids: Optional[set] = None) -> np.ndarray:
        """Compute traversable free-space mask.

        Free space = pixels labelled as background (0) OR floor (not
        explicitly labelled separately, so background that lies on the
        ground plane).

        Parameters
        ----------
        label_map    : ndarray (H, W) int32.
        obstacle_ids : set of class IDs considered obstacles.
                       Default: {1, 2, 3, 4, 5, 6} (everything except bg).

        Returns
        -------
        free_space : ndarray (H, W) bool.
        """
        if obstacle_ids is None:
            obstacle_ids = {1, 2, 3, 4, 5, 6}
        free = np.ones(label_map.shape, dtype=bool)
        for oid in obstacle_ids:
            free &= (label_map != oid)
        return free

    # ------------------------------------------------------------------
    #  Properties
    # ------------------------------------------------------------------

    @property
    def num_objects(self) -> int:
        return len(self._objects)

    @property
    def objects(self) -> List[dict]:
        return list(self._objects)
