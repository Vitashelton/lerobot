"""
Procedural generator for warehouse/lab RGB-D scenes.

Produces synthetic RGB, depth (clean + noisy), and ground-truth annotations
using pure numpy / OpenCV -- no external renderer is needed.

Scene types
-----------
- warehouse_aisle : parallel shelving units with boxes, far wall.
- lab_cluttered   : random boxes, cylinders, walls forming a lab room.
- pallet_pickup   : open area with a pallet near the middle and boxes nearby.

Camera model
------------
Pinhole camera looking along +Z (forward).  The camera sits at a prescribed
height and may pitch nose-down.  Intrinsics are derived from FOV and image
dimensions assuming square pixels (fx=fy).
"""

from __future__ import annotations

import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple

from .synthetic_depth_renderer import SyntheticDepthRenderer
from .label_generator import LabelGenerator


class SceneGenerator:
    """Procedural generator for warehouse/lab RGB-D scenes.

    Parameters
    ----------
    width, height : int
        Image dimensions in pixels.
    fov_h : float
        Horizontal field of view in degrees (RealSense D435i colour: ~69 deg;
        depth FOV is ~87 deg horizontal).  Matches the config default.
    camera_height_m : float
        Nominal camera height above ground (metres).
    camera_pitch_deg : float
        Nominal pitch angle -- positive = nose-down (degrees).
    depth_noise_std_mm : float
        Standard deviation of additive Gaussian noise on depth (mm).
    dropout_prob : float
        Probability that a depth pixel is dropped to 0 (simulates stereo
        shadow / IR absorption failures).
    """

    def __init__(self,
                 width: int = 480,
                 height: int = 640,
                 fov_h: float = 87.0,
                 camera_height_m: float = 0.4,
                 camera_pitch_deg: float = 0.0,
                 depth_noise_std_mm: float = 5.0,
                 dropout_prob: float = 0.02,
                 ):
        self.width = int(width)
        self.height = int(height)
        self.fov_h = float(fov_h)

        # Intrinsics
        self.fx = self.width / (2.0 * np.tan(np.radians(self.fov_h / 2.0)))
        self.fy = self.fx  # square pixels
        self.cx = self.width / 2.0
        self.cy = self.height / 2.0

        self.base_camera_height = float(camera_height_m)
        self.base_camera_pitch_deg = float(camera_pitch_deg)
        self.depth_noise_std_m = float(depth_noise_std_mm) / 1000.0
        self.dropout_prob = float(dropout_prob)

        # Sub-modules
        self.renderer = SyntheticDepthRenderer(
            self.width, self.height, self.fx, self.fy, self.cx, self.cy)
        self.label_gen = LabelGenerator()

        # RNG
        self._rng = np.random.RandomState()

    # ==================================================================
    #  Public API
    # ==================================================================

    def generate_scene(self, scene_type: str = "warehouse_aisle",
                       seed: Optional[int] = None,
                       ) -> dict:
        """Generate one complete synthetic scene.

        Parameters
        ----------
        scene_type : str
            One of {"warehouse_aisle", "lab_cluttered", "pallet_pickup"}.
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        scene : dict
            Keys:
                rgb          ndarray (H, W, 3) uint8
                depth        ndarray (H, W) float32  clean depth (metres)
                depth_noisy  ndarray (H, W) float32  noisy depth
                labels       ndarray (H, W) int32    semantic class IDs
                objects      list of per-object dicts
                scene_type   str
                camera_pose  dict {height_m, pitch_deg}
        """
        if seed is not None:
            self._rng.seed(seed)

        self.label_gen.reset()

        # Randomise camera pose within configured tolerance
        cam_pose = self._randomize_camera()

        # Build object list for the requested scene type
        objects = self._build_scene_objects(scene_type, cam_pose)

        # Render
        depth, labels, rgb = self.renderer.compose_scene(
            objects,
            camera_pos=np.array([0.0, cam_pose["height_m"], 0.0], dtype=np.float64),
            pitch_rad=np.radians(cam_pose["pitch_deg"]),
        )

        # Add noise
        depth_noisy = self._add_depth_noise(depth)

        # Build annotations
        annotation = self.label_gen.generate_annotation(labels)

        return {
            "rgb": rgb,
            "depth": depth,
            "depth_noisy": depth_noisy,
            "labels": labels,
            "objects": annotation["objects"],
            "free_space": annotation["free_space"],
            "scene_type": scene_type,
            "camera_pose": cam_pose,
        }

    # ==================================================================
    #  Scene builders
    # ==================================================================

    def _build_scene_objects(self, scene_type: str,
                             cam_pose: dict) -> List[dict]:
        """Return ordered list of renderable objects for a scene type."""
        rng = self._rng
        z_near = 0.3
        z_far = 20.0

        objects: List[dict] = []

        # --- floor (always present) ---
        floor_size = 30.0
        objects.append({
            "type": "floor",
            "origin": (0.0, 0.0, floor_size / 2),  # centre of floor quad
            "normal": (0.0, 1.0, 0.0),
            "width": floor_size,
            "height": floor_size,
            "color": (180, 180, 180),
            "label_id": 0,  # background
        })

        if scene_type == "warehouse_aisle":
            objects += self._build_warehouse_aisle(rng, cam_pose)
        elif scene_type == "lab_cluttered":
            objects += self._build_lab_cluttered(rng, cam_pose)
        elif scene_type == "pallet_pickup":
            objects += self._build_pallet_pickup(rng, cam_pose)
        else:
            raise ValueError(f"Unknown scene_type '{scene_type}'")

        return objects

    def _build_warehouse_aisle(self, rng: np.random.RandomState,
                               cam_pose: dict) -> List[dict]:
        """Parallel shelving units forming aisles, with far wall."""
        objects = []
        aisle_count = 3
        shelf_depth = 0.5
        shelf_height = 2.0
        aisle_width = 2.0

        # Far wall
        far_dist = 12.0
        objects.append({
            "type": "wall", "origin": (0.0, far_dist / 2, far_dist),
            "normal": (0.0, 0.0, -1.0), "width": 12.0, "height": far_dist,
            "color": (200, 190, 170), "label_id": 1,
        })

        # Left / right walls
        for sx, nx in [(-6.0, 1.0), (6.0, -1.0)]:
            objects.append({
                "type": "wall", "origin": (sx, 1.0, 6.0),
                "normal": (nx, 0.0, 0.0), "width": 12.0, "height": 2.0,
                "color": (190, 180, 160), "label_id": 1,
            })

        # Shelves (rows) -- each is two vertical side panels and horizontal planks
        for row in range(aisle_count):
            z_center = 3.0 + row * 3.0
            for x_sign in [-1, 1]:
                x_center = x_sign * (aisle_width / 2 + shelf_depth / 2)
                # Side panel (wall-like)
                objects.append({
                    "type": "wall",
                    "origin": (x_center, shelf_height / 2, z_center),
                    "normal": (-x_sign, 0.0, 0.0),
                    "width": 2.0, "height": shelf_height,
                    "color": (140, 130, 110), "label_id": 4,  # shelf
                })
                # Shelf plank (box)
                for shelf_level in [0.4, 1.0, 1.6]:
                    objects.append({
                        "type": "box",
                        "center": (x_center, shelf_level, z_center),
                        "size": (shelf_depth * 0.9, 0.05, 2.0),
                        "yaw": 0.0,
                        "color": (160, 140, 100),
                        "label_id": 4,
                    })

            # Boxes on shelves (random)
            for b in range(rng.randint(1, 4)):
                side = rng.choice([-1, 1])
                x_center = side * (aisle_width / 2 + shelf_depth / 4)
                bx = rng.uniform(0.15, 0.4)
                by = rng.uniform(0.1, 0.3)
                bz = rng.uniform(0.15, 0.4)
                shelf_y = rng.choice([0.45, 1.05, 1.65])
                objects.append({
                    "type": "box",
                    "center": (x_center + rng.uniform(-0.1, 0.1),
                               shelf_y + by / 2,
                               z_center + rng.uniform(-0.7, 0.7)),
                    "size": (bx, by, bz),
                    "yaw": rng.uniform(-0.3, 0.3),
                    "color": tuple(int(c) for c in rng.randint(80, 220, 3)),
                    "label_id": 2,
                })

        return objects

    def _build_lab_cluttered(self, rng: np.random.RandomState,
                             cam_pose: dict) -> List[dict]:
        """Lab room with walls and random obstacles."""
        objects = []
        room_size = 8.0
        room_height = 2.5

        # Walls
        for origin, normal, w, h in [
            ((0.0, room_height / 2, room_size), (0, 0, -1), room_size, room_height),
            ((room_size / 2, room_height / 2, room_size / 2), (-1, 0, 0), room_size, room_height),
            ((-room_size / 2, room_height / 2, room_size / 2), (1, 0, 0), room_size, room_height),
        ]:
            objects.append({
                "type": "wall", "origin": origin, "normal": normal,
                "width": w, "height": h, "color": (210, 210, 200), "label_id": 1,
            })

        # Random boxes
        for _ in range(rng.randint(5, 12)):
            bx = rng.uniform(0.15, 0.5)
            by = rng.uniform(0.15, 0.5)
            bz = rng.uniform(0.15, 0.5)
            objects.append({
                "type": "box",
                "center": (rng.uniform(-3.0, 3.0),
                           by / 2,  # sit on floor
                           rng.uniform(1.5, 7.0)),
                "size": (bx, by, bz),
                "yaw": rng.uniform(-np.pi, np.pi),
                "color": tuple(int(c) for c in rng.randint(80, 220, 3)),
                "label_id": 2,
            })

        # Cylinders (chair / person approximations)
        for _ in range(rng.randint(1, 4)):
            cyl_h = rng.uniform(0.5, 1.7)
            objects.append({
                "type": "cylinder",
                "center": (rng.uniform(-2.5, 2.5), cyl_h / 2,  # sit on floor
                           rng.uniform(2.0, 6.0)),
                "radius": rng.uniform(0.15, 0.3),
                "height": cyl_h,
                "num_faces": 16,
                "color": tuple(int(c) for c in rng.randint(40, 180, 3)),
                "label_id": rng.choice([5, 6]),  # chair or person
            })

        return objects

    def _build_pallet_pickup(self, rng: np.random.RandomState,
                             cam_pose: dict) -> List[dict]:
        """Open area with a pallet near centre and a few boxes nearby."""
        objects = []

        # Distant wall
        objects.append({
            "type": "wall", "origin": (0.0, 1.25, 10.0),
            "normal": (0, 0, -1), "width": 10.0, "height": 2.5,
            "color": (200, 195, 180), "label_id": 1,
        })
        # Side walls
        objects.append({
            "type": "wall", "origin": (-5.0, 1.25, 5.0),
            "normal": (1, 0, 0), "width": 10.0, "height": 2.5,
            "color": (190, 180, 160), "label_id": 1,
        })
        objects.append({
            "type": "wall", "origin": (5.0, 1.25, 5.0),
            "normal": (-1, 0, 0), "width": 10.0, "height": 2.5,
            "color": (190, 180, 160), "label_id": 1,
        })

        # Pallet: two layers of deck boards with stringers between
        pallet_z = 4.0
        board_thickness = 0.02
        pallet_total_height = 0.14  # standard EUR pallet
        pallet_color = (120, 80, 40)

        # Bottom deck boards (y=0.02 centre, sits on floor)
        for px in [-0.4, 0.0, 0.4]:
            objects.append({
                "type": "box",
                "center": (px, board_thickness, pallet_z),
                "size": (0.15, board_thickness * 2, 0.8),
                "yaw": 0.0,
                "color": pallet_color,
                "label_id": 3,  # pallet
            })

        # Top deck boards (y=0.12 centre, 0.10 above bottom boards)
        for px in [-0.4, 0.0, 0.4]:
            objects.append({
                "type": "box",
                "center": (px, pallet_total_height - board_thickness, pallet_z),
                "size": (0.15, board_thickness * 2, 0.8),
                "yaw": 0.0,
                "color": pallet_color,
                "label_id": 3,
            })

        # Stringer boards spanning between bottom and top deck
        for pz in [pallet_z - 0.3, pallet_z + 0.3]:
            objects.append({
                "type": "box",
                "center": (0.0, pallet_total_height / 2, pz),
                "size": (1.0, pallet_total_height - 2 * board_thickness * 2, 0.1),
                "yaw": 0.0,
                "color": pallet_color,
                "label_id": 3,
            })

        # Random boxes near/on pallet
        for _ in range(rng.randint(2, 5)):
            bx = rng.uniform(0.2, 0.5)
            by = rng.uniform(0.15, 0.4)
            bz = rng.uniform(0.2, 0.5)
            objects.append({
                "type": "box",
                "center": (rng.uniform(-1.5, 1.5),
                           by / 2,
                           pallet_z + rng.uniform(-0.8, 0.8)),
                "size": (bx, by, bz),
                "yaw": rng.uniform(-0.4, 0.4),
                "color": tuple(int(c) for c in rng.randint(60, 200, 3)),
                "label_id": 2,
            })

        return objects

    # ==================================================================
    #  Noise & camera helpers
    # ==================================================================

    def _add_depth_noise(self, depth: np.ndarray) -> np.ndarray:
        """Add Gaussian noise and random pixel dropout to a depth map.

        Parameters
        ----------
        depth : ndarray (H, W) float32, clean depth (metres).

        Returns
        -------
        noisy : ndarray (H, W) float32.
        """
        rng = self._rng
        noise = rng.randn(*depth.shape).astype(np.float32) * self.depth_noise_std_m
        noisy = depth + noise

        # Dropout -- set random pixels to 0 (invalid)
        dropout_mask = rng.rand(*depth.shape) < self.dropout_prob
        noisy[dropout_mask] = 0.0

        # Clamp negative depths to 0
        noisy = np.maximum(noisy, 0.0)

        return noisy.astype(np.float32)

    def _randomize_camera(self) -> Dict[str, float]:
        """Randomize camera height and pitch within configured ranges.

        Uses +/- 20 % variation on height and +/- 3 deg on pitch by default.
        """
        height = self.base_camera_height * (1.0 + self._rng.uniform(-0.2, 0.2))
        pitch = self.base_camera_pitch_deg + self._rng.uniform(-3.0, 3.0)
        return {"height_m": float(height), "pitch_deg": float(pitch)}
