"""
RealSense D435i RGB-D Reader with mock mode support.

Supports:
- Live RealSense capture via pyrealsense2
- Mock mode using procedurally generated scenes (no hardware needed)
- Mock mode using saved .npz data files

Output per frame:
    color_image: np.ndarray (H, W, 3) uint8
    depth_image_meters: np.ndarray (H, W) float32
    camera_intrinsics: dict with fx, fy, cx, cy, width, height
    timestamp: float
"""
import time
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import cv2

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Mock scene generators
# ---------------------------------------------------------------------------

def _generate_mock_scene(
    width: int = 640,
    height: int = 480,
    scene: str = "corridor",
    frame_idx: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic RGB and depth images for testing without a camera.

    Args:
        width, height: image resolution.
        scene: one of 'empty', 'single_box', 'corridor', 'pallet_marker', 'cluttered_lab'.
        frame_idx: frame number for animation.

    Returns:
        (color_bgr, depth_meters) tuple.
    """
    # Base depth: floor at ~2m, walls closer
    u = np.arange(width, dtype=np.float32)
    v = np.arange(height, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)

    # Normalized coordinates
    un = (uu - width / 2) / (width / 2)   # -1 .. 1
    vn = (vv - height / 2) / (height / 2)  # -1 .. 1

    color = np.zeros((height, width, 3), dtype=np.uint8)
    depth = np.full((height, width), 2.0, dtype=np.float32)

    # Animate box position slightly
    offset = 0.05 * np.sin(frame_idx * 0.1)

    if scene == "empty":
        # Flat floor, distant walls
        color[:] = (180, 180, 180)  # gray background
        # Floor gradient
        color[int(height * 0.6):] = (120, 120, 120)
        depth[int(height * 0.6):] = np.linspace(1.5, 3.0, int(height * 0.4))[:, None]

    elif scene == "single_box":
        color[:] = (180, 180, 180)
        color[int(height * 0.6):] = (120, 120, 120)
        # Box in center
        cx, cy = int(width * 0.5 + offset * width), int(height * 0.55)
        bw, bh = 80, 100
        x1, y1 = cx - bw // 2, cy - bh // 2
        x2, y2 = x1 + bw, y1 + bh
        color[y1:y2, x1:x2] = (60, 120, 180)  # brown-ish box
        depth[y1:y2, x1:x2] = 0.8
        # Draw an ArUco-like marker on the box
        marker_x = cx - 25
        marker_y = cy - 25
        cv2.rectangle(color, (marker_x, marker_y), (marker_x + 50, marker_y + 50), (0, 0, 0), -1)
        cv2.rectangle(color, (marker_x + 5, marker_y + 5), (marker_x + 20, marker_y + 20), (255, 255, 255), -1)

    elif scene == "corridor":
        # Walls on left and right, open ahead
        color[:] = (200, 200, 200)
        # Floor
        color[int(height * 0.55):] = (130, 130, 130)
        # Left wall
        left_edge = int(width * 0.05)
        right_wall_start = int(width * 0.15)
        color[:, left_edge:right_wall_start] = (100, 100, 140)
        depth[:, left_edge:right_wall_start] = 0.6 + np.abs(vn[:, left_edge:right_wall_start]) * 2.0
        # Right wall
        left_wall_start = int(width * 0.85)
        right_edge = int(width * 0.95)
        color[:, left_wall_start:right_edge] = (100, 140, 100)
        depth[:, left_wall_start:right_edge] = 0.6 + np.abs(vn[:, left_wall_start:right_edge]) * 2.0
        # Box ahead
        bx = int(width * 0.5)
        by = int(height * 0.55)
        box_w, box_h = 60, 70
        bx1, by1 = bx - box_w // 2, by - box_h // 2
        bx2, by2 = bx1 + box_w, by1 + box_h
        color[by1:by2, bx1:bx2] = (80, 80, 200)
        depth[by1:by2, bx1:bx2] = 1.2

    elif scene == "pallet_marker":
        color[:] = (200, 200, 200)
        color[int(height * 0.6):] = (130, 130, 130)
        # Simulated pallet with ArUco marker
        px = int(width * 0.5)
        py = int(height * 0.5)
        pw, ph = 120, 80
        px1, py1 = px - pw // 2, py - ph // 2
        px2, py2 = px1 + pw, py1 + ph
        color[py1:py2, px1:px2] = (50, 150, 200)  # blue-ish pallet
        depth[py1:py2, px1:px2] = 0.9
        # ArUco marker on pallet
        marker_size = 40
        mx1, my1 = px - marker_size // 2, py - marker_size // 2
        mx2, my2 = mx1 + marker_size, my1 + marker_size
        # Black border
        color[my1:my2, mx1:mx2] = (0, 0, 0)
        # White inner
        color[my1 + 8:my2 - 8, mx1 + 8:mx2 - 8] = (255, 255, 255)
        # Pattern inside
        color[my1 + 12:my1 + 20, mx1 + 12:mx1 + 20] = (0, 0, 0)
        color[my1 + 12:my1 + 20, mx2 - 20:mx2 - 12] = (0, 0, 0)
        color[my2 - 20:my2 - 12, mx1 + 12:mx1 + 20] = (0, 0, 0)

    elif scene == "cluttered_lab":
        color[:] = (190, 190, 190)
        color[int(height * 0.6):] = (120, 120, 120)
        # Multiple obstacles
        obstacles = [
            (int(width * 0.3), int(height * 0.5), 50, 60, 0.7, (80, 80, 180)),
            (int(width * 0.6 + offset * width * 2), int(height * 0.45), 45, 55, 1.0, (60, 140, 60)),
            (int(width * 0.15), int(height * 0.55), 35, 40, 1.5, (180, 80, 80)),
            (int(width * 0.8 + offset * width), int(height * 0.5), 40, 50, 0.8, (80, 160, 160)),
        ]
        for ox, oy, ow, oh, od, oc in obstacles:
            if 0 <= ox - ow // 2 < width and 0 <= oy - oh // 2 < height:
                x1c = max(0, ox - ow // 2)
                y1c = max(0, oy - oh // 2)
                x2c = min(width, x1c + ow)
                y2c = min(height, y1c + oh)
                color[y1c:y2c, x1c:x2c] = oc
                depth[y1c:y2c, x1c:x2c] = od
                # Draw ArUco-like square on first box
                if oc == (80, 80, 180):
                    ms = 25
                    mcx, mcy = ox, oy - 5
                    cv2.rectangle(color, (mcx - ms, mcy - ms), (mcx + ms, mcy + ms), (0, 0, 0), -1)
                    cv2.rectangle(color, (mcx - ms + 5, mcy - ms + 5), (mcx + ms - 5, mcy + ms - 5), (255, 255, 255), -1)

    # Add some Gaussian noise
    depth += np.random.normal(0, 0.01, (height, width)).astype(np.float32)
    depth = np.clip(depth, 0.0, 5.0)

    return color, depth


# ---------------------------------------------------------------------------
# Camera reader
# ---------------------------------------------------------------------------

class RealSenseReader:
    """Read RGB-D frames from Intel RealSense D435i with optional mock mode.

    Usage::

        reader = RealSenseReader(mock=True, scene="corridor")
        reader.start()
        data = reader.read()
        # data = {'color': ..., 'depth': ..., 'intrinsics': ..., 'timestamp': ...}
        reader.stop()
    """

    def __init__(
        self,
        mock: bool = False,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        serial: str = "",
        align_depth: bool = True,
        depth_scale: float = 0.001,
        min_depth: float = 0.15,
        max_depth: float = 5.0,
        scene: str = "corridor",
        mock_data_dir: str = "data/raw",
    ):
        self.mock = mock
        self.width = width
        self.height = height
        self.fps = fps
        self.serial = serial
        self.align_depth = align_depth
        self.depth_scale = depth_scale
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.scene = scene
        self.mock_data_dir = mock_data_dir

        self._pipeline = None
        self._align = None
        self._running = False
        self._frame_idx = 0
        self._mock_files: list[Path] = []

        # Cached intrinsics
        self._intrinsics: dict = {
            "fx": float(width) * 1.4,
            "fy": float(height) * 1.4,
            "cx": float(width) / 2,
            "cy": float(height) / 2,
            "width": width,
            "height": height,
            "model": "pinhole",
        }

    # ---- Public API -------------------------------------------------------

    def start(self):
        """Initialize the camera or mock source."""
        if self.mock:
            self._start_mock()
        else:
            self._start_realsense()
        self._running = True
        logger.info("RealSenseReader started (mock=%s, %dx%d @ %d FPS)",
                     self.mock, self.width, self.height, self.fps)

    def read(self) -> dict:
        """Read one RGB-D frame.

        Returns:
            dict with keys: color (BGR uint8 HxWx3), depth (float32 HxW meters),
            intrinsics (dict), timestamp (float).
        """
        if not self._running:
            raise RuntimeError("Reader not started. Call start() first.")

        if self.mock:
            return self._read_mock()
        return self._read_realsense()

    def stop(self):
        """Stop and release the camera."""
        self._running = False
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception:
                pass
            self._pipeline = None
        self._align = None
        logger.info("RealSenseReader stopped.")

    @property
    def intrinsics(self) -> dict:
        return dict(self._intrinsics)

    # ---- RealSense --------------------------------------------------------

    def _start_realsense(self):
        import pyrealsense2 as rs

        config = rs.config()
        if self.serial:
            config.enable_device(self.serial)
        config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)

        self._pipeline = rs.pipeline()
        profile = self._pipeline.start(config)

        # Get intrinsics
        color_stream = profile.get_stream(rs.stream.color)
        depth_stream = profile.get_stream(rs.stream.depth)
        color_intr = color_stream.as_video_stream_profile().get_intrinsics()
        self._intrinsics = {
            "fx": color_intr.fx, "fy": color_intr.fy,
            "cx": color_intr.ppx, "cy": color_intr.ppy,
            "width": color_intr.width, "height": color_intr.height,
            "model": str(color_intr.model),
        }

        if self.align_depth:
            self._align = rs.align(rs.stream.color)

        # Warmup: discard first N frames
        for _ in range(30):
            self._pipeline.wait_for_frames()

    def _read_realsense(self) -> dict:
        import pyrealsense2 as rs

        frames = self._pipeline.wait_for_frames()
        if self._align is not None:
            frames = self._align.process(frames)

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        color = np.asanyarray(color_frame.get_data())
        depth_raw = np.asanyarray(depth_frame.get_data())
        depth = depth_raw.astype(np.float32) * self.depth_scale
        depth[(depth < self.min_depth) | (depth > self.max_depth)] = 0.0

        return {
            "color": color,
            "depth": depth,
            "intrinsics": dict(self._intrinsics),
            "timestamp": time.time(),
        }

    # ---- Mock -------------------------------------------------------------

    def _start_mock(self):
        self._frame_idx = 0
        # Check for saved data files
        data_dir = Path(self.mock_data_dir)
        if data_dir.exists():
            self._mock_files = sorted(data_dir.glob("*.npz"))
            if self._mock_files:
                logger.info("Mock: found %d saved frames in %s", len(self._mock_files), data_dir)
                return
        logger.info("Mock: using procedurally generated '%s' scene", self.scene)

    def _read_mock(self) -> dict:
        # If we have saved data, play it back
        if self._mock_files:
            idx = self._frame_idx % len(self._mock_files)
            data = np.load(self._mock_files[idx])
            self._frame_idx += 1
            return {
                "color": data.get("color", np.zeros((self.height, self.width, 3), dtype=np.uint8)),
                "depth": data.get("depth", np.zeros((self.height, self.width), dtype=np.float32)),
                "intrinsics": dict(self._intrinsics),
                "timestamp": time.time(),
            }

        # Procedural generation
        color, depth = _generate_mock_scene(
            self.width, self.height, self.scene, self._frame_idx
        )
        self._frame_idx += 1
        return {
            "color": color,
            "depth": depth,
            "intrinsics": dict(self._intrinsics),
            "timestamp": time.time(),
        }


# ---------------------------------------------------------------------------
# Convenience function for config-based construction
# ---------------------------------------------------------------------------

def create_reader_from_config(config: dict, mock: bool = False) -> RealSenseReader:
    """Build a RealSenseReader from a camera config dictionary."""
    cam = config.get("camera", config)
    mock_cfg = config.get("mock", {})
    return RealSenseReader(
        mock=mock,
        width=cam.get("resolution", {}).get("width", 640),
        height=cam.get("resolution", {}).get("height", 480),
        fps=cam.get("fps", 30),
        serial=cam.get("serial", ""),
        align_depth=cam.get("align_depth_to_color", True),
        depth_scale=cam.get("depth_scale", 0.001),
        min_depth=cam.get("min_depth", 0.15),
        max_depth=cam.get("max_depth", 5.0),
        scene=mock_cfg.get("scene", "corridor"),
        mock_data_dir=mock_cfg.get("data_path", "data/raw"),
    )
