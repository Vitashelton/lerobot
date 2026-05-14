# Copyright 2025 The LeKiwi RGB-D Sim2Real AGV Project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
LeKiwi D435i Host -- on-board daemon for Raspberry Pi / Jetson.

Responsibilities
----------------
1. Connect to the LeKiwi Feetech motor bus (via ``LeKiwiClient`` patterns,
   but as a local robot -- the host runs the *actual* robot hardware).
2. Open the Intel RealSense D435i and stream RGB + depth.
3. Preprocess depth: clamp range, median filter, hole filling, temporal
   smoothing.
4. Convert depth to a 64-element scan via percentile pooling over the
   middle horizontal band.
5. Compute per-sector minimum distances (front / left / right).
6. Optionally detect ArUco pallet markers and estimate 6-DoF pose.
7. Pack everything into a JSON observation and PUSH it over ZMQ.
8. PULL commands over ZMQ and forward them to the base motors.
9. Watchdog: stop the base if no command received within a timeout.
10. Save raw depth frames locally (PNG, uint16) -- never sent over ZMQ.

Run this on the on-board computer:

.. code-block:: bash

    python communication/host.py \
        --robot.type lekiwi_d435i \
        --port_zmq_cmd 5555 \
        --port_zmq_observations 5556
"""

import base64
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    import zmq
except ImportError:
    zmq = None  # type: ignore

try:
    import draccus  # type: ignore
    HAS_DRACCUS = True
except ImportError:
    HAS_DRACCUS = False

from lerobot.cameras.utils import make_cameras_from_configs

# Project-local configs
from .config import (
    LeKiwiD435iConfig,
    LeKiwiD435iHostConfig,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")


# ===========================================================================
# Lightweight ArUco detection (no cv2.aruco required at import time)
# ===========================================================================

_ARUCO_DICT_LOOKUP: dict[str, int] = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50 if hasattr(cv2.aruco, "DICT_4X4_50") else 0,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50 if hasattr(cv2.aruco, "DICT_5X5_50") else 1,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50 if hasattr(cv2.aruco, "DICT_6X6_50") else 2,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50 if hasattr(cv2.aruco, "DICT_7X7_50") else 3,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL if hasattr(cv2.aruco, "DICT_ARUCO_ORIGINAL") else 4,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100 if hasattr(cv2.aruco, "DICT_4X4_100") else 5,
}


class ArucoPalletDetector:
    """
    Lightweight ArUco marker detector for pallet pose estimation.

    Uses a calibrated camera matrix (if provided) to solve PnP and
    return the 6-DoF pose of each detected marker, then averages the
    valid markers to produce a single pallet pose.
    """

    def __init__(
        self,
        dictionary_name: str = "DICT_4X4_50",
        marker_size_m: float = 0.12,
        camera_matrix: np.ndarray | None = None,
        dist_coeffs: np.ndarray | None = None,
        valid_ids: list[int] | None = None,
        min_confidence: float = 0.7,
    ):
        """
        Args:
            dictionary_name: One of the keys in ``_ARUCO_DICT_LOOKUP``.
            marker_size_m: Physical side length of one marker [meters].
            camera_matrix: 3x3 intrinsics.  If None, poses are skipped.
            dist_coeffs: Distortion coefficients (k1, k2, p1, p2, k3).
            valid_ids: If set, only these marker IDs are considered.
            min_confidence: Detection confidence threshold (0--1).
        """
        dict_id = _ARUCO_DICT_LOOKUP.get(dictionary_name, cv2.aruco.DICT_4X4_50)
        self._dict = cv2.aruco.getPredefinedDictionary(dict_id)
        self._params = cv2.aruco.DetectorParameters()
        self._params.minMarkerPerimeterRate = 0.02
        self._params.maxMarkerPerimeterRate = 0.5
        self._detector = cv2.aruco.ArucoDetector(self._dict, self._params)

        self.marker_size_m = marker_size_m
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs if dist_coeffs is not None else np.zeros((5, 1))
        self.valid_ids = set(valid_ids) if valid_ids else None
        self.min_confidence = min_confidence

        # Object points for a single marker (assuming Z=0 plane)
        half = self.marker_size_m / 2.0
        self._obj_points = np.array(
            [[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]],
            dtype=np.float32,
        )

    def detect(self, rgb: np.ndarray) -> dict[str, Any] | None:
        """
        Detect ArUco markers and return averaged pallet pose.

        Returns:
            Dict with keys ``x, y, z, yaw`` (meters/meters/radians) or
            ``None`` if no valid markers are found.
        """
        if self.camera_matrix is None:
            return None

        corners, ids, rejected = self._detector.detectMarkers(rgb)
        if ids is None or len(ids) == 0:
            return None

        poses: list[dict[str, float]] = []
        for i, marker_id in enumerate(ids.flatten()):
            if self.valid_ids is not None and int(marker_id) not in self.valid_ids:
                continue

            # Estimate pose via PnP
            success, rvec, tvec = cv2.solvePnP(
                self._obj_points,
                corners[i],
                self.camera_matrix,
                self.dist_coeffs,
                flags=cv2.SOLVEPNP_IPPE_SQUARE,
            )
            if not success:
                continue

            # Convert rotation vector to yaw angle
            R, _ = cv2.Rodrigues(rvec)
            yaw = float(np.arctan2(R[1, 0], R[0, 0]))

            poses.append({
                "x": float(tvec[0, 0]),
                "y": float(tvec[1, 0]),
                "z": float(tvec[2, 0]),
                "yaw": yaw,
            })

        if not poses:
            return None

        # Average over all valid markers
        n = len(poses)
        avg = {k: sum(p[k] for p in poses) / n for k in ("x", "y", "z", "yaw")}
        avg["n_markers"] = n
        return avg


# ===========================================================================
# Depth preprocessing utilities (run on the host)
# ===========================================================================

def preprocess_depth(
    depth: np.ndarray,
    min_m: float = 0.15,
    max_m: float = 5.0,
    median_ksize: int = 5,
    hole_fill: bool = True,
) -> np.ndarray:
    """
    Preprocess a raw uint16 depth image (mm).

    Steps:
      1. Clamp to [min_m, max_m] in metres, converting invalid (0) to max.
      2. Median filter (odd kernel size).
      3. Optional hole filling via inpainting (Navier-Stokes).
    """
    # mm -> m
    depth_m = depth.astype(np.float32) * 0.001
    # Clamp
    depth_m = np.clip(depth_m, 0.0, max_m)
    depth_m[depth_m < min_m] = max_m

    if median_ksize > 1:
        k = median_ksize if median_ksize % 2 == 1 else median_ksize + 1
        depth_m = cv2.medianBlur(depth_m, k)

    if hole_fill:
        # Identify holes (values near max_m and 0)
        hole_mask = ((depth_m >= max_m * 0.99) | (depth_m <= min_m)).astype(np.uint8)
        if hole_mask.any():
            depth_m = cv2.inpaint(depth_m, hole_mask, inpaintRadius=3, flags=cv2.INPAINT_NS)

    return depth_m  # metres


def depth_to_scan(
    depth_m: np.ndarray,
    scan_dim: int = 64,
    slice_fraction: float = 0.3,
    percentile: int = 10,
) -> np.ndarray:
    """
    Convert a depth image (metres) to a 1-D polar scan via percentile pooling.

    1. Crop to the middle horizontal band (``slice_fraction`` of height).
    2. Divide the image into ``scan_dim`` vertical columns.
    3. For each column take the ``percentile``-th smallest depth value.
       Using percentile (e.g. 10) instead of strict min avoids noise /
       single-pixel dropouts.
    4. Return (scan_dim,) float32 array in metres.
    """
    h, w = depth_m.shape
    # Middle band
    band_half = int(h * slice_fraction / 2)
    h_center = h // 2
    y0 = max(0, h_center - band_half)
    y1 = min(h, h_center + band_half)
    band = depth_m[y0:y1, :]

    col_width = max(1, w // scan_dim)
    scan = np.empty(scan_dim, dtype=np.float32)

    for i in range(scan_dim):
        x0 = i * col_width
        x1 = min(w, (i + 1) * col_width)
        col = band[:, x0:x1]
        if col.size == 0:
            scan[i] = np.nan
            continue
        # Drop infinite/nan values and use percentile
        valid = col[np.isfinite(col) & (col > 0)]
        if valid.size == 0:
            scan[i] = np.nan
        else:
            scan[i] = float(np.percentile(valid, percentile))

    return scan


# ===========================================================================
# Command receiver (base velocity control)
# ===========================================================================

class BaseCommandExecutor:
    """
    Minimal interface to the differential-drive base.

    The host implementation should connect to the Feetech bus and map
    body-frame velocities (x, y, theta) to wheel velocities using the
    LeKiwi kinematics ``_body_to_wheel_raw``.

    This stub provides a `send_velocity` method that can be replaced
    with the actual motor driver calls.
    """

    def __init__(self):
        # Kinematic constants (LeKiwi omni wheels)
        self.wheel_radius = 0.05   # m
        self.base_radius = 0.125   # m
        self.max_raw = 3000        # raw motor speed units

    def send_velocity(self, vx: float, vy: float, omega_deg_s: float) -> None:
        """
        Apply body-frame velocity command.

        Args:
            vx: Forward velocity [m/s].
            vy: Lateral velocity [m/s] (positive = left).
            omega_deg_s: Rotational velocity [deg/s] (CCW positive).
        """
        omega_rad_s = np.deg2rad(omega_deg_s)
        # Omni-wheel inverse kinematics (3 wheel)
        # Wheel speeds = A * [vx; vy; omega]
        w1 = vx + vy + self.base_radius * omega_rad_s
        w2 = vx - vy + self.base_radius * omega_rad_s
        w3 = -vx - vy + self.base_radius * omega_rad_s

        # Convert to raw motor units: raw = (wheel_speed / wheel_radius) * k
        # Where k maps m/s to raw units.  Actual constant depends on the
        # Feetech protocol; here we use a placeholder scale factor.
        raw_scale = 6000.0  # approximate
        raw = [
            int(np.clip(w * raw_scale / self.wheel_radius, -self.max_raw, self.max_raw))
            for w in (w1, w2, w3)
        ]
        logger.debug("Wheel raw: w1=%d w2=%d w3=%d", raw[0], raw[1], raw[2])
        # TODO(zbx): Replace with actual Feetech motor write calls.
        # feetech_bus.write(0x01, raw[0])
        # feetech_bus.write(0x02, raw[1])
        # feetech_bus.write(0x03, raw[2])

    def stop(self) -> None:
        """Emergency stop -- set all motors to 0."""
        logger.warning("Watchdog triggered -- stopping base.")
        # TODO(zbx): feetech_bus.write_all(0)
        pass


# ===========================================================================
# Merged config (host-only wrapper for draccus)
# ===========================================================================

@dataclass
class LeKiwiD435iHostMergedConfig:
    """
    Merged configuration consumed by the host ``main()``.

    Uses ``draccus`` default-factory injection to combine the robot-level
    ``LeKiwiD435iConfig`` with the host runtime ``LeKiwiD435iHostConfig``.
    """

    robot: LeKiwiD435iConfig = field(default_factory=LeKiwiD435iConfig)
    host: LeKiwiD435iHostConfig = field(default_factory=LeKiwiD435iHostConfig)


# ===========================================================================
# Main host loop
# ===========================================================================

class LeKiwiD435iHost:
    """
    Main on-board daemon.

    Lifecycle: ``__init__`` -> ``connect()`` -> ``run()`` -> ``disconnect()``.
    """

    def __init__(self, config: LeKiwiD435iHostMergedConfig):
        self.cfg = config
        self.host_cfg = config.host
        self.robot_cfg = config.robot

        # ZMQ
        self._zmq_context: zmq.Context | None = None
        self._zmq_cmd: zmq.Socket | None = None
        self._zmq_obs: zmq.Socket | None = None

        # Camera
        self._cameras: dict[str, Any] = {}
        self._depth_front_camera = None

        # Base command executor
        self._executor = BaseCommandExecutor()

        # State
        self._running = False
        self._last_cmd_time = time.monotonic()
        self._scan_smoothed: np.ndarray | None = None

        # ArUco
        self._aruco: ArucoPalletDetector | None = None

        # Depth save directory
        self._depth_save_path = Path(self.host_cfg.depth_save_dir)
        self._depth_frame_idx = 0

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Open camera, bind ZMQ sockets, and initialise ArUco."""
        logger.info("Connecting LeKiwi D435i host ...")

        # Cameras via LeRobot factory
        self._cameras = make_cameras_from_configs(self.robot_cfg.cameras)
        for name, cam in self._cameras.items():
            cam.connect()
            logger.info("  Camera '%s' connected.", name)

        if "front" in self._cameras:
            self._depth_front_camera = self._cameras["front"]

        # ZMQ
        zmq_ctx = zmq.Context()
        self._zmq_context = zmq_ctx

        self._zmq_cmd = zmq_ctx.socket(zmq.PULL)
        self._zmq_cmd.bind(f"tcp://*:{self.host_cfg.port_zmq_cmd}")
        self._zmq_cmd.setsockopt(zmq.CONFLATE, 1)
        logger.info("  ZMQ PULL bound on *:%d", self.host_cfg.port_zmq_cmd)

        self._zmq_obs = zmq_ctx.socket(zmq.PUSH)
        self._zmq_obs.bind(f"tcp://*:{self.host_cfg.port_zmq_observations}")
        self._zmq_obs.setsockopt(zmq.CONFLATE, 1)
        logger.info("  ZMQ PUSH bound on *:%d", self.host_cfg.port_zmq_observations)

        # ArUco detector
        if self.host_cfg.enable_aruco:
            self._aruco = ArucoPalletDetector(
                dictionary_name="DICT_4X4_50",
                marker_size_m=0.12,
                camera_matrix=None,  # auto-calibration not implemented yet
                valid_ids=[0, 1, 2, 3, 4],
                min_confidence=0.7,
            )
            logger.info("  ArUco detector initialised (auto-calibrate pending).")

        # Ensure depth save directory
        self._depth_save_path.mkdir(parents=True, exist_ok=True)

        logger.info("Host connected successfully.")

    def disconnect(self) -> None:
        """Clean up cameras, ZMQ sockets, and stop base."""
        self._running = False
        self._executor.stop()

        for cam in self._cameras.values():
            try:
                cam.disconnect()
            except Exception:
                pass
        self._cameras.clear()

        for sock in (self._zmq_cmd, self._zmq_obs):
            if sock is not None:
                try:
                    sock.close()
                except Exception:
                    pass
        if self._zmq_context is not None:
            try:
                self._zmq_context.term()
            except Exception:
                pass

        logger.info("Host disconnected.")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Blocking main loop."""
        self._running = True
        period_s = 1.0 / self.host_cfg.max_loop_freq_hz
        watchdog_s = self.host_cfg.watchdog_timeout_ms * 0.001

        logger.info(
            "Starting main loop @ %.1f Hz (watchdog: %.2f s)",
            self.host_cfg.max_loop_freq_hz,
            watchdog_s,
        )

        while self._running:
            loop_start = time.perf_counter()

            try:
                # 1. Receive and execute command
                self._process_incoming_command()

                # 2. Read sensors (RGB + depth)
                obs = self._read_sensors()

                # 3. Transmit observation
                self._send_observation(obs)

                # 4. Watchdog check
                if time.monotonic() - self._last_cmd_time > watchdog_s:
                    self._executor.stop()

            except KeyboardInterrupt:
                logger.info("KeyboardInterrupt received. Shutting down.")
                break
            except Exception:
                logger.exception("Unhandled error in main loop.")
                # Keep running; don't crash the daemon
                time.sleep(0.1)
                continue

            # Maintain loop rate
            elapsed = time.perf_counter() - loop_start
            if elapsed < period_s:
                time.sleep(period_s - elapsed)

    def _process_incoming_command(self) -> None:
        """Non-blocking read of the latest command from ZMQ."""
        try:
            while True:
                msg = self._zmq_cmd.recv_string(zmq.NOBLOCK)
                cmd = json.loads(msg)
                vx = float(cmd.get("x.vel", 0.0))
                vy = float(cmd.get("y.vel", 0.0))
                omega = float(cmd.get("theta.vel", 0.0))
                self._executor.send_velocity(vx, vy, omega)
                self._last_cmd_time = time.monotonic()
        except zmq.Again:
            pass  # No command waiting

    def _read_sensors(self) -> dict[str, Any]:
        """Capture RGB + depth from the D435i and build observation dict."""
        obs: dict[str, Any] = {}

        # Current velocity state.  In the real system these would be read
        # back from the motor encoders; here we stub with zeros.
        obs["x.vel"] = 0.0
        obs["y.vel"] = 0.0
        obs["theta.vel"] = 0.0

        if self._depth_front_camera is None:
            return obs

        cam = self._depth_front_camera
        cfg = self.robot_cfg.cameras["front"]

        # RGB
        try:
            rgb = cam.read()
        except Exception:
            logger.warning("Failed to read RGB frame; using placeholder.")
            rgb = np.zeros((cfg.height, cfg.width, 3), dtype=np.uint8)

        # JPEG-encode -> base64 for transport
        ok, jpg = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        if ok:
            obs["front"] = base64.b64encode(jpg).decode("ascii")
        else:
            obs["front"] = ""

        # Depth
        try:
            depth_raw = cam.read_depth()  # uint16, mm
        except Exception:
            logger.warning("Failed to read depth frame.")
            depth_raw = np.zeros((cfg.height, cfg.width), dtype=np.uint16)

        # Preprocess depth on the host
        depth_m = preprocess_depth(
            depth_raw,
            min_m=self.host_cfg.depth_min_m,
            max_m=self.host_cfg.depth_max_m,
            median_ksize=self.host_cfg.median_filter_ksize,
            hole_fill=self.host_cfg.hole_fill,
        )

        # Temporal smoothing of depth (EMA)
        # (Not applied here for simplicity; would require storing previous frame)

        # Depth-to-scan
        scan = depth_to_scan(
            depth_m,
            scan_dim=self.host_cfg.scan_dim,
            slice_fraction=self.host_cfg.slice_fraction,
            percentile=self.host_cfg.percentile,
        )

        # Temporal smoothing of scan (EMA)
        if self._scan_smoothed is not None:
            alpha = self.host_cfg.scan_temporal_alpha
            scan = alpha * scan + (1.0 - alpha) * self._scan_smoothed
        self._scan_smoothed = scan.copy()

        # Fill NaN with max range
        scan_filled = np.nan_to_num(scan, nan=self.host_cfg.depth_max_m)

        obs["scan64"] = scan_filled.tolist()

        # Per-sector minima
        n = self.host_cfg.scan_dim
        third = n // 3
        obs["left_min"] = float(np.min(scan_filled[:third])) if third > 0 else self.host_cfg.depth_max_m
        obs["front_min"] = float(np.min(scan_filled[third : 2 * third])) if third > 0 else self.host_cfg.depth_max_m
        obs["right_min"] = float(np.min(scan_filled[2 * third:])) if third > 0 else self.host_cfg.depth_max_m

        # ArUco detection
        obs["pallet_pose"] = None
        if self._aruco is not None:
            # Attempt to auto-calibrate camera matrix from image size
            if self._aruco.camera_matrix is None:
                h, w = rgb.shape[:2]
                fx = fy = max(h, w)  # rough pinhole guess
                cx, cy = w / 2.0, h / 2.0
                self._aruco.camera_matrix = np.array(
                    [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32
                )

            pallet = self._aruco.detect(rgb)
            if pallet is not None:
                obs["pallet_pose"] = {
                    "x": pallet["x"],
                    "y": pallet["y"],
                    "z": pallet["z"],
                    "yaw": pallet["yaw"],
                }

        # Save raw depth locally (PNG, uint16)
        self._save_depth_locally(depth_raw)

        return obs

    def _save_depth_locally(self, depth: np.ndarray) -> None:
        """Write raw uint16 depth as PNG to the local save directory."""
        try:
            fname = self._depth_save_path / f"depth_{self._depth_frame_idx:08d}.png"
            cv2.imwrite(str(fname), depth)
            self._depth_frame_idx += 1
        except Exception:
            logger.warning("Failed to save depth frame locally.", exc_info=True)

    def _send_observation(self, obs: dict[str, Any]) -> None:
        """Serialise observation dict to JSON and push over ZMQ."""
        try:
            payload = json.dumps(obs)
            self._zmq_obs.send_string(payload)
        except Exception:
            logger.warning("Failed to send observation.", exc_info=True)


# ===========================================================================
# Entry point
# ===========================================================================

def _has_draccus() -> bool:
    return HAS_DRACCUS


if _has_draccus():

    @draccus.wrap()
    def main(config: LeKiwiD435iHostMergedConfig) -> None:
        """Main entry point for the LeKiwi D435i host daemon."""
        host = LeKiwiD435iHost(config)
        try:
            host.connect()
            host.run()
        finally:
            host.disconnect()

else:

    def main(config: LeKiwiD435iHostMergedConfig) -> None:
        """Main entry point (without draccus -- pass config manually)."""
        host = LeKiwiD435iHost(config)
        try:
            host.connect()
            host.run()
        finally:
            host.disconnect()


if __name__ == "__main__":
    # Fallback: construct config manually when not using draccus CLI
    merged = LeKiwiD435iHostMergedConfig()
    main(merged)
