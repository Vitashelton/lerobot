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
LeKiwi D435i Client -- remote PC / laptop interface.

Extends the LeRobot ``Robot`` base class (like ``LeKiwiClient``) and adds
RGB-D perception decoding, scan-based safety fields, and ArUco pallet pose
parsing from the host's JSON observation stream.

Key responsibilities
--------------------
1. Connect to the D435i host via ZMQ (PUSH / PULL, same topology as vanilla
   LeKiwi).
2. Receive and parse JSON observations containing:
   - ``x.vel``, ``y.vel``, ``theta.vel`` (float)
   - ``front`` (base64 JPEG RGB image)
   - ``scan64`` (list of 64 floats)
   - ``front_min``, ``left_min``, ``right_min`` (float)
   - ``pallet_pose`` (dict | null)
3. Decode the RGB JPEG image back to a uint8 (H, W, 3) numpy array.
4. Expose ``get_observation()`` conforming to the LeRobot ``Robot`` interface.
5. Accept action dicts via ``send_action()`` and forward them to the host.
6. Apple safety limits on commanded velocities and optionally trigger an
   emergency stop based on front_min distance.
7. Support a ``mock_mode`` that synthesises plausible data for offline
   testing.
"""

import base64
import json
import logging
import time
from functools import cached_property
from typing import Any

import cv2
import numpy as np

from lerobot.robots.robot import Robot
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from .config import LeKiwiD435iClientConfig

logger = logging.getLogger(__name__)


# ===========================================================================
# Mock data generator (for ``mock_mode``)
# ===========================================================================

class _MockObservationGenerator:
    """
    Produces plausible synthetic observations when ``mock_mode`` is enabled.

    Generates a static RGB checkerboard, sinusoidal scan data, and randomized
    pallet poses to exercise the full pipeline without hardware.
    """

    def __init__(self, config: LeKiwiD435iClientConfig):
        n = config.scan_dim
        self._scan_dim = n

        # Static checkerboard RGB image
        if config.cameras:
            first_cam = list(config.cameras.values())[0]
            h, w = first_cam.height, first_cam.width
        else:
            h, w = 640, 480
        self._h, self._w = h, w
        checker = np.zeros((h, w, 3), dtype=np.uint8)
        block = 40
        for y in range(0, h, block):
            for x in range(0, w, block):
                if ((y // block) + (x // block)) % 2 == 0:
                    checker[y : y + block, x : x + block] = [100, 180, 100]
                else:
                    checker[y : y + block, x : x + block] = [60, 60, 140]
        self._checker = checker
        self._step = 0

    def generate(self) -> dict[str, Any]:
        """Return a dictionary matching the host JSON observation format."""
        self._step += 1
        t = self._step * 0.05  # pseudo-time

        # Simulate velocities (random walk)
        vx = 0.02 * np.sin(t * 0.5)
        vy = 0.01 * np.cos(t * 0.3)
        omega = 5.0 * np.sin(t * 0.7)

        # Scan: sinusoidal depth profile
        angles = np.linspace(-np.pi / 2, np.pi / 2, self._scan_dim)
        scan = 2.0 + 1.5 * np.sin(angles + t * 0.3) + 0.1 * np.random.randn(self._scan_dim)
        scan = np.clip(scan, 0.1, 5.0)

        # Sector minima
        third = self._scan_dim // 3
        front_min = float(np.min(scan[third : 2 * third])) if third > 0 else 5.0
        left_min = float(np.min(scan[:third])) if third > 0 else 5.0
        right_min = float(np.min(scan[2 * third:])) if third > 0 else 5.0

        # Random pallet (appears every ~100 steps)
        pallet = None
        if self._step % 97 == 0:
            pallet = {"x": np.random.uniform(-1, 1), "y": np.random.uniform(-1, 1),
                      "z": np.random.uniform(0.5, 2.0), "yaw": np.random.uniform(-np.pi, np.pi)}

        # RGB with varying brightness
        rgb = np.clip(self._checker.astype(np.float32) * (0.8 + 0.2 * np.sin(t * 0.1)),
                      0, 255).astype(np.uint8)
        _, jpg = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        front_b64 = base64.b64encode(jpg).decode("ascii")

        return {
            "x.vel": float(vx),
            "y.vel": float(vy),
            "theta.vel": float(omega),
            "front": front_b64,
            "scan64": scan.tolist(),
            "front_min": front_min,
            "left_min": left_min,
            "right_min": right_min,
            "pallet_pose": pallet,
        }


# ===========================================================================
# Client Robot class
# ===========================================================================

class LeKiwiD435iClient(Robot):
    """
    Remote client for the LeKiwi D435i robot.

    Implements the LeRobot ``Robot`` interface over ZMQ transport
    with RGB-D perception decoding and scan-based safety features.

    Usage::

        from communication.config import LeKiwiD435iClientConfig
        from communication.client import LeKiwiD435iClient

        cfg = LeKiwiD435iClientConfig(remote_ip="192.168.1.100")
        client = LeKiwiD435iClient(cfg)
        client.connect()
        obs = client.get_observation()
        # obs["front"] -> (H, W, 3) uint8 RGB image
        # obs["observation.state"] -> float32 array of state features
        client.send_action({"x.vel": 0.1, "y.vel": 0.0, "theta.vel": 0.0})
        client.disconnect()
    """

    config_class = LeKiwiD435iClientConfig
    name = "lekiwi_d435i_client"

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def __init__(self, config: LeKiwiD435iClientConfig):
        super().__init__(config)
        self.config = config
        self.id = config.id
        self.robot_type = config.type

        self.remote_ip = config.remote_ip
        self.port_zmq_cmd = config.port_zmq_cmd
        self.port_zmq_observations = config.port_zmq_observations
        self.polling_timeout_ms = config.polling_timeout_ms
        self.connect_timeout_s = config.connect_timeout_s

        self.mock_mode = config.mock_mode
        self.scan_dim = config.scan_dim

        # Safety limits
        self.max_linear_vel = config.max_linear_vel
        self.max_angular_vel = config.max_angular_vel
        self.emergency_stop_distance_m = config.emergency_stop_distance_m
        self.slow_down_distance_m = config.slow_down_distance_m

        # ZMQ state
        self._zmq_context: Any = None
        self._zmq_cmd_socket: Any = None
        self._zmq_observation_socket: Any = None
        self._is_connected: bool = False

        # Cached observation
        self._last_frames: dict[str, np.ndarray] = {}
        self._last_remote_state: dict[str, Any] = {}

        # Mock generator (lazily created)
        self._mock: _MockObservationGenerator | None = None

        # Speed level control (for direct teleoperation)
        self._speed_levels = [
            {"xy": 0.1, "theta": 30},
            {"xy": 0.2, "theta": 60},
            {"xy": 0.3, "theta": 90},
        ]
        self._speed_index = 0

    # ------------------------------------------------------------------
    # Feature descriptors (LeRobot protocol)
    # ------------------------------------------------------------------

    @cached_property
    def _state_ft(self) -> dict[str, type]:
        """Proprioceptive / scalar state features."""
        return {
            "x.vel": float,
            "y.vel": float,
            "theta.vel": float,
            "front_min": float,
            "left_min": float,
            "right_min": float,
        }

    @cached_property
    def _state_order(self) -> tuple[str, ...]:
        return tuple(self._state_ft.keys())

    @cached_property
    def _cameras_ft(self) -> dict[str, tuple[int, int, int]]:
        """Camera feature shapes: {name: (height, width, channels)}."""
        return {name: (cfg.height, cfg.width, 3) for name, cfg in self.config.cameras.items()}

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._state_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {
            "x.vel": float,
            "y.vel": float,
            "theta.vel": float,
        }

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def is_calibrated(self) -> bool:
        # Mobile base does not require calibration
        return True

    def connect(self) -> None:
        """
        Establish ZMQ sockets with the remote D435i host.

        Raises:
            DeviceAlreadyConnectedError: If already connected.
            DeviceNotConnectedError: If host doesn't respond within timeout.
        """
        if self.mock_mode:
            logger.info("Mock mode enabled -- ZMQ connection skipped.")
            self._mock = _MockObservationGenerator(self.config)
            self._is_connected = True
            return

        if self._is_connected:
            raise DeviceAlreadyConnectedError(
                "LeKiwiD435iClient is already connected. "
                "Do not call `connect()` twice."
            )

        import zmq

        self._zmq = zmq

        self._zmq_context = zmq.Context()

        # PUSH socket -> host command port
        self._zmq_cmd_socket = self._zmq_context.socket(zmq.PUSH)
        self._zmq_cmd_socket.connect(f"tcp://{self.remote_ip}:{self.port_zmq_cmd}")
        self._zmq_cmd_socket.setsockopt(zmq.CONFLATE, 1)

        # PULL socket <- host observation port
        self._zmq_observation_socket = self._zmq_context.socket(zmq.PULL)
        self._zmq_observation_socket.connect(
            f"tcp://{self.remote_ip}:{self.port_zmq_observations}"
        )
        self._zmq_observation_socket.setsockopt(zmq.CONFLATE, 1)

        # Wait for first observation
        poller = zmq.Poller()
        poller.register(self._zmq_observation_socket, zmq.POLLIN)
        socks = dict(poller.poll(self.connect_timeout_s * 1000))
        if (
            self._zmq_observation_socket not in socks
            or socks[self._zmq_observation_socket] != zmq.POLLIN
        ):
            raise DeviceNotConnectedError(
                "Timeout waiting for LeKiwi D435i host to respond."
            )

        self._is_connected = True
        logger.info(
            "Connected to LeKiwi D435i host at %s:%d/%d",
            self.remote_ip,
            self.port_zmq_cmd,
            self.port_zmq_observations,
        )

    def calibrate(self) -> None:
        """No-op: mobile base needs no calibration."""
        pass

    def configure(self) -> None:
        """No-op: runtime configuration not needed."""
        pass

    def disconnect(self) -> None:
        """Close ZMQ sockets and clean up."""
        if self.mock_mode:
            self._is_connected = False
            self._mock = None
            return

        if not self._is_connected:
            raise DeviceNotConnectedError(
                "LeKiwiD435iClient is not connected."
            )

        for sock in (self._zmq_observation_socket, self._zmq_cmd_socket):
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

        self._is_connected = False
        logger.info("Disconnected from LeKiwi D435i host.")

    # ------------------------------------------------------------------
    # ZMQ I/O
    # ------------------------------------------------------------------

    def _poll_and_get_latest_message(self) -> str | None:
        """Poll the observation socket and return the latest message (or None)."""
        zmq = self._zmq
        poller = zmq.Poller()
        poller.register(self._zmq_observation_socket, zmq.POLLIN)

        try:
            socks = dict(poller.poll(self.polling_timeout_ms))
        except zmq.ZMQError as e:
            logger.error("ZMQ polling error: %s", e)
            return None

        if self._zmq_observation_socket not in socks:
            return None

        last_msg = None
        while True:
            try:
                msg = self._zmq_observation_socket.recv_string(zmq.NOBLOCK)
                last_msg = msg
            except zmq.Again:
                break

        return last_msg

    def _decode_image_from_b64(self, image_b64: str) -> np.ndarray | None:
        """Decode a base64 JPEG string to an RGB uint8 numpy array."""
        if not image_b64:
            return None
        try:
            jpg_data = base64.b64decode(image_b64)
            np_arr = np.frombuffer(jpg_data, dtype=np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is not None:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return frame
        except (TypeError, ValueError) as e:
            logger.error("Error decoding base64 image: %s", e)
            return None

    def _remote_state_from_obs(
        self, observation: dict[str, Any]
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """
        Extract frames and flat state from a parsed JSON observation dict.

        Returns:
            (frames_dict, obs_dict) where ``obs_dict`` includes a
            ``"observation.state"`` key with a float32 numpy vector.
        """
        flat_state: dict[str, float] = {
            key: float(observation.get(key, 0.0)) for key in self._state_order
        }
        state_vec = np.array(
            [float(flat_state[k]) for k in self._state_order], dtype=np.float32
        )
        obs_dict: dict[str, Any] = {**flat_state, OBS_STATE: state_vec}

        # Decode RGB image
        frames: dict[str, np.ndarray] = {}
        front_b64 = observation.get("front", "")
        if front_b64:
            frame = self._decode_image_from_b64(str(front_b64))
            if frame is not None:
                frames["front"] = frame

        return frames, obs_dict

    def _get_data(self) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """
        Poll the ZMQ socket and return (frames, obs_dict).

        Falls back to cached data on any failure.
        """
        if self.mock_mode and self._mock is not None:
            observation = self._mock.generate()
            return self._remote_state_from_obs(observation)

        # 1. Latest message from socket
        latest_msg = self._poll_and_get_latest_message()
        if latest_msg is None:
            return self._last_frames, self._last_remote_state

        # 2. Parse JSON
        try:
            observation = json.loads(latest_msg)
        except json.JSONDecodeError as e:
            logger.error("JSON decode error: %s", e)
            return self._last_frames, self._last_remote_state

        # 3. Extract state / images
        try:
            new_frames, new_state = self._remote_state_from_obs(observation)
        except Exception as e:
            logger.error("Error processing observation, using cached data: %s", e)
            return self._last_frames, self._last_remote_state

        self._last_frames = new_frames
        self._last_remote_state = new_state
        return new_frames, new_state

    # ------------------------------------------------------------------
    # LeRobot interface
    # ------------------------------------------------------------------

    def get_observation(self) -> dict[str, Any]:
        """
        Retrieve the latest observation from the robot.

        Returns a flat dictionary whose keys match ``observation_features``.
        Includes the RGB image under ``"front"`` and a concatenated state
        vector under ``"observation.state"``.

        Raises:
            DeviceNotConnectedError: If the client is not connected.
        """
        if not self._is_connected:
            raise DeviceNotConnectedError("LeKiwiD435iClient is not connected.")

        frames, obs_dict = self._get_data()

        for cam_name, frame in frames.items():
            if cam_name not in self.config.cameras:
                continue

            cfg = self.config.cameras[cam_name]
            expected_h, expected_w = cfg.height, cfg.width

            if frame is None:
                frame = np.zeros((expected_h, expected_w, 3), dtype=np.uint8)

            if frame.shape[0] != expected_h or frame.shape[1] != expected_w:
                frame = cv2.resize(
                    frame, (expected_w, expected_h), interpolation=cv2.INTER_LINEAR
                )

            obs_dict[cam_name] = frame

        # Fill any missing camera keys with zeros
        for cam_name, cfg in self.config.cameras.items():
            if cam_name not in obs_dict:
                obs_dict[cam_name] = np.zeros(
                    (cfg.height, cfg.width, 3), dtype=np.uint8
                )

        return obs_dict

    # ------------------------------------------------------------------
    # Action sending with safety
    # ------------------------------------------------------------------

    def _apply_safety_constraints(
        self,
        action: dict[str, float],
        front_min: float | None = None,
    ) -> dict[str, float]:
        """
        Apply velocity limits and emergency-stop checks.

        Args:
            action: Raw action dict with ``x.vel``, ``y.vel``, ``theta.vel``.
            front_min: Current front sector minimum distance [m] (from scan).

        Returns:
            Safe (possibly zeroed) action dict.
        """
        safe = {}

        # Clip linear velocities
        safe["x.vel"] = float(np.clip(action.get("x.vel", 0.0),
                                       -self.max_linear_vel, self.max_linear_vel))
        safe["y.vel"] = float(np.clip(action.get("y.vel", 0.0),
                                       -self.max_linear_vel, self.max_linear_vel))

        # Clip angular velocity
        safe["theta.vel"] = float(np.clip(action.get("theta.vel", 0.0),
                                           -self.max_angular_vel, self.max_angular_vel))

        # Emergency stop if too close to obstacle in front
        if front_min is not None and front_min < self.emergency_stop_distance_m:
            logger.warning(
                "EMERGENCY STOP: front_min=%.3f m < %.3f m. Zeroing velocities.",
                front_min,
                self.emergency_stop_distance_m,
            )
            for k in safe:
                safe[k] = 0.0
            return safe

        # Slow-down zone: scale velocities linearly between stop and slow-down
        if front_min is not None and front_min < self.slow_down_distance_m:
            scale = max(
                0.0,
                (front_min - self.emergency_stop_distance_m)
                / (self.slow_down_distance_m - self.emergency_stop_distance_m),
            )
            for k in safe:
                safe[k] *= scale
            logger.debug(
                "Slow-down zone: front_min=%.3f m, scale=%.2f", front_min, scale
            )

        return safe

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """
        Send an action command to the robot host.

        Args:
            action: Dictionary with keys ``x.vel``, ``y.vel``, ``theta.vel``.

        Returns:
            The action dict actually sent (after safety clipping).

        Raises:
            DeviceNotConnectedError: If the client is not connected.
        """
        if not self._is_connected:
            raise DeviceNotConnectedError(
                "LeKiwiD435iClient is not connected. "
                "Call `connect()` before `send_action()`."
            )

        def _clean(value: Any) -> float:
            if hasattr(value, "item"):
                return float(value.item())
            return float(value)

        raw: dict[str, float] = {
            "x.vel": _clean(action.get("x.vel", 0.0)),
            "y.vel": _clean(action.get("y.vel", 0.0)),
            "theta.vel": _clean(action.get("theta.vel", 0.0)),
        }

        # Apply safety constraints using cached front_min
        front_min = float(self._last_remote_state.get("front_min", 5.0))
        safe = self._apply_safety_constraints(raw, front_min)

        if self.mock_mode:
            logger.debug("Mock send_action: %s", safe)
        else:
            self._zmq_cmd_socket.send_string(json.dumps(safe))

        return safe

    # ------------------------------------------------------------------
    # Convenience: direct teleoperation helpers
    # ------------------------------------------------------------------

    def set_speed_level(self, index: int) -> None:
        """Set speed preset index (0=slow, 1=medium, 2=fast)."""
        self._speed_index = max(0, min(2, index))

    def get_speed_setting(self) -> dict[str, float]:
        """Return (xy_speed, theta_speed) for the current speed level."""
        return dict(self._speed_levels[self._speed_index])

    @property
    def speed_index(self) -> int:
        return self._speed_index

    @speed_index.setter
    def speed_index(self, value: int) -> None:
        self._speed_index = max(0, min(2, value))

    def emergency_stop(self) -> None:
        """Send zero velocity immediately."""
        self.send_action({"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0})
        logger.warning("Emergency stop sent.")
