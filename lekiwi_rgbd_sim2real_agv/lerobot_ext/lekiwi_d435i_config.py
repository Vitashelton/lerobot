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
Configuration dataclasses for the LeKiwi D435i robot system.

Defines three configuration tiers:
  - LeKiwiD435iConfig:       Full host-side robot config (camera + base).
  - LeKiwiD435iHostConfig:   Host runtime parameters (ZMQ, watchdog, depth pipeline).
  - LeKiwiD435iClientConfig: Client-side robot config (remote connection, mock mode).

Uses the LeRobot ``RobotConfig`` subclass registry so that these configs can be
instantiated from YAML / draccus CLI argument parsing.
"""

from dataclasses import dataclass, field

from lerobot.cameras.configs import CameraConfig
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.robots.config import RobotConfig


def lekiwi_d435i_cameras_config() -> dict[str, CameraConfig]:
    """
    Default D435i camera configuration with depth enabled.

    Returns a single front-facing Intel RealSense D435i camera configured for
    480x640 @ 15 FPS with depth stream active and a 5-second warm-up period.
    """
    return {
        "front": RealSenseCameraConfig(
            serial_number_or_name="401622073080",
            fps=15,
            width=480,
            height=640,
            use_depth=True,
            warmup_s=5,
        ),
    }


# ---------------------------------------------------------------------------
# Host-side robot config (registered with RobotConfig registry)
# ---------------------------------------------------------------------------

@RobotConfig.register_subclass("lekiwi_d435i")
@dataclass
class LeKiwiD435iConfig(RobotConfig):
    """
    Full robot configuration for the LeKiwi D435i host (Pi / Jetson).

    Includes camera definitions used by :func:`make_cameras_from_configs`
    to instantiate the RealSense pipeline.
    """

    cameras: dict[str, CameraConfig] = field(default_factory=lekiwi_d435i_cameras_config)


# ---------------------------------------------------------------------------
# Host runtime parameters (NOT a RobotConfig subclass -- plain dataclass)
# ---------------------------------------------------------------------------

@dataclass
class LeKiwiD435iHostConfig:
    """
    Runtime parameters for the host-side main loop.

    These parameters are NOT part of the robot config registry; they are
    merged via a wrapper dataclass before passing to the host ``main()``.
    """

    # ---- ZMQ networking ----
    port_zmq_cmd: int = 5555
    port_zmq_observations: int = 5556
    connection_time_s: int = 30

    # ---- Safety / watchdog ----
    watchdog_timeout_ms: int = 500
    max_loop_freq_hz: int = 30

    # ---- Depth-to-scan pipeline ----
    scan_dim: int = 64
    depth_min_m: float = 0.15
    depth_max_m: float = 5.0

    # ---- Preprocessing ----
    median_filter_ksize: int = 5
    hole_fill: bool = True
    temporal_smoothing_alpha: float = 0.3

    # ---- Depth-to-scan quality ----
    slice_fraction: float = 0.3
    percentile: int = 10
    scan_temporal_alpha: float = 0.5

    # ---- ArUco detection ----
    enable_aruco: bool = True

    # ---- Local depth save dir (host only) ----
    depth_save_dir: str = "data/depth_raw"


# ---------------------------------------------------------------------------
# Client-side robot config (registered with RobotConfig registry)
# ---------------------------------------------------------------------------

@RobotConfig.register_subclass("lekiwi_d435i_client")
@dataclass
class LeKiwiD435iClientConfig(RobotConfig):
    """
    Client-side robot configuration for the D435i LeKiwi.

    Connects to a remote host via ZMQ.  Supports a ``mock_mode`` for
    offline testing without a physical robot.
    """

    # ---- Remote connection ----
    remote_ip: str = "192.168.1.100"
    port_zmq_cmd: int = 5555
    port_zmq_observations: int = 5556

    # ---- Camera definitions (used for feature specs on the client) ----
    cameras: dict[str, CameraConfig] = field(default_factory=lekiwi_d435i_cameras_config)

    # ---- Timing ----
    polling_timeout_ms: int = 15
    connect_timeout_s: int = 5

    # ---- Mock mode (no real robot) ----
    mock_mode: bool = False

    # ---- Perception (mirrors host config) ----
    scan_dim: int = 64

    # ---- Safety speed presets ----
    max_linear_vel: float = 0.3   # m/s
    max_angular_vel: float = 90.0  # deg/s
    emergency_stop_distance_m: float = 0.15
    slow_down_distance_m: float = 0.5
