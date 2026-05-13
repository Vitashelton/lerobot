"""Collect real-world dataset using LeKiwi + D435i."""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

import draccus

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")


@dataclass
class CollectRealConfig:
    """CLI configuration for real-world dataset collection."""

    remote_ip: str = "192.168.1.100"
    port_cmd: int = 5555
    port_obs: int = 5556
    output_dir: str = "data/real_logs"
    num_episodes: int = 10
    max_steps_per_episode: int = 500
    fps: int = 15
    teleop: bool = True  # True = human drives, False = autonomous
    save_depth: bool = True
    save_detections: bool = True
    display: bool = True
    # Mock mode for testing without hardware
    mock_mode: bool = False


# ---------------------------------------------------------------------------
# ZMQ client (shared with real demo)
# ---------------------------------------------------------------------------

class DatasetZMQClient:
    """Lightweight ZMQ client for dataset collection."""

    def __init__(
        self,
        remote_ip: str = "192.168.1.100",
        port_cmd: int = 5555,
        port_obs: int = 5556,
        mock_mode: bool = False,
    ):
        self.remote_ip = remote_ip
        self.port_cmd = port_cmd
        self.port_obs = port_obs
        self.mock_mode = mock_mode
        self._zmq = None
        self._connected = False

        try:
            import zmq as _zmq
            self._zmq = _zmq
        except ImportError:
            logger.warning("zmq not available; using mock mode.")
            self.mock_mode = True

        self._ctx = None
        self._cmd_sock = None
        self._obs_sock = None
        self._mock_step = 0

    def connect(self) -> None:
        if self.mock_mode or self._zmq is None:
            self._connected = True
            return
        self._ctx = self._zmq.Context()
        self._cmd_sock = self._ctx.socket(self._zmq.PUSH)
        self._cmd_sock.connect(f"tcp://{self.remote_ip}:{self.port_cmd}")
        self._cmd_sock.setsockopt(self._zmq.CONFLATE, 1)
        self._obs_sock = self._ctx.socket(self._zmq.PULL)
        self._obs_sock.connect(f"tcp://{self.remote_ip}:{self.port_obs}")
        self._obs_sock.setsockopt(self._zmq.CONFLATE, 1)
        self._obs_sock.setsockopt(self._zmq.RCVTIMEO, 100)
        self._connected = True

    def disconnect(self) -> None:
        for sock in (self._cmd_sock, self._obs_sock):
            if sock:
                try:
                    sock.close()
                except Exception:
                    pass
        if self._ctx:
            try:
                self._ctx.term()
            except Exception:
                pass
        self._connected = False

    def send_action(self, action: dict) -> None:
        if self.mock_mode or self._cmd_sock is None:
            return
        try:
            self._cmd_sock.send_string(json.dumps(action))
        except Exception:
            pass

    def receive_observation(self) -> dict | None:
        if self.mock_mode:
            return self._mock_observation()
        if self._obs_sock is None:
            return None
        try:
            return json.loads(self._obs_sock.recv_string(self._zmq.NOBLOCK))
        except Exception:
            return None

    @property
    def is_connected(self) -> bool:
        return self._connected

    def _mock_observation(self) -> dict:
        import base64
        h, w = 640, 480
        t = self._mock_step * 0.1
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        rgb[:, :] = (180, 180, 180)
        rgb[int(h * 0.6):] = (120, 120, 120)
        obstacles = [
            (int(w * 0.3), int(h * 0.5), 50, 60, (60, 120, 180)),
            (int(w * 0.6 + 10 * np.sin(t)), int(h * 0.45), 45, 55, (60, 140, 60)),
            (int(w * 0.75), int(h * 0.5), 40, 50, (180, 80, 80)),
        ]
        for ox, oy, ow, oh, oc in obstacles:
            x1, y1 = ox - ow // 2, oy - oh // 2
            x2, y2 = x1 + ow, y1 + oh
            rgb[max(0, y1):min(h, y2), max(0, x1):min(w, x2)] = oc

        ok, jpg = cv2.imencode(".jpg", rgb)
        front_b64 = base64.b64encode(jpg).decode("ascii") if ok else ""
        scan = np.full(64, float(np.random.uniform(1.0, 3.0)), dtype=np.float32)
        for i in range(25, 35):
            scan[i] = 0.5 + np.random.normal(0, 0.1)
        third = 64 // 3
        self._mock_step += 1

        return {
            "x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0,
            "front": front_b64,
            "scan64": scan.tolist(),
            "left_min": float(np.min(scan[:third])),
            "front_min": float(np.min(scan[third:2 * third])),
            "right_min": float(np.min(scan[2 * third:])),
            "pallet_pose": None,
        }


# ---------------------------------------------------------------------------
# Keyboard teleoperation
# ---------------------------------------------------------------------------

class KeyboardTeleop:
    """Read keyboard input for teleoperation."""

    # Default key-to-action mapping
    KEY_MAP = {
        ord("w"): (0.2, 0.0, 0.0),   # forward
        ord("s"): (-0.2, 0.0, 0.0),  # backward
        ord("a"): (0.0, 0.2, 0.0),   # left
        ord("d"): (0.0, -0.2, 0.0),  # right
        ord("q"): (0.0, 0.0, 30.0),  # rotate left
        ord("e"): (0.0, 0.0, -30.0), # rotate right
        ord("z"): (0.0, 0.0, 0.0),   # stop
    }

    def __init__(self, speed_factor: float = 1.0):
        self.speed_factor = speed_factor

    def get_action(self) -> dict | None:
        """Read keyboard and return action, or None to quit."""
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            return None
        if key in self.KEY_MAP:
            vx, vy, w = self.KEY_MAP[key]
            return {
                "x.vel": vx * self.speed_factor,
                "y.vel": vy * self.speed_factor,
                "theta.vel": w * self.speed_factor,
            }
        return {"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0}


# ---------------------------------------------------------------------------
# Autonomous navigation (simple reactive)
# ---------------------------------------------------------------------------

class SimpleAutonomous:
    """Basic reactive navigation for autonomous data collection."""

    def __init__(self, speed: float = 0.15):
        self.speed = speed

    def get_action(self, front_min: float, left_min: float, right_min: float) -> dict:
        if front_min > 0.5:
            return {"x.vel": self.speed, "y.vel": 0.0, "theta.vel": 0.0}
        elif left_min > right_min and left_min > 0.3:
            return {"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 45.0}
        elif right_min > 0.3:
            return {"x.vel": 0.0, "y.vel": 0.0, "theta.vel": -45.0}
        else:
            return {"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0}


# ---------------------------------------------------------------------------
# Data saver
# ---------------------------------------------------------------------------

class EpisodeSaver:
    """Save episode data to disk."""

    def __init__(self, base_dir: Path, episode_idx: int, save_depth: bool = True):
        self.base_dir = base_dir
        self.episode_dir = base_dir / f"episode_{episode_idx:04d}"
        self.episode_dir.mkdir(parents=True, exist_ok=True)
        self.save_depth = save_depth
        self.frames: list[dict] = []
        self.detections: list[dict] = []

    def add_frame(self, obs: dict, action: dict, step: int) -> None:
        """Record one frame."""
        frame_data = {
            "step": step,
            "timestamp": time.time(),
            "x.vel": obs.get("x.vel", 0.0),
            "y.vel": obs.get("y.vel", 0.0),
            "theta.vel": obs.get("theta.vel", 0.0),
            "action_x.vel": action.get("x.vel", 0.0),
            "action_y.vel": action.get("y.vel", 0.0),
            "action_theta.vel": action.get("theta.vel", 0.0),
            "scan64": obs.get("scan64", []),
            "front_min": obs.get("front_min", float("inf")),
            "left_min": obs.get("left_min", float("inf")),
            "right_min": obs.get("right_min", float("inf")),
            "front_b64": obs.get("front", ""),
        }
        if obs.get("pallet_pose"):
            frame_data["pallet_pose"] = obs["pallet_pose"]
        self.frames.append(frame_data)

    def add_detections(self, detections: dict) -> None:
        if detections:
            self.detections.append(detections)

    def save(self) -> Path:
        """Write all episode data."""
        # Save frames as JSONL
        frames_path = self.episode_dir / "observations.jsonl"
        with open(frames_path, "w") as f:
            for frame in self.frames:
                f.write(json.dumps(frame) + "\n")

        # Save detections
        if self.detections:
            det_path = self.episode_dir / "detections.jsonl"
            with open(det_path, "w") as f:
                for d in self.detections:
                    f.write(json.dumps(d) + "\n")

        # Save episode metadata
        meta = {
            "num_frames": len(self.frames),
            "episode_dir": str(self.episode_dir),
        }
        with open(self.episode_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        logger.info("  Saved %d frames to %s", len(self.frames), self.episode_dir)
        return self.episode_dir


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@draccus.wrap()
def main(cfg: CollectRealConfig) -> None:
    """Collect real-world dataset:

    1. Connect to LeKiwi host
    2. For each episode:
       - Start recording
       - If teleop: read keyboard, send actions
       - If autonomous: run reactive navigation
       - Save observation + action + depth + detections
       - End episode on keypress or max steps
    3. Write dataset summary
    """
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    client = DatasetZMQClient(
        remote_ip=cfg.remote_ip,
        port_cmd=cfg.port_cmd,
        port_obs=cfg.port_obs,
        mock_mode=cfg.mock_mode,
    )
    client.connect()

    teleop = KeyboardTeleop(speed_factor=1.0) if cfg.teleop else None
    autonomous = SimpleAutonomous(speed=0.15) if not cfg.teleop else None

    all_episodes: list[Path] = []
    period = 1.0 / cfg.fps

    logger.info("Starting dataset collection: %d episodes, %d steps max, teleop=%s",
                cfg.num_episodes, cfg.max_steps_per_episode, cfg.teleop)
    logger.info("Keys: W/S=forward/back, A/D=strafe, Q/E=rotate, Z=stop, ESC=end episode")

    for ep in range(cfg.num_episodes):
        logger.info("=== Episode %d/%d ===", ep + 1, cfg.num_episodes)
        saver = EpisodeSaver(out_dir, ep, save_depth=cfg.save_depth)

        # Warmup: discard first few observations
        for _ in range(5):
            client.receive_observation()
            time.sleep(0.02)

        for step in range(cfg.max_steps_per_episode):
            loop_start = time.perf_counter()

            # Get observation
            obs = client.receive_observation()
            if obs is None:
                time.sleep(0.01)
                continue

            # Get action
            if cfg.teleop:
                action = teleop.get_action()
                if action is None:
                    break
            else:
                action = autonomous.get_action(
                    front_min=obs.get("front_min", float("inf")),
                    left_min=obs.get("left_min", float("inf")),
                    right_min=obs.get("right_min", float("inf")),
                )

            # Send action
            client.send_action(action)

            # Save data
            saver.add_frame(obs, action, step)

            # Display
            if cfg.display:
                # Simple display showing RGB if available
                front_b64 = obs.get("front", "")
                if front_b64:
                    import base64
                    try:
                        jpg_bytes = base64.b64decode(front_b64)
                        rgb = cv2.imdecode(np.frombuffer(jpg_bytes, np.uint8), cv2.IMREAD_COLOR)
                        cv2.imshow("Dataset Collection", rgb)
                    except Exception:
                        pass

                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    break

            # Maintain FPS
            elapsed = time.perf_counter() - loop_start
            if elapsed < period:
                time.sleep(period - elapsed)

        # Save episode
        ep_dir = saver.save()
        all_episodes.append(ep_dir)

    # Write dataset summary
    summary = {
        "num_episodes": len(all_episodes),
        "episodes": [str(p) for p in all_episodes],
        "config": {
            "remote_ip": cfg.remote_ip,
            "teleop": cfg.teleop,
            "save_depth": cfg.save_depth,
            "max_steps": cfg.max_steps_per_episode,
        },
    }
    with open(out_dir / "dataset_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    client.disconnect()
    if cfg.display:
        cv2.destroyAllWindows()
    logger.info("Dataset collection finished. %d episodes saved to %s",
                len(all_episodes), out_dir)


if __name__ == "__main__":
    main()
