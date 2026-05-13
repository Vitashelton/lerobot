"""Run real LeKiwi + D435i demo with safety monitoring."""
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
class RealDemoConfig:
    """CLI configuration for the real-robot demo."""

    remote_ip: str = "192.168.1.100"
    port_cmd: int = 5555
    port_obs: int = 5556
    safe_mode: bool = True  # perception only, no base movement
    low_speed_factor: float = 0.3
    max_duration_s: int = 120
    display: bool = True
    record: bool = True
    output_dir: str = "demo_output/real"
    # Mock mode: use synthetic camera data instead of real ZMQ connection
    mock_mode: bool = False
    mock_scene: str = "cluttered_lab"


# ---------------------------------------------------------------------------
# ZMQ client for LeKiwi host communication
# ---------------------------------------------------------------------------

class LeKiwiZMQClient:
    """Lightweight ZMQ client to communicate with the LeKiwi D435i host.

    Uses external zmq library. Falls back to mock data if unavailable or
    mock_mode is set.
    """

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
        try:
            import zmq as _zmq
            self._zmq = _zmq
        except ImportError:
            logger.warning("zmq not available; switching to mock mode.")
            self.mock_mode = True

        self._ctx = None
        self._cmd_sock = None
        self._obs_sock = None
        self._connected = False

        # Mock state
        self._mock_frame_idx = 0

    def connect(self) -> None:
        """Connect ZMQ sockets to the host."""
        if self.mock_mode or self._zmq is None:
            logger.info("LeKiwiZMQClient: using mock mode.")
            self._connected = True
            return

        self._ctx = self._zmq.Context()
        self._cmd_sock = self._ctx.socket(self._zmq.PUSH)
        self._cmd_sock.connect(f"tcp://{self.remote_ip}:{self.port_cmd}")
        self._cmd_sock.setsockopt(self._zmq.CONFLATE, 1)

        self._obs_sock = self._ctx.socket(self._zmq.PULL)
        self._obs_sock.connect(f"tcp://{self.remote_ip}:{self.port_obs}")
        self._obs_sock.setsockopt(self._zmq.CONFLATE, 1)
        self._obs_sock.setsockopt(self._zmq.RCVTIMEO, 100)  # ms

        self._connected = True
        logger.info("LeKiwiZMQClient connected to %s", self.remote_ip)

    def disconnect(self) -> None:
        """Close sockets and context."""
        for sock in (self._cmd_sock, self._obs_sock):
            if sock is not None:
                try:
                    sock.close()
                except Exception:
                    pass
        if self._ctx is not None:
            try:
                self._ctx.term()
            except Exception:
                pass
        self._connected = False

    def send_action(self, action: dict) -> None:
        """Send a velocity action to the host."""
        if self.mock_mode or self._cmd_sock is None:
            return
        try:
            payload = json.dumps(action)
            self._cmd_sock.send_string(payload)
        except Exception:
            logger.warning("Failed to send action.", exc_info=True)

    def receive_observation(self) -> dict | None:
        """Receive observation from the host (non-blocking). Returns None
        if no data available.
        """
        if self.mock_mode:
            return self._mock_observation()

        if self._obs_sock is None:
            return None
        try:
            msg = self._obs_sock.recv_string(self._zmq.NOBLOCK)
            return json.loads(msg)
        except self._zmq.Again:
            return None
        except Exception:
            logger.warning("Failed to receive observation.", exc_info=True)
            return None

    @property
    def is_connected(self) -> bool:
        return self._connected

    # Mock observation generator (matches host observation format)
    def _mock_observation(self) -> dict:
        import base64

        h, w = 640, 480
        # Generate synthetic RGB
        t = self._mock_frame_idx * 0.1
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        rgb[:, :] = (180, 180, 180)  # gray
        rgb[int(h * 0.6):] = (120, 120, 120)  # floor

        # Add some obstacles
        obstacles = [
            (int(w * 0.3), int(h * 0.5), 50, 60, (60, 120, 180)),
            (int(w * 0.6 + 10 * np.sin(t)), int(h * 0.45), 45, 55, (60, 140, 60)),
            (int(w * 0.75), int(h * 0.5), 40, 50, (180, 80, 80)),
        ]
        for ox, oy, ow, oh, oc in obstacles:
            x1, y1 = ox - ow // 2, oy - oh // 2
            x2, y2 = x1 + ow, y1 + oh
            if 0 <= x1 < w and 0 <= y1 < h:
                rgb[max(0, y1):min(h, y2), max(0, x1):min(w, x2)] = oc

        ok, jpg = cv2.imencode(".jpg", rgb)
        front_b64 = base64.b64encode(jpg).decode("ascii") if ok else ""

        # Generate scan (64 bins)
        scan = np.full(64, 5.0, dtype=np.float32)
        for i in range(64):
            bearing = np.deg2rad(-87.0 / 2 + i * 87.0 / 64)
            base_dist = 1.5 + np.random.normal(0, 0.3)
            # Obstacles at certain angles
            if 25 < i < 35:
                base_dist = 0.5 + np.random.normal(0, 0.1)
            elif 40 < i < 50:
                base_dist = 0.8 + np.random.normal(0, 0.15)
            scan[i] = max(0.1, base_dist)

        third = 64 // 3
        self._mock_frame_idx += 1

        return {
            "x.vel": 0.0,
            "y.vel": 0.0,
            "theta.vel": 0.0,
            "front": front_b64,
            "scan64": scan.tolist(),
            "left_min": float(np.min(scan[:third])),
            "front_min": float(np.min(scan[third:2 * third])),
            "right_min": float(np.min(scan[2 * third:])),
            "pallet_pose": None,
        }


# ---------------------------------------------------------------------------
# Perception pipeline (client-side)
# ---------------------------------------------------------------------------

class PerceptionPipeline:
    """Simple perception pipeline on the client side for safety monitoring.

    Processes observations from the host and evaluates risk.
    """

    def __init__(self):
        self.risk_level = "unknown"
        self.front_min = float("inf")
        self.left_min = float("inf")
        self.right_min = float("inf")
        self.pallet_detected = False
        self.shield_active = False

    def process(self, obs: dict) -> dict:
        """Process observation and return enriched data."""
        scan = np.array(obs.get("scan64", []), dtype=np.float32)
        front_min = obs.get("front_min", float("inf"))
        left_min = obs.get("left_min", float("inf"))
        right_min = obs.get("right_min", float("inf"))
        pallet_pose = obs.get("pallet_pose")

        # Risk assessment
        if front_min < 0.2:
            risk = "danger"
        elif front_min < 0.5:
            risk = "warning"
        else:
            risk = "safe"

        self.risk_level = risk
        self.front_min = front_min
        self.left_min = left_min
        self.right_min = right_min
        self.pallet_detected = pallet_pose is not None

        return {
            "risk_level": risk,
            "front_min": front_min,
            "left_min": left_min,
            "right_min": right_min,
            "pallet_detected": self.pallet_detected,
            "pallet_pose": pallet_pose,
            "scan": scan,
            "rgb_b64": obs.get("front", ""),
        }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def draw_real_dashboard(
    rgb: np.ndarray | None,
    scan: np.ndarray,
    observation: dict,
    fps: float,
    latency_ms: float,
    safe_mode: bool,
) -> np.ndarray:
    """Draw the real-demo dashboard."""
    panel_w, panel_h = 480, 640
    total_h = panel_h * 2
    total_w = panel_w * 2
    canvas = np.ones((total_h, total_w, 3), dtype=np.uint8) * 220

    # --- Top-left: RGB ---
    if rgb is not None:
        rgb_resized = cv2.resize(rgb, (panel_w, panel_h))
        canvas[:panel_h, :panel_w] = rgb_resized
    else:
        cv2.putText(canvas, "NO RGB", (panel_w // 3, panel_h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # --- Top-right: Depth / status ---
    status = canvas[0:panel_h, panel_w:total_w]
    status[:] = (240, 240, 240)
    y = 30

    def draw_row(label, value, color=(0, 0, 0)):
        nonlocal y
        cv2.putText(status, f"{label}: {value}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        y += 25

    draw_row("Mode", "SAFE (perception only)" if safe_mode else "LIVE (base active)")
    draw_row("FPS", f"{fps:.1f}")
    draw_row("Latency", f"{latency_ms:.1f} ms")
    draw_row("Risk", observation.get("risk_level", "?").upper(),
             (0, 255, 0) if observation.get("risk_level") == "safe" else
             (0, 165, 255) if observation.get("risk_level") == "warning" else (0, 0, 255))
    draw_row("Front Min", f"{observation.get('front_min', float('inf')):.2f} m",
             (0, 0, 255) if observation.get('front_min', float('inf')) < 0.3 else (0, 100, 0))
    draw_row("Left Min", f"{observation.get('left_min', float('inf')):.2f} m")
    draw_row("Right Min", f"{observation.get('right_min', float('inf')):.2f} m")
    draw_row("Pallet", "DETECTED" if observation.get("pallet_detected") else "none",
             (0, 180, 0) if observation.get("pallet_detected") else (100, 100, 100))
    draw_row("Shield", "ACTIVE" if observation.get("shield_active") else "inactive",
             (0, 0, 255) if observation.get("shield_active") else (100, 100, 100))

    # --- Bottom-left: Scan ---
    scan_plot = canvas[panel_h:total_h, :panel_w]
    scan_plot[:] = (240, 240, 240)
    n = len(scan)
    if n > 0:
        max_range = max(np.nanmax(scan) if np.any(np.isfinite(scan)) else 5.0, 1.0)
        bar_w = max(1, panel_w // n)

        # Grid lines
        for d in np.arange(1, int(max_range) + 1):
            y_grid = int(panel_h * (1 - d / max_range))
            cv2.line(scan_plot, (0, y_grid), (panel_w, y_grid), (200, 200, 200), 1)

        for i, val in enumerate(scan):
            if np.isfinite(val):
                bar_h = int(panel_h * min(1.0, val / max_range))
                x0 = i * bar_w
                color = (0, 180, 0) if val > 1.0 else (0, 100, 255) if val > 0.3 else (0, 0, 255)
                cv2.rectangle(scan_plot, (x0, panel_h - bar_h), (x0 + bar_w - 1, panel_h), color, -1)

        # Sector dividers
        third = n // 3
        for div in [third, 2 * third]:
            x = div * bar_w
            cv2.line(scan_plot, (x, 0), (x, panel_h), (100, 100, 100), 1)

        cv2.putText(scan_plot, "LiDAR Scan (64 bins)", (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # --- Bottom-right: Top-down safety map ---
    top_down = canvas[panel_h:total_h, panel_w:total_w]
    top_down[:] = (240, 240, 240)
    cx, cy = panel_w // 2, panel_h // 2
    scale = 40  # pixels per meter

    # Robot marker
    cv2.circle(top_down, (cx, cy), 10, (0, 0, 255), -1)
    cv2.line(top_down, (cx, cy), (cx, cy - 15), (0, 0, 255), 2)  # heading

    # Safety zones
    cv2.circle(top_down, (cx, cy), int(0.15 * scale), (0, 0, 255), 1)  # emergency
    cv2.circle(top_down, (cx, cy), int(0.5 * scale), (0, 165, 255), 1)  # slow-down

    # Draw scan points
    if n > 0:
        for i, val in enumerate(scan):
            if np.isfinite(val) and val < 3.0:
                angle = np.deg2rad(-87.0 / 2 + i * 87.0 / n)
                px = cx + int(scale * val * np.sin(angle))
                py = cy - int(scale * val * np.cos(angle))
                if 0 <= px < panel_w and 0 <= py < panel_h:
                    cv2.circle(top_down, (px, py), 1, (255, 0, 0), -1)

    cv2.putText(top_down, "Safety Map", (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return canvas


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@draccus.wrap()
def main(cfg: RealDemoConfig) -> None:
    """Run real demo:

    1. Connect to LeKiwi host via ZMQ
    2. Read RGB + depth from D435i (via host)
    3. Run perception pipeline (obstacles, ArUco, safety)
    4. If not safe_mode: send low-speed safe actions
    5. Display dashboard
    6. Record data
    """
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Initialize ZMQ client
    client = LeKiwiZMQClient(
        remote_ip=cfg.remote_ip,
        port_cmd=cfg.port_cmd,
        port_obs=cfg.port_obs,
        mock_mode=cfg.mock_mode,
    )
    client.connect()

    # Initialize perception
    perception = PerceptionPipeline()

    # Data recorder
    recorded_frames = [] if cfg.record else None
    video_writer = None
    if cfg.record:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_path = str(out_dir / "real_demo.mp4")
        video_writer = cv2.VideoWriter(video_path, fourcc, 15.0, (960, 1280))
        logger.info("Recording to %s", video_path)

    # FPS tracking
    fps_times: list[float] = []

    logger.info("Starting real demo loop (safe_mode=%s, duration=%ds)...",
                cfg.safe_mode, cfg.max_duration_s)

    start_time = time.monotonic()
    prev_scan: np.ndarray | None = None
    step = 0

    try:
        while (time.monotonic() - start_time) < cfg.max_duration_s:
            loop_start = time.perf_counter()

            # Receive observation
            obs = client.receive_observation()
            if obs is None:
                time.sleep(0.01)
                continue

            # Process perception
            processed = perception.process(obs)

            # Decode RGB from base64 if available
            rgb = None
            rgb_b64 = obs.get("front", "")
            if rgb_b64:
                try:
                    import base64
                    jpg_bytes = base64.b64decode(rgb_b64)
                    rgb = cv2.imdecode(np.frombuffer(jpg_bytes, np.uint8), cv2.IMREAD_COLOR)
                except Exception:
                    pass

            # Simple safety action (velocity zero for safe mode)
            action = {"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0}
            shield_active = False

            if processed["risk_level"] == "danger":
                action = {"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0}
                shield_active = True
            elif not cfg.safe_mode:
                # Basic reactive navigation: avoid obstacles, move forward
                front = processed["front_min"]
                left = processed["left_min"]
                right = processed["right_min"]

                if front > 0.5:
                    action = {"x.vel": 0.1 * cfg.low_speed_factor, "y.vel": 0.0,
                              "theta.vel": 0.0}
                elif left > right and left > 0.3:
                    action = {"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 30.0 * cfg.low_speed_factor}
                elif right > 0.3:
                    action = {"x.vel": 0.0, "y.vel": 0.0, "theta.vel": -30.0 * cfg.low_speed_factor}
                else:
                    action = {"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0}

            # Send action (only if not safe mode)
            if not cfg.safe_mode:
                client.send_action(action)

            processed["shield_active"] = shield_active

            # Visualization
            scan = processed.get("scan", np.array([]))
            latency_ms = (time.perf_counter() - loop_start) * 1000

            # FPS
            now = time.perf_counter()
            fps_times.append(now)
            if len(fps_times) > 30:
                fps_times = fps_times[-30:]
            fps = len(fps_times) / (fps_times[-1] - fps_times[0]) if len(fps_times) > 1 else 0.0

            if cfg.display or video_writer is not None:
                dashboard = draw_real_dashboard(
                    rgb=rgb,
                    scan=scan,
                    observation=processed,
                    fps=fps,
                    latency_ms=latency_ms,
                    safe_mode=cfg.safe_mode,
                )

                if cfg.display:
                    cv2.imshow("LeKiwi Real Demo", dashboard)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                if video_writer is not None:
                    video_writer.write(dashboard)

            # Record frame data
            if recorded_frames is not None:
                record_entry = {
                    "step": step,
                    "timestamp": time.time(),
                    "risk": processed["risk_level"],
                    "front_min": processed["front_min"],
                    "left_min": processed["left_min"],
                    "right_min": processed["right_min"],
                    "action": action,
                    "shield_active": shield_active,
                }
                if scan is not None and len(scan) > 0:
                    record_entry["scan"] = np.nan_to_num(scan, nan=5.0).tolist()
                recorded_frames.append(record_entry)

            step += 1

            # Maintain 15 Hz
            elapsed = time.perf_counter() - loop_start
            if elapsed < 1.0 / 15.0:
                time.sleep(1.0 / 15.0 - elapsed)

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")

    finally:
        client.disconnect()
        if video_writer is not None:
            video_writer.release()
        if recorded_frames:
            # Save recorded data as JSONL
            jsonl_path = out_dir / "real_demo_data.jsonl"
            with open(jsonl_path, "w") as f:
                for entry in recorded_frames:
                    f.write(json.dumps(entry) + "\n")
            logger.info("Recorded %d frames to %s", len(recorded_frames), jsonl_path)
        if cfg.display:
            cv2.destroyAllWindows()
        logger.info("Real demo finished. Output in %s", out_dir)


if __name__ == "__main__":
    main()
