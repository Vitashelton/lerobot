"""Run simulation demo with synthetic RGB-D and LeKiwi DWA navigation."""

import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

import draccus

# Ensure project root is on sys.path for sibling imports.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from perception.realsense_reader import RealSenseReader
from perception.obstacle_detector import ObstacleDetector

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SimDemoConfig:
    """CLI configuration for the simulation demo."""

    scene_type: str = "warehouse_aisle"
    num_episodes: int = 5
    max_steps: int = 500
    display: bool = True
    record_video: bool = False
    output_dir: str = "demo_output/sim"


# ---------------------------------------------------------------------------
# Mock synthetic RGB-D reader for simulation demo
# ---------------------------------------------------------------------------

class MockSyntheticRGBDReader:
    """Generate simple synthetic RGB-D frames for the simulation demo.

    This class is intentionally independent from RealSenseReader.
    RealSenseReader should be used only for real D435i hardware.
    """

    def __init__(self, width: int = 480, height: int = 640, fps: int = 15, scene: str = "corridor"):
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.scene = scene
        self.frame_id = 0
        self.rng = np.random.RandomState(42)

    def start(self) -> None:
        self.frame_id = 0

    def stop(self) -> None:
        pass

    def read(self) -> dict:
        h, w = self.height, self.width

        # BGR image
        rgb = np.full((h, w, 3), 235, dtype=np.uint8)
        depth = np.full((h, w), 4.5, dtype=np.float32)

        # Add weak depth gradient so the visualization is not flat.
        yy = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
        depth = depth - 0.4 * yy

        if self.scene in ["corridor", "warehouse_aisle"]:
            # Two side shelves / walls.
            left_w = int(0.18 * w)
            right_w = int(0.82 * w)
            depth[:, :left_w] = 1.0
            depth[:, right_w:] = 1.0
            cv2.rectangle(rgb, (0, 0), (left_w, h), (90, 90, 90), -1)
            cv2.rectangle(rgb, (right_w, 0), (w, h), (90, 90, 90), -1)

            # A pallet/box in the middle distance.
            x1, x2 = int(0.43 * w), int(0.57 * w)
            y1, y2 = int(0.45 * h), int(0.72 * h)
            depth[y1:y2, x1:x2] = 1.8
            cv2.rectangle(rgb, (x1, y1), (x2, y2), (40, 120, 220), -1)

        elif self.scene in ["cluttered_lab", "cluttered"]:
            boxes = [
                (0.15, 0.35, 0.32, 0.70, 1.2, (50, 130, 220)),
                (0.62, 0.25, 0.78, 0.55, 1.6, (70, 180, 70)),
                (0.40, 0.58, 0.55, 0.82, 0.9, (180, 80, 80)),
            ]
            for x1f, y1f, x2f, y2f, d, color in boxes:
                x1, y1, x2, y2 = int(x1f*w), int(y1f*h), int(x2f*w), int(y2f*h)
                depth[y1:y2, x1:x2] = d
                cv2.rectangle(rgb, (x1, y1), (x2, y2), color, -1)

        elif self.scene in ["pallet_marker", "pallet_pickup"]:
            x1, x2 = int(0.35 * w), int(0.65 * w)
            y1, y2 = int(0.38 * h), int(0.75 * h)
            depth[y1:y2, x1:x2] = 1.2
            cv2.rectangle(rgb, (x1, y1), (x2, y2), (40, 140, 230), -1)
            # Fake marker
            mx1, my1, mx2, my2 = int(0.47*w), int(0.50*h), int(0.53*w), int(0.58*h)
            cv2.rectangle(rgb, (mx1, my1), (mx2, my2), (0, 0, 0), -1)

        elif self.scene in ["empty", "lab_empty"]:
            pass

        # Add mild depth noise and invalid pixels.
        noise = self.rng.normal(0.0, 0.015, size=depth.shape).astype(np.float32)
        depth = np.clip(depth + noise, 0.15, 5.0)

        invalid_mask = self.rng.rand(h, w) < 0.002
        depth[invalid_mask] = np.nan

        self.frame_id += 1
        return {
            "color": rgb,
            "depth": depth,
            "timestamp": time.time(),
            "intrinsics": {
                "fx": 430.0,
                "fy": 430.0,
                "cx": w / 2.0,
                "cy": h / 2.0,
            },
        }


# ---------------------------------------------------------------------------
# Simple DWA (Dynamic Window Approach) controller
# ---------------------------------------------------------------------------

class DWAController:
    """Minimal Dynamic Window Approach for differential-drive navigation.

    Given a goal (x, y) in robot frame and a 1-D scan, it evaluates a set
    of candidate (linear, angular) velocity pairs, scores them on:
      - heading toward the goal,
      - obstacle clearance,
      - forward speed,
    and returns the best action.
    """

    def __init__(
        self,
        max_linear: float = 0.3,
        max_angular: float = 90.0,
        linear_samples: int = 11,
        angular_samples: int = 21,
        dt: float = 0.1,
        predict_steps: int = 15,
        goal_weight: float = 0.5,
        clearance_weight: float = 1.0,
        speed_weight: float = 0.1,
        safe_distance: float = 0.3,
    ):
        self.max_linear = max_linear
        self.max_angular = max_angular
        self.linear_samples = linear_samples
        self.angular_samples = angular_samples
        self.dt = dt
        self.predict_steps = predict_steps
        self.goal_weight = goal_weight
        self.clearance_weight = clearance_weight
        self.speed_weight = speed_weight
        self.safe_distance = safe_distance

    def plan(self, goal_x: float, goal_y: float, scan_m: np.ndarray) -> dict:
        """Return (vx, omega_deg) and auxiliary info."""
        lin_vels = np.linspace(-self.max_linear, self.max_linear, self.linear_samples)
        ang_vels = np.linspace(-self.max_angular, self.max_angular, self.angular_samples)

        best_score = -np.inf
        best_action = (0.0, 0.0)
        best_info = {}

        for v in lin_vels:
            for w in ang_vels:
                score, info = self._score(v, w, goal_x, goal_y, scan_m)
                if score > best_score:
                    best_score = score
                    best_action = (v, w)
                    best_info = info

        return {
            "x.vel": best_action[0],
            "theta.vel": best_action[1],
            "score": best_score,
            "info": best_info,
        }

    def _score(self, v: float, w_deg: float, gx: float, gy: float, scan: np.ndarray) -> tuple[float, dict]:
        """Score a (v, omega) pair by forward-simulating the trajectory."""
        w_rad = np.deg2rad(w_deg)
        x, y, theta = 0.0, 0.0, 0.0
        min_clearance = float("inf")

        for _ in range(self.predict_steps):
            # Kinematic update
            x += v * np.cos(theta) * self.dt
            y += v * np.sin(theta) * self.dt
            theta += w_rad * self.dt

            # Check clearance: project position to polar and find closest scan bin
            dist = np.hypot(x, y)
            bearing = np.arctan2(y, x)
            min_clearance = min(min_clearance, dist)

            # Check against scan: using bearing to index into scan
            scan_bearing_deg = np.rad2deg(bearing)
            # Map bearing to scan index
            n = len(scan)
            fov_half = 87.0 / 2.0
            idx = int((scan_bearing_deg + fov_half) / (87.0 / n))
            if 0 <= idx < n and not np.isnan(scan[idx]):
                if dist < scan[idx]:
                    min_clearance = min(min_clearance, dist)

        # Collision check
        if min_clearance < self.safe_distance * 0.3:
            return -1000.0, {"min_clearance": min_clearance, "heading_error": 0.0}

        # Heading score: how well does final heading point toward goal?
        goal_dist = np.hypot(gx, gy)
        goal_bearing = np.arctan2(gy, gx)
        heading_error = abs(theta - goal_bearing)
        heading_error = min(heading_error, 2 * np.pi - heading_error)
        heading_score = max(0.0, 1.0 - heading_error / np.pi)

        # Clearance score
        clearance_score = min(1.0, min_clearance / self.safe_distance) if np.isfinite(min_clearance) else 0.0

        # Speed score
        speed_score = abs(v) / self.max_linear if self.max_linear > 0 else 0.0

        score = (
            self.goal_weight * heading_score
            + self.clearance_weight * clearance_score
            + self.speed_weight * speed_score
        )

        return score, {
            "min_clearance": min_clearance,
            "heading_error": heading_error,
            "heading_score": heading_score,
            "clearance_score": clearance_score,
            "speed_score": speed_score,
        }


# ---------------------------------------------------------------------------
# Simple residual safety controller
# ---------------------------------------------------------------------------

class SafetyResidualController:
    """Safety residual that overrides the DWA action if danger is detected."""

    def __init__(self, emergency_stop_m: float = 0.15, slow_down_m: float = 0.5):
        self.emergency_stop_m = emergency_stop_m
        self.slow_down_m = slow_down_m

    def apply(self, action: dict, sectors: dict) -> dict:
        """Return a (possibly modified) action dict."""
        front_min = sectors.get("front", {}).get("min_dist", float("inf"))
        risk = sectors.get("front", {}).get("risk", "safe")

        if risk == "danger" or front_min < self.emergency_stop_m:
            return {"x.vel": 0.0, "theta.vel": 0.0, "shield_active": True, "reason": "emergency_stop"}
        elif risk == "warning" or front_min < self.slow_down_m:
            factor = max(0.1, (front_min - self.emergency_stop_m) / (self.slow_down_m - self.emergency_stop_m))
            return {
                "x.vel": action.get("x.vel", 0.0) * factor,
                "theta.vel": action.get("theta.vel", 0.0) * factor,
                "shield_active": True,
                "reason": "slow_down",
            }
        return {**action, "shield_active": False, "reason": "normal"}


# ---------------------------------------------------------------------------
# Goal generation
# ---------------------------------------------------------------------------

def generate_goal(scene_type: str, step: int, max_steps: int) -> np.ndarray:
    """Generate a goal position in robot frame based on the scene."""
    rng = np.random.RandomState(42 + step // max_steps)

    if scene_type == "warehouse_aisle":
        # Goal straight ahead with small lateral offset
        gx = 2.0 + rng.uniform(-0.5, 0.5)
        gy = rng.uniform(-0.8, 0.8)
    elif scene_type == "lab_empty":
        gx = rng.uniform(1.0, 2.5)
        gy = rng.uniform(-1.0, 1.0)
    elif scene_type == "pallet_pickup":
        # Goal near center for pallet approach
        gx = 1.0 + rng.uniform(-0.2, 0.2)
        gy = rng.uniform(-0.3, 0.3)
    elif scene_type == "cluttered_lab":
        gx = 2.0 + rng.uniform(-0.3, 0.3)
        gy = rng.uniform(-0.5, 0.5)
    else:
        gx = 1.5
        gy = 0.0

    return np.array([gx, gy], dtype=np.float32)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def visualize(
    rgb: np.ndarray,
    depth: np.ndarray,
    scan: np.ndarray,
    top_down: np.ndarray | None,
    action: dict,
    safety: dict,
    step: int,
    text: str = "",
) -> np.ndarray:
    """Create a 2x2 display: RGB | Depth colormap; Scan curve | Top-down map."""
    h_rgb, w_rgb = rgb.shape[:2]
    # --- RGB (top-left) ---
    rgb_disp = rgb.copy() if rgb.shape[2] == 3 else cv2.cvtColor(rgb, cv2.COLOR_GRAY2BGR)

    # --- Depth colormap (top-right) ---
    # Normalize depth for display
    depth_valid = depth[np.isfinite(depth) & (depth > 0)]
    if len(depth_valid) > 0:
        d_min, d_max = np.percentile(depth_valid, [5, 95])
    else:
        d_min, d_max = 0.0, 5.0
    d_max = max(d_max, d_min + 0.1)
    depth_norm = np.clip((depth - d_min) / (d_max - d_min), 0, 1)
    depth_norm = np.nan_to_num(depth_norm, nan=0.0).astype(np.float32)
    depth_color = cv2.applyColorMap((depth_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    depth_disp = cv2.resize(depth_color, (w_rgb, h_rgb))

    # --- Scan curve (bottom-left) ---
    scan_h, scan_w = h_rgb, w_rgb
    scan_plot = np.ones((scan_h, scan_w, 3), dtype=np.uint8) * 240
    if scan is not None and len(scan) > 0:
        valid_scan = np.where(np.isfinite(scan), scan, np.nan)
        max_range = np.nanmax(valid_scan) if np.any(np.isfinite(valid_scan)) else 5.0
        max_range = max(max_range, 1.0)

        # Draw grid
        for d in np.arange(1, int(max_range) + 1):
            y = int(scan_h * (1 - d / max_range))
            if 0 <= y < scan_h:
                cv2.line(scan_plot, (0, y), (scan_w, y), (200, 200, 200), 1)

        # Draw scan values as bars
        n = len(scan)
        bar_w = max(1, scan_w // n)
        for i, val in enumerate(scan):
            if np.isfinite(val):
                bar_h = int(scan_h * min(1.0, val / max_range))
                x0 = i * bar_w
                x1 = x0 + bar_w
                y0 = scan_h - bar_h
                color = (0, 180, 0) if val > 1.0 else (0, 100, 255) if val > 0.3 else (0, 0, 255)
                cv2.rectangle(scan_plot, (x0, y0), (x1 - 1, scan_h), color, -1)

        # Draw sector dividers
        third = n // 3
        for div in [third, 2 * third]:
            x = div * bar_w
            cv2.line(scan_plot, (x, 0), (x, scan_h), (100, 100, 100), 1)

        # Label
        cv2.putText(scan_plot, f"Scan (64 bins)", (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # --- Top-down map (bottom-right) ---
    if top_down is None:
        top_down = np.ones((scan_h, scan_w, 3), dtype=np.uint8) * 240
        # Draw robot at center
        cx, cy = scan_w // 2, scan_h // 2
        cv2.circle(top_down, (cx, cy), 8, (0, 0, 255), -1)
        # Draw goal as green star
        gx_pix = cx + int(30 * action.get("goal_x", 0))
        gy_pix = cy - int(30 * action.get("goal_y", 0))
        cv2.circle(top_down, (gx_pix, gy_pix), 6, (0, 255, 0), -1)
        # Draw scan obstacles
        if scan is not None and len(scan) > 0:
            for i, val in enumerate(scan):
                if np.isfinite(val) and val < 3.0:
                    angle = np.deg2rad(-87.0 / 2.0 + i * 87.0 / len(scan))
                    px = cx + int(30 * val * np.sin(angle))
                    py = cy - int(30 * val * np.cos(angle))
                    if 0 <= px < scan_w and 0 <= py < scan_h:
                        cv2.circle(top_down, (px, py), 1, (255, 0, 0), -1)

    # --- Assemble 2x2 grid ---
    row1 = np.hstack([rgb_disp, depth_disp])
    row2 = np.hstack([scan_plot, top_down])
    dashboard = np.vstack([row1, row2])

    # --- Overlay step info ---
    cv2.putText(dashboard, f"Step: {step}", (5, dashboard.shape[0] - 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(dashboard, f"vx: {action.get('x.vel', 0):.2f} m/s",
                (5, dashboard.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(dashboard, f"w: {action.get('theta.vel', 0):.1f} deg/s",
                (5, dashboard.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(dashboard, f"Shield: {safety.get('shield_active', False)} {safety.get('reason', '')}",
                (5, dashboard.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (0, 0, 255) if safety.get("shield_active") else (100, 100, 100), 1)
    if text:
        cv2.putText(dashboard, text, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return dashboard


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@draccus.wrap()
def main(cfg: SimDemoConfig) -> None:
    """Run the sim demo:

    1. Create synthetic scene via RealSenseReader mock mode
    2. Initialize obstacle detector, DWA controller, safety residual
    3. Run DWA + residual controller loop
    4. Display RGB, depth, scan visualization, top-down view
    5. Save video if requested
    """
    # Create output directory
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Initialize mock camera reader for synthetic scene generation
    scene_map = {
        "warehouse_aisle": "corridor",
        "lab_empty": "empty",
        "pallet_pickup": "pallet_marker",
        "cluttered_lab": "cluttered_lab",
    }
    mock_scene = scene_map.get(cfg.scene_type, "corridor")

    reader = MockSyntheticRGBDReader(
        width=480,
        height=640,
        fps=15,
        scene=mock_scene,
    )
    reader.start()

    # Initialize perception
    obstacle_detector = ObstacleDetector(
        scan_dim=64,
        safe_threshold=1.0,
        warning_threshold=0.5,
        danger_threshold=0.2,
    )

    # Initialize controllers
    dwa = DWAController()
    safety_controller = SafetyResidualController()

    # Video writer
    video_writer = None
    if cfg.record_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_path = str(out_dir / f"sim_demo_{cfg.scene_type}.mp4")
        video_writer = cv2.VideoWriter(video_path, fourcc, 15.0, (960, 1280))
        logger.info("Recording video to %s", video_path)

    for episode in range(cfg.num_episodes):
        logger.info("=== Episode %d/%d ===", episode + 1, cfg.num_episodes)

        # Generate a goal for this episode
        goal = generate_goal(cfg.scene_type, episode * cfg.max_steps, cfg.max_steps * cfg.num_episodes)

        # Simple scan simulation from depth image
        prev_scan_raw: np.ndarray | None = None

        for step in range(cfg.max_steps):
            loop_start = time.perf_counter()

            # Read synthetic frame
            frame = reader.read()
            rgb = frame["color"]  # BGR, HxWx3
            depth_m = frame["depth"]  # float32 meters, HxW

            # Convert depth to pseudo-scan (simplified: take min in vertical columns)
            h, w = depth_m.shape
            scan_dim = 64
            col_width = max(1, w // scan_dim)
            scan_m = np.full(scan_dim, np.nan, dtype=np.float32)
            for i in range(scan_dim):
                x0 = i * col_width
                x1 = min(w, x0 + col_width)
                col = depth_m[h // 2 - 20:h // 2 + 20, x0:x1]
                valid = col[np.isfinite(col) & (col > 0.01)]
                if len(valid) > 0:
                    scan_m[i] = float(np.percentile(valid, 10))

            # EMA smoothing
            if prev_scan_raw is not None:
                alpha = 0.5
                smoothed = np.where(
                    np.isnan(scan_m) & ~np.isnan(prev_scan_raw),
                    prev_scan_raw,
                    np.where(~np.isnan(scan_m) & np.isnan(prev_scan_raw),
                             scan_m,
                             alpha * scan_m + (1 - alpha) * prev_scan_raw),
                )
                scan_m = smoothed.astype(np.float32)
            prev_scan_raw = scan_m.copy()

            # Obstacle detection on scan
            obs_result = obstacle_detector.detect(np.nan_to_num(scan_m, nan=5.0))

            # DWA planning
            action = dwa.plan(goal[0], goal[1], scan_m)

            # Safety residual
            safe_action = safety_controller.apply(action, obs_result["sectors"])

            # Store goal info for visualization
            safe_action["goal_x"] = float(goal[0])
            safe_action["goal_y"] = float(goal[1])

            # Visualization
            if cfg.display or video_writer is not None:
                top_down_map = None  # Will be generated in visualize()
                vis = visualize(
                    rgb=rgb,
                    depth=depth_m,
                    scan=scan_m,
                    top_down=top_down_map,
                    action=safe_action,
                    safety={
                        "shield_active": safe_action.get("shield_active", False),
                        "reason": safe_action.get("reason", "normal"),
                    },
                    step=step,
                    text=f"Episode {episode + 1}/{cfg.num_episodes}",
                )

                if cfg.display:
                    cv2.imshow("LeKiwi Sim Demo", vis)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        reader.stop()
                        cv2.destroyAllWindows()
                        return

                if video_writer is not None:
                    video_writer.write(vis)

            # Check if goal reached (within 0.3m)
            dist_to_goal = np.hypot(goal[0], goal[1])
            if dist_to_goal < 0.3:
                logger.info("  Goal reached at step %d", step)
                break

            # Maintain loop rate
            elapsed = time.perf_counter() - loop_start
            target_period = 1.0 / 15.0
            if elapsed < target_period:
                time.sleep(target_period - elapsed)

        logger.info("  Episode finished after %d steps", min(step + 1, cfg.max_steps))

    # Cleanup
    reader.stop()
    if video_writer is not None:
        video_writer.release()
        logger.info("Video saved.")
    if cfg.display:
        cv2.destroyAllWindows()
    logger.info("Sim demo finished. Output in %s", out_dir)


if __name__ == "__main__":
    main()
