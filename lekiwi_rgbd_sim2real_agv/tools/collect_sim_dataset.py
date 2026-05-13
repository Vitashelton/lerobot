"""Collect synthetic training dataset using LeKiwiScanEnv."""
from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

import draccus

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from camera.realsense_reader import RealSenseReader
from perception.obstacle_detector import ObstacleDetector

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")


@dataclass
class CollectSimConfig:
    """CLI configuration for synthetic dataset collection."""

    output_dir: str = "data/sim_logs"
    num_episodes: int = 100
    max_steps: int = 500
    scene_types: list[str] = field(default_factory=lambda: ["warehouse_aisle", "cluttered_lab"])
    fps: int = 15
    save_depth: bool = True
    save_detections: bool = True
    display: bool = False


# ---------------------------------------------------------------------------
# Scene-to-mock mapping
# ---------------------------------------------------------------------------

SCENE_MAP = {
    "warehouse_aisle": "corridor",
    "cluttered_lab": "cluttered_lab",
    "lab_empty": "empty",
    "pallet_pickup": "pallet_marker",
}


# ---------------------------------------------------------------------------
# Data recorder
# ---------------------------------------------------------------------------

class SimEpisodeRecorder:
    """Record synthetic episode data."""

    def __init__(self, base_dir: Path, episode_idx: int, scene_type: str):
        self.base_dir = base_dir
        self.scene_type = scene_type
        self.episode_dir = base_dir / scene_type / f"episode_{episode_idx:05d}"
        self.episode_dir.mkdir(parents=True, exist_ok=True)
        self.rgb_dir = self.episode_dir / "rgb"
        self.depth_dir = self.episode_dir / "depth"
        self.rgb_dir.mkdir(exist_ok=True)
        self.depth_dir.mkdir(exist_ok=True)
        self.observations: list[dict] = []
        self.detections: list[dict] = []

    def add_frame(
        self,
        rgb: np.ndarray,
        depth_m: np.ndarray,
        scan: np.ndarray,
        action: dict,
        detections: dict,
        step: int,
    ) -> None:
        """Record one frame."""
        # Save RGB as PNG
        rgb_path = self.rgb_dir / f"frame_{step:06d}.png"
        cv2.imwrite(str(rgb_path), rgb)

        # Save depth as NPY
        depth_path = self.depth_dir / f"frame_{step:06d}.npy"
        np.save(str(depth_path), depth_m)

        # Observation record
        obs = {
            "step": step,
            "timestamp": time.time(),
            "scene_type": self.scene_type,
            "action": action,
            "scan_tail": np.nan_to_num(scan, nan=5.0).tolist(),
            "front_min": float(np.nanmin(scan[21:43])) if len(scan) >= 43 else float("inf"),
            "left_min": float(np.nanmin(scan[:21])) if len(scan) >= 21 else float("inf"),
            "right_min": float(np.nanmin(scan[43:])) if len(scan) >= 43 else float("inf"),
            "rgb_path": str(rgb_path.relative_to(self.episode_dir)),
            "depth_path": str(depth_path.relative_to(self.episode_dir)),
        }
        if detections:
            obs["detections"] = detections
        self.observations.append(obs)

    def save(self) -> Path:
        """Write all episode data."""
        obs_path = self.episode_dir / "observations.jsonl"
        with open(obs_path, "w") as f:
            for o in self.observations:
                f.write(json.dumps(o) + "\n")

        meta = {
            "num_frames": len(self.observations),
            "scene_type": self.scene_type,
            "episode_dir": str(self.episode_dir),
        }
        with open(self.episode_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        logger.info("  Saved %d frames to %s", len(self.observations), self.episode_dir)
        return self.episode_dir


# ---------------------------------------------------------------------------
# DWA controller (same as in run_sim_demo.py)
# ---------------------------------------------------------------------------

class DWAController:
    """Minimal DWA controller for synthetic navigation."""

    def __init__(
        self,
        max_linear: float = 0.3,
        max_angular: float = 90.0,
        linear_samples: int = 11,
        angular_samples: int = 21,
        dt: float = 0.1,
        predict_steps: int = 15,
    ):
        self.max_linear = max_linear
        self.max_angular = max_angular
        self.linear_samples = linear_samples
        self.angular_samples = angular_samples
        self.dt = dt
        self.predict_steps = predict_steps

    def plan(self, goal_x: float, goal_y: float, scan_m: np.ndarray) -> dict:
        """Return best (vx, omega_deg) action."""
        lin_vels = np.linspace(-self.max_linear, self.max_linear, self.linear_samples)
        ang_vels = np.linspace(-self.max_angular, self.max_angular, self.angular_samples)

        best_score = -np.inf
        best_action = (0.0, 0.0)

        for v in lin_vels:
            for w in ang_vels:
                w_rad = np.deg2rad(w)
                x, y, theta = 0.0, 0.0, 0.0
                min_clearance = float("inf")

                for _ in range(self.predict_steps):
                    x += v * np.cos(theta) * self.dt
                    y += v * np.sin(theta) * self.dt
                    theta += w_rad * self.dt
                    dist = np.hypot(x, y)
                    min_clearance = min(min_clearance, dist)

                if min_clearance < 0.1:
                    continue  # collides

                # Heading toward goal
                goal_bearing = np.arctan2(goal_y, goal_x)
                heading_error = abs(theta - goal_bearing)
                heading_error = min(heading_error, 2 * np.pi - heading_error)
                heading_score = max(0.0, 1.0 - heading_error / np.pi)

                # Speed score
                speed_score = abs(v) / self.max_linear if self.max_linear > 0 else 0

                score = 0.5 * heading_score + 0.3 * min(1.0, min_clearance / 0.5) + 0.2 * speed_score
                if score > best_score:
                    best_score = score
                    best_action = (v, w)

        return {"x.vel": best_action[0], "theta.vel": best_action[1]}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@draccus.wrap()
def main(cfg: CollectSimConfig) -> None:
    """Run LeKiwiScanEnv with DWA policy, record trajectories."""
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_episodes: list[Path] = []
    episode_idx = 0

    for scene_type in cfg.scene_types:
        mock_scene = SCENE_MAP.get(scene_type, "corridor")
        logger.info("=== Scene: %s (mock: %s) ===", scene_type, mock_scene)

        # Per-scene episode count
        n_ep = cfg.num_episodes // len(cfg.scene_types)
        if scene_type == cfg.scene_types[-1]:
            n_ep = cfg.num_episodes - episode_idx  # remainder

        reader = RealSenseReader(
            mock=True,
            width=480,
            height=640,
            fps=cfg.fps,
            scene=mock_scene,
        )
        reader.start()

        dwa = DWAController()
        obstacle_detector = ObstacleDetector()

        for ep in range(n_ep):
            logger.info("  Episode %d/%d", ep + 1, n_ep)
            recorder = SimEpisodeRecorder(out_dir, episode_idx, scene_type)

            # Random goal
            goal = np.array([
                np.random.uniform(1.0, 2.5),
                np.random.uniform(-1.0, 1.0),
            ], dtype=np.float32)

            prev_scan: np.ndarray | None = None

            for step in range(cfg.max_steps):
                frame = reader.read()
                rgb = frame["color"]
                depth_m = frame["depth"]

                # Convert depth to scan
                h, w = depth_m.shape
                scan_dim = 64
                col_width = max(1, w // scan_dim)
                scan_m = np.full(scan_dim, np.nan, dtype=np.float32)
                band_start, band_end = h // 2 - 20, h // 2 + 20
                for i in range(scan_dim):
                    x0 = i * col_width
                    x1 = min(w, x0 + col_width)
                    col = depth_m[band_start:band_end, x0:x1]
                    valid = col[np.isfinite(col) & (col > 0.01)]
                    if len(valid) > 0:
                        scan_m[i] = float(np.percentile(valid, 10))

                # EMA
                if prev_scan is not None:
                    scan_m = np.where(
                        np.isnan(scan_m), prev_scan,
                        0.5 * scan_m + 0.5 * prev_scan
                    ).astype(np.float32)
                prev_scan = scan_m.copy()

                # DWA planning
                action = dwa.plan(goal[0], goal[1], scan_m)

                # Detect obstacles
                detections = obstacle_detector.detect(np.nan_to_num(scan_m, nan=5.0))

                # Save frame
                recorder.add_frame(
                    rgb=rgb,
                    depth_m=depth_m,
                    scan=scan_m,
                    action=action,
                    detections=detections,
                    step=step,
                )

                # Display
                if cfg.display and step % 5 == 0:
                    cv2.imshow("Sim Collection", rgb)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        reader.stop()
                        cv2.destroyAllWindows()
                        return

                # Check goal reached
                if np.hypot(goal[0], goal[1]) < 0.3:
                    break

            recorder.save()
            all_episodes.append(recorder.episode_dir)
            episode_idx += 1

        reader.stop()

    # Dataset summary
    summary = {
        "num_episodes": len(all_episodes),
        "scene_types": cfg.scene_types,
        "episodes": [str(p) for p in all_episodes],
        "config": {
            "max_steps": cfg.max_steps,
            "save_depth": cfg.save_depth,
        },
    }
    with open(out_dir / "dataset_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    if cfg.display:
        cv2.destroyAllWindows()
    logger.info("Sim dataset collection finished. %d episodes saved to %s",
                len(all_episodes), out_dir)


if __name__ == "__main__":
    main()
