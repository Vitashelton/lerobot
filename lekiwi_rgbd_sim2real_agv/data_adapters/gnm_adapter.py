"""GNM / ViNT / NoMaD dataset adapter.

These datasets typically provide:
    - RGB images from forward-facing camera
    - Actions (linear + angular velocity)
    - GPS or relative position
    - May or may not have depth

If depth is not available, Scan64 is populated with NaN and
the model falls back to RGB-only encoding.
"""

from __future__ import annotations

import os
import glob
from typing import Any, Dict, List

import numpy as np

from data_adapters.base_adapter import BaseDatasetAdapter


class GNMAdapter(BaseDatasetAdapter):
    """Adapter for GNM/ViNT-style trajectory datasets.

    Expects data in one of these formats:

    1. Single HDF5 file:
       data_dir/trajectories.h5

    2. Directory of npz files:
       data_dir/ep_000.npz, ep_001.npz, ...

    3. Directory with subdirectories:
       data_dir/ep_000/rgb/*.jpg + data.npz

    Parameters
    ----------
    data_dir : str
        Path to raw dataset.
    scan_dim : int
        Number of Scan64 beams.
    image_size : tuple[int, int]
        Target (H, W) for RGB.
    file_format : str
        One of "h5", "npz", "dirs".
    depth_available : bool
        Whether depth images exist.
    """

    def __init__(
        self,
        data_dir: str,
        scan_dim: int = 64,
        image_size: tuple[int, int] = (224, 224),
        file_format: str = "h5",
        depth_available: bool = False,
    ) -> None:
        super().__init__(data_dir, scan_dim, image_size)
        self.file_format = file_format
        self.depth_available = depth_available

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_episodes(self) -> List[Dict[str, Any]]:
        if self.file_format == "h5":
            return self._load_h5()
        elif self.file_format == "npz":
            return self._load_npz()
        elif self.file_format == "dirs":
            return self._load_dirs()
        else:
            raise ValueError(f"Unknown file_format: {self.file_format}")

    def _load_h5(self) -> List[Dict[str, Any]]:
        """Load episodes from a single HDF5 file."""
        import h5py

        h5_path = os.path.join(self.data_dir, "trajectories.h5")
        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"H5 file not found: {h5_path}")

        with h5py.File(h5_path, "r") as f:
            episodes = []
            for key in sorted(f.keys()):
                grp = f[key]
                episodes.append({
                    "ep_id": key,
                    "rgb": np.array(grp["rgb"]),
                    "actions": np.array(grp["actions"]),
                    "positions": np.array(grp.get("positions", grp.get("position", None))),
                    "goal": np.array(grp.get("goal", grp.get("goal_position", None))),
                    "dones": np.array(grp.get("dones", np.zeros(len(grp["actions"]), dtype=bool))),
                })
            return episodes

    def _load_npz(self) -> List[Dict[str, Any]]:
        """Load episodes from individual .npz files."""
        npz_paths = sorted(glob.glob(os.path.join(self.data_dir, "ep_*.npz")))
        episodes = []
        for p in npz_paths:
            data = dict(np.load(p, allow_pickle=True))
            episodes.append({
                "ep_id": os.path.basename(p),
                "rgb": data["rgb"],
                "actions": data["actions"],
                "positions": data.get("positions", data.get("position", None)),
                "goal": data.get("goal", data.get("goal_position", None)),
                "dones": data.get("dones", np.zeros(len(data["actions"]), dtype=bool)),
            })
        return episodes

    def _load_dirs(self) -> List[Dict[str, Any]]:
        """Load episodes from subdirectory structure."""
        import cv2

        ep_dirs = sorted(glob.glob(os.path.join(self.data_dir, "ep_*")))
        episodes = []
        for ep_dir in ep_dirs:
            # Load metadata
            meta = dict(np.load(os.path.join(ep_dir, "data.npz"), allow_pickle=True))
            # Load RGB frames
            rgb_paths = sorted(glob.glob(os.path.join(ep_dir, "rgb", "*.jpg")))
            if not rgb_paths:
                rgb_paths = sorted(glob.glob(os.path.join(ep_dir, "rgb", "*.png")))

            rgb_frames = []
            for rp in rgb_paths:
                img = cv2.imread(rp)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = self._resize_rgb(img, self.image_size)
                    rgb_frames.append(img)

            episodes.append({
                "ep_id": os.path.basename(ep_dir),
                "rgb": np.array(rgb_frames),
                "actions": meta["actions"],
                "positions": meta.get("positions", meta.get("position", None)),
                "goal": meta.get("goal", meta.get("goal_position", None)),
                "dones": meta.get("dones", np.zeros(len(meta["actions"]), dtype=bool)),
            })
        return episodes

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    def convert_episode(self, raw_episode: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a GNM episode to unified format."""
        rgb_data = raw_episode["rgb"]
        actions_data = raw_episode["actions"]
        positions_data = raw_episode.get("positions")
        goal_data = raw_episode.get("goal")
        dones_data = raw_episode.get("dones")

        T = len(actions_data)

        # RGB
        rgb_frames = np.zeros((T, *self.image_size, 3), dtype=np.uint8)
        for t in range(min(T, len(rgb_data))):
            rgb = rgb_data[t]
            if rgb.shape[:2] != self.image_size:
                rgb = self._resize_rgb(rgb, self.image_size)
            rgb_frames[t] = rgb

        # No depth → Scan64 is NaN
        depth_frames = None
        scan64 = np.full((T, self.scan_dim), np.nan, dtype=np.float32)

        # Actions: ensure shape (T, 3)
        actions = np.zeros((T, 3), dtype=np.float32)
        a_data = np.asarray(actions_data, dtype=np.float32)
        if a_data.ndim == 1:
            # [v, omega] → [v, 0, omega]
            for t in range(min(T, len(a_data))):
                if len(a_data) >= T:
                    pass
            if a_data.shape[0] == 2 * T:
                a_data = a_data.reshape(T, 2)
        if a_data.shape[-1] == 2:
            actions[:, 0] = a_data[:, 0]
            actions[:, 2] = a_data[:, 1]
        elif a_data.shape[-1] >= 3:
            actions[:, :3] = a_data[:, :3]

        # State from positions
        state = np.zeros((T, 3), dtype=np.float32)
        robot_positions = np.zeros((T, 2), dtype=np.float32)
        if positions_data is not None:
            pos = np.asarray(positions_data, dtype=np.float32)
            robot_positions[: len(pos)] = pos[:, :2]
            for t in range(1, min(T, len(pos))):
                dp = robot_positions[t] - robot_positions[t - 1]
                state[t, 0] = dp[0] / 0.1  # approximate vx
                state[t, 1] = dp[1] / 0.1  # approximate vy
                if pos.shape[-1] >= 3:
                    dtheta = pos[t, 2] - pos[t - 1, 2]
                    state[t, 2] = np.rad2deg(dtheta) / 0.1

        # Goal: broadcast per-episode goal to each timestep
        goal = np.zeros((T, 3), dtype=np.float32)
        goal_positions = np.zeros((T, 2), dtype=np.float32)
        if goal_data is not None:
            g = np.asarray(goal_data, dtype=np.float32).flatten()
            goal_positions[:] = g[:2]
            for t in range(T):
                dx = g[0] - robot_positions[t, 0]
                dy = g[1] - robot_positions[t, 1]
                goal[t] = [dx, dy, 0.0]

        # Dones, collisions
        dones = np.zeros(T, dtype=bool)
        if dones_data is not None:
            dones[: len(dones_data)] = np.asarray(dones_data, dtype=bool)
        collisions = np.zeros(T, dtype=bool)
        interventions = np.zeros(T, dtype=bool)
        goal_reached = np.zeros(T, dtype=bool)

        # Detect goal reached from position proximity
        if positions_data is not None and goal_data is not None:
            for t in range(T):
                if np.linalg.norm(robot_positions[t] - goal_positions[t]) < 0.5:
                    goal_reached[t] = True

        # Previous action
        prev_actions = np.zeros((T, 3), dtype=np.float32)
        prev_actions[1:] = actions[:-1]

        return {
            "observations": {
                "rgb": rgb_frames,
                "depth": depth_frames,
                "scan64": scan64,
                "state": state,
                "goal": goal,
                "prev_action": prev_actions,
            },
            "actions": actions,
            "rewards": np.zeros(T, dtype=np.float32),
            "dones": dones,
            "info": {
                "collision": collisions,
                "intervention": interventions,
                "goal_reached": goal_reached,
                "robot_position": robot_positions,
                "goal_position": goal_positions,
            },
        }
