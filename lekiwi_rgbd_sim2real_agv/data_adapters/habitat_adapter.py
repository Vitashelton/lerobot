"""Habitat / HM3D / RoboTHOR dataset adapter.

Habitat datasets provide RGB-D observations, robot pose, goal position,
and discrete/continuous actions. This adapter converts them to the
unified multimodal format for downstream offline RL training.
"""

from __future__ import annotations

import os
import glob
from typing import Any, Dict, List

import numpy as np

from data_adapters.base_adapter import BaseDatasetAdapter


class HabitatAdapter(BaseDatasetAdapter):
    """Adapter for Habitat-style embodied navigation datasets.

    Expects the dataset to be organized as::

        data_dir/
            episode_000/
                rgb/*.png
                depth/*.png  or  depth/*.npy
                metadata.npz  (contains actions, pose, goal, done, etc.)
            episode_001/
                ...

    Parameters
    ----------
    data_dir : str
        Path to the dataset root directory.
    scan_dim : int
        Number of Scan64 beams.
    image_size : tuple[int, int]
        Target (H, W) for RGB images.
    depth_available : bool
        Whether depth images exist in the dataset.
    fps : int
        Frames per second for computing Scan64 (used for velocity estimation).
    """

    def __init__(
        self,
        data_dir: str,
        scan_dim: int = 64,
        image_size: tuple[int, int] = (224, 224),
        depth_available: bool = True,
        fps: int = 10,
    ) -> None:
        super().__init__(data_dir, scan_dim, image_size)
        self.depth_available = depth_available
        self.fps = fps
        self.dt = 1.0 / fps

    # ------------------------------------------------------------------
    # Episode discovery
    # ------------------------------------------------------------------

    def load_episodes(self) -> List[Dict[str, Any]]:
        """Find and load all episode directories."""
        episode_dirs = sorted(glob.glob(os.path.join(self.data_dir, "episode_*")))
        if not episode_dirs:
            # Try flat structure: rgb/ depth/ metadata.npz in data_dir
            episode_dirs = [self.data_dir]

        episodes = []
        for ep_dir in episode_dirs:
            ep = self._load_single_episode(ep_dir)
            if ep is not None:
                episodes.append(ep)
        return episodes

    def _load_single_episode(self, ep_dir: str) -> Dict[str, Any] | None:
        """Load raw data for one episode."""
        # Load metadata
        meta_path = os.path.join(ep_dir, "metadata.npz")
        if not os.path.exists(meta_path):
            meta_path = os.path.join(ep_dir, "data.npz")
        if not os.path.exists(meta_path):
            print(f"[HabitatAdapter] WARNING: no metadata found in {ep_dir}, skipping")
            return None

        meta = dict(np.load(meta_path, allow_pickle=True))

        # Load RGB frames
        rgb_dir = os.path.join(ep_dir, "rgb")
        rgb_paths = sorted(glob.glob(os.path.join(rgb_dir, "*.png")))
        if not rgb_paths:
            rgb_dir = os.path.join(ep_dir, "color")
            rgb_paths = sorted(glob.glob(os.path.join(rgb_dir, "*.png")))
        if not rgb_paths:
            print(f"[HabitatAdapter] WARNING: no RGB frames in {ep_dir}, skipping")
            return None

        # Load depth frames (optional)
        depth_frames = None
        if self.depth_available:
            depth_dir = os.path.join(ep_dir, "depth")
            depth_paths = sorted(glob.glob(os.path.join(depth_dir, "*.png")))
            if not depth_paths:
                depth_paths = sorted(glob.glob(os.path.join(depth_dir, "*.npy")))

        return {
            "ep_dir": ep_dir,
            "rgb_paths": rgb_paths,
            "depth_paths": depth_paths if depth_paths else [],
            "meta": meta,
        }

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    def convert_episode(self, raw_episode: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a Habitat episode to unified format."""
        import cv2

        rgb_paths = raw_episode["rgb_paths"]
        depth_paths = raw_episode["depth_paths"]
        meta = raw_episode["meta"]
        T = len(rgb_paths)

        # Pre-allocate arrays
        rgb_frames = np.zeros((T, *self.image_size, 3), dtype=np.uint8)
        depth_frames = (
            np.zeros((T, *self.image_size), dtype=np.float32)
            if self.depth_available
            else None
        )
        scan64 = np.full((T, self.scan_dim), np.nan, dtype=np.float32)
        actions = np.zeros((T, 3), dtype=np.float32)
        state = np.zeros((T, 3), dtype=np.float32)
        goal = np.zeros((T, 3), dtype=np.float32)
        rewards = np.zeros(T, dtype=np.float32)
        dones = np.zeros(T, dtype=bool)
        collisions = np.zeros(T, dtype=bool)
        interventions = np.zeros(T, dtype=bool)
        goal_reached = np.zeros(T, dtype=bool)
        robot_positions = np.zeros((T, 2), dtype=np.float32)
        goal_positions = np.zeros((T, 2), dtype=np.float32)
        prev_actions = np.zeros((T, 3), dtype=np.float32)

        # Extract metadata arrays
        meta_actions = meta.get("actions", meta.get("action", None))
        meta_positions = meta.get("positions", meta.get("position", meta.get("poses", None)))
        meta_goal = meta.get("goal", meta.get("goal_position", None))
        meta_dones = meta.get("dones", meta.get("done", None))
        meta_collisions = meta.get("collisions", meta.get("collision", None))

        # If goal is a single vector for the whole episode, broadcast it
        if meta_goal is not None and meta_goal.ndim == 1:
            goal_positions[:] = meta_goal[:2]
            for t in range(T):
                if meta_positions is not None and t < len(meta_positions):
                    dx = meta_goal[0] - meta_positions[t][0]
                    dy = meta_goal[1] - meta_positions[t][1]
                    dtheta = 0.0  # Habitat goal usually only has position
                    goal[t] = [dx, dy, dtheta]

        for t in range(T):
            # RGB
            rgb = cv2.imread(rgb_paths[t])
            if rgb is not None:
                rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
                rgb = self._resize_rgb(rgb, self.image_size)
                rgb_frames[t] = rgb

            # Depth
            if depth_paths and t < len(depth_paths):
                depth_path = depth_paths[t]
                if depth_path.endswith(".npy"):
                    d = np.load(depth_path)
                else:
                    d = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
                    if d is not None and d.dtype == np.uint16:
                        d = d.astype(np.float32) / 1000.0  # mm → m
                if d is not None:
                    d = cv2.resize(d, (self.image_size[1], self.image_size[0]))
                    depth_frames[t] = d
                    # Compute Scan64 from depth
                    scan64[t] = self.compute_scan64_from_depth(d, self.scan_dim)

            # Actions
            if meta_actions is not None and t < len(meta_actions):
                a = meta_actions[t]
                if isinstance(a, (int, np.integer)):
                    # Discrete action → convert to continuous
                    a_cont = self._discrete_to_continuous(a, meta)
                    actions[t] = a_cont
                else:
                    actions[t][: len(a)] = a

            # State (velocity estimation from position differences)
            if meta_positions is not None and t < len(meta_positions):
                robot_positions[t] = meta_positions[t][:2]
                if t > 0:
                    dp = robot_positions[t] - robot_positions[t - 1]
                    state[t, 0] = dp[0] / self.dt  # vx
                    state[t, 1] = dp[1] / self.dt  # vy
                    # omega from heading change
                    if meta_positions[t].shape[0] >= 3:
                        dtheta = meta_positions[t][2] - meta_positions[t - 1][2]
                        state[t, 2] = np.rad2deg(dtheta) / self.dt

            # Previous action
            if t > 0:
                prev_actions[t] = actions[t - 1]

            # Done / collision / success
            if meta_dones is not None and t < len(meta_dones):
                dones[t] = bool(meta_dones[t])
            if meta_collisions is not None and t < len(meta_collisions):
                collisions[t] = bool(meta_collisions[t])

            # Goal reached detection
            if t > 0 and meta_goal is not None:
                dist = np.linalg.norm(robot_positions[t] - goal_positions[t])
                if dist < 0.5:
                    goal_reached[t] = True

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
            "rewards": rewards,  # Will be relabeled later
            "dones": dones,
            "info": {
                "collision": collisions,
                "intervention": interventions,
                "goal_reached": goal_reached,
                "robot_position": robot_positions,
                "goal_position": goal_positions,
            },
        }

    @staticmethod
    def _discrete_to_forward(
        action_idx: int, meta: Dict[str, Any]
    ) -> np.ndarray:
        """Convert discrete Habitat action to continuous [vx, vy, omega].

        Habitat discrete actions:
            0 = STOP, 1 = MOVE_FORWARD, 2 = TURN_LEFT, 3 = TURN_RIGHT
        """
        step_size = meta.get("step_size", 0.25)  # meters per step
        turn_angle = meta.get("turn_angle", 30.0)  # degrees per step
        dt = meta.get("dt", 0.1)

        if action_idx == 0:  # STOP
            return np.array([0.0, 0.0, 0.0], dtype=np.float32)
        elif action_idx == 1:  # MOVE_FORWARD
            return np.array([step_size / dt, 0.0, 0.0], dtype=np.float32)
        elif action_idx == 2:  # TURN_LEFT
            return np.array([0.0, 0.0, turn_angle / dt], dtype=np.float32)
        elif action_idx == 3:  # TURN_RIGHT
            return np.array([0.0, 0.0, -turn_angle / dt], dtype=np.float32)
        else:
            return np.array([0.0, 0.0, 0.0], dtype=np.float32)
