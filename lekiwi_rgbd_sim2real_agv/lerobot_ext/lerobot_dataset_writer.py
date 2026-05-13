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
LeKiwiDatasetWriter -- saves robot episodes in a LeRobot-compatible layout.

Produces a directory tree that mirrors the LeRobotDataset v3.0 format so
that recorded data can later be loaded via ``LeRobotDataset`` or packaged
for HuggingFace Hub upload.

Directory layout for one episode (episode_000000)::

    output_dir/
        meta/
            episodes/
                chunk-000/
                    file-00000.parquet          # episode metadata table
            info.json                            # dataset-level metadata
            stats.json                           # (optional) stats
            tasks.json                           # (optional) tasks
        data/
            chunk-000/
                episode_000000/
                    observation.state.npy        # (N, state_dim) float32
                    action.npy                   # (N, action_dim) float32
                    observation.images.front/
                        img_00000000.jpg         # (H, W, 3) uint8 RGB
                        img_00000001.jpg
                        ...
                    depth/
                        depth_00000000.png       # uint16 sidecar depth
                        depth_00000001.png
                        ...
                episode_000001/
                    ...
        sidecar/
            episode_000000/
                detections.jsonl                 # YOLO / ArUco per-frame
                tracks.jsonl                     # tracking output

State vector layout (observation.state)::

    [x.vel, y.vel, theta.vel,
     scan64[0], ..., scan64[63],
     front_min, left_min, right_min,
     pallet_x, pallet_y, pallet_z, pallet_yaw]

Action vector layout::

    [x.vel, y.vel, theta.vel]

All vectors are float32.  Images are saved as JPEG (RGB) for compactness.
Depth sidecar images are saved as 16-bit unsigned PNG.

Usage::

    from lerobot_ext.lerobot_dataset_writer import LeKiwiDatasetWriter

    writer = LeKiwiDatasetWriter("data/lekiwi_rgbd_agv", fps=15)
    writer.start_episode()
    for step in range(500):
        obs = client.get_observation()
        action = compute_action(obs)
        writer.add_step(obs, action)
    writer.end_episode()
    writer.finalize()
"""

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Number of state dimensions kept inside observation.state
STATE_DIM = 64 + 3 + 3 + 4   # scan64 + velocity + sector_mins + pallet_pose
ACTION_DIM = 3                # x.vel, y.vel, theta.vel


class LeKiwiDatasetWriter:
    """
    Buffers episode data in memory and flushes it to disk on episode end.

    Designed to be called in a real-time loop: ``add_step()`` is cheap
    (list append + numpy array copy for images/state).  The heavy I/O
    happens when ``end_episode()`` is called.
    """

    def __init__(
        self,
        output_dir: str | Path,
        fps: int = 15,
        max_steps_per_episode: int = 2000,
    ):
        """
        Args:
            output_dir: Root directory where the dataset tree will be created.
            fps: Recording frame rate (used for metadata).
            max_steps_per_episode: Safety limit; episodes longer than this
                are auto-truncated with a warning.
        """
        self._root = Path(output_dir)
        self._fps = fps
        self._max_steps = max_steps_per_episode

        # Episode counter (global across all episodes)
        self._episode_idx: int = 0
        self._total_frames: int = 0

        # Per-episode buffers
        self._active: bool = False
        self._episode_start_time: float = 0.0
        self._state_list: list[np.ndarray] = []      # (STATE_DIM,) float32 each
        self._action_list: list[np.ndarray] = []      # (ACTION_DIM,) float32 each
        self._rgb_list: list[np.ndarray] = []         # (H, W, 3) uint8 each
        self._depth_list: list[np.ndarray] = []       # (H, W) uint16 each
        self._timestamps: list[float] = []
        self._detections: list[dict[str, Any]] = []

        # Cached image dimensions (set on first add_step)
        self._img_h: int | None = None
        self._img_w: int | None = None

        # Accumulated episode metadata
        self._episode_meta: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_episode(self) -> None:
        """
        Begin a new episode.

        Raises:
            RuntimeError: If an episode is already active.
        """
        if self._active:
            raise RuntimeError(
                "Episode already active. Call `end_episode()` first."
            )
        self._active = True
        self._episode_start_time = time.perf_counter()
        self._state_list.clear()
        self._action_list.clear()
        self._rgb_list.clear()
        self._depth_list.clear()
        self._timestamps.clear()
        self._detections.clear()

        logger.info("Episode %06d started.", self._episode_idx)

    def add_step(
        self,
        obs: dict[str, Any],
        action: dict[str, Any] | np.ndarray,
        detections: list[dict[str, Any]] | None = None,
        depth: np.ndarray | None = None,
    ) -> None:
        """
        Append one time step to the current episode buffer.

        Args:
            obs: Observation dict from ``LeKiwiD435iClient.get_observation()``.
                 Expected keys: ``front`` (uint8 RGB), ``x.vel``, ``y.vel``,
                 ``theta.vel``, ``scan64``, ``front_min``, ``left_min``,
                 ``right_min``, ``pallet_pose``.
            action: Action dict or (3,) numpy array.
            detections: Optional per-frame detection metadata (YOLO / ArUco).
            depth: Optional raw uint16 depth image (saved as sidecar PNG).
        """
        if not self._active:
            raise RuntimeError("No active episode. Call `start_episode()` first.")

        # Enforce max steps
        if len(self._state_list) >= self._max_steps:
            logger.warning(
                "Episode %06d reached max steps (%d). Auto-ending.",
                self._episode_idx,
                self._max_steps,
            )
            self.end_episode()
            self.start_episode()

        # ---- Build state vector ----
        state = self._build_state_vector(obs)

        # ---- Build action vector ----
        if isinstance(action, np.ndarray):
            act_vec = action.astype(np.float32).ravel()[:ACTION_DIM]
        else:
            act_vec = np.array(
                [
                    float(action.get("x.vel", 0.0)),
                    float(action.get("y.vel", 0.0)),
                    float(action.get("theta.vel", 0.0)),
                ],
                dtype=np.float32,
            )

        # ---- RGB image ----
        rgb = obs.get("front", None)
        if rgb is None:
            # Fill with zeros if missing
            if self._img_h and self._img_w:
                rgb = np.zeros((self._img_h, self._img_w, 3), dtype=np.uint8)
            else:
                rgb = np.zeros((640, 480, 3), dtype=np.uint8)
        if self._img_h is None and rgb.ndim == 3:
            self._img_h, self._img_w = rgb.shape[:2]

        self._state_list.append(state)
        self._action_list.append(act_vec)
        self._rgb_list.append(rgb.copy())

        # ---- Depth sidecar ----
        if depth is not None:
            self._depth_list.append(depth.copy())
        else:
            # Keep aligned with state/action length via sentinel
            self._depth_list.append(np.zeros((1, 1), dtype=np.uint16))

        # ---- Timestamp ----
        self._timestamps.append(time.perf_counter() - self._episode_start_time)

        # ---- Detections ----
        if detections:
            self._detections.append(detections)
        else:
            self._detections.append({})

    def end_episode(self) -> None:
        """
        Flush the current episode buffer to disk and reset.

        Raises:
            RuntimeError: If no episode is active.
        """
        if not self._active:
            raise RuntimeError("No active episode. Call `start_episode()` first.")

        n_steps = len(self._state_list)
        if n_steps == 0:
            logger.warning("Episode %06d has 0 steps. Skipping.", self._episode_idx)
            self._active = False
            self._episode_idx += 1
            return

        ep_idx = self._episode_idx
        ep_dir, img_dir, depth_dir, sidecar_dir = self._episode_dirs(ep_idx)

        # Create directories
        ep_dir.mkdir(parents=True, exist_ok=True)
        img_dir.mkdir(parents=True, exist_ok=True)
        sidecar_dir.mkdir(parents=True, exist_ok=True)

        # ---- Write state array ----
        state_all = np.stack(self._state_list, axis=0).astype(np.float32)  # (N, STATE_DIM)
        np.save(str(ep_dir / "observation.state.npy"), state_all)

        # ---- Write action array ----
        action_all = np.stack(self._action_list, axis=0).astype(np.float32)  # (N, ACTION_DIM)
        np.save(str(ep_dir / "action.npy"), action_all)

        # ---- Write RGB images ----
        for i, rgb in enumerate(self._rgb_list):
            if rgb.shape[-1] == 3:
                # Ensure BGR for cv2.imwrite
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            else:
                bgr = rgb
            fname = img_dir / f"img_{i:08d}.jpg"
            cv2.imwrite(str(fname), bgr)

        # ---- Write depth sidecar PNGs ----
        if any(d.size > 1 for d in self._depth_list):
            depth_dir.mkdir(parents=True, exist_ok=True)
            for i, depth in enumerate(self._depth_list):
                if depth.size <= 1:
                    continue
                fname = depth_dir / f"depth_{i:08d}.png"
                cv2.imwrite(str(fname), depth)

        # ---- Write detections JSONL sidecar ----
        det_path = sidecar_dir / "detections.jsonl"
        with open(det_path, "w") as f:
            for i, det in enumerate(self._detections):
                rec = {"frame_index": i, "timestamp_s": self._timestamps[i],
                       "detections": det}
                f.write(json.dumps(rec, default=_json_default) + "\n")

        # ---- Accumulate episode metadata ----
        duration_s = self._timestamps[-1] if self._timestamps else 0.0
        self._episode_meta.append({
            "episode_index": ep_idx,
            "length": n_steps,
            "duration_s": round(duration_s, 3),
            "fps": self._fps,
            "img_height": self._img_h,
            "img_width": self._img_w,
            "state_dim": STATE_DIM,
            "action_dim": ACTION_DIM,
        })

        self._total_frames += n_steps
        logger.info(
            "Episode %06d ended: %d steps, %.1f s",
            ep_idx, n_steps, duration_s,
        )

        self._active = False
        self._episode_idx += 1

    def finalize(self) -> None:
        """
        Write dataset-level metadata files.

        Should be called once after all episodes have been recorded.
        Does NOT close any active episode -- call ``end_episode()`` first.
        """
        meta_dir = self._root / "meta"
        meta_dir.mkdir(parents=True, exist_ok=True)

        now = datetime.now(timezone.utc).isoformat()

        # ---- info.json ----
        info = {
            "codebase_version": "v3.0",
            "robot_type": "lekiwi_d435i",
            "total_episodes": self._episode_idx,
            "total_frames": self._total_frames,
            "total_tasks": 0,
            "total_chunks": 1,
            "chunks_size": self._total_frames,
            "fps": self._fps,
            "splits": {"train": f"0:{self._episode_idx}"},
            "data_path": "data/chunk-000",
            "video_keys": [],
            "features": {
                "observation.state": {
                    "dtype": "float32",
                    "shape": [STATE_DIM],
                    "names": [
                        "x.vel", "y.vel", "theta.vel",
                        *[f"scan_{i}" for i in range(64)],
                        "front_min", "left_min", "right_min",
                        "pallet_x", "pallet_y", "pallet_z", "pallet_yaw",
                    ],
                },
                "action": {
                    "dtype": "float32",
                    "shape": [ACTION_DIM],
                    "names": ["x.vel", "y.vel", "theta.vel"],
                },
                "observation.images.front": {
                    "dtype": "video",
                    "shape": [self._img_h or 640, self._img_w or 480, 3],
                },
            },
            "info": {
                "creation_timestamp": now,
                "repo_id": "lekiwi_rgbd_agv",
                "version": "1.0.0",
                "description": (
                    "LeKiwi D435i AGV dataset with RGB-D perception, "
                    "64-D depth scan, ArUco pallet detection, and safe "
                    "navigation demonstrations."
                ),
            },
        }
        write_json(meta_dir / "info.json", info)

        # ---- stats.json (placeholder) ----
        stats = {
            "observation.state": {
                "min": None, "max": None, "mean": None, "std": None
            },
            "action": {
                "min": None, "max": None, "mean": None, "std": None
            },
        }
        write_json(meta_dir / "stats.json", stats)

        logger.info(
            "Dataset finalised: %d episodes, %d total frames in %s",
            self._episode_idx,
            self._total_frames,
            self._root,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_state_vector(self, obs: dict[str, Any]) -> np.ndarray:
        """Pack the observation into a flat float32 state vector."""
        # Velocities (3)
        x_vel = float(obs.get("x.vel", 0.0))
        y_vel = float(obs.get("y.vel", 0.0))
        theta_vel = float(obs.get("theta.vel", 0.0))

        # Scan64 (64 floats)
        scan_raw = obs.get("scan64", [])
        if isinstance(scan_raw, np.ndarray):
            scan = scan_raw.astype(np.float32).ravel()
        else:
            scan = np.array(scan_raw, dtype=np.float32).ravel()
        if len(scan) < 64:
            scan = np.pad(scan, (0, 64 - len(scan)), constant_values=5.0)
        else:
            scan = scan[:64]

        # Sector minima (3 floats)
        front_min = float(obs.get("front_min", 5.0))
        left_min = float(obs.get("left_min", 5.0))
        right_min = float(obs.get("right_min", 5.0))

        # Pallet pose (4 floats; pad with NaN if missing)
        pallet = obs.get("pallet_pose", None)
        if pallet and isinstance(pallet, dict):
            pallet_vec = np.array([
                float(pallet.get("x", np.nan)),
                float(pallet.get("y", np.nan)),
                float(pallet.get("z", np.nan)),
                float(pallet.get("yaw", np.nan)),
            ], dtype=np.float32)
        else:
            pallet_vec = np.full(4, np.nan, dtype=np.float32)

        return np.concatenate([
            np.array([x_vel, y_vel, theta_vel], dtype=np.float32),
            scan,
            np.array([front_min, left_min, right_min], dtype=np.float32),
            pallet_vec,
        ])

    def _episode_dirs(self, ep_idx: int) -> tuple[Path, Path, Path, Path]:
        """Return (episode_dir, images_dir, depth_dir, sidecar_dir)."""
        ep_name = f"episode_{ep_idx:06d}"
        chunk = "chunk-000"
        ep_dir = self._root / "data" / chunk / ep_name
        img_dir = ep_dir / "observation.images.front"
        depth_dir = ep_dir / "depth"
        sidecar_dir = self._root / "sidecar" / ep_name
        return ep_dir, img_dir, depth_dir, sidecar_dir


# ===========================================================================
# Utility
# ===========================================================================

def write_json(path: Path, data: Any) -> None:
    """Write a JSON file with consistent formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=_json_default)


def _json_default(obj: Any) -> Any:
    """JSON serialiser fallback for numpy types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable.")
