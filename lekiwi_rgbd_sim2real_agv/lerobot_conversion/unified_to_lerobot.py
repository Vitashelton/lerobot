"""Convert unified format dataset to LeRobot-compatible HuggingFace Dataset.

Writes a LeRobotDataset that can be loaded by LeRobot's training pipeline,
or used standalone with our offline RL algorithms.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm


class UnifiedToLeRobotConverter:
    """Convert unified format → LeRobotDataset (HuggingFace datasets).

    Parameters
    ----------
    output_dir : str
        Directory to save the LeRobot dataset.
    repo_id : str, optional
        HuggingFace Hub repo ID for uploading.
    fps : int
        Frames per second for video encoding metadata.
    chunk_size : int
        Number of frames per video chunk.
    """

    def __init__(
        self,
        output_dir: str,
        repo_id: Optional[str] = None,
        fps: int = 10,
        chunk_size: int = 100,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.repo_id = repo_id
        self.fps = fps
        self.chunk_size = chunk_size
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Main conversion
    # ------------------------------------------------------------------

    def convert(
        self,
        unified_data: Dict[str, Any],
        episode_metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Convert unified data to LeRobot format.

        Parameters
        ----------
        unified_data : dict
            Output of ``BaseDatasetAdapter.convert_all()``.
        episode_metadata : list of dict, optional
            Per-episode metadata (tasks, languages, etc.).

        Returns
        -------
        str
            Path to the saved dataset directory.
        """
        obs = unified_data["observations"]
        actions = unified_data["actions"]
        rewards = unified_data.get("rewards")
        dones = unified_data.get("dones")
        episode_ids = unified_data.get("episode_ids")
        n_frames = len(actions)

        # Build per-frame dicts
        frames: List[Dict[str, Any]] = []
        for t in tqdm(range(n_frames), desc="Converting frames"):
            frame = self._build_frame(t, obs, actions, rewards, dones, episode_ids)
            frames.append(frame)

        # Write dataset to disk
        return self._write_dataset(frames, episode_metadata)

    def _build_frame(
        self,
        t: int,
        obs: Dict[str, np.ndarray],
        actions: np.ndarray,
        rewards: Optional[np.ndarray],
        dones: Optional[np.ndarray],
        episode_ids: Optional[np.ndarray],
    ) -> Dict[str, Any]:
        """Build a single frame dict in LeRobot format."""
        frame: Dict[str, Any] = {}

        # RGB image → (C, H, W) float32 [0, 1]
        if obs.get("rgb") is not None:
            rgb = obs["rgb"][t]
            # (H, W, C) uint8 → (C, H, W) float32
            rgb_t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
            frame["observation.images.rgb"] = rgb_t

        # Depth → (C, H, W) float32
        if obs.get("depth") is not None:
            depth = obs["depth"][t]
            depth_t = torch.from_numpy(depth).unsqueeze(0).float()
            frame["observation.images.depth"] = depth_t

        # Scan64
        if obs.get("scan64") is not None:
            s = obs["scan64"][t]
            scan_t = torch.from_numpy(np.nan_to_num(s, nan=5.0)).float()
            frame["observation.scan64"] = scan_t

        # State
        if obs.get("state") is not None:
            frame["observation.state"] = torch.from_numpy(obs["state"][t]).float()

        # Goal
        if obs.get("goal") is not None:
            frame["observation.goal"] = torch.from_numpy(obs["goal"][t]).float()

        # Previous action
        if obs.get("prev_action") is not None:
            frame["observation.prev_action"] = torch.from_numpy(
                obs["prev_action"][t]
            ).float()

        # Action
        frame["action"] = torch.from_numpy(actions[t]).float()

        # Reward
        if rewards is not None:
            frame["reward"] = torch.tensor([rewards[t]], dtype=torch.float32)
        else:
            frame["reward"] = torch.tensor([0.0], dtype=torch.float32)

        # Done
        if dones is not None:
            frame["done"] = torch.tensor([dones[t]], dtype=torch.bool)
        else:
            frame["done"] = torch.tensor([False], dtype=torch.bool)

        # Episode index
        if episode_ids is not None:
            frame["episode_index"] = torch.tensor(
                [int(episode_ids[t])], dtype=torch.int64
            )
        else:
            frame["episode_index"] = torch.tensor([0], dtype=torch.int64)

        # Frame index
        frame["frame_index"] = torch.tensor([t], dtype=torch.int64)
        frame["timestamp"] = torch.tensor(
            [float(t) / self.fps], dtype=torch.float32
        )
        frame["index"] = torch.tensor([t], dtype=torch.int64)

        return frame

    def _write_dataset(
        self,
        frames: List[Dict[str, Any]],
        episode_metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Write frames to disk in a simple torch-save format.

        Also produces a LeRobot-compatible info.json and stats.json.
        """
        data_path = self.output_dir / "data"
        data_path.mkdir(parents=True, exist_ok=True)

        # Save frames in chunks
        n_frames = len(frames)
        n_chunks = (n_frames + self.chunk_size - 1) // self.chunk_size

        for c in tqdm(range(n_chunks), desc="Writing chunks"):
            start = c * self.chunk_size
            end = min(start + self.chunk_size, n_frames)
            chunk_frames = frames[start:end]

            chunk_dict: Dict[str, Any] = {}
            for key in chunk_frames[0]:
                tensors = [f[key] for f in chunk_frames]
                chunk_dict[key] = torch.stack(tensors)

            torch.save(chunk_dict, data_path / f"chunk-{c:03d}.pt")

        # Compute statistics
        stats = self._compute_stats(frames)

        # Write info.json
        info = {
            "codebase_version": "v1.0",
            "fps": self.fps,
            "total_frames": n_frames,
            "total_episodes": int(
                max(f["episode_index"].item() for f in frames) + 1
            ),
            "total_chunks": n_chunks,
            "chunks_size": self.chunk_size,
            "data_path": str(data_path),
            "features": {
                k: {
                    "shape": list(v.shape),
                    "dtype": str(v.dtype).replace("torch.", ""),
                }
                for k, v in frames[0].items()
            },
        }
        with open(self.output_dir / "info.json", "w") as f:
            json.dump(info, f, indent=2)

        # Write stats.json
        with open(self.output_dir / "stats.json", "w") as f:
            json.dump(stats, f, indent=2)

        # Write meta.json (episode metadata)
        if episode_metadata:
            with open(self.output_dir / "meta.json", "w") as f:
                json.dump(episode_metadata, f, indent=2)

        return str(self.output_dir)

    @staticmethod
    def _compute_stats(frames: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute dataset statistics."""
        n = len(frames)
        stats: Dict[str, Any] = {}

        for key in frames[0]:
            if key.startswith("observation.images"):
                # Image stats across all frames
                all_vals = torch.stack([f[key] for f in frames[: min(1000, n)]])
                stats[key] = {
                    "mean": all_vals.mean(dim=(0, 2, 3)).tolist(),
                    "std": all_vals.std(dim=(0, 2, 3)).tolist(),
                    "min": all_vals.min().item(),
                    "max": all_vals.max().item(),
                }
            elif key == "observation.scan64":
                all_vals = torch.stack([f[key] for f in frames])
                stats[key] = {
                    "mean": all_vals.nanmean(dim=0).tolist(),
                    "std": torch.nan_to_num(all_vals.nanstd(dim=0), nan=0.0).tolist(),
                }
            elif key in ("action", "observation.state", "observation.goal"):
                all_vals = torch.stack([f[key] for f in frames])
                stats[key] = {
                    "mean": all_vals.mean(dim=0).tolist(),
                    "std": all_vals.std(dim=0).tolist(),
                    "min": all_vals.min().item(),
                    "max": all_vals.max().item(),
                }

        return stats

    @staticmethod
    def load_lerobot_dataset(data_dir: str) -> List[Dict[str, torch.Tensor]]:
        """Load a saved LeRobot dataset from disk.

        Parameters
        ----------
        data_dir : str
            Path to the dataset directory.

        Returns
        -------
        list of dict
            All frames as a list of tensordict-like dicts.
        """
        import glob

        data_path = Path(data_dir) / "data"
        chunk_paths = sorted(glob.glob(str(data_path / "chunk-*.pt")))

        all_frames = []
        for cp in chunk_paths:
            chunk = torch.load(cp, weights_only=False)
            n_frames = len(next(iter(chunk.values())))
            for i in range(n_frames):
                frame = {k: v[i] for k, v in chunk.items()}
                all_frames.append(frame)

        return all_frames
