"""
Script to generate a full synthetic RGB-D dataset.

Usage:
    python -m sim.synthetic_rgbd.render_dataset \\
        --output_dir data/synthetic \\
        --num_scenes 1000 \\
        --seed 42

Output structure::

    output_dir/
        scene_000000/
            rgb.png
            depth_m.npy
            depth_noisy_m.npy
            labels.npy
            annotation.json
        scene_000001/
            ...
        dataset_info.json
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np
from draccus import wrap

from .scene_generator import SceneGenerator


@dataclass
class SyntheticDatasetConfig:
    """Configuration for synthetic dataset generation."""

    output_dir: str = "data/synthetic"
    """Root output directory."""

    num_scenes: int = 1000
    """Total number of scenes to generate."""

    width: int = 480
    """Image width in pixels (portrait orientation for AGV)."""

    height: int = 640
    """Image height in pixels."""

    seed: int = 42
    """Base random seed (incremented per scene)."""

    scene_types: List[str] = field(default_factory=lambda: [
        "warehouse_aisle",
        "lab_cluttered",
        "pallet_pickup",
    ])
    """Scene types to generate, sampled uniformly."""

    camera_height_range: Tuple[float, float] = (0.3, 0.6)
    """Min / max camera height (metres)."""

    camera_pitch_range: Tuple[float, float] = (-5.0, 10.0)
    """Min / max camera pitch in degrees (positive = nose-down)."""

    depth_noise_std_mm: float = 5.0
    """Gaussian depth noise standard deviation (mm)."""

    dropout_prob: float = 0.02
    """Probability a depth pixel is dropped to 0."""

    fov_h: float = 87.0
    """Horizontal field of view in degrees (D435i depth FOV)."""


def _serialize_annotation(annotation: dict) -> dict:
    """Convert numpy arrays / non-JSON types to JSON-safe values."""
    out = {}
    for k, v in annotation.items():
        if k == "free_space":
            # Free-space mask is large; we store shape + stats instead of the full matrix.
            mask = np.asarray(v)
            out["free_space_shape"] = list(mask.shape)
            out["free_space_fraction"] = float(mask.mean())
        elif isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, list):
            out[k] = v  # already serializable
        else:
            out[k] = v
    return out


@wrap
def main(cfg: SyntheticDatasetConfig):
    """Generate the synthetic dataset.

    This function is the entry point when using ``draccus.wrap``.
    It can also be called directly from Python code.
    """
    os.makedirs(cfg.output_dir, exist_ok=True)

    # Resolve mean camera params from range
    cam_height_mid = (cfg.camera_height_range[0] + cfg.camera_height_range[1]) / 2
    cam_pitch_mid = (cfg.camera_pitch_range[0] + cfg.camera_pitch_range[1]) / 2

    generator = SceneGenerator(
        width=cfg.width,
        height=cfg.height,
        fov_h=cfg.fov_h,
        camera_height_m=cam_height_mid,
        camera_pitch_deg=cam_pitch_mid,
        depth_noise_std_mm=cfg.depth_noise_std_mm,
        dropout_prob=cfg.dropout_prob,
    )

    scene_types = cfg.scene_types
    counts: dict = {t: 0 for t in scene_types}

    print(f"Generating {cfg.num_scenes} scenes → {cfg.output_dir}")
    t_start = time.perf_counter()

    for idx in range(cfg.num_scenes):
        scene_seed = cfg.seed + idx
        stype = scene_types[idx % len(scene_types)]

        # Additionally randomise camera within configured ranges
        height = cfg.camera_height_range[0] + np.random.uniform() * (
            cfg.camera_height_range[1] - cfg.camera_height_range[0])
        pitch = cfg.camera_pitch_range[0] + np.random.uniform() * (
            cfg.camera_pitch_range[1] - cfg.camera_pitch_range[0])
        generator.base_camera_height = height
        generator.base_camera_pitch_deg = pitch

        scene = generator.generate_scene(scene_type=stype, seed=scene_seed)

        # ---- Save ----
        scene_dir = os.path.join(cfg.output_dir, f"scene_{idx:06d}")
        os.makedirs(scene_dir, exist_ok=True)

        # RGB
        rgb_bgr = cv2.cvtColor(scene["rgb"], cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(scene_dir, "rgb.png"), rgb_bgr)

        # Depth (clean)
        np.save(os.path.join(scene_dir, "depth_m.npy"), scene["depth"])

        # Depth (noisy)
        np.save(os.path.join(scene_dir, "depth_noisy_m.npy"), scene["depth_noisy"])

        # Labels
        np.save(os.path.join(scene_dir, "labels.npy"), scene["labels"])

        # Annotation JSON
        ann = _serialize_annotation({
            "objects": scene["objects"],
            "free_space": scene["free_space"],
            "scene_type": scene["scene_type"],
            "camera_pose": scene["camera_pose"],
            "seed": scene_seed,
        })
        with open(os.path.join(scene_dir, "annotation.json"), "w") as f:
            json.dump(ann, f, indent=2)

        counts[stype] += 1

        if (idx + 1) % 100 == 0:
            elapsed = time.perf_counter() - t_start
            rate = (idx + 1) / elapsed
            eta = (cfg.num_scenes - (idx + 1)) / rate
            print(f"  [{idx + 1:6d}/{cfg.num_scenes}]  {rate:.1f} scenes/s  "
                  f"ETA {eta:.0f}s  counts={counts}")

    # ---- Dataset info ----
    total_elapsed = time.perf_counter() - t_start
    info = {
        "num_scenes": cfg.num_scenes,
        "width": cfg.width,
        "height": cfg.height,
        "fov_h": cfg.fov_h,
        "camera_height_range": list(cfg.camera_height_range),
        "camera_pitch_range": list(cfg.camera_pitch_range),
        "depth_noise_std_mm": cfg.depth_noise_std_mm,
        "dropout_prob": cfg.dropout_prob,
        "scene_type_counts": counts,
        "total_time_s": round(total_elapsed, 1),
        "scenes_per_second": round(cfg.num_scenes / total_elapsed, 1),
        "seed": cfg.seed,
    }
    with open(os.path.join(cfg.output_dir, "dataset_info.json"), "w") as f:
        json.dump(info, f, indent=2)

    print(f"\nDone.  {cfg.num_scenes} scenes in {total_elapsed:.1f}s  "
          f"({cfg.num_scenes / total_elapsed:.1f} scenes/s)")
    print(f"Output: {cfg.output_dir}")


if __name__ == "__main__":
    main()
