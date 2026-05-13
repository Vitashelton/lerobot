"""Offline replay of recorded dataset with visualization."""
from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import dataclass
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
class ReplayConfig:
    """CLI configuration for offline replay."""

    data_dir: str = "data/real_logs"
    fps: int = 15
    loop: bool = False
    export_video: bool = False
    output_video_path: str = "replay_output.mp4"
    display: bool = True
    # Optional JSONL sidecar file with perception annotations
    annotations_file: str = ""


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_npz_frames(data_dir: Path) -> list[dict]:
    """Load frames stored as individual .npz files."""
    npz_files = sorted(data_dir.glob("*.npz"))
    if not npz_files:
        return []
    frames = []
    for fpath in npz_files:
        data = np.load(fpath)
        frame = {
            "color": data.get("color"),
            "depth": data.get("depth"),
            "timestamp": data.get("timestamp", 0.0),
        }
        frames.append(frame)
    logger.info("Loaded %d npz frames from %s", len(frames), data_dir)
    return frames


def load_png_npy_frames(data_dir: Path) -> list[dict]:
    """Load frames stored as color PNG + depth NPY pairs."""
    color_files = sorted(data_dir.glob("*_color.png"))
    frames = []
    for cf in color_files:
        stem = cf.stem.replace("_color", "")
        df = cf.parent / f"{stem}_depth.npy"
        if not df.exists():
            continue
        color = cv2.imread(str(cf))
        depth = np.load(str(df))
        frames.append({"color": color, "depth": depth, "timestamp": 0.0})
    logger.info("Loaded %d png+npy frames from %s", len(frames), data_dir)
    return frames


def load_jsonl_annotations(path: Path) -> list[dict]:
    """Load JSONL annotations file."""
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    logger.info("Loaded %d annotations from %s", len(entries), path)
    return entries


def auto_detect_and_load(data_dir: str) -> list[dict]:
    """Auto-detect format and load frames."""
    dpath = Path(data_dir)
    if not dpath.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # Try NPZ format first
    npz_files = list(dpath.glob("*.npz"))
    if npz_files:
        return load_npz_frames(dpath)

    # Try PNG+NPY format
    png_files = list(dpath.glob("*_color.png"))
    if png_files:
        return load_png_npy_frames(dpath)

    # Try episode subdirectories
    subdirs = [p for p in dpath.iterdir() if p.is_dir()]
    if subdirs:
        # Try NPZ in each subdir
        for sd in subdirs:
            npz_files = list(sd.glob("*.npz"))
            if npz_files:
                return load_npz_frames(sd)

    raise ValueError(f"No recognizable frame data found in {data_dir}")


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def render_frame(
    frame: dict,
    annotations: dict | None = None,
    draw_depth: bool = True,
    draw_scan: bool = True,
) -> np.ndarray:
    """Render a single frame with optional overlays."""
    color = frame.get("color")
    depth = frame.get("depth")

    if color is None:
        color = np.zeros((480, 640, 3), dtype=np.uint8)

    h, w = color.shape[:2]

    # Create a canvas with room for depth and scan panels
    panel_w = w
    panel_h = h

    if draw_depth and depth is not None:
        # Side-by-side: RGB | Depth colormap
        depth_valid = depth[np.isfinite(depth) & (depth > 0)]
        if len(depth_valid) > 0:
            lo, hi = np.percentile(depth_valid, [2, 98])
        else:
            lo, hi = 0.0, 5.0
        hi = max(hi, lo + 0.1)
        depth_norm = np.clip((depth - lo) / (hi - lo), 0, 1)
        depth_norm = np.nan_to_num(depth_norm, nan=0.0).astype(np.float32)
        depth_color = cv2.applyColorMap((depth_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
        depth_color = cv2.resize(depth_color, (panel_w, panel_h))
        display = np.hstack([color, depth_color])
    else:
        display = color.copy()

    # Overlay scan data if available from annotations
    if draw_scan and annotations:
        scan = annotations.get("scan")
        if scan is not None and len(scan) > 0:
            scan_arr = np.array(scan, dtype=np.float32)
            # Draw scan as a small inset bar chart
            scan_h = 120
            scan_w = 300
            scan_inset = np.ones((scan_h, scan_w, 3), dtype=np.uint8) * 240
            n = len(scan_arr)
            bar_w = max(1, scan_w // n)
            max_range = max(np.nanmax(scan_arr) if np.any(np.isfinite(scan_arr)) else 5.0, 1.0)
            for i, val in enumerate(scan_arr):
                if np.isfinite(val):
                    bar_h = int(scan_h * min(1.0, val / max_range))
                    x0 = i * bar_w
                    color_bar = (0, 180, 0) if val > 1.0 else (0, 100, 255) if val > 0.3 else (0, 0, 255)
                    cv2.rectangle(scan_inset, (x0, scan_h - bar_h), (x0 + bar_w - 1, scan_h), color_bar, -1)
            cv2.putText(scan_inset, "Scan", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

            # Overlay inset on display
            y_offset = max(0, display.shape[0] - scan_h - 10)
            x_offset = max(0, display.shape[1] - scan_w - 10)
            display[y_offset:y_offset + scan_h, x_offset:x_offset + scan_w] = scan_inset

    # Overlay annotation text
    if annotations:
        ann = annotations
        y = 20
        for key in ["risk_level", "step", "front_min"]:
            val = ann.get(key)
            if val is not None:
                text = f"{key}: {val}"
                cv2.putText(display, text, (5, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                y += 18

    return display


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@draccus.wrap()
def main(cfg: ReplayConfig) -> None:
    """Load recorded data, replay with visualization."""
    # Load frames
    try:
        frames = auto_detect_and_load(cfg.data_dir)
    except Exception as e:
        logger.error("Failed to load data: %s", e)
        sys.exit(1)

    if not frames:
        logger.error("No frames found in %s", cfg.data_dir)
        sys.exit(1)

    # Load annotations if provided
    annotations: list[dict] = []
    if cfg.annotations_file:
        annot_path = Path(cfg.annotations_file)
        if annot_path.exists():
            annotations = load_jsonl_annotations(annot_path)

    # Video writer
    video_writer = None
    if cfg.export_video:
        sample = frames[0].get("color")
        h, w = (sample.shape[:2]) if sample is not None else (480, 640)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(cfg.output_video_path, fourcc, cfg.fps, (w * 2, h))
        logger.info("Exporting video to %s", cfg.output_video_path)

    period = 1.0 / cfg.fps
    frame_idx = 0
    paused = False
    total_frames = len(frames)

    logger.info("Replaying %d frames at %d FPS. Keys: q=quit, space=pause, left/right=seek",
                total_frames, cfg.fps)

    while True:
        if not paused:
            frame = frames[frame_idx]
            annot = annotations[frame_idx] if frame_idx < len(annotations) else None
            rendered = render_frame(frame, annotations=annot, draw_depth=True, draw_scan=True)

            # Overlay frame counter
            cv2.putText(rendered, f"Frame: {frame_idx + 1}/{total_frames}",
                        (rendered.shape[1] - 220, rendered.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            if paused:
                cv2.putText(rendered, "PAUSED", (10, rendered.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            if cfg.display:
                cv2.imshow("Offline Replay", rendered)

            if video_writer is not None:
                video_writer.write(rendered)

            # Advance frame
            frame_idx += 1
            if frame_idx >= total_frames:
                if cfg.loop:
                    frame_idx = 0
                    logger.info("Looping replay.")
                else:
                    logger.info("Replay finished.")
                    break

        # Handle keyboard input
        key = cv2.waitKey(int(period * 1000)) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            paused = not paused
        elif key == 81:  # left arrow (some systems)
            frame_idx = max(0, frame_idx - 2)
        elif key == 83:  # right arrow
            frame_idx = min(total_frames - 1, frame_idx)
        elif key == ord("p"):
            paused = not paused

    # Cleanup
    if video_writer is not None:
        video_writer.release()
        logger.info("Video exported to %s", cfg.output_video_path)
    if cfg.display:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
