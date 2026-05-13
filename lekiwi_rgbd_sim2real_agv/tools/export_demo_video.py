"""Export demo video from recorded data with overlays."""
from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

import draccus

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")


@dataclass
class ExportVideoConfig:
    """CLI configuration for video export."""

    input_dir: str = "demo_output/real"
    output_path: str = "demo.mp4"
    fps: int = 15
    overlay_detection: bool = True
    overlay_scan: bool = True
    overlay_safety: bool = True
    codec: str = "mp4v"  # mp4v, avc1, XVID
    resolution: str = "auto"  # auto, or WxH like "1280x720"


# ---------------------------------------------------------------------------
# Frame loaders
# ---------------------------------------------------------------------------

def load_frames_npz(data_dir: Path) -> list[dict]:
    """Load frames from .npz files."""
    frames = []
    for npz_path in sorted(data_dir.glob("*.npz")):
        data = np.load(npz_path)
        frame = {
            "color": data.get("color"),
            "depth": data.get("depth"),
            "timestamp": float(data.get("timestamp", 0.0)),
        }
        frames.append(frame)
    return frames


def load_frames_png_npy(data_dir: Path) -> list[dict]:
    """Load frames from PNG color + NPY depth pairs."""
    frames = []
    for color_path in sorted(data_dir.glob("*_color.png")):
        stem = color_path.stem.replace("_color", "")
        depth_path = color_path.parent / f"{stem}_depth.npy"
        color = cv2.imread(str(color_path))
        depth = None
        if depth_path.exists():
            depth = np.load(str(depth_path))
        frames.append({"color": color, "depth": depth, "timestamp": 0.0})
    return frames


def load_annotations(data_dir: Path) -> list[dict]:
    """Load JSONL annotations."""
    annotations = []
    for jsonl_path in sorted(data_dir.glob("*.jsonl")):
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        annotations.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    return annotations


def auto_load_frames(input_dir: str) -> tuple[list[dict], list[dict]]:
    """Auto-detect format and load frames + annotations."""
    dpath = Path(input_dir)
    if not dpath.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    frames = []

    # NPZ
    npz_files = list(dpath.glob("*.npz"))
    if npz_files:
        frames = load_frames_npz(dpath)

    # PNG+NPY
    if not frames:
        png_files = list(dpath.glob("*_color.png"))
        if png_files:
            frames = load_frames_png_npy(dpath)

    # Episode subdirectories
    if not frames:
        for subdir in sorted(dpath.iterdir()):
            if subdir.is_dir():
                sub_frames = load_frames_npz(subdir)
                if not sub_frames:
                    sub_frames = load_frames_png_npy(subdir)
                frames.extend(sub_frames)

    # Annotations
    annotations = load_annotations(dpath)

    return frames, annotations


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_frame_with_overlays(
    frame: dict,
    annotation: dict | None = None,
    overlay_detection: bool = True,
    overlay_scan: bool = True,
    overlay_safety: bool = True,
) -> np.ndarray:
    """Render a single frame with detection, scan, and safety overlays.

    Returns a BGR image suitable for video encoding.
    """
    color = frame.get("color")
    depth = frame.get("depth")

    # Fallback to blank image
    if color is None:
        color = np.zeros((480, 640, 3), dtype=np.uint8)
    if color.shape[2] != 3:
        color = cv2.cvtColor(color, cv2.COLOR_GRAY2BGR)

    h, w = color.shape[:2]
    vis = color.copy()

    # Overlay detection bounding boxes
    if overlay_detection and annotation:
        detections = annotation.get("detections") or annotation.get("obstacles", [])
        for det in detections:
            bbox = det.get("bbox")
            if bbox and len(bbox) == 4:
                x1, y1, x2, y2 = [int(v) for v in bbox]
                # Clamp to image bounds
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw distance label
            dist = det.get("distance")
            bearing = det.get("bearing_deg")
            if dist is not None and bbox:
                label = f"{dist:.2f}m"
                if bearing is not None:
                    label += f" {bearing:.0f}deg"
                cv2.putText(vis, label, (int(bbox[0]), max(10, int(bbox[1]) - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # Draw pallet detections
        pallets = annotation.get("pallets") or []
        for pallet in pallets:
            pos = pallet.get("position_3d")
            center_uv = pallet.get("center_uv")
            if center_uv:
                cx, cy = int(center_uv[0]), int(center_uv[1])
                cv2.circle(vis, (cx, cy), 10, (255, 0, 0), 2)
                label = f"Pallet {pallet.get('id', '?')}"
                if pallet.get("distance_m"):
                    label += f" {pallet['distance_m']:.2f}m"
                cv2.putText(vis, label, (cx - 20, cy - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # Overlay scan visualization (inset bar chart)
    if overlay_scan and annotation:
        scan = annotation.get("scan") or annotation.get("scan64") or annotation.get("scan_tail")
        if scan and isinstance(scan, list) and len(scan) > 0:
            scan_arr = np.array(scan, dtype=np.float32)

            # Inset dimensions
            inset_h = 100
            inset_w = min(300, w - 20)
            inset_x = w - inset_w - 10
            inset_y = h - inset_h - 10

            # Draw scan background
            cv2.rectangle(vis, (inset_x - 1, inset_y - 1),
                          (inset_x + inset_w + 1, inset_y + inset_h + 1),
                          (0, 0, 0), -1)

            n = len(scan_arr)
            bar_w = max(1, inset_w // n)
            max_range = max(np.nanmax(scan_arr) if np.any(np.isfinite(scan_arr)) else 5.0, 1.0)

            for i, val in enumerate(scan_arr):
                if np.isfinite(val):
                    bar_h = int(inset_h * min(1.0, val / max_range))
                    x0 = inset_x + i * bar_w
                    color_bar = (
                        (0, 180, 0) if val > 1.0 else
                        (0, 100, 255) if val > 0.3 else
                        (0, 0, 255)
                    )
                    cv2.rectangle(vis, (x0, inset_y + inset_h - bar_h),
                                  (x0 + bar_w - 1, inset_y + inset_h), color_bar, -1)

            # Sector dividers
            third = n // 3
            for div in [third, 2 * third]:
                x_div = inset_x + div * bar_w
                cv2.line(vis, (x_div, inset_y), (x_div, inset_y + inset_h), (80, 80, 80), 1)

            cv2.putText(vis, f"Scan {scan_arr.shape[0]} bins",
                        (inset_x + 3, inset_y + inset_h - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

    # Overlay safety zone indicators
    if overlay_safety and annotation:
        # Draw safety zone boundaries at bottom of frame
        risk = annotation.get("risk_level") or annotation.get("risk", "unknown")
        shield = annotation.get("shield_active", False)

        y_bar = h - 15
        # Risk indicator bar
        if risk == "danger":
            risk_color = (0, 0, 255)
            risk_text = "DANGER"
        elif risk == "warning":
            risk_color = (0, 165, 255)
            risk_text = "WARNING"
        else:
            risk_color = (0, 200, 0)
            risk_text = "SAFE"

        cv2.putText(vis, risk_text, (10, h - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, risk_color, 2)

        if shield:
            cv2.putText(vis, "SHIELD ACTIVE", (w - 150, h - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Draw front/left/right distances
        front = annotation.get("front_min", float("inf"))
        left = annotation.get("left_min", float("inf"))
        right = annotation.get("right_min", float("inf"))

        status_text = f"F:{front:.2f} L:{left:.2f} R:{right:.2f}"
        cv2.putText(vis, status_text, (w // 2 - 60, h - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    # Overlay step/frame counter
    step = annotation.get("step", 0) if annotation else 0
    cv2.putText(vis, f"Frame: {step}", (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return vis


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@draccus.wrap()
def main(cfg: ExportVideoConfig) -> None:
    """Load recorded frames, render overlays, encode video."""
    # Load frames and annotations
    try:
        frames, annotations = auto_load_frames(cfg.input_dir)
    except Exception as e:
        logger.error("Failed to load input data: %s", e)
        sys.exit(1)

    if not frames:
        logger.error("No frames found in %s", cfg.input_dir)
        sys.exit(1)

    logger.info("Loaded %d frames, %d annotations", len(frames), len(annotations))

    # Determine video resolution
    if cfg.resolution == "auto":
        sample = frames[0].get("color")
        if sample is not None:
            out_h, out_w = sample.shape[:2]
        else:
            out_h, out_w = 480, 640
    else:
        parts = cfg.resolution.split("x")
        out_w, out_h = int(parts[0]), int(parts[1])

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*cfg.codec)
    output_path = Path(cfg.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), fourcc, cfg.fps, (out_w, out_h))

    if not writer.isOpened():
        logger.error("Failed to open video writer for %s (codec=%s)",
                     output_path, cfg.codec)
        sys.exit(1)

    logger.info("Exporting %d frames to %s at %d FPS (%dx%d)...",
                len(frames), output_path, cfg.fps, out_w, out_h)

    for i, frame in enumerate(frames):
        # Get corresponding annotation
        annot = annotations[i] if i < len(annotations) else None

        # Render frame with overlays
        rendered = render_frame_with_overlays(
            frame,
            annotation=annot,
            overlay_detection=cfg.overlay_detection,
            overlay_scan=cfg.overlay_scan,
            overlay_safety=cfg.overlay_safety,
        )

        # Resize if needed
        if rendered.shape[0] != out_h or rendered.shape[1] != out_w:
            rendered = cv2.resize(rendered, (out_w, out_h))

        writer.write(rendered)

        if (i + 1) % 100 == 0:
            logger.info("  Exported %d/%d frames...", i + 1, len(frames))

    writer.release()
    logger.info("Video exported to %s (%d frames)", output_path, len(frames))


if __name__ == "__main__":
    main()
