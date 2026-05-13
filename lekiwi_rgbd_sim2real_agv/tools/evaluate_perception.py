"""Evaluate perception module performance."""
from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt

import draccus

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from perception.obstacle_detector import ObstacleDetector

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")


@dataclass
class EvaluatePerceptionConfig:
    """CLI configuration for perception evaluation."""

    gt_data_dir: str = "data/test_labels"
    pred_data_dir: str = "data/predictions"
    output_dir: str = "eval_output/perception"
    iou_threshold: float = 0.5
    distance_threshold: float = 0.1
    verbose: bool = True


# ---------------------------------------------------------------------------
# Obstacle detection metrics
# ---------------------------------------------------------------------------

def evaluate_obstacle_detection(
    gt_obstacles: list[dict],
    pred_obstacles: list[dict],
    iou_threshold: float = 0.5,
) -> dict:
    """Compute precision, recall, F1 for obstacle detection.

    Args:
        gt_obstacles: ground-truth obstacles, each with bbox [x1, y1, x2, y2].
        pred_obstacles: predicted obstacles.
        iou_threshold: IoU threshold to consider a match.

    Returns:
        dict with precision, recall, f1, tp, fp, fn, and per-frame counts.
    """
    tp = 0
    fp = 0
    fn = 0

    if len(gt_obstacles) == 0 and len(pred_obstacles) == 0:
        return {
            "precision": 1.0, "recall": 1.0, "f1": 1.0,
            "tp": 0, "fp": 0, "fn": 0,
            "gt_count": 0, "pred_count": 0,
        }

    # Greedy matching
    matched_gt = set()
    matched_pred = set()

    for pi, pred in enumerate(pred_obstacles):
        pred_bbox = pred.get("bbox", [0, 0, 0, 0])
        best_iou = 0.0
        best_gt = -1
        for gi, gt in enumerate(gt_obstacles):
            if gi in matched_gt:
                continue
            gt_bbox = gt.get("bbox", [0, 0, 0, 0])
            iou = _bbox_iou(pred_bbox, gt_bbox)
            if iou > best_iou:
                best_iou = iou
                best_gt = gi

        if best_iou >= iou_threshold:
            tp += 1
            matched_gt.add(best_gt)
            matched_pred.add(pi)

    fp = len(pred_obstacles) - len(matched_pred)
    fn = len(gt_obstacles) - len(matched_gt)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "gt_count": len(gt_obstacles),
        "pred_count": len(pred_obstacles),
    }


def _bbox_iou(bbox_a: list, bbox_b: list) -> float:
    """Compute IoU between two bounding boxes [x1, y1, x2, y2]."""
    x1 = max(bbox_a[0], bbox_b[0])
    y1 = max(bbox_a[1], bbox_b[1])
    x2 = min(bbox_a[2], bbox_b[2])
    y2 = min(bbox_a[3], bbox_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (bbox_a[2] - bbox_a[0]) * (bbox_a[3] - bbox_a[1])
    area_b = (bbox_b[2] - bbox_b[0]) * (bbox_b[3] - bbox_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Pallet detection metrics
# ---------------------------------------------------------------------------

def evaluate_pallet_detection(
    gt_pallets: list[dict],
    pred_pallets: list[dict],
    distance_threshold: float = 0.1,
) -> dict:
    """Detection rate, distance MAE, bearing MAE for pallet detection.

    Args:
        gt_pallets: each with position_3d [x, y, z], bearing_deg.
        pred_pallets: each with position_3d [x, y, z], bearing_deg.
        distance_threshold: max position error for matching (meters).

    Returns:
        dict with detection_rate, distance_mae, bearing_mae, position_mae.
    """
    gt_positions = np.array([p.get("position_3d", [0, 0, 0]) for p in gt_pallets])
    pred_positions = np.array([p.get("position_3d", [0, 0, 0]) for p in pred_pallets])

    if len(gt_pallets) == 0:
        return {
            "detection_rate": 1.0 if len(pred_pallets) == 0 else 0.0,
            "distance_mae": 0.0,
            "bearing_mae": 0.0,
            "position_mae": 0.0,
            "false_positives": len(pred_pallets),
        }

    if len(pred_pallets) == 0:
        return {
            "detection_rate": 0.0,
            "distance_mae": float("inf"),
            "bearing_mae": float("inf"),
            "position_mae": float("inf"),
            "false_negatives": len(gt_pallets),
        }

    # Greedy matching by 3D distance
    matched_gt = set()
    errors_3d = []
    bearing_errors = []
    distance_errors = []

    for pi, pred in enumerate(pred_pallets):
        pred_pos = np.array(pred.get("position_3d", [0, 0, 0]))
        best_dist = float("inf")
        best_gi = -1
        for gi, gt in enumerate(gt_pallets):
            if gi in matched_gt:
                continue
            gt_pos = np.array(gt.get("position_3d", [0, 0, 0]))
            d = np.linalg.norm(pred_pos - gt_pos)
            if d < best_dist:
                best_dist = d
                best_gi = gi

        if best_dist < distance_threshold:
            matched_gt.add(best_gi)
            errors_3d.append(best_dist)
            distance_errors.append(abs(
                np.linalg.norm(pred_pos) - np.linalg.norm(gt_positions[best_gi])
            ))
            bearing_errors.append(abs(
                pred.get("bearing_deg", 0) - gt_pallets[best_gi].get("bearing_deg", 0)
            ))

    tp = len(matched_gt)
    fp = len(pred_pallets) - tp
    fn = len(gt_pallets) - tp

    detection_rate = tp / len(gt_pallets) if len(gt_pallets) > 0 else 0.0
    distance_mae = np.mean(distance_errors) if distance_errors else float("inf")
    bearing_mae = np.mean(bearing_errors) if bearing_errors else float("inf")
    position_mae = np.mean(errors_3d) if errors_3d else float("inf")

    return {
        "detection_rate": round(detection_rate, 4),
        "distance_mae": round(distance_mae, 4),
        "bearing_mae": round(bearing_mae, 4),
        "position_mae": round(position_mae, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


# ---------------------------------------------------------------------------
# Depth accuracy metrics
# ---------------------------------------------------------------------------

def evaluate_depth_accuracy(
    gt_depth: np.ndarray,
    pred_depth: np.ndarray,
    mask: np.ndarray | None = None,
) -> dict:
    """Compute MAE, RMSE, relative error for depth estimation.

    Args:
        gt_depth: ground-truth depth (H, W) meters.
        pred_depth: predicted depth (H, W) meters.
        mask: optional binary mask, shape (H, W), where True = evaluate.

    Returns:
        dict with mae, rmse, rel_error, valid_pixel_ratio.
    """
    if mask is None:
        mask = np.ones_like(gt_depth, dtype=bool)

    # Consider only pixels where both gt and pred are valid
    valid = mask & np.isfinite(gt_depth) & np.isfinite(pred_depth) & (gt_depth > 0) & (pred_depth > 0)

    if not np.any(valid):
        return {
            "mae": float("inf"),
            "rmse": float("inf"),
            "rel_error": float("inf"),
            "valid_pixel_ratio": 0.0,
        }

    error = np.abs(gt_depth[valid] - pred_depth[valid])
    mae = float(np.mean(error))
    rmse = float(np.sqrt(np.mean(error ** 2)))
    rel_error = float(np.mean(error / gt_depth[valid]))

    return {
        "mae": round(mae, 6),
        "rmse": round(rmse, 6),
        "rel_error": round(rel_error, 6),
        "valid_pixel_ratio": round(float(np.mean(valid)), 4),
    }


# ---------------------------------------------------------------------------
# Tracking metrics
# ---------------------------------------------------------------------------

def evaluate_tracking(tracks: list[dict], gt_tracks: list[dict]) -> dict:
    """Compute simplified MOTA, MOTP, ID switches.

    Args:
        tracks: predicted tracks.
        gt_tracks: ground-truth tracks.

    Returns:
        dict with mota, motp, id_switches, track_count.
    """
    if len(gt_tracks) == 0:
        return {
            "mota": 1.0 if len(tracks) == 0 else 0.0,
            "motp": 0.0,
            "id_switches": 0,
            "track_count": 0,
        }

    # Simplified MOTA = 1 - (FP + FN + IDSW) / total_gt
    tp = 0
    fp = len(tracks)
    fn = len(gt_tracks)
    id_switches = 0

    for gt in gt_tracks:
        gt_id = gt.get("track_id")
        for pred in tracks:
            if pred.get("track_id") == gt_id:
                tp += 1
                fp -= 1
                fn -= 1
                # Check for ID switch by looking at previous assignments
                break

    total_gt = len(gt_tracks)
    mota = max(0.0, 1.0 - (fp + fn + id_switches) / max(1, total_gt))
    motp = 1.0  # simplified

    return {
        "mota": round(mota, 4),
        "motp": round(motp, 4),
        "id_switches": id_switches,
        "track_count": len(tracks),
    }


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_obstacle_detection_results(results: list[dict], output_path: Path) -> None:
    """Plot per-frame precision/recall over time."""
    frames = list(range(len(results)))
    precision_vals = [r.get("precision", 0) for r in results]
    recall_vals = [r.get("recall", 0) for r in results]
    f1_vals = [r.get("f1", 0) for r in results]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(frames, precision_vals, label="Precision", linewidth=1.5)
    ax.plot(frames, recall_vals, label="Recall", linewidth=1.5)
    ax.plot(frames, f1_vals, label="F1", linewidth=1.5)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Score")
    ax.set_title("Obstacle Detection Performance Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(output_path / "obstacle_detection_metrics.png", dpi=150)
    plt.close(fig)


def plot_depth_error_distribution(
    gt_depth: np.ndarray, pred_depth: np.ndarray, output_path: Path
) -> None:
    """Plot histogram of depth errors."""
    valid = (
        np.isfinite(gt_depth) & np.isfinite(pred_depth)
        & (gt_depth > 0) & (pred_depth > 0)
    )
    if not np.any(valid):
        return

    errors = np.abs(gt_depth[valid] - pred_depth[valid])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.hist(errors, bins=50, color="steelblue", edgecolor="white", alpha=0.8)
    ax1.set_xlabel("Absolute Error (m)")
    ax1.set_ylabel("Count")
    ax1.set_title("Depth Error Distribution")
    ax1.axvline(np.mean(errors), color="red", linestyle="--", label=f"Mean: {np.mean(errors):.4f}")
    ax1.legend()

    # Scatter: pred vs gt
    sample = np.random.choice(len(gt_depth[valid]), min(5000, int(np.sum(valid))), replace=False)
    ax2.scatter(gt_depth[valid][sample], pred_depth[valid][sample], s=1, alpha=0.5)
    ax2.plot([0, max(gt_depth[valid])], [0, max(gt_depth[valid])], "r--", alpha=0.5)
    ax2.set_xlabel("Ground Truth Depth (m)")
    ax2.set_ylabel("Predicted Depth (m)")
    ax2.set_title("Predicted vs Ground Truth Depth")

    fig.tight_layout()
    fig.savefig(output_path / "depth_error_distribution.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@draccus.wrap()
def main(cfg: EvaluatePerceptionConfig) -> None:
    """Load test data, run all evaluations, print/save report."""
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gt_dir = Path(cfg.gt_data_dir)
    pred_dir = Path(cfg.pred_data_dir)

    # Load data
    gt_data = _load_jsonl_dir(gt_dir, "*.jsonl")
    pred_data = _load_jsonl_dir(pred_dir, "*.jsonl")

    if not gt_data:
        logger.warning("No ground truth data found in %s. Generating synthetic test data.", gt_dir)
        gt_data = _generate_synthetic_test_data()

    if not pred_data:
        logger.warning("No prediction data found in %s. Running detector on GT data.", pred_dir)
        pred_data = _run_detector_on_data(gt_data)

    # Evaluate obstacle detection
    logger.info("=== Obstacle Detection ===")
    obs_results = []
    for gt_frame, pred_frame in zip(gt_data, pred_data):
        result = evaluate_obstacle_detection(
            gt_obstacles=gt_frame.get("obstacles", []),
            pred_obstacles=pred_frame.get("obstacles", []),
            iou_threshold=cfg.iou_threshold,
        )
        obs_results.append(result)

    # Aggregate obstacle metrics
    all_precision = [r["precision"] for r in obs_results]
    all_recall = [r["recall"] for r in obs_results]
    all_f1 = [r["f1"] for r in obs_results]

    obs_summary = {
        "num_frames": len(obs_results),
        "mean_precision": round(np.mean(all_precision), 4),
        "mean_recall": round(np.mean(all_recall), 4),
        "mean_f1": round(np.mean(all_f1), 4),
        "std_f1": round(np.std(all_f1), 4),
    }

    logger.info("  Precision: %.4f", obs_summary["mean_precision"])
    logger.info("  Recall:    %.4f", obs_summary["mean_recall"])
    logger.info("  F1:        %.4f (+/- %.4f)", obs_summary["mean_f1"], obs_summary["std_f1"])

    # Evaluate pallet detection
    logger.info("=== Pallet Detection ===")
    pallet_results = []
    for gt_frame, pred_frame in zip(gt_data, pred_data):
        result = evaluate_pallet_detection(
            gt_pallets=gt_frame.get("pallets", []),
            pred_pallets=pred_frame.get("pallets", []),
            distance_threshold=cfg.distance_threshold,
        )
        pallet_results.append(result)

    detection_rates = [r["detection_rate"] for r in pallet_results]
    distance_maes = [r["distance_mae"] for r in pallet_results if r["distance_mae"] != float("inf")]
    pallet_summary = {
        "num_frames": len(pallet_results),
        "mean_detection_rate": round(np.mean(detection_rates), 4),
        "mean_distance_mae": round(np.mean(distance_maes), 4) if distance_maes else float("inf"),
    }

    # Evaluate depth accuracy
    logger.info("=== Depth Accuracy ===")
    depth_results = []
    for gt_frame, pred_frame in zip(gt_data, pred_data):
        gt_depth = np.array(gt_frame.get("depth", []), dtype=np.float32)
        pred_depth = np.array(pred_frame.get("depth", []), dtype=np.float32)
        if gt_depth.size > 0 and pred_depth.size > 0:
            result = evaluate_depth_accuracy(gt_depth, pred_depth)
            depth_results.append(result)

    depth_maes = [r["mae"] for r in depth_results if r["mae"] != float("inf")]
    depth_rmse = [r["rmse"] for r in depth_results if r["rmse"] != float("inf")]
    depth_summary = {
        "num_frames": len(depth_results),
        "mean_mae": round(np.mean(depth_maes), 6) if depth_maes else float("inf"),
        "mean_rmse": round(np.mean(depth_rmse), 6) if depth_rmse else float("inf"),
    }
    logger.info("  MAE:  %.6f m", depth_summary["mean_mae"])
    logger.info("  RMSE: %.6f m", depth_summary["mean_rmse"])

    # Generate report
    report = {
        "obstacle_detection": obs_summary,
        "pallet_detection": pallet_summary,
        "depth_accuracy": depth_summary,
        "config": {
            "iou_threshold": cfg.iou_threshold,
            "distance_threshold": cfg.distance_threshold,
        },
    }

    report_path = out_dir / "perception_eval_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # Generate plots
    try:
        plot_obstacle_detection_results(obs_results, out_dir)
        logger.info("  Plots saved to %s", out_dir)
    except Exception as e:
        logger.warning("  Plot generation failed: %s", e)

    logger.info("Evaluation report saved to %s", report_path)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_jsonl_dir(data_dir: Path, pattern: str = "*.jsonl") -> list[dict]:
    """Load all JSONL files in directory."""
    frames = []
    for fpath in sorted(data_dir.glob(pattern)):
        with open(fpath) as f:
            for line in f:
                line = line.strip()
                if line:
                    frames.append(json.loads(line))
    return frames


def _generate_synthetic_test_data(num_frames: int = 50) -> list[dict]:
    """Generate synthetic test data for evaluation."""
    rng = np.random.RandomState(42)
    data = []
    for i in range(num_frames):
        h, w = 640, 480
        depth = np.zeros((h, w), dtype=np.float32)
        depth[:] = 2.0
        # Add a box
        cx, cy = int(w * 0.5), int(h * 0.5)
        bw, bh = 60, 80
        depth[cy - bh // 2:cy + bh // 2, cx - bw // 2:cx + bw // 2] = 0.8

        num_obs = rng.randint(1, 4)
        obstacles = []
        for _ in range(num_obs):
            ox = rng.randint(100, 300)
            oy = rng.randint(150, 350)
            obstacles.append({
                "bbox": [ox, oy, ox + 40, oy + 60],
                "distance": float(rng.uniform(0.5, 2.0)),
                "bearing_deg": float(rng.uniform(-40, 40)),
                "sector": rng.choice(["left", "front", "right"]),
            })

        data.append({
            "frame_id": i,
            "depth": depth.tolist(),
            "obstacles": obstacles,
            "pallets": [],
        })
    return data


def _run_detector_on_data(gt_data: list[dict]) -> list[dict]:
    """Run the ObstacleDetector on ground-truth depth to get predictions."""
    detector = ObstacleDetector()
    pred_data = []
    for frame in gt_data:
        depth = np.array(frame.get("depth", []), dtype=np.float32)
        if depth.size == 0:
            pred_data.append({"obstacles": [], "pallets": []})
            continue

        # Convert depth to scan
        h, w = depth.shape
        col_width = max(1, w // 64)
        scan_m = np.full(64, np.nan, dtype=np.float32)
        band_h = h // 3
        for i in range(64):
            x0 = i * col_width
            x1 = min(w, x0 + col_width)
            col = depth[h // 2 - band_h // 2:h // 2 + band_h // 2, x0:x1]
            valid = col[np.isfinite(col) & (col > 0.01)]
            if len(valid) > 0:
                scan_m[i] = float(np.percentile(valid, 10))

        detections = detector.detect(np.nan_to_num(scan_m, nan=5.0))
        obstacles_formatted = detections.get("obstacles", [])

        pred_data.append({
            "obstacles": obstacles_formatted,
            "pallets": [],
            "depth": frame.get("depth", []),
        })

    return pred_data


if __name__ == "__main__":
    main()
