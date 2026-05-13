"""Compare simulation vs real depth/scan distributions."""
from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from scipy.spatial.distance import jensenshannon

import draccus

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")


@dataclass
class SimToRealCompareConfig:
    """CLI configuration for sim-to-real comparison."""

    sim_data_dir: str = "data/sim_logs"
    real_data_dir: str = "data/real_logs"
    output_dir: str = "eval_output/sim2real"
    num_bins: int = 50
    depth_range_m: tuple[float, float] = (0.0, 5.0)


# ---------------------------------------------------------------------------
# Depth comparison
# ---------------------------------------------------------------------------

def compare_depth_distributions(
    sim_depths: list[np.ndarray],
    real_depths: list[np.ndarray],
    num_bins: int = 50,
    depth_range: tuple[float, float] = (0.0, 5.0),
) -> dict:
    """Compare histograms, KL divergence, Wasserstein distance.

    Args:
        sim_depths: list of depth images (H, W) from simulation.
        real_depths: list of depth images (H, W) from real sensor.
        num_bins: number of histogram bins.
        depth_range: (min, max) range for histogram.

    Returns:
        dict with histogram, kl_divergence, wasserstein_distance, per-sample stats.
    """
    # Flatten all valid pixels
    sim_all = np.concatenate([
        d[np.isfinite(d) & (d > 0.001)].ravel()
        for d in sim_depths if d.size > 0
    ])
    real_all = np.concatenate([
        d[np.isfinite(d) & (d > 0.001)].ravel()
        for d in real_depths if d.size > 0
    ])

    if len(sim_all) == 0 or len(real_all) == 0:
        return {
            "error": "Insufficient data for comparison",
            "sim_samples": len(sim_all),
            "real_samples": len(real_all),
        }

    # Compute histograms
    bins = np.linspace(depth_range[0], depth_range[1], num_bins + 1)
    sim_hist, _ = np.histogram(sim_all, bins=bins, density=True)
    real_hist, _ = np.histogram(real_all, bins=bins, density=True)

    # Add small epsilon to avoid zero bins
    eps = 1e-10
    sim_hist = sim_hist + eps
    real_hist = real_hist + eps
    sim_hist /= sim_hist.sum()
    real_hist /= real_hist.sum()

    # Jensen-Shannon divergence
    js_div = float(jensenshannon(sim_hist, real_hist) ** 2)

    # Wasserstein distance (1D Earth Mover's Distance)
    sim_sorted = np.sort(sim_all)
    real_sorted = np.sort(real_all)
    # Sub-sample to equal length for comparison
    min_len = min(len(sim_sorted), len(real_sorted), 10000)
    if min_len > 0:
        sim_sample = sim_sorted[:min_len]
        real_sample = real_sorted[:min_len]
        wasserstein = float(stats.wasserstein_distance(sim_sample, real_sample))
    else:
        wasserstein = float("inf")

    # Per-sample statistics
    sim_stats = {
        "mean": float(np.mean(sim_all)),
        "std": float(np.std(sim_all)),
        "median": float(np.median(sim_all)),
        "p10": float(np.percentile(sim_all, 10)),
        "p90": float(np.percentile(sim_all, 90)),
        "num_pixels": int(len(sim_all)),
    }
    real_stats = {
        "mean": float(np.mean(real_all)),
        "std": float(np.std(real_all)),
        "median": float(np.median(real_all)),
        "p10": float(np.percentile(real_all, 10)),
        "p90": float(np.percentile(real_all, 90)),
        "num_pixels": int(len(real_all)),
    }

    return {
        "js_divergence": round(js_div, 6),
        "wasserstein_distance_m": round(wasserstein, 6),
        "sim_stats": sim_stats,
        "real_stats": real_stats,
        "bin_edges": bins.tolist(),
        "sim_histogram": sim_hist.tolist(),
        "real_histogram": real_hist.tolist(),
    }


# ---------------------------------------------------------------------------
# Scan comparison
# ---------------------------------------------------------------------------

def compare_scan_distributions(
    sim_scans: list[np.ndarray],
    real_scans: list[np.ndarray],
    scan_dim: int = 64,
) -> dict:
    """Per-bin statistics, distribution plots.

    Args:
        sim_scans: list of scan arrays (N,).
        real_scans: list of scan arrays (N,).

    Returns:
        dict with per-bin mean, std, wasserstein distances.
    """
    if not sim_scans or not real_scans:
        return {"error": "No scan data provided"}

    sim_stack = np.stack([s for s in sim_scans if len(s) == scan_dim])
    real_stack = np.stack([s for s in real_scans if len(s) == scan_dim])

    if sim_stack.size == 0 or real_stack.size == 0:
        return {"error": "No valid scan data after filtering"}

    # Per-bin statistics
    sim_bin_mean = np.nanmean(sim_stack, axis=0)
    sim_bin_std = np.nanstd(sim_stack, axis=0)
    real_bin_mean = np.nanmean(real_stack, axis=0)
    real_bin_std = np.nanstd(real_stack, axis=0)

    # Per-bin Wasserstein distance
    per_bin_ws = []
    for i in range(scan_dim):
        sim_col = sim_stack[:, i]
        real_col = real_stack[:, i]
        sim_valid = sim_col[np.isfinite(sim_col)]
        real_valid = real_col[np.isfinite(real_col)]
        if len(sim_valid) > 5 and len(real_valid) > 5:
            ws = float(stats.wasserstein_distance(sim_valid, real_valid))
        else:
            ws = float("nan")
        per_bin_ws.append(ws)

    # Global Wasserstein (flattened)
    sim_all = sim_stack.ravel()
    real_all = real_stack.ravel()
    sim_all = sim_all[np.isfinite(sim_all)]
    real_all = real_all[np.isfinite(real_all)]
    if len(sim_all) > 0 and len(real_all) > 0:
        global_ws = float(stats.wasserstein_distance(sim_all, real_all))
    else:
        global_ws = float("nan")

    return {
        "global_wasserstein_distance": round(global_ws, 6),
        "mean_per_bin_wasserstein": round(float(np.nanmean(per_bin_ws)), 6),
        "per_bin": {
            "sim_mean": [round(float(x), 4) for x in sim_bin_mean],
            "sim_std": [round(float(x), 4) for x in sim_bin_std],
            "real_mean": [round(float(x), 4) for x in real_bin_mean],
            "real_std": [round(float(x), 4) for x in real_bin_std],
            "per_bin_wasserstein": [round(float(x), 4) for x in per_bin_ws],
        },
        "sim_frames": int(sim_stack.shape[0]),
        "real_frames": int(real_stack.shape[0]),
    }


# ---------------------------------------------------------------------------
# Invalid ratio comparison
# ---------------------------------------------------------------------------

def compare_invalid_ratios(
    sim_quality: list[dict],
    real_quality: list[dict],
) -> dict:
    """Compare invalid pixel ratios, dropout patterns.

    Args:
        sim_quality: list of per-frame quality dicts from sim.
        real_quality: list of per-frame quality dicts from real.

    Returns:
        dict comparing invalid_ratio and dropout_ratio distributions.
    """
    sim_invalid = [
        q.get("invalid_ratio", float("nan"))
        for q in sim_quality
        if not np.isnan(q.get("invalid_ratio", float("nan")))
    ]
    real_invalid = [
        q.get("invalid_ratio", float("nan"))
        for q in real_quality
        if not np.isnan(q.get("invalid_ratio", float("nan")))
    ]
    sim_dropout = [
        q.get("dropout_ratio", float("nan"))
        for q in sim_quality
        if not np.isnan(q.get("dropout_ratio", float("nan")))
    ]
    real_dropout = [
        q.get("dropout_ratio", float("nan"))
        for q in real_quality
        if not np.isnan(q.get("dropout_ratio", float("nan")))
    ]

    def _stats(arr):
        if not arr:
            return {}
        return {
            "mean": round(float(np.mean(arr)), 6),
            "std": round(float(np.std(arr)), 6),
            "median": round(float(np.median(arr)), 6),
            "p95": round(float(np.percentile(arr, 95)), 6),
            "max": round(float(np.max(arr)), 6),
        }

    return {
        "sim_invalid_ratio": _stats(sim_invalid),
        "real_invalid_ratio": _stats(real_invalid),
        "sim_dropout_ratio": _stats(sim_dropout),
        "real_dropout_ratio": _stats(real_dropout),
        "sim_frames_with_quality": len(sim_quality),
        "real_frames_with_quality": len(real_quality),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_depth_comparison(depth_result: dict, bin_edges: list, output_path: Path) -> None:
    """Plot depth distribution comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram overlay
    ax1 = axes[0]
    centers = (np.array(bin_edges[:-1]) + np.array(bin_edges[1:])) / 2
    ax1.plot(centers, depth_result.get("sim_histogram", []), "b-", linewidth=1.5, label="Simulation")
    ax1.plot(centers, depth_result.get("real_histogram", []), "r-", linewidth=1.5, label="Real")
    ax1.set_xlabel("Depth (m)")
    ax1.set_ylabel("Density")
    ax1.set_title(f"Depth Distribution Comparison (JS Div: {depth_result.get('js_divergence', 0):.6f})")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Stats comparison bar
    ax2 = axes[1]
    sim_stats = depth_result.get("sim_stats", {})
    real_stats = depth_result.get("real_stats", {})
    metrics = ["mean", "std", "median", "p10", "p90"]
    x = np.arange(len(metrics))
    width = 0.35
    sim_vals = [sim_stats.get(m, 0) for m in metrics]
    real_vals = [real_stats.get(m, 0) for m in metrics]
    bars1 = ax2.bar(x - width / 2, sim_vals, width, label="Simulation", color="steelblue")
    bars2 = ax2.bar(x + width / 2, real_vals, width, label="Real", color="coral")
    ax2.set_xticks(x)
    ax2.set_xticklabels([m.capitalize() for m in metrics])
    ax2.set_ylabel("Depth (m)")
    ax2.set_title("Depth Statistics Comparison")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(output_path / "depth_comparison.png", dpi=150)
    plt.close(fig)


def plot_scan_comparison(scan_result: dict, output_path: Path) -> None:
    """Plot scan distribution comparison."""
    per_bin = scan_result.get("per_bin", {})
    if not per_bin:
        return

    bins = np.arange(len(per_bin.get("sim_mean", [])))
    sim_mean = np.array(per_bin.get("sim_mean", []))
    sim_std = np.array(per_bin.get("sim_std", []))
    real_mean = np.array(per_bin.get("real_mean", []))
    real_std = np.array(per_bin.get("real_std", []))
    per_bin_ws = np.array(per_bin.get("per_bin_wasserstein", []))

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Per-bin mean with error bands
    ax1 = axes[0]
    ax1.fill_between(bins, sim_mean - sim_std, sim_mean + sim_std,
                     alpha=0.2, color="blue", label="Sim +/- 1std")
    ax1.fill_between(bins, real_mean - real_std, real_mean + real_std,
                     alpha=0.2, color="red", label="Real +/- 1std")
    ax1.plot(bins, sim_mean, "b-", linewidth=1.5, label="Sim Mean")
    ax1.plot(bins, real_mean, "r-", linewidth=1.5, label="Real Mean")
    ax1.set_ylabel("Distance (m)")
    ax1.set_title("Per-Bin Scan Distribution Comparison")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Per-bin Wasserstein distance
    ax2 = axes[1]
    ax2.bar(bins, np.nan_to_num(per_bin_ws, nan=0), color="purple", alpha=0.7, width=1.0)
    ax2.axhline(y=np.nanmean(per_bin_ws), color="orange", linestyle="--",
                label=f"Mean WS: {np.nanmean(per_bin_ws):.4f}")
    ax2.set_xlabel("Scan Bin Index")
    ax2.set_ylabel("Wasserstein Distance")
    ax2.set_title("Per-Bin Sim-to-Real Wasserstein Distance")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path / "scan_comparison.png", dpi=150)
    plt.close(fig)


def plot_invalid_comparison(invalid_result: dict, output_path: Path) -> None:
    """Plot invalid ratio comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    categories = ["Invalid Ratio", "Dropout Ratio"]
    sim_means = [
        invalid_result.get("sim_invalid_ratio", {}).get("mean", 0),
        invalid_result.get("sim_dropout_ratio", {}).get("mean", 0),
    ]
    real_means = [
        invalid_result.get("real_invalid_ratio", {}).get("mean", 0),
        invalid_result.get("real_dropout_ratio", {}).get("mean", 0),
    ]

    x = np.arange(len(categories))
    width = 0.35

    axes[0].bar(x - width / 2, sim_means, width, label="Simulation", color="steelblue")
    axes[0].bar(x + width / 2, real_means, width, label="Real", color="coral")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(categories)
    axes[0].set_ylabel("Mean Ratio")
    axes[0].set_title("Invalid & Dropout Ratio Comparison")
    axes[0].legend()

    # Text summary
    axes[1].axis("off")
    text_lines = ["Depth Quality Comparison", ""]
    for domain, label in [("sim", "Simulation"), ("real", "Real")]:
        for metric in ["invalid_ratio", "dropout_ratio"]:
            key = f"{domain}_{metric}"
            stats = invalid_result.get(key, {})
            text_lines.append(
                f"{label} {metric}: "
                f"mean={stats.get('mean', 0):.4f}, "
                f"std={stats.get('std', 0):.4f}, "
                f"p95={stats.get('p95', 0):.4f}"
            )
    axes[1].text(0.05, 0.95, "\n".join(text_lines), transform=axes[1].transAxes,
                 fontsize=10, verticalalignment="top", fontfamily="monospace")

    fig.tight_layout()
    fig.savefig(output_path / "invalid_comparison.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(
    depth_result: dict,
    scan_result: dict,
    invalid_result: dict,
    output_path: Path,
) -> None:
    """Generate comparison report with plots."""
    report = {
        "depth_comparison": depth_result,
        "scan_comparison": scan_result,
        "invalid_ratio_comparison": invalid_result,
        "summary": {
            "js_divergence": depth_result.get("js_divergence"),
            "wasserstein_depth_m": depth_result.get("wasserstein_distance_m"),
            "wasserstein_scan": scan_result.get("global_wasserstein_distance"),
            "sim_invalid_mean": invalid_result.get("sim_invalid_ratio", {}).get("mean"),
            "real_invalid_mean": invalid_result.get("real_invalid_ratio", {}).get("mean"),
        },
    }

    with open(output_path / "sim_to_real_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("Comparison report saved to %s", output_path / "sim_to_real_report.json")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_depth_frames(data_dir: Path) -> list[np.ndarray]:
    """Load depth frames from NPZ, NPY, or JSONL."""
    depths = []

    # Try NPZ files
    for npz_path in sorted(data_dir.glob("*.npz")):
        data = np.load(npz_path)
        depth = data.get("depth")
        if depth is not None:
            depths.append(np.asarray(depth, dtype=np.float32))

    # Try NPY files
    for npy_path in sorted(data_dir.glob("*depth*.npy")):
        depths.append(np.load(str(npy_path)).astype(np.float32))

    # Try episode subdirectories
    for subdir in sorted(data_dir.iterdir()):
        if not subdir.is_dir():
            continue
        for depth_path in sorted(subdir.glob("*depth*.npy")):
            depths.append(np.load(str(depth_path)).astype(np.float32))
        for npz_path in sorted(subdir.glob("*.npz")):
            data = np.load(npz_path)
            depth = data.get("depth")
            if depth is not None:
                depths.append(np.asarray(depth, dtype=np.float32))

    return depths


def _load_scan_frames(data_dir: Path) -> list[np.ndarray]:
    """Load scan data from observations JSONL."""
    scans = []
    for jsonl_path in sorted(data_dir.glob("*observations*.jsonl")):
        with open(jsonl_path) as f:
            for line in f:
                try:
                    obs = json.loads(line.strip())
                    scan = obs.get("scan64") or obs.get("scan") or obs.get("scan_tail")
                    if scan and isinstance(scan, list) and len(scan) > 0:
                        scans.append(np.array(scan, dtype=np.float32))
                except (json.JSONDecodeError, KeyError):
                    continue

    # Try episode subdirectories
    for subdir in sorted(data_dir.iterdir()):
        if not subdir.is_dir():
            continue
        for jsonl_path in sorted(subdir.glob("*observations*.jsonl")):
            with open(jsonl_path) as f:
                for line in f:
                    try:
                        obs = json.loads(line.strip())
                        scan = obs.get("scan64") or obs.get("scan") or obs.get("scan_tail")
                        if scan and isinstance(scan, list) and len(scan) > 0:
                            scans.append(np.array(scan, dtype=np.float32))
                    except (json.JSONDecodeError, KeyError):
                        continue

    return scans


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@draccus.wrap()
def main(cfg: SimToRealCompareConfig) -> None:
    """Load sim and real data, compare distributions, output report."""
    sim_dir = Path(cfg.sim_data_dir)
    real_dir = Path(cfg.real_data_dir)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load depth frames
    sim_depths = _load_depth_frames(sim_dir)
    real_depths = _load_depth_frames(real_dir)
    logger.info("Loaded %d sim depth frames, %d real depth frames",
                len(sim_depths), len(real_depths))

    # Load scan frames
    sim_scans = _load_scan_frames(sim_dir)
    real_scans = _load_scan_frames(real_dir)
    logger.info("Loaded %d sim scan frames, %d real scan frames",
                len(sim_scans), len(real_scans))

    if not sim_depths and not real_depths:
        logger.warning("No depth data found. Generating synthetic demo data.")
        sim_depths = _generate_synthetic_depth(sim=True, n=100)
        real_depths = _generate_synthetic_depth(sim=False, n=100)

    if not sim_scans and not real_scans:
        logger.warning("No scan data found. Generating synthetic demo data.")
        sim_scans = _generate_synthetic_scans(sim=True, n=200)
        real_scans = _generate_synthetic_scans(sim=False, n=200)

    # Compare depth distributions
    logger.info("=== Depth Distribution Comparison ===")
    depth_result = compare_depth_distributions(
        sim_depths, real_depths,
        num_bins=cfg.num_bins,
        depth_range=cfg.depth_range_m,
    )
    if "js_divergence" in depth_result:
        logger.info("  JS Divergence:        %.6f", depth_result["js_divergence"])
        logger.info("  Wasserstein Distance: %.6f", depth_result["wasserstein_distance_m"])
        logger.info("  Sim mean depth: %.3f, Real mean depth: %.3f",
                    depth_result["sim_stats"]["mean"],
                    depth_result["real_stats"]["mean"])

    # Compare scan distributions
    logger.info("=== Scan Distribution Comparison ===")
    scan_result = compare_scan_distributions(sim_scans, real_scans)
    if "global_wasserstein_distance" in scan_result:
        logger.info("  Global Wasserstein:     %.6f", scan_result["global_wasserstein_distance"])
        logger.info("  Mean Per-Bin Wasserstein: %.6f", scan_result["mean_per_bin_wasserstein"])

    # Compare invalid ratios
    logger.info("=== Invalid Ratio Comparison ===")
    # Generate quality data from depth/scan if not available
    sim_quality = _compute_quality_from_depth(sim_depths)
    real_quality = _compute_quality_from_depth(real_depths)
    invalid_result = compare_invalid_ratios(sim_quality, real_quality)
    logger.info("  Sim invalid ratio: %.4f +/- %.4f",
                invalid_result.get("sim_invalid_ratio", {}).get("mean", 0),
                invalid_result.get("sim_invalid_ratio", {}).get("std", 0))
    logger.info("  Real invalid ratio: %.4f +/- %.4f",
                invalid_result.get("real_invalid_ratio", {}).get("mean", 0),
                invalid_result.get("real_invalid_ratio", {}).get("std", 0))

    # Generate report
    generate_report(depth_result, scan_result, invalid_result, out_dir)

    # Generate plots
    try:
        bin_edges = depth_result.get("bin_edges", np.linspace(0, 5, cfg.num_bins + 1).tolist())
        plot_depth_comparison(depth_result, bin_edges, out_dir)
        plot_scan_comparison(scan_result, out_dir)
        plot_invalid_comparison(invalid_result, out_dir)
        logger.info("  Plots saved to %s", out_dir)
    except Exception as e:
        logger.warning("  Plot generation failed: %s", e)


# ---------------------------------------------------------------------------
# Synthetic data generators
# ---------------------------------------------------------------------------

def _generate_synthetic_depth(sim: bool = True, n: int = 100) -> list[np.ndarray]:
    """Generate synthetic depth frames for demo."""
    rng = np.random.RandomState(42 if sim else 99)
    h, w = 640, 480
    depths = []
    for _ in range(n):
        depth = rng.normal(loc=1.8, scale=0.4, size=(h, w)).astype(np.float32)
        depth = np.clip(depth, 0.15, 5.0)
        if not sim:
            # Real data: add more noise and dropouts
            depth[rng.rand(h, w) < 0.05] = 0.0
            depth += rng.normal(0, 0.02, (h, w)).astype(np.float32)
        depths.append(depth)
    return depths


def _generate_synthetic_scans(sim: bool = True, n: int = 200) -> list[np.ndarray]:
    """Generate synthetic scan frames for demo."""
    rng = np.random.RandomState(123 if sim else 456)
    scans = []
    for _ in range(n):
        scan = rng.normal(loc=1.5, scale=0.5, size=64).astype(np.float32)
        scan = np.clip(scan, 0.1, 5.0)
        if not sim:
            # Real data: add some dropout, more front obstacles
            scan[rng.rand(64) < 0.03] = np.nan
            scan[25:40] *= rng.uniform(0.5, 0.9, size=15)
        scans.append(scan)
    return scans


def _compute_quality_from_depth(depths: list[np.ndarray]) -> list[dict]:
    """Compute per-frame quality metrics from depth images."""
    quality_list = []
    for depth in depths:
        if depth.size == 0:
            continue
        total = depth.size
        invalid = np.sum(~np.isfinite(depth) | (depth <= 0.001))
        quality_list.append({
            "invalid_ratio": float(invalid / total) if total > 0 else 1.0,
            "dropout_ratio": 0.0,
            "min_range_over_time": float(np.min(depth[np.isfinite(depth) & (depth > 0.001)])) if np.any(np.isfinite(depth)) else float("inf"),
        })
    return quality_list


if __name__ == "__main__":
    main()
