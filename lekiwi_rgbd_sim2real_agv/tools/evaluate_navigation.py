"""Evaluate navigation performance."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import draccus

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")


@dataclass
class EvalNavConfig:
    """CLI configuration for navigation evaluation."""

    results_dir: str = "data/eval_results"
    output_dir: str = "eval_output/navigation"
    goal_distance_threshold: float = 0.3


# ---------------------------------------------------------------------------
# Navigation metrics
# ---------------------------------------------------------------------------

def evaluate_navigation(
    episode_results: list[dict],
    goal_distance_threshold: float = 0.3,
) -> dict:
    """Compute navigation performance metrics.

    Args:
        episode_results: list of episode dicts, each containing:
            - success: bool
            - collision: bool
            - timeout: bool
            - steps: int
            - min_clearance_per_step: list[float]
            - actions: list[dict] with x.vel, y.vel, theta.vel
            - latencies_ms: list[float]
            - goal_distance: float (final distance to goal)
        goal_distance_threshold: distance threshold for success.

    Returns:
        dict of aggregate metrics.
    """
    n = len(episode_results)
    if n == 0:
        return _empty_metrics()

    successes = sum(1 for e in episode_results if e.get("success", False))
    collisions = sum(1 for e in episode_results if e.get("collision", False))
    timeouts = sum(1 for e in episode_results if e.get("timeout", False))

    all_clearances = []
    all_jerks = []
    all_latencies = []
    all_goal_distances = []
    all_steps = []

    for ep in episode_results:
        clearances = ep.get("min_clearance_per_step", [])
        if clearances:
            all_clearances.append(np.mean(clearances))

        actions = ep.get("actions", [])
        if len(actions) > 1:
            jerk = _compute_jerk(actions)
            all_jerks.append(jerk)

        latencies = ep.get("latencies_ms", [])
        if latencies:
            all_latencies.append(np.mean(latencies))

        gd = ep.get("goal_distance")
        if gd is not None:
            all_goal_distances.append(gd)

        steps = ep.get("steps", 0)
        all_steps.append(steps)

    return {
        "num_episodes": n,
        "success_rate": round(successes / n, 4),
        "collision_rate": round(collisions / n, 4),
        "timeout_rate": round(timeouts / n, 4),
        "avg_clearance_m": round(np.mean(all_clearances), 4) if all_clearances else float("nan"),
        "avg_smoothness_jerk": round(np.mean(all_jerks), 4) if all_jerks else float("nan"),
        "avg_latency_ms": round(np.mean(all_latencies), 2) if all_latencies else float("nan"),
        "avg_steps": round(np.mean(all_steps), 1) if all_steps else 0,
        "avg_goal_distance_m": round(np.mean(all_goal_distances), 4) if all_goal_distances else float("nan"),
        "success_count": successes,
        "collision_count": collisions,
        "timeout_count": timeouts,
    }


def _compute_jerk(actions: list[dict]) -> float:
    """Compute average action jerk (sum of absolute accelerations)."""
    if len(actions) < 2:
        return 0.0

    jerk_sum = 0.0
    for i in range(1, len(actions)):
        prev = actions[i - 1]
        curr = actions[i]
        dvx = abs(curr.get("x.vel", 0) - prev.get("x.vel", 0))
        dvy = abs(curr.get("y.vel", 0) - prev.get("y.vel", 0))
        dw = abs(curr.get("theta.vel", 0) - prev.get("theta.vel", 0))
        jerk_sum += dvx + dvy + dw / 90.0  # normalize angular
    return jerk_sum / (len(actions) - 1)


def _empty_metrics() -> dict:
    return {
        "num_episodes": 0,
        "success_rate": 0.0,
        "collision_rate": 0.0,
        "timeout_rate": 0.0,
        "avg_clearance_m": float("nan"),
        "avg_smoothness_jerk": float("nan"),
        "avg_latency_ms": float("nan"),
        "avg_steps": 0,
        "avg_goal_distance_m": float("nan"),
        "success_count": 0,
        "collision_count": 0,
        "timeout_count": 0,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_navigation_results(metrics: dict, output_path: Path) -> None:
    """Generate summary plots."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Pie chart: outcomes
    ax1 = axes[0, 0]
    labels = ["Success", "Collision", "Timeout"]
    values = [
        metrics.get("success_count", 0),
        metrics.get("collision_count", 0),
        metrics.get("timeout_count", 0),
    ]
    colors = ["#2ecc71", "#e74c3c", "#f39c12"]
    if sum(values) > 0:
        ax1.pie(values, labels=labels, colors=colors, autopct="%1.1f%%",
                startangle=90)
    ax1.set_title("Episode Outcomes")

    # Bar chart: rates
    ax2 = axes[0, 1]
    rates = [
        metrics.get("success_rate", 0),
        metrics.get("collision_rate", 0),
        metrics.get("timeout_rate", 0),
    ]
    bars = ax2.bar(labels, rates, color=colors, edgecolor="white")
    for bar, val in zip(bars, rates):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{val:.2%}", ha="center", fontsize=10)
    ax2.set_ylabel("Rate")
    ax2.set_title("Navigation Rates")
    ax2.set_ylim(0, max(1.0, max(rates) * 1.2))

    # Key metrics table
    ax3 = axes[1, 0]
    ax3.axis("off")
    table_data = [
        ["Metric", "Value"],
        ["Success Rate", f"{metrics.get('success_rate', 0):.2%}"],
        ["Collision Rate", f"{metrics.get('collision_rate', 0):.2%}"],
        ["Timeout Rate", f"{metrics.get('timeout_rate', 0):.2%}"],
        ["Avg Clearance", f"{metrics.get('avg_clearance_m', 0):.3f} m"],
        ["Avg Smoothness", f"{metrics.get('avg_smoothness_jerk', 0):.4f}"],
        ["Avg Latency", f"{metrics.get('avg_latency_ms', 0):.1f} ms"],
        ["Avg Steps", f"{metrics.get('avg_steps', 0):.1f}"],
    ]
    table = ax3.table(cellText=table_data, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    ax3.set_title("Key Metrics")

    # Episodes summary text
    ax4 = axes[1, 1]
    ax4.axis("off")
    ax4.text(0.5, 0.5,
             f"Total Episodes: {metrics.get('num_episodes', 0)}\n\n"
             f"Successes: {metrics.get('success_count', 0)}\n"
             f"Collisions: {metrics.get('collision_count', 0)}\n"
             f"Timeouts: {metrics.get('timeout_count', 0)}",
             transform=ax4.transAxes, fontsize=12,
             verticalalignment="center", horizontalalignment="center",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    ax4.set_title("Summary")

    fig.tight_layout()
    fig.savefig(output_path / "navigation_metrics.png", dpi=150)
    plt.close(fig)


def plot_path_visualizations(
    episode_results: list[dict], output_path: Path
) -> None:
    """Plot XYZ paths for each episode."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Select up to 4 representative episodes
    selected = []
    # prioritize: 1 success, 1 collision, 1 timeout
    for label, cond in [("success", lambda e: e.get("success")),
                         ("collision", lambda e: e.get("collision")),
                         ("timeout", lambda e: e.get("timeout"))]:
        for e in episode_results:
            if cond(e) and e not in selected:
                selected.append(e)
                break
    # Fill remaining with any
    for e in episode_results:
        if len(selected) >= 4:
            break
        if e not in selected:
            selected.append(e)

    for ax, ep in zip(axes.flat, selected):
        path_xy = ep.get("path_xy", [])
        if not path_xy:
            ax.text(0.5, 0.5, "No path data", transform=ax.transAxes, ha="center")
            ax.set_title(f"Episode: {ep.get('episode_id', '?')}")
            continue

        path = np.array(path_xy)
        ax.plot(path[:, 0], path[:, 1], "b-", linewidth=1, alpha=0.7, label="Path")
        # Start
        ax.plot(path[0, 0], path[0, 1], "go", markersize=8, label="Start")
        # End
        ax.plot(path[-1, 0], path[-1, 1], "r*", markersize=10, label="End")
        # Goal
        goal = ep.get("goal", [0, 0])
        ax.plot(goal[0], goal[1], "m*", markersize=10, label="Goal")

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.legend(fontsize=7)
        ax.set_aspect("equal")
        status = "Success" if ep.get("success") else "Fail"
        ax.set_title(f"E{ep.get('episode_id', '?')} [{status}]")

    fig.tight_layout()
    fig.savefig(output_path / "navigation_paths.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@draccus.wrap()
def main(cfg: EvalNavConfig) -> None:
    """Load episode results, compute metrics, generate report and plots."""
    results_dir = Path(cfg.results_dir)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load episode results
    episode_results = _load_episode_results(results_dir)

    if not episode_results:
        logger.warning("No episode results found. Generating synthetic data for demo.")
        episode_results = _generate_synthetic_episodes()

    # Compute metrics
    metrics = evaluate_navigation(episode_results, goal_distance_threshold=cfg.goal_distance_threshold)

    # Print summary
    logger.info("=== Navigation Evaluation ===")
    logger.info("  Episodes:       %d", metrics["num_episodes"])
    logger.info("  Success Rate:   %.2f%%", metrics["success_rate"] * 100)
    logger.info("  Collision Rate: %.2f%%", metrics["collision_rate"] * 100)
    logger.info("  Timeout Rate:   %.2f%%", metrics["timeout_rate"] * 100)
    logger.info("  Avg Clearance:  %.3f m", metrics["avg_clearance_m"])
    logger.info("  Avg Smoothness: %.4f", metrics["avg_smoothness_jerk"])
    logger.info("  Avg Latency:    %.1f ms", metrics["avg_latency_ms"])

    # Save report
    report_path = out_dir / "navigation_eval_report.json"
    with open(report_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Report saved to %s", report_path)

    # Plots
    try:
        plot_navigation_results(metrics, out_dir)
        plot_path_visualizations(episode_results, out_dir)
        logger.info("Plots saved to %s", out_dir)
    except Exception as e:
        logger.warning("Plot generation failed: %s", e)


def _load_episode_results(results_dir: Path) -> list[dict]:
    """Load episode results from JSONL files."""
    episodes = []
    for fpath in sorted(results_dir.glob("*.jsonl")):
        with open(fpath) as f:
            for line in f:
                line = line.strip()
                if line:
                    episodes.append(json.loads(line))
    return episodes


def _generate_synthetic_episodes(n: int = 20) -> list[dict]:
    """Generate synthetic episode data for demonstration."""
    rng = np.random.RandomState(123)
    episodes = []

    for i in range(n):
        steps = rng.randint(50, 500)
        success = rng.rand() < 0.7
        collision = rng.rand() < 0.1 if not success else False
        timeout = not success and not collision

        # Generate path
        path_len = steps
        path = np.zeros((path_len, 2))
        path[:, 0] = np.cumsum(rng.normal(0.02, 0.01, path_len))
        path[:, 1] = np.cumsum(rng.normal(0, 0.02, path_len))

        actions = []
        for _ in range(steps):
            actions.append({
                "x.vel": rng.uniform(0, 0.2),
                "y.vel": rng.uniform(-0.05, 0.05),
                "theta.vel": rng.uniform(-10, 10),
            })

        episodes.append({
            "episode_id": i,
            "success": bool(success),
            "collision": bool(collision),
            "timeout": bool(timeout),
            "steps": steps,
            "min_clearance_per_step": rng.uniform(0.1, 2.0, size=steps).tolist(),
            "actions": actions,
            "latencies_ms": rng.normal(15, 5, size=steps).tolist(),
            "goal_distance": float(rng.uniform(0.05, 2.0)),
            "goal": [2.0, rng.uniform(-0.5, 0.5)],
            "path_xy": path.tolist(),
        })

    return episodes


if __name__ == "__main__":
    main()
