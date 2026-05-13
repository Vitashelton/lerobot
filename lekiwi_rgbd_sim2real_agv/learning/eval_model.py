"""
Evaluate a trained Residual Safety Model on held-out test data.

Metrics:
  - MSE (delta prediction error)
  - Collision-rate reduction (with vs without residual correction)
  - Mean clearance improvement
  - Action smoothness
  - Inference latency
"""

from __future__ import annotations

import os
import time
import json
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

import draccus

try:
    from lekiwi_rgbd_sim2real_agv.learning.residual_model import ResidualSafetyModel
    from lekiwi_rgbd_sim2real_agv.learning.risk_scorer import RiskScorer
except ImportError:
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from learning.residual_model import ResidualSafetyModel  # type: ignore[import-not-found]
    from learning.risk_scorer import RiskScorer  # type: ignore[import-not-found]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class EvalModelConfig:
    checkpoint: str = "checkpoints/best.pt"
    test_data: str = "data/training/test.npz"
    device: str = "cpu"
    batch_size: int = 256
    scan_dim: int = 64
    action_dim: int = 3


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def load_model(
    checkpoint_path: str,
    device: str = "cpu",
    scan_dim: int = 64,
    action_dim: int = 3,
) -> ResidualSafetyModel:
    """Load a trained model from checkpoint.

    Supports both full checkpoint dicts (with ``model_state_dict`` key) and
    bare state dicts.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt

    # Infer hidden dims from state dict if not provided
    hidden_dims = []
    for key in sorted(state_dict.keys()):
        if key.startswith("net.") and key.endswith(".weight"):
            dim = state_dict[key].shape[0]
            if dim != action_dim:  # skip final layer
                hidden_dims.append(dim)

    model = ResidualSafetyModel(
        scan_dim=scan_dim,
        action_dim=action_dim,
        hidden_dims=hidden_dims if hidden_dims else [128, 64, 32],
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def evaluate_model(
    checkpoint_path: str,
    test_data_path: str,
    device: str = "cpu",
    batch_size: int = 256,
    scan_dim: int = 64,
    action_dim: int = 3,
) -> dict:
    """Run full evaluation and return a metrics dict.

    Parameters
    ----------
    checkpoint_path : str
        Path to the trained model checkpoint.
    test_data_path : str
        Path to the test .npz file.
    device : str
        Torch device string.
    batch_size : int
        Batch size for inference.
    scan_dim, action_dim : int
        Dimensionality of scan and action vectors.

    Returns
    -------
    dict
        Metrics:
        - ``mse`` : float
        - ``mae`` : float
        - ``collision_rate_before`` : float
        - ``collision_rate_after`` : float
        - ``collision_rate_reduction_pct`` : float
        - ``mean_clearance_before`` : float (m)
        - ``mean_clearance_after`` : float (m)
        - ``mean_clearance_improvement_pct`` : float
        - ``action_smoothness_before`` : float
        - ``action_smoothness_after`` : float
        - ``inference_latency_ms_mean`` : float
        - ``inference_latency_ms_std`` : float
        - ``num_samples`` : int
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not os.path.isfile(test_data_path):
        raise FileNotFoundError(f"Test data not found: {test_data_path}")

    model = load_model(checkpoint_path, device, scan_dim, action_dim)
    scorer = RiskScorer()

    # Load test data
    data = dict(np.load(test_data_path, allow_pickle=True))
    scans = torch.as_tensor(data["scan"], dtype=torch.float32, device=device)
    raw_actions = torch.as_tensor(data["raw_action"], dtype=torch.float32, device=device)
    goals = torch.as_tensor(data["goal"], dtype=torch.float32, device=device)
    velocities = torch.as_tensor(data["velocity"], dtype=torch.float32, device=device)
    last_actions = torch.as_tensor(data["last_action"], dtype=torch.float32, device=device)
    delta_targets = torch.as_tensor(data["delta"], dtype=torch.float32, device=device)

    n = len(scans)

    # ------------------------------------------------------------------
    # Metrics accumulators
    # ------------------------------------------------------------------
    mse_sum = 0.0
    mae_sum = 0.0

    collisions_before = 0
    collisions_after = 0
    clearance_before_sum = 0.0
    clearance_after_sum = 0.0

    smoothness_before_sum = 0.0
    smoothness_after_sum = 0.0

    latency_list: list[float] = []

    # Process in batches
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        bs = end - start

        s = scans[start:end]
        ra = raw_actions[start:end]
        g = goals[start:end]
        v = velocities[start:end]
        la = last_actions[start:end]
        dt = delta_targets[start:end]

        # Inference latency measurement
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            delta_pred = model(s, ra, g, v, la)
        if device == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        latency_list.append((t1 - t0) / bs * 1000)  # ms per sample

        # MSE / MAE
        mse_sum += torch.nn.functional.mse_loss(delta_pred, dt, reduction="sum").item()
        mae_sum += torch.nn.functional.l1_loss(delta_pred, dt, reduction="sum").item()

        # Per-sample collision / clearance
        sa = ra + delta_pred  # safe action
        for i in range(bs):
            scan_np = s[i].cpu().numpy()
            risk = scorer.compute_risk(scan_np)
            clearance_before_sum += risk["min_distance"]

            if risk["is_emergency"]:
                collisions_before += 1

            # Simulate one step with the safe action to estimate new clearance
            # Simplified: use scan directly (the residual does not change the
            # environment in-place here; we compute a heuristic improvement).
            safe_clearance = _estimate_clearance_improvement(
                scan_np, ra[i].cpu().numpy(), sa[i].cpu().numpy(), scorer
            )
            clearance_after_sum += safe_clearance

            if safe_clearance < scorer.collision_threshold:
                collisions_after += 1

            # Smoothness: magnitude of raw action vs safe action change
            smoothness_before_sum += float(np.linalg.norm(ra[i].cpu().numpy()))
            smoothness_after_sum += float(np.linalg.norm(sa[i].cpu().numpy()))

    # ------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------
    mse = mse_sum / n
    mae = mae_sum / n
    collision_rate_before = collisions_before / n
    collision_rate_after = collisions_after / n
    collision_reduction_pct = (
        100.0 * (collisions_before - collisions_after) / max(collisions_before, 1)
    )
    mean_clearance_before = clearance_before_sum / n
    mean_clearance_after = clearance_after_sum / n
    clearance_improvement_pct = (
        100.0 * (mean_clearance_after - mean_clearance_before) / max(mean_clearance_before, 1e-6)
    )
    smoothness_before = smoothness_before_sum / n
    smoothness_after = smoothness_after_sum / n
    latency_mean = np.mean(latency_list) if latency_list else 0.0
    latency_std = np.std(latency_list) if latency_list else 0.0

    return {
        "mse": float(mse),
        "mae": float(mae),
        "collision_rate_before": float(collision_rate_before),
        "collision_rate_after": float(collision_rate_after),
        "collision_rate_reduction_pct": float(collision_reduction_pct),
        "mean_clearance_before_m": float(mean_clearance_before),
        "mean_clearance_after_m": float(mean_clearance_after),
        "mean_clearance_improvement_pct": float(clearance_improvement_pct),
        "action_smoothness_before": float(smoothness_before),
        "action_smoothness_after": float(smoothness_after),
        "inference_latency_ms_mean": float(latency_mean),
        "inference_latency_ms_std": float(latency_std),
        "num_samples": n,
    }


def _estimate_clearance_improvement(
    scan: np.ndarray,
    raw_action: np.ndarray,
    safe_action: np.ndarray,
    scorer: RiskScorer,
) -> float:
    """Heuristically estimate clearance after applying the safe action.

    The idea: if the safe action reduces forward velocity (raw vx >
    safe vx), clearance improves proportionally.  This is a rough
    approximation used for evaluation purposes.
    """
    vx_raw = float(raw_action[0])
    vx_safe = float(safe_action[0])
    risk = scorer.compute_risk(scan)
    current_clearance = risk["min_distance"]

    if vx_raw <= vx_safe:
        # No speed reduction; clearance unchanged.
        return current_clearance

    # Simulate one time-step (DT = 0.1 s) to estimate position change.
    dt = 0.1
    # The difference in forward travel over one step.
    delta_x = (vx_raw - vx_safe) * dt  # metres of additional forward movement avoided

    # Clamp: clearance cannot exceed scan_max, and improvement is bounded.
    improved = min(current_clearance + delta_x, 5.0)
    return max(improved, 0.01)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


@draccus.wrap()
def main(cfg: EvalModelConfig) -> None:  # type: ignore[no-untyped-def]
    metrics = evaluate_model(
        checkpoint_path=cfg.checkpoint,
        test_data_path=cfg.test_data,
        device=cfg.device,
        batch_size=cfg.batch_size,
        scan_dim=cfg.scan_dim,
        action_dim=cfg.action_dim,
    )
    print("\n" + "=" * 50)
    print("  Residual Safety Model - Evaluation")
    print("=" * 50)
    for key, value in metrics.items():
        print(f"  {key:<36s} : {value}")
    print("=" * 50)

    # Optionally save metrics
    metrics_path = os.path.join(os.path.dirname(cfg.checkpoint), "eval_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
