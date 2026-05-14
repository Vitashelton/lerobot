"""Navigation evaluation metrics."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np


def compute_all_metrics(
    true_actions: np.ndarray,
    pred_actions: np.ndarray,
    rewards: np.ndarray | None = None,
    dones: np.ndarray | None = None,
    scan64: np.ndarray | None = None,
    collision_labels: np.ndarray | None = None,
    goal_reached: np.ndarray | None = None,
    q_values: np.ndarray | None = None,
) -> Dict[str, float]:
    """Compute all offline evaluation metrics.

    Parameters
    ----------
    true_actions : (T, 3)
        Dataset actions.
    pred_actions : (T, 3)
        Predicted actions from the policy.
    rewards : (T,) or None
        Per-timestep rewards.
    dones : (T,) or None
        Episode termination flags.
    scan64 : (T, 64) or None
        Scan64 observations for collision risk.
    collision_labels : (T,) or None
        Ground-truth collision flags.
    goal_reached : (T,) or None
        Goal reached flags.
    q_values : (T,) or None
        Q-values for Q-value distribution analysis.

    Returns
    -------
    dict
        Metric name → value.
    """
    T = len(true_actions)
    metrics: Dict[str, float] = {}

    # Action MSE
    metrics["action_mse"] = float(np.mean((true_actions - pred_actions) ** 2))

    # Action smoothness (mean absolute difference)
    if T > 1:
        true_smooth = np.mean(np.abs(np.diff(true_actions, axis=0)))
        pred_smooth = np.mean(np.abs(np.diff(pred_actions, axis=0)))
        metrics["action_smoothness_true"] = float(true_smooth)
        metrics["action_smoothness_pred"] = float(pred_smooth)
        metrics["action_smoothness_diff"] = float(abs(true_smooth - pred_smooth))

    # OOD action deviation (max deviation)
    metrics["action_max_deviation"] = float(
        np.max(np.abs(true_actions - pred_actions))
    )

    # Return
    if rewards is not None:
        metrics["mean_reward"] = float(np.mean(rewards))
        metrics["total_return"] = float(np.sum(rewards))
        if dones is not None:
            # Per-episode return
            ep_returns = []
            ep_return = 0.0
            for t in range(T):
                ep_return += rewards[t]
                if dones[t]:
                    ep_returns.append(ep_return)
                    ep_return = 0.0
            if ep_returns:
                metrics["mean_episode_return"] = float(np.mean(ep_returns))

    # Collision risk
    if scan64 is not None:
        n = scan64.shape[1]
        front = scan64[:, max(0, n // 2 - 8) : min(n, n // 2 + 8)]
        front_min = np.nanmin(front, axis=1)
        metrics["collision_risk_ratio"] = float(
            np.mean(front_min < 0.15)
        )
        metrics["min_obstacle_distance_mean"] = float(np.nanmean(front_min))

    # Unsafe action rate: action that would move toward close obstacle
    if scan64 is not None and T > 0:
        unsafe_count = 0
        front = scan64[:, max(0, n // 2 - 8) : min(n, n // 2 + 8)]
        front_min = np.nanmin(front, axis=1)
        for t in range(T):
            # Forward motion while obstacle is critically close
            if pred_actions[t, 0] > 0.01 and front_min[t] < 0.3:
                unsafe_count += 1
            # Lateral motion toward close side obstacle
            left_min = np.nanmin(scan64[t, : n // 3])
            right_min = np.nanmin(scan64[t, 2 * n // 3 :])
            if pred_actions[t, 1] > 0.01 and left_min < 0.3:
                unsafe_count += 1
            if pred_actions[t, 1] < -0.01 and right_min < 0.3:
                unsafe_count += 1
        metrics["unsafe_action_rate"] = float(unsafe_count / max(T, 1))

    # Success proxy
    if goal_reached is not None:
        metrics["success_rate"] = float(np.mean(goal_reached))

    # Q-value distribution
    if q_values is not None:
        metrics["q_value_mean"] = float(np.mean(q_values))
        metrics["q_value_std"] = float(np.std(q_values))
        metrics["q_value_min"] = float(np.min(q_values))
        metrics["q_value_max"] = float(np.max(q_values))

    return metrics
