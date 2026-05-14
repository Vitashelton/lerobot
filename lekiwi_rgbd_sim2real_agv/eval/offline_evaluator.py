"""Offline evaluation on held-out dataset."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch
from tqdm import tqdm

from eval.metrics import compute_all_metrics


class OfflineEvaluator:
    """Evaluate trained policies on offline data.

    Parameters
    ----------
    trainer : TD3BC or IQL or BehaviorCloning
        Trained policy (must have a ``predict`` method).
    safety_filter : SafetyFilter or None
        Optional safety filter to apply during evaluation.
    device : str
    """

    def __init__(
        self,
        trainer,
        safety_filter=None,
        device: str = "cuda",
    ) -> None:
        self.trainer = trainer
        self.safety_filter = safety_filter
        self.device = device

    def evaluate(
        self,
        eval_data: Dict[str, Any],
        batch_size: int = 256,
        use_safety_filter: bool = False,
        verbose: bool = True,
    ) -> Dict[str, float]:
        """Evaluate policy on entire evaluation dataset.

        Parameters
        ----------
        eval_data : dict
            Unified format dataset (evaluation split).
        batch_size : int
            Batch size for inference.
        use_safety_filter : bool
            Whether to apply safety filter.
        verbose : bool
            Show progress bar.

        Returns
        -------
        dict
            Metric name → value.
        """
        obs = eval_data["observations"]
        true_actions = eval_data["actions"]
        T = len(true_actions)

        all_pred_actions = np.zeros((T, 3), dtype=np.float32)
        all_q_values = np.zeros(T, dtype=np.float32)

        # Batch inference
        n_batches = (T + batch_size - 1) // batch_size
        iterator = range(n_batches)
        if verbose:
            iterator = tqdm(iterator, desc="Evaluating")

        for b in iterator:
            start = b * batch_size
            end = min(start + batch_size, T)

            # Prepare observation batch
            rgb = None
            if obs.get("rgb") is not None:
                rgb_batch = obs["rgb"][start:end]
                rgb = torch.from_numpy(rgb_batch).float() / 255.0
                if rgb.dim() == 4:  # (B, H, W, C) → (B, C, H, W)
                    rgb = rgb.permute(0, 3, 1, 2)
                rgb = rgb.to(self.device)

            scan64 = None
            if obs.get("scan64") is not None:
                s = obs["scan64"][start:end]
                s = np.nan_to_num(s, nan=5.0)
                scan64 = torch.from_numpy(s).float().to(self.device)

            state = None
            if obs.get("state") is not None:
                state = torch.from_numpy(obs["state"][start:end]).float().to(self.device)

            goal = None
            if obs.get("goal") is not None:
                goal = torch.from_numpy(obs["goal"][start:end]).float().to(self.device)

            # Predict
            pred = self.trainer.predict(rgb, scan64, state, goal)

            # Apply safety filter
            if use_safety_filter and self.safety_filter is not None:
                for i in range(len(pred)):
                    s_i = obs.get("scan64", np.zeros((T, 64)))[start + i]
                    safe_action, _ = self.safety_filter.filter(pred[i], s_i)
                    pred[i] = safe_action

            all_pred_actions[start:end] = pred

            # Q-values (only for RL methods)
            if hasattr(self.trainer, "get_q_value"):
                for i in range(len(pred)):
                    act_t = torch.from_numpy(true_actions[start + i : start + i + 1]).float().to(self.device)
                    try:
                        q = self.trainer.get_q_value(
                            rgb[i:i+1] if rgb is not None else None,
                            scan64[i:i+1] if scan64 is not None else None,
                            state[i:i+1] if state is not None else None,
                            goal[i:i+1] if goal is not None else None,
                            act_t,
                        )
                        all_q_values[start + i] = q
                    except Exception:
                        pass

        # Compute metrics
        rewards = eval_data.get("rewards")
        dones = eval_data.get("dones")
        scan64_all = obs.get("scan64")
        collision_labels = eval_data.get("info", {}).get("collision")
        goal_reached = eval_data.get("info", {}).get("goal_reached")

        metrics = compute_all_metrics(
            true_actions=true_actions,
            pred_actions=all_pred_actions,
            rewards=rewards,
            dones=dones,
            scan64=scan64_all,
            collision_labels=collision_labels,
            goal_reached=goal_reached,
            q_values=all_q_values,
        )

        return metrics
