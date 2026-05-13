"""
Placeholder: Residual safety model evaluation.

In Phase 2, this module will evaluate the trained residual safety model
on held-out synthetic scenes and on MuJoCo rollouts.

Metrics (planned):
    - AUC-ROC / AUC-PR for collision prediction
    - False-positive rate at 95 % recall
    - Mean time-to-collision (TTC) estimates vs ground truth
    - Policy degradation: how much does the safety filter slow progress?

Evaluation protocol:
    1. Load exported TorchScript model.
    2. Run N episodes in LeKiwiScanEnv (all four worlds).
    3. At each step, query the residual model.
    4. If residual risk > threshold, override action (zero velocity).
    5. Report: success rate, avg steps to goal, collision rate,
       intervention rate, and spatial heatmaps of interventions.

Usage (planned):
    python -m sim.mujoco_lekiwi.evaluate_residual \\
        --model_path outputs/residual_model.pt \\
        --num_episodes 200 \\
        --render

Currently stubbed out.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def main():
    """Placeholder entry point for residual model evaluation."""
    logger.info(
        "Residual safety model evaluation is not yet implemented.\n"
        "This will be added in Phase 2.\n"
        "For environment testing, use:\n"
        "    python -c \"from sim.mujoco_lekiwi.envs import LeKiwiScanEnv; "
        "env = LeKiwiScanEnv(world='lab_empty'); "
        "obs, _ = env.reset(); print(obs.shape)\""
    )


if __name__ == "__main__":
    main()
