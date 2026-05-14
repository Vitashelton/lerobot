"""Reward relabeling for offline navigation datasets.

Computes dense rewards from trajectory data: progress toward goal,
obstacle avoidance, action smoothness, collision and intervention penalties.
"""

from reward.reward_calculator import RewardCalculator
from reward.progress_estimator import ProgressEstimator
from reward.collision_detector import CollisionDetector
from reward.intervention_labeler import InterventionLabeler

__all__ = [
    "RewardCalculator",
    "ProgressEstimator",
    "CollisionDetector",
    "InterventionLabeler",
]
