"""Baseline methods for comparison.

1. Behavior Cloning (BC)
2. DWA traditional baseline
3. TD3+BC without multimodal fusion (ablation)
4. IQL without safety filter (ablation)
"""

from baselines.behavior_cloning import BehaviorCloning
from baselines.dwa_baseline import DWABaseline

__all__ = ["BehaviorCloning", "DWABaseline"]
