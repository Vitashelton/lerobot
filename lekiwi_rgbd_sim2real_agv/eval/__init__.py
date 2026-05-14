"""Evaluation: offline metrics, real-robot validation, latency benchmarks."""

from eval.offline_evaluator import OfflineEvaluator
from eval.metrics import compute_all_metrics

__all__ = ["OfflineEvaluator", "compute_all_metrics"]
