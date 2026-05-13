"""
Utility functions: YAML config loading, path resolution, FPS counter.
"""
import os
import time
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent


def load_config(config_name: str) -> dict:
    """Load a YAML config file from the configs/ directory.

    Args:
        config_name: config filename without .yaml extension (e.g., 'camera').

    Returns:
        Parsed config dictionary.
    """
    config_path = PROJECT_ROOT / "configs" / f"{config_name}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path) as f:
        return yaml.safe_load(f)


def resolve_path(path: str) -> Path:
    """Resolve a path relative to project root if not absolute."""
    p = Path(path)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


class FPSCounter:
    """Simple moving-average FPS counter."""

    def __init__(self, window: int = 30):
        self.window = window
        self._times: list[float] = []
        self._last = time.perf_counter()

    def tick(self) -> float:
        """Call once per frame. Returns current FPS."""
        now = time.perf_counter()
        self._times.append(now - self._last)
        self._last = now
        if len(self._times) > self.window:
            self._times = self._times[-self.window:]
        total = sum(self._times)
        return len(self._times) / total if total > 0 else 0.0

    @property
    def fps(self) -> float:
        if len(self._times) < 2:
            return 0.0
        total = sum(self._times)
        return len(self._times) / total if total > 0 else 0.0


class LatencyTracker:
    """Track per-stage latency."""

    def __init__(self):
        self._starts: dict[str, float] = {}
        self._measures: dict[str, list[float]] = {}

    def start(self, stage: str):
        self._starts[stage] = time.perf_counter()

    def stop(self, stage: str) -> float:
        elapsed = time.perf_counter() - self._starts.pop(stage, 0.0)
        self._measures.setdefault(stage, []).append(elapsed)
        return elapsed

    def avg(self, stage: str) -> float:
        vals = self._measures.get(stage, [])
        return sum(vals) / len(vals) if vals else 0.0

    def summary(self) -> dict[str, float]:
        return {k: sum(v) / len(v) for k, v in self._measures.items() if v}


def ensure_dir(path: Path | str):
    """Ensure directory exists, creating it if needed."""
    Path(path).mkdir(parents=True, exist_ok=True)
