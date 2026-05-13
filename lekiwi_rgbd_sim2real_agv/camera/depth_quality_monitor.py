"""
Real-time depth quality monitor for AGV perception.

Maintains a sliding window of per-frame quality metrics and provides
aggregate statistics, reliability checks, and alert generation to help
higher-level planning modules decide when depth data can be trusted.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np


class DepthQualityMonitor:
    """Monitor depth stream quality over a sliding window of frames.

    The monitor accumulates ``quality`` dictionaries produced by
    :func:`scan_quality_diagnostics` (or equivalent) and exposes:

    * Aggregate statistics over the window.
    * A binary reliability flag.
    * Human-readable alert strings for degraded conditions.

    Parameters
    ----------
    window_size:
        Number of recent frames to retain for aggregate statistics.
    invalid_ratio_threshold:
        Fraction of bins allowed to be invalid before depth is
        considered unreliable (default 0.5 = 50 %).
    dropout_ratio_threshold:
        Fraction of newly-dropped bins before a dropout alert fires
        (default 0.3 = 30 %).
    """

    # ------------------------------------------------------------------
    # Predefined quality thresholds
    # ------------------------------------------------------------------
    DEFAULT_INVALID_RATIO_THRESHOLD: float = 0.50
    DEFAULT_DROPOUT_RATIO_THRESHOLD: float = 0.30

    def __init__(
        self,
        window_size: int = 30,
        invalid_ratio_threshold: float | None = None,
        dropout_ratio_threshold: float | None = None,
    ) -> None:
        if window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}")

        self.window_size: int = window_size
        self.invalid_ratio_threshold: float = (
            invalid_ratio_threshold
            if invalid_ratio_threshold is not None
            else self.DEFAULT_INVALID_RATIO_THRESHOLD
        )
        self.dropout_ratio_threshold: float = (
            dropout_ratio_threshold
            if dropout_ratio_threshold is not None
            else self.DEFAULT_DROPOUT_RATIO_THRESHOLD
        )

        self.history: List[Dict] = []
        self.warnings: List[str] = []
        self._frame_count: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, quality_dict: Dict) -> None:
        """Ingest a new quality dictionary and maintain the sliding window.

        Parameters
        ----------
        quality_dict:
            A dict with (at minimum) keys ``"invalid_ratio"``,
            ``"dropout_ratio"``, and ``"min_range_over_time"``, as
            produced by :func:`scan_quality_diagnostics`.
        """
        self.history.append(dict(quality_dict))
        if len(self.history) > self.window_size:
            self.history = self.history[-self.window_size :]
        self._frame_count += 1

    def is_reliable(self) -> bool:
        """Return ``True`` if depth data is currently considered reliable.

        Reliability is determined by the *mean* invalid ratio over the
        sliding window.  If the mean exceeds ``invalid_ratio_threshold``
        the stream is treated as unreliable.

        Returns
        -------
        bool
        """
        if not self.history:
            # No data yet — assume unreliable until proven otherwise.
            return False

        ratios = [
            d["invalid_ratio"]
            for d in self.history
            if "invalid_ratio" in d and not np.isnan(d["invalid_ratio"])
        ]
        if not ratios:
            return False

        mean_invalid = float(np.mean(ratios))
        return mean_invalid < self.invalid_ratio_threshold

    def get_stats(self) -> Dict:
        """Return aggregate statistics over the sliding window.

        Returns
        -------
        dict
            Keys:

            - ``num_frames``: frames in the window.
            - ``total_frames``: total frames ingested since creation.
            - ``mean_invalid_ratio``: average fraction of invalid
              scan bins.
            - ``mean_dropout_ratio``: average fraction of bins that
              newly dropped out between consecutive frames.
            - ``min_min_range``: global minimum range across the window.
            - ``max_invalid_ratio``: worst-case invalid ratio in window.
        """
        n = len(self.history)
        if n == 0:
            return {
                "num_frames": 0,
                "total_frames": self._frame_count,
                "mean_invalid_ratio": float("nan"),
                "mean_dropout_ratio": float("nan"),
                "min_min_range": float("inf"),
                "max_invalid_ratio": float("nan"),
            }

        invalid_vals = [
            d.get("invalid_ratio", float("nan")) for d in self.history
        ]
        dropout_vals = [
            d.get("dropout_ratio", float("nan")) for d in self.history
        ]
        min_range_vals = [
            d.get("min_range_over_time", float("inf")) for d in self.history
        ]

        invalid_arr = np.array(invalid_vals, dtype=np.float64)
        dropout_arr = np.array(dropout_vals, dtype=np.float64)
        min_range_arr = np.array(min_range_vals, dtype=np.float64)

        return {
            "num_frames": n,
            "total_frames": self._frame_count,
            "mean_invalid_ratio": float(np.nanmean(invalid_arr)) if n > 0 else float("nan"),
            "mean_dropout_ratio": float(np.nanmean(dropout_arr)) if n > 0 else float("nan"),
            "min_min_range": float(np.nanmin(min_range_arr)) if n > 0 else float("inf"),
            "max_invalid_ratio": float(np.nanmax(invalid_arr)) if n > 0 else float("nan"),
        }

    def check_alerts(self) -> List[str]:
        """Generate alert strings describing any degraded quality conditions.

        Alerts are generated when the latest frame's invalid ratio or
        dropout ratio crosses the configured thresholds.

        Returns
        -------
        list[str]
            Human-readable alert strings (empty if quality is nominal).
        """
        alerts: List[str] = []

        if not self.history:
            return alerts

        latest = self.history[-1]

        invalid_ratio = latest.get("invalid_ratio", 0.0)
        dropout_ratio = latest.get("dropout_ratio", 0.0)
        min_range = latest.get("min_range_over_time", float("inf"))

        # Check invalid ratio.
        if (
            not np.isnan(invalid_ratio)
            and invalid_ratio >= self.invalid_ratio_threshold
        ):
            alerts.append(
                f"High invalid ratio: {invalid_ratio:.2%} "
                f"(threshold: {self.invalid_ratio_threshold:.2%})"
            )

        # Check dropout ratio.
        if (
            not np.isnan(dropout_ratio)
            and dropout_ratio >= self.dropout_ratio_threshold
        ):
            alerts.append(
                f"Sudden depth dropout: {dropout_ratio:.2%} "
                f"(threshold: {self.dropout_ratio_threshold:.2%})"
            )

        # Check for dangerously close obstacles across the window.
        if min_range < 0.3:
            alerts.append(
                f"Obstacle very close: min range = {min_range:.2f} m"
            )

        # Track cumulative warnings.
        self.warnings.extend(alerts)
        # Keep warnings bounded.
        if len(self.warnings) > 100:
            self.warnings = self.warnings[-100:]

        return alerts

    def reset(self) -> None:
        """Clear all accumulated history and warnings."""
        self.history.clear()
        self.warnings.clear()
        self._frame_count = 0
