"""
Convert a depth image into a 1D polar scan representation for AGV perception.

The module extracts a horizontal band from the depth image, divides it
into angular bins, and pools each bin into a single range value (by
default the 10th percentile to suppress outliers).  It also provides
spatial sector analysis, temporal smoothing, and quality diagnostics.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np


def percentile_pool(bin_values: np.ndarray, percentile: float = 10.0) -> float:
    """Return the *p*-th percentile of a bin's depth values.

    This is more robust than taking the minimum, which is sensitive to
    speckle noise and single-pixel drop-outs common in stereo / active-IR
    depth sensors.

    Parameters
    ----------
    bin_values:
        1-D array of depth values (meters) belonging to one angular bin.
        May contain ``NaN`` entries which are ignored.
    percentile:
        Percentile to compute (0-100).  Default 10 (= near-minimum).

    Returns
    -------
    float
        The specified percentile of the valid values, or ``NaN`` if the
        bin has no valid pixels.
    """
    valid = bin_values[~np.isnan(bin_values)]
    if len(valid) == 0:
        return np.nan
    if len(valid) == 1:
        return float(valid[0])
    return float(np.percentile(valid, percentile))


def depth_to_scan_polar(
    depth_m: np.ndarray,
    scan_dim: int = 64,
    fov_horizontal_deg: float = 87.0,
    slice_fraction: float = 0.3,
    percentile: float = 10.0,
) -> np.ndarray:
    """Convert a depth image into a 1-D polar scan of ``scan_dim`` angular bins.

    Only a horizontal band in the middle of the image is used (specified
    by ``slice_fraction``).  The band is divided into ``scan_dim`` equal
    angular bins across ``fov_horizontal_deg``, and each bin is pooled
    via ``percentile_pool``.

    Parameters
    ----------
    depth_m:
        Processed depth image in meters, shape ``(H, W)``, dtype
        ``float32``.  Invalid pixels should be ``NaN``.
    scan_dim:
        Number of angular bins in the output scan.
    fov_horizontal_deg:
        Horizontal field of view of the depth sensor in degrees.
        D435i is ~87deg (HFOV of the depth stream at 848x480).
    slice_fraction:
        Fraction of the image height to use (centered vertically).
        E.g. 0.3 means take the middle 30 % of rows.

    Returns
    -------
    np.ndarray
        1-D array of shape ``(scan_dim,)``, dtype ``float32``, where
        each element is the pooled depth (meters) in that angular sector.
        Bins with no valid pixels are ``NaN``.
    """
    h, w = depth_m.shape
    if h < 3 or w < 3:
        raise ValueError(
            f"Depth image too small for scan extraction: {(h, w)}"
        )

    # Extract the horizontal band.
    band_half = max(1, int(h * slice_fraction / 2.0))
    row_center = h // 2
    r_start = max(0, row_center - band_half)
    r_end = min(h, row_center + band_half)
    band = depth_m[r_start:r_end, :]  # shape (band_h, W)

    # Divide the band columns into scan_dim angular bins.
    # Each column corresponds to an angle that is linearly distributed
    # across the HFOV from left (-fov/2) to right (+fov/2).
    bin_edges = np.linspace(0, w, scan_dim + 1, dtype=np.int32)
    meter_scan = np.full(scan_dim, np.nan, dtype=np.float32)

    for i in range(scan_dim):
        col_start = bin_edges[i]
        col_end = bin_edges[i + 1]
        if col_end <= col_start:
            continue
        bin_pixels = band[:, col_start:col_end].ravel()
        meter_scan[i] = percentile_pool(bin_pixels, percentile=percentile)

    return meter_scan


def compute_sector_mins(scan_m: np.ndarray) -> Dict[str, float]:
    """Split the scan into left / center / right thirds and compute the
    minimum distance in each sector.

    Parameters
    ----------
    scan_m:
        Polar scan array of shape ``(N,)`` in meters.

    Returns
    -------
    dict
        Keys ``"front_min"``, ``"left_min"``, ``"right_min"``.  Each
        value is the minimum distance (meters) in that sector, or
        ``float("inf")`` if all bins are invalid.
    """
    n = len(scan_m)
    if n < 3:
        # Degenerate case: treat everything as front.
        valid = scan_m[~np.isnan(scan_m)]
        val = float(np.min(valid)) if len(valid) > 0 else float("inf")
        return {"front_min": val, "left_min": val, "right_min": val}

    third = max(1, n // 3)
    left = scan_m[:third]
    center = scan_m[third : 2 * third]
    right = scan_m[2 * third :]

    def _safe_min(arr: np.ndarray) -> float:
        valid = arr[~np.isnan(arr)]
        return float(np.min(valid)) if len(valid) > 0 else float("inf")

    return {
        "front_min": _safe_min(center),
        "left_min": _safe_min(left),
        "right_min": _safe_min(right),
    }


def temporal_smooth(
    scan_m: np.ndarray,
    prev_scan: np.ndarray | None,
    alpha: float = 0.5,
) -> np.ndarray:
    """Apply exponential moving average smoothing across consecutive scans.

    Per-bin smoothing::

        smoothed[i] = alpha * scan[i] + (1 - alpha) * prev[i]

    Parameters
    ----------
    scan_m:
        Current scan of shape ``(N,)`` in meters.
    prev_scan:
        Previous smoothed scan of shape ``(N,)``, or ``None`` on the
        first call (returns ``scan_m`` unchanged).
    alpha:
        Smoothing factor in (0, 1].  Higher = more weight on new data.

    Returns
    -------
    np.ndarray
        Temporally smoothed scan, same shape/dtype as ``scan_m``.
    """
    scan_m = np.asarray(scan_m, dtype=np.float32)
    if prev_scan is None:
        return scan_m.copy()

    prev_scan = np.asarray(prev_scan, dtype=np.float32)
    if scan_m.shape != prev_scan.shape:
        raise ValueError(
            f"Scan shape mismatch: {scan_m.shape} vs prev {prev_scan.shape}"
        )

    alpha = float(np.clip(alpha, 0.0, 1.0))

    # Handle NaN: if either frame has NaN, use the other's value.
    mask_cur_nan = np.isnan(scan_m)
    mask_prev_nan = np.isnan(prev_scan)

    smoothed = np.where(
        mask_cur_nan & ~mask_prev_nan,
        prev_scan,
        np.where(
            ~mask_cur_nan & mask_prev_nan,
            scan_m,
            np.where(
                mask_cur_nan & mask_prev_nan,
                np.nan,
                alpha * scan_m + (1.0 - alpha) * prev_scan,
            ),
        ),
    )
    return smoothed.astype(np.float32)


def scan_quality_diagnostics(
    scan_m: np.ndarray,
    prev_scans: List[np.ndarray] | None = None,
) -> Dict:
    """Compute real-time quality diagnostics for the current scan.

    Parameters
    ----------
    scan_m:
        Current polar scan, shape ``(N,)``, dtype ``float32``.
    prev_scans:
        List of previous scans (most recent first) for dropout and
        trend analysis.  May be empty or ``None``.

    Returns
    -------
    dict
        Keys:

        - ``invalid_ratio`` (float): fraction of bins that are NaN/inf.
        - ``dropout_ratio`` (float): fraction of bins that transitioned
          from valid to invalid relative to the most recent previous scan
          (0.0 if no previous data).
        - ``min_range_over_time`` (float): minimum distance across the
          history window (meters).
    """
    scan = np.asarray(scan_m, dtype=np.float32)
    n = len(scan)

    invalid = np.isnan(scan) | np.isinf(scan)
    invalid_ratio = float(np.mean(invalid)) if n > 0 else 1.0

    # Dropout ratio: bins that were valid in prev but became invalid now.
    dropout_ratio = 0.0
    if prev_scans and len(prev_scans) > 0:
        prev = np.asarray(prev_scans[0], dtype=np.float32)
        prev_valid = ~np.isnan(prev) & ~np.isinf(prev)
        now_invalid = invalid
        dropout = prev_valid & now_invalid
        total_prev_valid = int(np.sum(prev_valid))
        if total_prev_valid > 0:
            dropout_ratio = float(np.sum(dropout)) / total_prev_valid

    # Minimum range over history.
    all_valid = scan[~invalid]
    if prev_scans:
        for ps in prev_scans:
            ps_arr = np.asarray(ps, dtype=np.float32)
            ps_valid = ps_arr[~np.isnan(ps_arr) & ~np.isinf(ps_arr)]
            if len(ps_valid) > 0:
                all_valid = (
                    np.concatenate([all_valid, ps_valid])
                    if len(all_valid) > 0
                    else ps_valid
                )
    min_range = float(np.min(all_valid)) if len(all_valid) > 0 else float("inf")

    return {
        "invalid_ratio": invalid_ratio,
        "dropout_ratio": dropout_ratio,
        "min_range_over_time": min_range,
    }


def depth_to_scan_pipeline(
    depth_m: np.ndarray,
    scan_dim: int = 64,
    fov_horizontal_deg: float = 87.0,
    slice_fraction: float = 0.3,
    percentile: float = 10.0,
    alpha: float = 0.5,
    prev_scan: np.ndarray | None = None,
    prev_scans: List[np.ndarray] | None = None,
) -> Dict:
    """Full pipeline: depth image -> polar scan + sector analysis + diagnostics.

    Parameters
    ----------
    depth_m:
        Preprocessed depth image in meters, shape ``(H, W)``.
    scan_dim:
        Number of angular bins.
    fov_horizontal_deg:
        Horizontal field of view in degrees.
    slice_fraction:
        Vertical fraction of the image to use for scan extraction.
    percentile:
        Percentile for bin pooling (default 10).
    alpha:
        EMA smoothing factor for temporal smoothing.
    prev_scan:
        Previous smoothed scan for temporal smoothing, or ``None``.
    prev_scans:
        Previous raw scans for quality diagnostics, or ``None``.

    Returns
    -------
    dict
        Keys:

        - ``scan_m``: raw polar scan ``(scan_dim,)``, float32, meters.
        - ``scan_smoothed``: temporally smoothed scan.
        - ``front_min``, ``left_min``, ``right_min``: sector minimums.
        - ``quality``: dict from ``scan_quality_diagnostics``.
    """
    # Step 1: convert depth image to polar scan.
    scan_raw = depth_to_scan_polar(
        depth_m,
        scan_dim=scan_dim,
        fov_horizontal_deg=fov_horizontal_deg,
        slice_fraction=slice_fraction,
        percentile=percentile,
    )

    # Step 2: temporal smoothing.
    scan_smoothed = temporal_smooth(scan_raw, prev_scan, alpha=alpha)

    # Step 3: sector analysis.
    sectors = compute_sector_mins(scan_smoothed)

    # Step 4: quality diagnostics.
    quality = scan_quality_diagnostics(scan_raw, prev_scans=prev_scans)

    return {
        "scan_m": scan_raw,
        "scan_smoothed": scan_smoothed,
        "front_min": sectors["front_min"],
        "left_min": sectors["left_min"],
        "right_min": sectors["right_min"],
        "quality": quality,
    }
