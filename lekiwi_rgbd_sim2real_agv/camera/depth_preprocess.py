"""
Depth preprocessing pipeline for Intel RealSense D435i.

Converts raw ``uint16`` depth maps (millimeters) into clean, filtered
``float32`` depth images in meters suitable for downstream perception
tasks (e.g., polar scan extraction, obstacle avoidance).

The typical call sequence is::

    depth_m = preprocess_pipeline(depth_raw)
"""

from __future__ import annotations

import cv2
import numpy as np


def raw_to_meters(depth_raw: np.ndarray) -> np.ndarray:
    """Convert raw ``uint16`` depth (millimeters) to ``float32`` meters.

    Parameters
    ----------
    depth_raw:
        Raw depth image from the RealSense pipeline, shape ``(H, W)``,
        dtype ``uint16``.  Zero values indicate invalid / no-return
        regions.

    Returns
    -------
    np.ndarray
        Depth image in meters, shape ``(H, W)``, dtype ``float32``.
    """
    return depth_raw.astype(np.float32) * 0.001


def filter_invalid(
    depth_m: np.ndarray,
    min_range: float = 0.15,
    max_range: float = 5.0,
) -> np.ndarray:
    """Set invalid pixels (zero, NaN, inf, out-of-range) to ``NaN``.

    Parameters
    ----------
    depth_m:
        Depth image in meters, shape ``(H, W)``, dtype ``float32``.
    min_range:
        Minimum valid depth in meters.  Values below this are masked out.
    max_range:
        Maximum valid depth in meters.  Values above this are masked out.

    Returns
    -------
    np.ndarray
        Depth image with invalid regions set to ``NaN``, same shape/dtype.
    """
    # Work on a copy to avoid mutating the caller's array.
    out = depth_m.copy()

    # Vectorised mask: zero, NaN, inf, sub-minimum, supra-maximum.
    invalid = (
        (out <= 0.0)
        | np.isnan(out)
        | np.isinf(out)
        | (out < min_range)
        | (out > max_range)
    )
    out[invalid] = np.nan
    return out


def clamp_range(
    depth_m: np.ndarray,
    min_range: float = 0.15,
    max_range: float = 5.0,
) -> np.ndarray:
    """Clamp depth values to ``[min_range, max_range]`` (non-destructive).

    NaN pixels are left untouched.

    Parameters
    ----------
    depth_m:
        Depth image in meters, shape ``(H, W)``, dtype ``float32``.
    min_range:
        Lower bound in meters.
    max_range:
        Upper bound in meters.

    Returns
    -------
    np.ndarray
        Clamped depth image, same shape/dtype.
    """
    out = depth_m.copy()
    valid = ~np.isnan(out)
    np.clip(out, min_range, max_range, out=out, where=valid)
    return out


def median_filter(depth_m: np.ndarray, ksize: int = 5) -> np.ndarray:
    """Apply an OpenCV median filter, propagating ``NaN`` through the
    neighbourhood correctly.

    Because ``cv2.medianBlur`` does not handle ``NaN``, we first replace
    ``NaN`` with a large sentinel, filter, and then re-insert ``NaN`` for
    any pixel whose original neighbourhood was entirely invalid.

    Parameters
    ----------
    depth_m:
        Depth image in meters, shape ``(H, W)``, dtype ``float32``.
    ksize:
        Kernel size (must be odd, >= 3).

    Returns
    -------
    np.ndarray
        Median-filtered depth image, same shape/dtype.
    """
    if ksize < 3 or ksize % 2 == 0:
        raise ValueError(f"ksize must be an odd integer >= 3, got {ksize}.")

    # If the entire image is NaN we cannot meaningfully filter.
    if np.all(np.isnan(depth_m)):
        return depth_m.copy()

    nan_mask = np.isnan(depth_m)

    # Replace NaN with a value far outside the valid range.
    sentinel = -9999.0
    filled = np.where(nan_mask, sentinel, depth_m.astype(np.float32))

    # Apply median filter.  cv2.medianBlur works on float32 in OpenCV >= 4.
    blurred = cv2.medianBlur(filled, ksize)

    # Build a mask indicating where the median filter received *only*
    # sentinel (NaN) values in its kernel, i.e. the entire neighbourhood
    # was invalid.
    sentinel_mask = (filled == sentinel).astype(np.uint8)
    kernel = np.ones((ksize, ksize), dtype=np.uint8)
    neighbour_sentinel_count = cv2.filter2D(sentinel_mask, -1, kernel, borderType=cv2.BORDER_REPLICATE)
    all_invalid = neighbour_sentinel_count >= (ksize * ksize)

    blurred[all_invalid] = np.nan
    return blurred


def hole_filling(depth_m: np.ndarray) -> np.ndarray:
    """Fill small holes in the depth image using nearest-neighbour
    inpainting.

    Parameters
    ----------
    depth_m:
        Depth image in meters, shape ``(H, W)``, dtype ``float32``.
        NaN pixels are treated as holes.

    Returns
    -------
    np.ndarray
        Depth image with small holes filled via inpainting, same
        shape/dtype.
    """
    if np.all(np.isnan(depth_m)):
        return depth_m.copy()

    # Convert NaN mask to uint8 inpainting mask (0 = good, 255 = hole).
    hole_mask = np.isnan(depth_m).astype(np.uint8) * 255

    # Replace NaN with 0 for the inpainting algorithm input.
    src = np.where(np.isnan(depth_m), 0.0, depth_m).astype(np.float32)

    # Nearest-neighbour (NS) inpainting.  A radius of 3 pixels is a
    # reasonable default for filling small sensor drop-outs.
    inpainted = cv2.inpaint(src, hole_mask, inpaintRadius=3, flags=cv2.INPAINT_NS)

    # After inpainting, re-apply NaN for any pixels that are still at
    # zero (no valid neighbour found).
    inpainted[inpainted <= 0.0] = np.nan

    return inpainted  # type: ignore[return-value]


def preprocess_pipeline(
    depth_raw: np.ndarray,
    min_range: float = 0.15,
    max_range: float = 5.0,
    median_ksize: int = 5,
    hole_fill: bool = True,
) -> np.ndarray:
    """Full depth preprocessing pipeline.

    Processing steps (in order):

    1. Convert raw ``uint16`` (mm) to ``float32`` (meters).
    2. Mask out invalid / out-of-range pixels (set to ``NaN``).
    3. Clamp remaining values to ``[min_range, max_range]``.
    4. Apply a median filter to suppress salt-and-pepper noise.
    5. Optionally fill small holes via nearest-neighbour inpainting.

    Parameters
    ----------
    depth_raw:
        Raw depth image of shape ``(H, W)``, dtype ``uint16``.
    min_range:
        Minimum valid depth in meters (default 0.15 m).
    max_range:
        Maximum valid depth in meters (default 5.0 m).
    median_ksize:
        Kernel size for the median filter (must be odd, >= 3).
    hole_fill:
        If ``True``, apply hole-filling inpainting as the final step.

    Returns
    -------
    np.ndarray
        Processed depth image of shape ``(H, W)``, dtype ``float32``,
        in meters.  Remaining holes are ``NaN``.
    """
    # Step 1: unit conversion.
    depth_m = raw_to_meters(depth_raw)

    # Step 2: mark invalid regions.
    depth_m = filter_invalid(depth_m, min_range=min_range, max_range=max_range)

    # Step 3: clamp the valid range.
    depth_m = clamp_range(depth_m, min_range=min_range, max_range=max_range)

    # Step 4: median denoising.
    depth_m = median_filter(depth_m, ksize=median_ksize)

    # Step 5: optional hole-filling.
    if hole_fill:
        depth_m = hole_filling(depth_m)

    return depth_m
