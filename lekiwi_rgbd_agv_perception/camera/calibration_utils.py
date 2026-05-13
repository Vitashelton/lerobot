"""
Camera calibration utilities for RealSense D435i.

Provides:
- Intrinsics extraction from RealSense profile
- Depth-to-3D projection helpers
- Simple stereo rectification (placeholder)
"""
import numpy as np


def intrinsics_to_matrix(intrinsics: dict) -> np.ndarray:
    """Convert intrinsics dict to 3x3 camera matrix K.

    Args:
        intrinsics: dict with fx, fy, cx, cy.

    Returns:
        (3, 3) float32 camera matrix.
    """
    K = np.eye(3, dtype=np.float32)
    K[0, 0] = intrinsics["fx"]
    K[1, 1] = intrinsics["fy"]
    K[0, 2] = intrinsics["cx"]
    K[1, 2] = intrinsics["cy"]
    return K


def intrinsics_from_matrix(K: np.ndarray) -> dict:
    """Extract intrinsics dict from a 3x3 camera matrix."""
    return {
        "fx": float(K[0, 0]),
        "fy": float(K[1, 1]),
        "cx": float(K[0, 2]),
        "cy": float(K[1, 2]),
        "width": int(K[0, 2] * 2),
        "height": int(K[1, 2] * 2),
        "model": "pinhole",
    }


def get_default_intrinsics(width: int = 640, height: int = 480) -> dict:
    """Return approximate D435i intrinsics for the given resolution."""
    # D435i color sensor ~69° HFOV x 42° VFOV
    fx = width * 1.4   # approximate
    fy = height * 1.4
    return {
        "fx": fx, "fy": fy,
        "cx": width / 2.0, "cy": height / 2.0,
        "width": width, "height": height,
        "model": "pinhole",
    }
