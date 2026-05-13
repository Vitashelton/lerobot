"""
Depth-to-3D point cloud utilities.

- Pixel + depth -> camera 3D coordinates
- Depth image -> point cloud (Nx3)
- Downsampling and PLY export
"""
import logging
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def pixel_to_camera(
    u: float, v: float, z: float, intrinsics: dict,
) -> tuple[float, float, float]:
    """Project a pixel coordinate and depth to camera 3D coordinates.

    Args:
        u, v: pixel coordinates (column, row).
        z: depth in meters.
        intrinsics: dict with fx, fy, cx, cy.

    Returns:
        (X, Y, Z) in camera coordinate frame:
            X = right, Y = down, Z = forward.
    """
    fx, fy = intrinsics["fx"], intrinsics["fy"]
    cx, cy = intrinsics["cx"], intrinsics["cy"]
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return float(x), float(y), float(z)


def depth_to_pointcloud(
    depth: np.ndarray,
    intrinsics: dict,
    mask: Optional[np.ndarray] = None,
    downsample: int = 1,
) -> np.ndarray:
    """Convert depth image to 3D point cloud.

    Args:
        depth: HxW float32 depth in meters.
        intrinsics: dict with fx, fy, cx, cy.
        mask: optional boolean mask of valid pixels.
        downsample: stride for downsampling (1 = full, 2 = half, etc.).

    Returns:
        Nx3 float32 array of (X, Y, Z) camera coordinates.
    """
    h, w = depth.shape
    fx, fy = intrinsics["fx"], intrinsics["fy"]
    cx, cy = intrinsics["cx"], intrinsics["cy"]

    vv, uu = np.mgrid[0:h:downsample, 0:w:downsample]
    z = depth[::downsample, ::downsample]

    valid = (z > 0) & np.isfinite(z)
    if mask is not None:
        valid &= mask[::downsample, ::downsample]

    uu, vv, z = uu[valid], vv[valid], z[valid]

    x = (uu - cx) * z / fx
    y = (vv - cy) * z / fy

    return np.column_stack([x, y, z]).astype(np.float32)


def export_ply(filepath: str, points: np.ndarray, colors: Optional[np.ndarray] = None):
    """Export point cloud to PLY format.

    Args:
        filepath: output .ply path.
        points: Nx3 float32 XYZ array.
        colors: Nx3 uint8 RGB array (optional).
    """
    n = len(points)
    has_color = colors is not None and len(colors) == n

    with open(filepath, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        if has_color:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
        f.write("end_header\n")

        for i in range(n):
            line = f"{points[i,0]:.6f} {points[i,1]:.6f} {points[i,2]:.6f}"
            if has_color:
                line += f" {int(colors[i,0])} {int(colors[i,1])} {int(colors[i,2])}"
            f.write(line + "\n")

    logger.info("Exported %d points to %s", n, filepath)


def bearing_angle(x: float, z: float) -> float:
    """Compute horizontal bearing angle from camera XZ coordinates.

    Args:
        x: lateral offset (right positive).
        z: forward distance.

    Returns:
        Angle in radians: 0 = straight ahead, positive = right.
    """
    return float(np.arctan2(x, z))
