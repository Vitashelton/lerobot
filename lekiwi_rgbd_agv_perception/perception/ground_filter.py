"""
Ground plane filtering.

Two approaches:
    1. Simple: assume ground occupies the bottom portion of the image,
       filter by height threshold.
    2. RANSAC: fit a plane to the point cloud and remove inliers (optional).
"""
import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def filter_ground_simple(
    depth: np.ndarray,
    intrinsics: dict,
    ground_ratio: float = 0.5,
    height_threshold: float = 0.05,
) -> np.ndarray:
    """Filter ground points using a simple height threshold.

    Assumes the camera is mounted at a known height above ground.
    Pixels in the lower portion of the image and with Y (height) near 0
    (in camera frame, Y points down) are classified as ground.

    Args:
        depth: HxW float32 depth in meters.
        intrinsics: dict with fx, fy, cx, cy.
        ground_ratio: fraction of image height considered potential ground.
        height_threshold: maximum |Y| to classify as ground (meters).

    Returns:
        HxW boolean mask where True = non-ground (obstacle).
    """
    from .pointcloud_utils import depth_to_pointcloud

    h, w = depth.shape
    # Only process lower portion
    ground_region = int(h * (1 - ground_ratio))
    cloud = depth_to_pointcloud(depth[ground_region:, :], intrinsics)

    # Points with Y close to 0 (camera height plane) are ground
    non_ground = np.abs(cloud[:, 1]) > height_threshold

    # Build full mask
    mask = np.ones((h, w), dtype=bool)
    ground_mask_region = np.zeros(depth[ground_region:, :].shape, dtype=bool)
    valid = (depth[ground_region:, :] > 0).ravel()
    ng_idx = 0
    for i in range(ground_mask_region.size):
        if valid[i]:
            ground_mask_region.flat[i] = not non_ground[ng_idx]
            ng_idx += 1
    mask[ground_region:, :] = ~ground_mask_region
    return mask


def filter_ground_ransac(
    cloud: np.ndarray,
    distance_threshold: float = 0.02,
    max_iterations: int = 200,
    min_height: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a ground plane using RANSAC and separate ground from obstacles.

    Args:
        cloud: Nx3 point cloud (X, Y, Z) with Y pointing down.
        distance_threshold: max distance to plane for inliers.
        max_iterations: RANSAC iterations.
        min_height: additional height offset above plane to include.

    Returns:
        (ground_mask, obstacle_mask): boolean masks for the input cloud.
    """
    n = len(cloud)
    if n < 3:
        return np.zeros(n, dtype=bool), np.ones(n, dtype=bool)

    best_inliers = np.zeros(n, dtype=bool)
    best_count = 0

    rng = np.random.RandomState(42)

    for _ in range(max_iterations):
        # Sample 3 points
        idx = rng.choice(n, 3, replace=False)
        p1, p2, p3 = cloud[idx]

        # Fit plane: normal = (p2-p1) x (p3-p1)
        normal = np.cross(p2 - p1, p3 - p1)
        norm_len = np.linalg.norm(normal)
        if norm_len < 1e-10:
            continue
        normal /= norm_len
        d = -np.dot(normal, p1)

        # Compute distances
        distances = np.abs(np.dot(cloud, normal) + d)
        inliers = distances < distance_threshold
        count = np.sum(inliers)

        if count > best_count:
            best_count = count
            best_inliers = inliers

    # Ground = inliers with Y near plane
    ground_mask = best_inliers.copy()
    obstacle_mask = ~ground_mask

    logger.info("RANSAC ground filter: %d ground, %d obstacle points",
                int(np.sum(ground_mask)), int(np.sum(obstacle_mask)))
    return ground_mask, obstacle_mask
