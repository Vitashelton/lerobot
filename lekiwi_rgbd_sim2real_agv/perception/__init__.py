"""
Perception module (Paper Section III-C): RGB-D processing and obstacle detection.

Provides the complete perception pipeline:
  - D435i depth capture and preprocessing (median filter, hole filling)
  - RGB-D-to-Scan64 virtual LiDAR projection via percentile pooling
  - Sector-based obstacle detection and safety zone evaluation
  - ArUco marker detection for pallet localization
  - YOLO object detection with depth-based 3D localization
  - Multi-object tracking with Kalman filters
"""

from .realsense_reader import RealSenseReader
from .depth_preprocess import (
    preprocess_pipeline,
    raw_to_meters,
    filter_invalid,
    clamp_range,
    median_filter,
    hole_filling,
)
from .depth_to_scan import (
    depth_to_scan_pipeline,
    depth_to_scan_polar,
    percentile_pool,
    compute_sector_mins,
    temporal_smooth,
    scan_quality_diagnostics,
)
from .depth_quality_monitor import DepthQualityMonitor
from .obstacle_detector import ObstacleDetector
from .aruco_pallet_detector import ArUcoPalletDetector
from .yolo_detector import YOLODetector
from .depth_localizer import DepthLocalizer
from .tracker import CentroidTracker, KalmanFilter3D
from .safety_zone import SafetyZone

__all__ = [
    # Depth preprocessing
    "RealSenseReader",
    "preprocess_pipeline",
    "raw_to_meters",
    "filter_invalid",
    "clamp_range",
    "median_filter",
    "hole_filling",
    # Scan64 projection
    "depth_to_scan_pipeline",
    "depth_to_scan_polar",
    "percentile_pool",
    "compute_sector_mins",
    "temporal_smooth",
    "scan_quality_diagnostics",
    "DepthQualityMonitor",
    # Obstacle detection
    "ObstacleDetector",
    "ArUcoPalletDetector",
    "YOLODetector",
    "DepthLocalizer",
    "CentroidTracker",
    "KalmanFilter3D",
    "SafetyZone",
]
