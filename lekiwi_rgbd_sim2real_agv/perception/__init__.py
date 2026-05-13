"""
Perception module for lekiwi_rgbd_sim2real_agv.

Provides obstacle detection, ArUco pallet detection, YOLO detection,
depth localization, multi-object tracking, and safety zone evaluation
for AGV navigation.
"""

from .obstacle_detector import ObstacleDetector
from .aruco_pallet_detector import ArUcoPalletDetector
from .yolo_detector import YOLODetector
from .depth_localizer import DepthLocalizer
from .tracker import CentroidTracker, KalmanFilter3D
from .safety_zone import SafetyZone

__all__ = [
    "ObstacleDetector",
    "ArUcoPalletDetector",
    "YOLODetector",
    "DepthLocalizer",
    "CentroidTracker",
    "KalmanFilter3D",
    "SafetyZone",
]
