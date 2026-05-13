# lekiwi_rgbd_sim2real_agv camera module
#
# Provides utilities for RealSense D435i depth processing and
# 1D scan representation for AGV perception.
#

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

__all__ = [
    "RealSenseReader",
    "preprocess_pipeline",
    "raw_to_meters",
    "filter_invalid",
    "clamp_range",
    "median_filter",
    "hole_filling",
    "depth_to_scan_pipeline",
    "depth_to_scan_polar",
    "percentile_pool",
    "compute_sector_mins",
    "temporal_smooth",
    "scan_quality_diagnostics",
    "DepthQualityMonitor",
]
