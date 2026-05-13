"""
Synthetic RGB-D data generation module.

Produces procedurally-generated warehouse/lab scenes with:
- RGB images
- Depth maps (clean + noisy)
- Semantic / instance label maps
- Ground-truth object annotations

Pure numpy/OpenCV implementation -- no external renderer needed.
"""

from .synthetic_depth_renderer import SyntheticDepthRenderer
from .label_generator import LabelGenerator
from .scene_generator import SceneGenerator
from .render_dataset import SyntheticDatasetConfig, main as generate_dataset

__all__ = [
    "SyntheticDepthRenderer",
    "LabelGenerator",
    "SceneGenerator",
    "SyntheticDatasetConfig",
    "generate_dataset",
]
