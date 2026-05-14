"""Public dataset adapters: convert various public navigation datasets
into a unified multimodal format.

Supported datasets:
    - Habitat / HM3D (RGB-D + pose + goal)
    - RoboTHOR (RGB-D + pose + ObjectNav via ai2thor)
    - GNM / ViNT (RGB + action + GPS)
    - Generic h5/npz replay buffers
"""

from data_adapters.base_adapter import BaseDatasetAdapter
from data_adapters.habitat_adapter import HabitatAdapter
from data_adapters.robothor_adapter import RoboTHORAdapter
from data_adapters.gnm_adapter import GNMAdapter
from data_adapters.data_splitter import split_trajectories
from data_adapters.observation_normalizer import ObservationNormalizer

__all__ = [
    "BaseDatasetAdapter",
    "HabitatAdapter",
    "RoboTHORAdapter",
    "GNMAdapter",
    "split_trajectories",
    "ObservationNormalizer",
]
