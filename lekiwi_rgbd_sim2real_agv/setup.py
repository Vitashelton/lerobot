#!/usr/bin/env python
"""Setup script for lekiwi_rgbd_sim2real_agv."""

import sys
from pathlib import Path

from setuptools import find_packages, setup

if sys.version_info < (3, 10):
    raise RuntimeError("Python >= 3.10 is required.")

_here = Path(__file__).resolve().parent

install_requires = [
    "numpy>=1.24",
    "opencv-python>=4.8",
    "pyzmq>=25",
    "draccus>=0.6",
    "matplotlib>=3.7",
    "scipy>=1.10",
    "tqdm",
]

extras_require = {
    "camera": ["pyrealsense2>=2.54"],
    "sim": ["gymnasium>=0.29", "mujoco>=3.0"],
    "learning": ["torch>=2.0"],
    "detection": ["ultralytics>=8.0"],
    "dash": ["flask>=3.0"],
    "all": [
        "pyrealsense2>=2.54",
        "gymnasium>=0.29",
        "mujoco>=3.0",
        "torch>=2.0",
        "ultralytics>=8.0",
        "flask>=3.0",
    ],
}

setup(
    name="lekiwi-public-offline-nav-rl",
    version="0.2.0",
    description="Multimodal Offline RL for Safe Navigation of Low-Cost Mobile Robots from Public Visual Navigation Datasets",
    author="LeKiwi Public Offline Nav RL Project",
    python_requires=">=3.10",
    install_requires=install_requires,
    extras_require=extras_require,
    packages=find_packages(
        include=[
            "communication", "communication.*",
            "perception", "perception.*",
            "sim", "sim.*",
            "learning", "learning.*",
            "control", "control.*",
            "app", "app.*",
            "tools", "tools.*",
            "experiments", "experiments.*",
            "data_adapters", "data_adapters.*",
            "lerobot_conversion", "lerobot_conversion.*",
            "reward", "reward.*",
            "models", "models.*",
            "rl", "rl.*",
            "safety", "safety.*",
            "lekiwi_deployment", "lekiwi_deployment.*",
            "baselines", "baselines.*",
            "eval", "eval.*",
            "scripts", "scripts.*",
        ]
    ),
    include_package_data=True,
    zip_safe=False,
)
