"""LeRobot-compatible observation/action space schema for navigation.

Defines the spaces used in the LeRobot dataset:
    - observation spaces (RGB, depth, Scan64, state, goal)
    - action space (continuous velocity commands)
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch

# Each entry maps a key to (shape, dtype, low, high)
OBSERVATION_SPACES: Dict[str, tuple] = {
    "observation.images.rgb": ((3, 224, 224), np.float32, 0.0, 1.0),
    "observation.images.depth": ((1, 224, 224), np.float32, 0.0, 5.0),
    "observation.scan64": ((64,), np.float32, 0.0, 5.0),
    "observation.state": ((3,), np.float32, -1.0, 1.0),
    "observation.goal": ((3,), np.float32, -10.0, 10.0),
    "observation.prev_action": ((3,), np.float32, -1.0, 1.0),
}

ACTION_SPACE: Dict[str, tuple] = {
    "action": ((3,), np.float32, -0.3, 0.3),  # [vx, vy, omega_norm]
}

# LeRobot convention: images stored as (C, H, W), float32 [0, 1]
# Actions stored as normalized [-1, 1] for each dimension

NAVIGATION_OBS_SCHEMA = {
    "features": {
        "observation.images.rgb": {
            "shape": [3, 224, 224],
            "dtype": "float32",
            "names": ["channels", "height", "width"],
        },
        "observation.images.depth": {
            "shape": [1, 224, 224],
            "dtype": "float32",
            "names": ["channels", "height", "width"],
        },
        "observation.scan64": {
            "shape": [64],
            "dtype": "float32",
            "names": ["beam"],
        },
        "observation.state": {
            "shape": [3],
            "dtype": "float32",
            "names": ["vx", "vy", "omega"],
        },
        "observation.goal": {
            "shape": [3],
            "dtype": "float32",
            "names": ["dx", "dy", "dtheta"],
        },
        "observation.prev_action": {
            "shape": [3],
            "dtype": "float32",
            "names": ["vx_prev", "vy_prev", "omega_prev"],
        },
        "action": {
            "shape": [3],
            "dtype": "float32",
            "names": ["vx", "vy", "omega"],
        },
        "reward": {
            "shape": [1],
            "dtype": "float32",
            "names": ["reward"],
        },
        "done": {
            "shape": [1],
            "dtype": "bool",
            "names": ["done"],
        },
        "episode_index": {
            "shape": [1],
            "dtype": "int64",
            "names": ["episode_index"],
        },
        "frame_index": {
            "shape": [1],
            "dtype": "int64",
            "names": ["frame_index"],
        },
        "timestamp": {
            "shape": [1],
            "dtype": "float32",
            "names": ["timestamp"],
        },
        "index": {
            "shape": [1],
            "dtype": "int64",
            "names": ["index"],
        },
    }
}
