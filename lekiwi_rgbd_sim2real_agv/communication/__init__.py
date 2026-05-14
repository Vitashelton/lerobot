"""
Communication module (Paper Section III-D): ZMQ Host/Client architecture.

Provides low-latency communication between the LeKiwi onboard computer (host)
and the remote control laptop (client), including observation streaming,
command transmission, dataset writing, and watchdog safety monitoring.
"""

from .config import (
    LeKiwiD435iConfig,
    LeKiwiD435iHostConfig,
    LeKiwiD435iClientConfig,
    lekiwi_d435i_cameras_config,
)
from .host import LeKiwiD435iHost
from .client import LeKiwiD435iClient
from .dataset_writer import LeKiwiDatasetWriter

__all__ = [
    "LeKiwiD435iConfig",
    "LeKiwiD435iHostConfig",
    "LeKiwiD435iClientConfig",
    "lekiwi_d435i_cameras_config",
    "LeKiwiD435iHost",
    "LeKiwiD435iClient",
    "LeKiwiDatasetWriter",
]
