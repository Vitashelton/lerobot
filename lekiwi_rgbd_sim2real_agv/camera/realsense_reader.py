"""
Unified reader for Intel RealSense D435i RGB + depth streams.

Wraps LeRobot's ``RealSenseCamera`` to provide a single call interface
for synchronously reading both color and depth frames.  Depth is
returned as a ``uint16`` array in millimeters when ``use_depth`` is
enabled, or ``None`` otherwise.
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np

from lerobot.cameras.realsense.camera_realsense import RealSenseCamera
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

logger = logging.getLogger(__name__)


class RealSenseReader:
    """Unified reader for RealSense D435i RGB and depth images.

    This class wraps LeRobot's ``RealSenseCamera`` and exposes a simpler
    interface tailored to AGV perception pipelines where the caller
    typically needs a paired (rgb, depth) tuple each cycle.

    Parameters
    ----------
    config:
        A fully populated ``RealSenseCameraConfig``.  Set ``use_depth=True``
        on the config to enable the depth stream.
    """

    def __init__(self, config: RealSenseCameraConfig) -> None:
        self._config = config
        self._camera = RealSenseCamera(config)
        self.use_depth: bool = config.use_depth

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"s/n={self._config.serial_number_or_name!r}, "
            f"depth={'on' if self.use_depth else 'off'}"
            f")"
        )

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Open the RealSense pipeline and warm up the camera.

        Raises
        ------
        DeviceAlreadyConnectedError
            If the camera is already connected.
        ConnectionError
            If no compatible RealSense device is found.
        """
        self._camera.connect()

    def disconnect(self) -> None:
        """Stop the RealSense pipeline and release resources.

        Raises
        ------
        DeviceNotConnectedError
            If the camera was not connected.
        """
        self._camera.disconnect()

    @property
    def is_connected(self) -> bool:
        """Return ``True`` if the underlying pipeline is active."""
        return self._camera.is_connected

    # ------------------------------------------------------------------
    # Frame reading
    # ------------------------------------------------------------------

    def read_rgb(self, timeout_ms: int = 500) -> np.ndarray:
        """Return a single RGB color frame.

        Parameters
        ----------
        timeout_ms:
            Maximum time (milliseconds) to wait for a coherent frame set.

        Returns
        -------
        np.ndarray
            RGB image of shape ``(H, W, 3)``, dtype ``uint8``.
        """
        return self._camera.read(timeout_ms=timeout_ms)

    def read_depth(self, timeout_ms: int = 500) -> np.ndarray | None:
        """Return a single depth frame.

        Parameters
        ----------
        timeout_ms:
            Maximum time (milliseconds) to wait for a coherent frame set.

        Returns
        -------
        np.ndarray or None
            Depth image of shape ``(H, W)``, dtype ``uint16`` (millimeters).
            Returns ``None`` when ``use_depth`` is ``False``.

        Raises
        ------
        RuntimeError
            If ``use_depth`` is ``True`` but the depth stream was not
            enabled in the pipeline configuration.
        """
        if not self.use_depth:
            return None
        return self._camera.read_depth(timeout_ms=timeout_ms)

    def read_all(
        self, timeout_ms: int = 500
    ) -> Tuple[np.ndarray, np.ndarray | None]:
        """Return a synchronized ``(rgb, depth)`` tuple in a single call.

        Parameters
        ----------
        timeout_ms:
            Maximum time (milliseconds) to wait for each frame read.

        Returns
        -------
        tuple[np.ndarray, np.ndarray | None]
            ``(rgb, depth)`` where *rgb* is ``uint8 (H, W, 3)`` and
            *depth* is ``uint16 (H, W)`` in millimeters, or ``None`` if
            the depth stream is disabled.
        """
        rgb = self._camera.read(timeout_ms=timeout_ms)
        depth: np.ndarray | None = None
        if self.use_depth:
            depth = self._camera.read_depth(timeout_ms=timeout_ms)
        return rgb, depth
