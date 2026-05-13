"""
RGB-D data recorder for saving RealSense streams to disk.

Supported formats:
    - png (RGB) + npy (depth) per frame
    - npz (single compressed archive per frame)
    - jsonl metadata (per-episode summary)

Each recording session creates an episode folder under the output directory.
"""
import json
import logging
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import cv2

from ..utils import ensure_dir

logger = logging.getLogger(__name__)


class RGBDRecorder:
    """Save RGB-D frames and metadata to disk.

    Usage::

        recorder = RGBDRecorder(output_dir="data/raw", format="npz")
        recorder.start_episode(scene="single_box")
        for frame_data in reader_stream:
            recorder.write_frame(frame_data)
            if recorder.should_stop():
                break
        episode_dir = recorder.stop_episode()
    """

    def __init__(
        self,
        output_dir: str = "data/raw",
        format: str = "npz",          # npz | png_npy
        save_metadata: bool = True,
        compress: bool = True,
    ):
        self.output_dir = Path(output_dir)
        self.format = format
        self.save_metadata = save_metadata
        self.compress = compress

        self._episode_dir: Path | None = None
        self._frame_count = 0
        self._episode_start_time = 0.0
        self._recording = False
        self._paused = False
        self._metadata: dict = {}

    # ---- Public API -------------------------------------------------------

    def start_episode(self, scene: str = "unknown", notes: str = ""):
        """Begin a new recording episode."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._episode_dir = ensure_dir(self.output_dir / f"{scene}_{timestamp}")
        self._frame_count = 0
        self._episode_start_time = time.time()
        self._recording = True
        self._paused = False
        self._metadata = {
            "scene": scene,
            "notes": notes,
            "start_time": timestamp,
            "format": self.format,
        }
        logger.info("Recording started: %s", self._episode_dir)

    def write_frame(self, frame_data: dict):
        """Write one RGB-D frame to the current episode.

        Args:
            frame_data: dict with 'color' (HxWx3 uint8 BGR), 'depth' (HxW float32 meters),
                        'intrinsics', 'timestamp'.
        """
        if not self._recording or self._paused:
            return

        color = frame_data.get("color")
        depth = frame_data.get("depth")
        if color is None or depth is None:
            return

        fname = f"frame_{self._frame_count:06d}"

        if self.format == "npz":
            save_kwargs = {}
            if self.compress:
                pass  # np.savez_compressed is the default we'll use
            np.savez_compressed(
                self._episode_dir / f"{fname}.npz",
                color=color,
                depth=depth,
                timestamp=frame_data.get("timestamp", 0.0),
            )
        elif self.format == "png_npy":
            cv2.imwrite(str(self._episode_dir / f"{fname}_color.png"), color)
            np.save(str(self._episode_dir / f"{fname}_depth.npy"), depth)

        self._frame_count += 1

    def pause(self):
        self._paused = True
        logger.info("Recording paused.")

    def resume(self):
        self._paused = False
        logger.info("Recording resumed.")

    def stop_episode(self) -> Path:
        """Stop recording and save metadata. Returns episode directory path."""
        self._recording = False
        duration = time.time() - self._episode_start_time
        self._metadata.update({
            "frame_count": self._frame_count,
            "duration_s": round(duration, 2),
            "fps_avg": round(self._frame_count / duration, 2) if duration > 0 else 0,
        })
        if self.save_metadata and self._episode_dir is not None:
            meta_path = self._episode_dir / "metadata.json"
            # Also save intrinsics from first frame
            self._metadata.setdefault("intrinsics", {})
            with open(meta_path, "w") as f:
                json.dump(self._metadata, f, indent=2)
        logger.info("Episode saved: %d frames to %s", self._frame_count, self._episode_dir)
        ep_dir = self._episode_dir
        self._episode_dir = None
        return ep_dir

    def should_stop(self) -> bool:
        """Check for keyboard stop signal (press 'q' to stop, 'space' to pause/resume)."""
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            return True
        if key == ord(' '):
            if self._paused:
                self.resume()
            else:
                self.pause()
        return False

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def is_paused(self) -> bool:
        return self._paused
