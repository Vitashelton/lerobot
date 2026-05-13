"""
Multi-object centroid tracker with Kalman filtering.

This module provides:

- :class:`KalmanFilter3D`: A simple constant-velocity 3D Kalman filter
  for smoothing position estimates and predicting future states.
- :class:`CentroidTracker`: Maintains persistent track IDs across frames
  by matching incoming detections to existing tracks using Hungarian
  assignment on Euclidean distance.  Supports track aging, lost-frame
  counting, and velocity estimation.

Typical usage:

    tracker = CentroidTracker(max_disappeared=30, dt=0.1)
    detections = yolo_detector.detect(rgb, depth)
    tracks = tracker.update(detections)
    for t in tracks:
        print(f"Track {t['track_id']}: {t['position_3d']}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# KalmanFilter3D: constant-velocity 3-DoF position filter
# ---------------------------------------------------------------------------

class KalmanFilter3D:
    """
    Simple constant-velocity 3D Kalman filter for object position tracking.

    State vector: ``[x, vx, y, vy, z, vz]`` (6 elements).
    Measurement vector: ``[x, y, z]`` (3 elements).

    Parameters
    ----------
    dt : float
        Time step between predictions (seconds).
    process_noise : float
        Process noise standard deviation (position), used to build Q.
    measurement_noise : float
        Measurement noise standard deviation, used to build R.
    initial_uncertainty : float
        Initial diagonal value for the state covariance matrix P.
    """

    def __init__(
        self,
        dt: float = 0.1,
        process_noise: float = 0.03,
        measurement_noise: float = 0.1,
        initial_uncertainty: float = 1.0,
    ):
        self.dt = dt
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise

        # State: [x, vx, y, vy, z, vz]
        self._x = np.zeros((6, 1), dtype=np.float64)
        self._initialized = False

        # State transition matrix (constant velocity)
        self._F = np.array(
            [
                [1, dt, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, dt, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, dt],
                [0, 0, 0, 0, 0, 1],
            ],
            dtype=np.float64,
        )

        # Measurement matrix (observing position only)
        self._H = np.array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0],
            ],
            dtype=np.float64,
        )

        # Process noise covariance Q
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt2 * dt2
        q = process_noise * process_noise
        self._Q = np.array(
            [
                [dt4 / 4, dt3 / 2, 0, 0, 0, 0],
                [dt3 / 2, dt2, 0, 0, 0, 0],
                [0, 0, dt4 / 4, dt3 / 2, 0, 0],
                [0, 0, dt3 / 2, dt2, 0, 0],
                [0, 0, 0, 0, dt4 / 4, dt3 / 2],
                [0, 0, 0, 0, dt3 / 2, dt2],
            ],
            dtype=np.float64,
        ) * q

        # Measurement noise covariance R
        r = measurement_noise * measurement_noise
        self._R = np.eye(3, dtype=np.float64) * r

        # State covariance
        self._P = np.eye(6, dtype=np.float64) * initial_uncertainty

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self) -> np.ndarray:
        """
        Advance state estimate by one time step (constant velocity).

        Returns the predicted position ``[x, y, z]`` as a (3,) array.
        """
        if not self._initialized:
            raise RuntimeError("KalmanFilter3D must be initialized via update() before predict().")

        self._x = self._F @ self._x
        self._P = self._F @ self._P @ self._F.T + self._Q
        return self._x[[0, 2, 4]].flatten()

    def update(self, measurement: np.ndarray) -> np.ndarray:
        """
        Incorporate a new position measurement.

        Parameters
        ----------
        measurement : np.ndarray
            Shape (3,) array ``[x, y, z]``.

        Returns
        -------
        np.ndarray
            Updated position estimate, shape (3,).
        """
        z = np.asarray(measurement, dtype=np.float64).reshape(3, 1)

        if not self._initialized:
            # Initialize state with measurement, zero velocity
            self._x[0, 0] = z[0, 0]
            self._x[2, 0] = z[1, 0]
            self._x[4, 0] = z[2, 0]
            self._initialized = True
            return self._x[[0, 2, 4]].flatten()

        # Innovation
        y = z - self._H @ self._x  # (3,1)
        S = self._H @ self._P @ self._H.T + self._R  # (3,3)
        K = self._P @ self._H.T @ np.linalg.inv(S)  # (6,3)

        self._x = self._x + K @ y
        self._P = (np.eye(6) - K @ self._H) @ self._P

        return self._x[[0, 2, 4]].flatten()

    def get_state(self) -> Dict[str, Any]:
        """
        Return the current filter state as a dictionary.

        Returns
        -------
        dict
            With keys ``position``, ``velocity``, ``covariance``,
            ``initialized``.
        """
        return {
            "position": (
                float(self._x[0, 0]),
                float(self._x[2, 0]),
                float(self._x[4, 0]),
            ),
            "velocity": (
                float(self._x[1, 0]),
                float(self._x[3, 0]),
                float(self._x[5, 0]),
            ),
            "covariance": self._P.copy(),
            "initialized": self._initialized,
        }

    def reset(self) -> None:
        """Reset the filter to an uninitialized state."""
        self._x = np.zeros((6, 1), dtype=np.float64)
        self._P = np.eye(6, dtype=np.float64)
        self._initialized = False

    @property
    def position(self) -> Tuple[float, float, float]:
        return (
            float(self._x[0, 0]),
            float(self._x[2, 0]),
            float(self._x[4, 0]),
        )

    @property
    def velocity(self) -> Tuple[float, float, float]:
        return (
            float(self._x[1, 0]),
            float(self._x[3, 0]),
            float(self._x[5, 0]),
        )

    @property
    def initialized(self) -> bool:
        return self._initialized


# ---------------------------------------------------------------------------
# CentroidTracker
# ---------------------------------------------------------------------------

@dataclass
class _Track:
    """Internal track representation."""

    track_id: int
    class_name: str = "unknown"
    kf: KalmanFilter3D = field(default_factory=KalmanFilter3D)
    frames_lost: int = 0
    age: int = 0
    position_3d: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    position_history: List[Tuple[float, float, float]] = field(default_factory=list)
    max_history: int = 20


class CentroidTracker:
    """
    Multi-object centroid tracker with Kalman-filtered position smoothing.

    Uses Hungarian assignment (via ``scipy.optimize.linear_sum_assignment``)
    to match incoming detections to existing tracks based on Euclidean
    distance between 3D centroids.  Unmatched tracks are aged and removed
    after ``max_disappeared`` consecutive frames without a match.

    Parameters
    ----------
    max_disappeared : int
        Maximum number of consecutive frames a track can go unmatched
        before it is deleted.
    dt : float
        Time step (seconds) passed to :class:`KalmanFilter3D`.
    max_distance : float
        Maximum Euclidean distance (meters) for associating a detection
        with a track.  Detections beyond this distance cannot be matched.
    kalman_process_noise : float
        Process noise for Kalman filters.
    kalman_measurement_noise : float
        Measurement noise for Kalman filters.
    """

    def __init__(
        self,
        max_disappeared: int = 30,
        dt: float = 0.1,
        max_distance: float = 2.0,
        kalman_process_noise: float = 0.03,
        kalman_measurement_noise: float = 0.1,
    ):
        self.max_disappeared = max_disappeared
        self.dt = dt
        self.max_distance = max_distance

        self.next_track_id: int = 0
        self.tracks: Dict[int, _Track] = {}

        self._kf_process_noise = kalman_process_noise
        self._kf_measurement_noise = kalman_measurement_noise

        # Lazily check for scipy
        self._scipy_available: Optional[bool] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, detections: List[dict]) -> List[dict]:
        """
        Match new detections to existing tracks and return active tracks.

        Parameters
        ----------
        detections : list[dict]
            Each detection must contain at least one of ``position_3d``
            (tuple of 3 floats) or ``center_uv`` (tuple of 2 floats).
            May also contain ``class_name`` (str) and ``confidence`` (float).

        Returns
        -------
        list[dict]
            Each dict with keys ``track_id``, ``class_name``,
            ``position_3d``, ``velocity``, ``frames_lost``, ``age``,
            ``confidence``.
        """
        if not detections:
            # Age all tracks
            self._age_all_tracks()
            self._prune_lost_tracks()
            return self._compile_active_tracks()

        # Predict positions for all existing tracks
        self._predict_all_tracks()

        # Build detection centroids
        det_centroids, det_metadata = self._extract_detection_data(detections)

        if len(det_centroids) == 0:
            self._age_all_tracks()
            self._prune_lost_tracks()
            return self._compile_active_tracks()

        # Match
        track_ids = list(self.tracks.keys())
        matches, unmatched_tracks, unmatched_dets = self._associate(
            det_centroids, track_ids
        )

        # Update matched tracks
        for track_id, det_idx in matches:
            pos_3d = det_centroids[det_idx]
            track = self.tracks[track_id]
            track.kf.update(np.array(pos_3d, dtype=np.float64))
            track.position_3d = track.kf.position
            track.frames_lost = 0
            track.age += 1
            track.class_name = det_metadata[det_idx].get(
                "class_name", track.class_name
            )
            track.position_history.append(track.position_3d)
            if len(track.position_history) > track.max_history:
                track.position_history.pop(0)

        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            pos_3d = det_centroids[det_idx]
            kf = KalmanFilter3D(
                dt=self.dt,
                process_noise=self._kf_process_noise,
                measurement_noise=self._kf_measurement_noise,
            )
            kf.update(np.array(pos_3d, dtype=np.float64))

            track = _Track(
                track_id=self.next_track_id,
                class_name=det_metadata[det_idx].get("class_name", "unknown"),
                kf=kf,
                frames_lost=0,
                age=1,
                position_3d=pos_3d,
            )
            self.tracks[self.next_track_id] = track
            self.next_track_id += 1

        # Age unmatched tracks
        for track_id in unmatched_tracks:
            track = self.tracks[track_id]
            track.frames_lost += 1
            track.age += 1

        self._prune_lost_tracks()
        return self._compile_active_tracks()

    def get_track_by_id(self, track_id: int) -> Optional[dict]:
        """Return a single track dict, or None if not found."""
        track = self.tracks.get(track_id)
        if track is None:
            return None
        return self._track_to_dict(track)

    def reset(self) -> None:
        """Reset the tracker, clearing all tracks."""
        self.tracks.clear()
        self.next_track_id = 0

    def track_count(self) -> int:
        """Return the number of active (non-lost) tracks."""
        return len([t for t in self.tracks.values() if t.frames_lost == 0])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_detection_data(
        self, detections: List[dict]
    ) -> Tuple[List[Tuple[float, float, float]], List[dict]]:
        """Extract 3D centroids and metadata from raw detections."""
        centroids: List[Tuple[float, float, float]] = []
        metadata: List[dict] = []

        for det in detections:
            if "position_3d" in det:
                pos = det["position_3d"]
                if all(np.isfinite(v) for v in pos):
                    centroids.append(tuple(float(v) for v in pos))
                    metadata.append(det)
                    continue

            # Fallback: use center_uv with distance
            if "center_uv" in det and "distance_m" in det:
                u, v = det["center_uv"]
                d = det["distance_m"]
                if np.isfinite(u) and np.isfinite(v) and np.isfinite(d) and d > 0:
                    centroids.append((float(u), float(v), float(d)))
                    metadata.append(det)
                    continue

            # Cannot extract a usable centroid; skip
            continue

        return centroids, metadata

    def _predict_all_tracks(self) -> None:
        """Run Kalman predict step for all existing tracks."""
        for track in self.tracks.values():
            if track.kf.initialized:
                predicted = track.kf.predict()
                track.position_3d = (
                    float(predicted[0]),
                    float(predicted[1]),
                    float(predicted[2]),
                )

    def _age_all_tracks(self) -> None:
        """Age every track by one frame (used when no detections)."""
        for track in list(self.tracks.values()):
            track.frames_lost += 1
            track.age += 1

    def _prune_lost_tracks(self) -> None:
        """Remove tracks that have been lost for too many frames."""
        lost_ids = [
            tid
            for tid, t in self.tracks.items()
            if t.frames_lost > self.max_disappeared
        ]
        for tid in lost_ids:
            del self.tracks[tid]

    def _associate(
        self,
        det_centroids: List[Tuple[float, float, float]],
        track_ids: List[int],
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Match detections to tracks using Hungarian algorithm.

        Returns
        -------
        matches : list of (track_id, det_idx)
        unmatched_tracks : list of track_ids
        unmatched_dets : list of det indices
        """
        if len(track_ids) == 0:
            return [], [], list(range(len(det_centroids)))

        if len(det_centroids) == 0:
            return [], list(track_ids), []

        # Build cost matrix (Euclidean distance)
        cost = np.zeros((len(track_ids), len(det_centroids)), dtype=np.float64)
        for i, tid in enumerate(track_ids):
            track_pos = np.array(self.tracks[tid].position_3d, dtype=np.float64)
            for j, det_pos in enumerate(det_centroids):
                cost[i, j] = np.linalg.norm(track_pos - np.array(det_pos, dtype=np.float64))

        # Apply max_distance gate
        cost[cost > self.max_distance] = cost.max() + 1.0

        # Hungarian assignment
        row_ind, col_ind = self._linear_sum_assignment(cost)

        matches: List[Tuple[int, int]] = []
        used_tracks: set[int] = set()
        used_dets: set[int] = set()

        for r, c in zip(row_ind, col_ind):
            if cost[r, c] <= self.max_distance:
                matches.append((track_ids[r], c))
                used_tracks.add(track_ids[r])
                used_dets.add(c)

        unmatched_tracks = [tid for tid in track_ids if tid not in used_tracks]
        unmatched_dets = [j for j in range(len(det_centroids)) if j not in used_dets]

        return matches, unmatched_tracks, unmatched_dets

    def _linear_sum_assignment(
        self, cost_matrix: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Solve the linear sum assignment problem.

        Uses scipy if available, otherwise falls back to a greedy algorithm.
        """
        if self._scipy_available is None:
            try:
                import scipy.optimize  # noqa: F401

                self._scipy_available = True
            except ImportError:
                self._scipy_available = False

        if self._scipy_available:
            from scipy.optimize import linear_sum_assignment

            return linear_sum_assignment(cost_matrix)

        # Greedy fallback
        n_rows, n_cols = cost_matrix.shape
        used_cols = set()
        row_ind: List[int] = []
        col_ind: List[int] = []

        # Sort rows by minimum cost
        row_order = sorted(range(n_rows), key=lambda r: cost_matrix[r].min())
        for r in row_order:
            # Find best available column
            best_c = -1
            best_val = float("inf")
            for c in range(n_cols):
                if c not in used_cols and cost_matrix[r, c] < best_val:
                    best_val = cost_matrix[r, c]
                    best_c = c
            if best_c >= 0 and best_val <= self.max_distance:
                row_ind.append(r)
                col_ind.append(best_c)
                used_cols.add(best_c)

        return np.array(row_ind), np.array(col_ind)

    def _compile_active_tracks(self) -> List[dict]:
        """Build the public list-of-dicts for all tracks that are not lost."""
        result: List[dict] = []
        for track in sorted(
            self.tracks.values(), key=lambda t: t.track_id
        ):
            if track.frames_lost > 0:
                continue  # skip lost tracks in output
            result.append(self._track_to_dict(track))
        return result

    def _track_to_dict(self, track: _Track) -> dict:
        """Convert internal _Track to a public dictionary."""
        vel = track.kf.velocity if track.kf.initialized else (0.0, 0.0, 0.0)
        return {
            "track_id": track.track_id,
            "class_name": track.class_name,
            "position_3d": track.position_3d,
            "velocity": vel,
            "frames_lost": track.frames_lost,
            "age": track.age,
        }
