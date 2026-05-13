"""
Dynamic Window Approach for omnidirectional AGV.

Searches over (vx, vy, omega) velocity space to find the best action
given the current LiDAR scan, goal position, and current velocity.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


class DWAPolicy:
    """Dynamic Window Approach for omniwheel AGV.

    The DWA searches a discrete set of velocity samples within the
    achievable dynamic window, forward-simulates each candidate for
    ``predict_time`` seconds, scores every trajectory, and returns
    the action (vx, vy, omega) with the highest score.

    Scoring combines:
      - **goal_progress** -- how much closer the robot gets to the goal.
      - **clearance** -- minimum distance to obstacles along the trajectory.
      - **speed** -- preference for faster trajectories when safe.
      - **heading_alignment** -- how well the final heading matches the goal heading.

    Parameters
    ----------
    vx_limits : tuple[float, float]
        Min / max forward velocity (m/s).
    vy_limits : tuple[float, float]
        Min / max lateral velocity (m/s).
    omega_limits : tuple[float, float]
        Min / max angular velocity (deg/s).
    vx_samples : int
        Number of discrete vx values.
    vy_samples : int
        Number of discrete vy values.
    omega_samples : int
        Number of discrete omega values.
    predict_time : float
        Lookahead time for trajectory simulation (s).
    dt : float
        Simulation time step (s).
    goal_weight : float
        Weight for the goal-progress term.
    clearance_weight : float
        Weight for obstacle clearance.
    speed_weight : float
        Weight for speed preference.
    heading_weight : float
        Weight for goal-heading alignment.
    safety_distance : float
        Minimum acceptable distance to obstacles (m).  Trajectories
        that come closer than this are heavily penalised.
    max_accel_v : float
        Max linear acceleration (m/s^2) used to define the dynamic window.
    max_accel_omega : float
        Max angular acceleration (deg/s^2) used to define the dynamic window.
    scan_angle_min : float
        First beam angle (rad), default -pi (beam 0 = -180 deg).
    scan_angle_max : float
        Last beam angle (rad), default pi (beam N-1 = +180 deg).
    """

    def __init__(
        self,
        vx_limits: tuple[float, float] = (-0.3, 0.3),
        vy_limits: tuple[float, float] = (-0.3, 0.3),
        omega_limits: tuple[float, float] = (-90.0, 90.0),
        vx_samples: int = 7,
        vy_samples: int = 7,
        omega_samples: int = 15,
        predict_time: float = 1.5,
        dt: float = 0.1,
        goal_weight: float = 1.0,
        clearance_weight: float = 0.5,
        speed_weight: float = 0.1,
        heading_weight: float = 0.3,
        safety_distance: float = 0.2,
        max_accel_v: float = 0.5,
        max_accel_omega: float = 180.0,
        scan_angle_min: float = -np.pi,
        scan_angle_max: float = np.pi,
    ) -> None:
        self.vx_limits = vx_limits
        self.vy_limits = vy_limits
        self.omega_limits = omega_limits
        self.vx_samples = vx_samples
        self.vy_samples = vy_samples
        self.omega_samples = omega_samples
        self.predict_time = predict_time
        self.dt = dt
        self.goal_weight = goal_weight
        self.clearance_weight = clearance_weight
        self.speed_weight = speed_weight
        self.heading_weight = heading_weight
        self.safety_distance = safety_distance
        self.max_accel_v = max_accel_v
        self.max_accel_omega = max_accel_omega
        self.scan_angle_min = scan_angle_min
        self.scan_angle_max = scan_angle_max

        self._steps = max(1, int(predict_time / dt))

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------

    def compute_action(
        self,
        scan_m: np.ndarray,
        goal_position: np.ndarray,
        current_velocity: np.ndarray,
    ) -> dict:
        """Compute the best action.

        Parameters
        ----------
        scan_m : np.ndarray, shape (N,)
            LiDAR ranges in metres (typically N=64, 360 degrees).
        goal_position : np.ndarray, shape (3,)
            Goal (dx, dy, dtheta) in **robot frame**.
        current_velocity : np.ndarray, shape (3,)
            Current robot velocity [vx, vy, omega_deg_s].

        Returns
        -------
        dict
            ``{"x.vel": vx, "y.vel": vy, "theta.vel": omega_deg_s, "score": best_score}``
        """
        scan_m = np.asarray(scan_m, dtype=np.float32)
        goal = np.asarray(goal_position, dtype=np.float32)
        vel = np.asarray(current_velocity, dtype=np.float32)

        # 1. Generate velocity samples within dynamic window
        vx_vals, vy_vals, omega_vals = self._dynamic_window(vel)

        best_score = -np.inf
        best_action = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        # 2. Evaluate all samples
        for vx in vx_vals:
            for vy in vy_vals:
                for omega in omega_vals:
                    trajectory = self._simulate_trajectory(
                        np.array([0.0, 0.0, 0.0], dtype=np.float32),
                        vx, vy, omega,
                        self.predict_time,
                        self.dt,
                    )
                    score = self._score_trajectory(trajectory, goal, scan_m)

                    if score > best_score:
                        best_score = score
                        best_action = np.array([vx, vy, omega], dtype=np.float32)

        return {
            "x.vel": float(best_action[0]),
            "y.vel": float(best_action[1]),
            "theta.vel": float(best_action[2]),
            "score": float(best_score),
        }

    # ------------------------------------------------------------------
    # Dynamic window
    # ------------------------------------------------------------------

    def _dynamic_window(self, current_velocity: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute the achievable velocity window given acceleration limits.

        Returns three 1-D arrays of sample values for (vx, vy, omega).
        """
        vx, vy, omega = current_velocity.astype(np.float32)

        # Window: current +/- max_accel * dt
        vx_min = max(self.vx_limits[0], vx - self.max_accel_v * self.dt)
        vx_max = min(self.vx_limits[1], vx + self.max_accel_v * self.dt)
        vy_min = max(self.vy_limits[0], vy - self.max_accel_v * self.dt)
        vy_max = min(self.vy_limits[1], vy + self.max_accel_v * self.dt)
        omega_min = max(self.omega_limits[0], omega - self.max_accel_omega * self.dt)
        omega_max = min(self.omega_limits[1], omega + self.max_accel_omega * self.dt)

        vx_vals = np.linspace(vx_min, vx_max, self.vx_samples, dtype=np.float32)
        vy_vals = np.linspace(vy_min, vy_max, self.vy_samples, dtype=np.float32)
        omega_vals = np.linspace(omega_min, omega_max, self.omega_samples, dtype=np.float32)

        return vx_vals, vy_vals, omega_vals

    # ------------------------------------------------------------------
    # Trajectory simulation
    # ------------------------------------------------------------------

    def _simulate_trajectory(
        self,
        start_pose: np.ndarray,
        vx: float,
        vy: float,
        omega: float,
        predict_time: float,
        dt: float,
    ) -> np.ndarray:
        """Forward-simulate a trajectory from *start_pose*.

        Parameters
        ----------
        start_pose : np.ndarray, shape (3,)
            (x, y, theta_rad) in world/robot frame.
        vx, vy : float
            Linear velocities (m/s).
        omega : float
            Angular velocity (deg/s).
        predict_time : float
            Total simulation time (s).
        dt : float
            Step size (s).

        Returns
        -------
        np.ndarray, shape (steps+1, 3)
            List of poses (x, y, theta_rad), including start pose.
        """
        omega_rad_s = np.deg2rad(omega)
        steps = max(1, int(predict_time / dt))
        traj = np.zeros((steps + 1, 3), dtype=np.float32)
        traj[0] = start_pose

        x, y, theta = start_pose
        for i in range(1, steps + 1):
            # Update heading first (midpoint approximation for rotation)
            theta = theta + omega_rad_s * dt
            # Transform body-frame velocities to world frame
            dx = vx * np.cos(theta) - vy * np.sin(theta)
            dy = vx * np.sin(theta) + vy * np.cos(theta)
            x += dx * dt
            y += dy * dt
            traj[i] = [x, y, theta]

        return traj

    # ------------------------------------------------------------------
    # Collision checking
    # ------------------------------------------------------------------

    def _check_collision_at_pose(
        self, pose: np.ndarray, scan_m: np.ndarray
    ) -> float:
        """Return the estimated clearance at a given pose using the scan.

        We approximate by looking up the scan beam closest to the
        direction of travel.  This is a *conservative* check: we
        return the minimum range in a forward-facing cone.

        Returns
        -------
        float
            Estimated clearance (m).  Smaller = closer to obstacle.
        """
        # Use a forward-facing cone of beams (approx 90 deg).
        n = len(scan_m)
        cone_beams = max(4, n // 4)  # ~16 for 64-beam scan
        # Centre on beam 0 (forward).  We take beams [-cone_beams//2 : cone_beams//2]
        half = cone_beams // 2
        front = np.concatenate([scan_m[-half:], scan_m[:half]]) if half > 0 else scan_m
        return float(front.min())

    def _check_collision(
        self, pose: np.ndarray, obstacles: np.ndarray
    ) -> bool:
        """Check if a pose collides with known obstacle points.

        Parameters
        ----------
        pose : np.ndarray, shape (3,)
            Robot pose (x, y, theta).
        obstacles : np.ndarray, shape (M, 2)
            Obstacle point cloud in world frame.

        Returns
        -------
        bool
            True if any obstacle is within *safety_distance*.
        """
        if obstacles is None or len(obstacles) == 0:
            return False
        dists = np.linalg.norm(obstacles - pose[:2], axis=1)
        return bool(np.any(dists < self.safety_distance))

    # ------------------------------------------------------------------
    # Trajectory scoring
    # ------------------------------------------------------------------

    def _score_trajectory(
        self,
        trajectory: np.ndarray,
        goal: np.ndarray,
        scan_m: Optional[np.ndarray] = None,
    ) -> float:
        """Score a trajectory.

        Parameters
        ----------
        trajectory : np.ndarray, shape (T, 3)
            Simulated poses.
        goal : np.ndarray, shape (3,)
            Goal (dx, dy, dtheta) in robot frame.
        scan_m : np.ndarray, optional
            Current LiDAR scan for clearance checking.

        Returns
        -------
        float
            Combined score (higher is better).
        """
        # 1. Goal progress: distance from final pose to goal
        final_pose = trajectory[-1]
        goal_dist = np.linalg.norm(final_pose[:2] - goal[:2])
        # Normalise: 0 at 0 distance, decays to 0 at large distance
        goal_score = np.exp(-goal_dist / 2.0)  # characteristic length 2m

        # 2. Clearance along trajectory
        if scan_m is not None:
            clearances = []
            for pose in trajectory[1:]:  # skip start
                c = self._check_collision_at_pose(pose, scan_m)
                clearances.append(c)
            min_clearance = min(clearances) if clearances else 5.0

            if min_clearance < self.safety_distance:
                # Heavy penalty for unsafe trajectories
                clearance_score = -100.0 / max(min_clearance, 0.01)
            else:
                # Sigmoid-ish: prefers larger clearance, saturates
                clearance_score = 1.0 - np.exp(-min_clearance / self.safety_distance)
        else:
            clearance_score = 1.0

        # 3. Speed: prefer faster speeds when safe
        # Average speed over trajectory
        if len(trajectory) > 1:
            displacements = np.linalg.norm(np.diff(trajectory[:, :2], axis=0), axis=1)
            avg_speed = np.mean(displacements) / self.dt
            # Normalise by max possible speed
            max_speed = np.sqrt(self.vx_limits[1] ** 2 + self.vy_limits[1] ** 2)
            speed_score = avg_speed / max(max_speed, 1e-6)
        else:
            speed_score = 0.0

        # 4. Heading alignment: how close final heading is to goal heading
        final_heading = trajectory[-1, 2]
        goal_heading = np.arctan2(goal[1], goal[0])
        heading_error = _normalize_angle(final_heading - goal_heading)
        heading_score = np.cos(heading_error)  # 1 when aligned, -1 when opposite

        return float(
            self.goal_weight * goal_score
            + self.clearance_weight * clearance_score
            + self.speed_weight * speed_score
            + self.heading_weight * heading_score
        )


def _normalize_angle(angle: float) -> float:
    """Normalise angle to [-pi, pi)."""
    return float((angle + np.pi) % (2 * np.pi) - np.pi)
