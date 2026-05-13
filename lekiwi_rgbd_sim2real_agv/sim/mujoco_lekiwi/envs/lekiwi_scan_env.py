"""
Gymnasium environment for LeKiwi omniwheel AGV navigation with scan observations.

Action space (3-D continuous):
    [vx (m/s), vy (m/s), omega (deg/s)]

Observation space (73-D):
    scan64 (64) + goal_vec_3d (3) + velocity_3d (3) + last_action_3d (3)

Worlds are defined procedurally via obstacle lists (walls as line segments,
boxes as axis-aligned rectangles).  A 64-ray scan is simulated by ray-casting
against all world obstacles.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np


# ======================================================================
#  World definitions
# ======================================================================

def _make_world_lab_empty() -> dict:
    """Sparse world: four walls forming a 20x20 m room."""
    s = 10.0  # half-size
    return {
        "walls": [
            ((-s, -s), ( s, -s)),  # bottom
            (( s, -s), ( s,  s)),  # right
            (( s,  s), (-s,  s)),  # top
            ((-s,  s), (-s, -s)),  # left
        ],
        "boxes": [],   # (cx, cy, w, h)
        "cylinders": [],  # (cx, cy, radius)
        "goal_zone": {"center": (7.0, 7.0), "radius": 1.5},
        "start_zone": {"center": (-7.0, -7.0), "radius": 1.0},
        "name": "lab_empty",
    }


def _make_world_warehouse_aisle() -> dict:
    """Parallel shelving units creating aisles in a warehouse."""
    s = 12.0
    walls = [
        ((-s, -s), ( s, -s)),
        (( s, -s), ( s,  s)),
        (( s,  s), (-s,  s)),
        ((-s,  s), (-s, -s)),
    ]
    boxes = []
    # 3 rows of shelves (long rectangles), 2 columns each
    shelf_w = 0.6
    shelf_h = 3.0
    aisle_gap = 3.0
    for row in range(2):
        y_center = -4.0 + row * 8.0
        for col in range(3):
            x_center = -6.0 + col * 6.0
            boxes.append((x_center, y_center, shelf_w, shelf_h))
            # second shelf in the same aisle
            boxes.append((x_center, y_center + aisle_gap, shelf_w, shelf_h))

    return {
        "walls": walls,
        "boxes": boxes,
        "cylinders": [],
        "goal_zone": {"center": (8.0, 8.0), "radius": 1.5},
        "start_zone": {"center": (-8.0, -8.0), "radius": 1.0},
        "name": "warehouse_aisle",
    }


def _make_world_pallet_pickup() -> dict:
    """Approach zone: large open room with a pallet area near centre."""
    s = 10.0
    walls = [
        ((-s, -s), ( s, -s)),
        (( s, -s), ( s,  s)),
        (( s,  s), (-s,  s)),
        ((-s,  s), (-s, -s)),
    ]
    boxes = [
        # Pallet area represented as a few boxes
        (-0.4, 3.8, 0.15, 0.8),
        (0.0, 3.8, 0.15, 0.8),
        (0.4, 3.8, 0.15, 0.8),
        (-0.4, 4.2, 0.15, 0.8),
        (0.0, 4.2, 0.15, 0.8),
        (0.4, 4.2, 0.15, 0.8),
    ]
    # A couple of nearby boxes
    boxes.extend([
        (2.0, 3.0, 0.4, 0.4),
        (-2.0, 5.0, 0.5, 0.5),
    ])
    return {
        "walls": walls,
        "boxes": boxes,
        "cylinders": [],
        "goal_zone": {"center": (0.0, 4.0), "radius": 1.0},
        "start_zone": {"center": (0.0, -6.0), "radius": 1.0},
        "name": "pallet_pickup",
    }


def _make_world_cluttered_lab() -> dict:
    """Lab room with random boxes and cylindrical obstacles."""
    s = 8.0
    walls = [
        ((-s, -s), ( s, -s)),
        (( s, -s), ( s,  s)),
        (( s,  s), (-s,  s)),
        ((-s,  s), (-s, -s)),
    ]
    rng = np.random.RandomState(42)
    boxes = []
    for _ in range(12):
        bx = rng.uniform(0.2, 0.6)
        by = rng.uniform(0.2, 0.6)
        cx = rng.uniform(-6.0, 6.0)
        cy = rng.uniform(-6.0, 6.0)
        boxes.append((cx, cy, bx, by))

    cylinders = []
    for _ in range(4):
        cx = rng.uniform(-5.0, 5.0)
        cy = rng.uniform(-5.0, 5.0)
        r = rng.uniform(0.15, 0.35)
        cylinders.append((cx, cy, r))

    return {
        "walls": walls,
        "boxes": boxes,
        "cylinders": cylinders,
        "goal_zone": {"center": (6.0, 6.0), "radius": 1.5},
        "start_zone": {"center": (-6.0, -6.0), "radius": 1.0},
        "name": "cluttered_lab",
    }


WORLD_BUILDERS = {
    "lab_empty": _make_world_lab_empty,
    "warehouse_aisle": _make_world_warehouse_aisle,
    "pallet_pickup": _make_world_pallet_pickup,
    "cluttered_lab": _make_world_cluttered_lab,
}


# ======================================================================
#  Ray-casting helpers (2-D)
# ======================================================================

def _ray_segment_intersection(
    origin: np.ndarray,
    direction: np.ndarray,
    seg_a: np.ndarray,
    seg_b: np.ndarray,
    max_range: float = 30.0,
) -> Optional[float]:
    """Return distance to intersection of ray with line segment, or None.

    origin   : (2,)  ray origin.
    direction: (2,)  unit ray direction.
    seg_a, seg_b: (2,)  segment end-points.
    max_range: float  ignore hits beyond this.
    """
    edge = seg_b - seg_a
    # Solve  origin + t * dir = seg_a + u * edge
    # [dir, -edge] @ [t, u] = seg_a - origin
    A = np.column_stack([direction, -edge])
    b = seg_a - origin
    det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
    if abs(det) < 1e-12:
        return None
    t = (b[0] * A[1, 1] - b[1] * A[0, 1]) / det
    u = (A[0, 0] * b[1] - A[1, 0] * b[0]) / det
    if t > 1e-6 and 0.0 <= u <= 1.0 and t <= max_range:
        return float(t)
    return None


def _ray_box_intersection(
    origin: np.ndarray,
    direction: np.ndarray,
    box: Tuple[float, float, float, float],
    max_range: float = 30.0,
) -> Optional[float]:
    """Return distance to intersection of ray with axis-aligned box.

    box = (cx, cy, width, height).
    """
    cx, cy, w, h = box
    half_w, half_h = w / 2, h / 2
    x_min, x_max = cx - half_w, cx + half_w
    y_min, y_max = cy - half_h, cy + half_h

    t_min = -1e12
    t_max = 1e12

    eps = 1e-12

    # X slabs
    if abs(direction[0]) < eps:
        if origin[0] < x_min or origin[0] > x_max:
            return None
    else:
        t1 = (x_min - origin[0]) / direction[0]
        t2 = (x_max - origin[0]) / direction[0]
        if t1 > t2:
            t1, t2 = t2, t1
        t_min = max(t_min, t1)
        t_max = min(t_max, t2)
        if t_min > t_max:
            return None

    # Y slabs
    if abs(direction[1]) < eps:
        if origin[1] < y_min or origin[1] > y_max:
            return None
    else:
        t1 = (y_min - origin[1]) / direction[1]
        t2 = (y_max - origin[1]) / direction[1]
        if t1 > t2:
            t1, t2 = t2, t1
        t_min = max(t_min, t1)
        t_max = min(t_max, t2)
        if t_min > t_max:
            return None

    if t_max < 1e-6:
        return None  # behind the ray

    t_hit = t_min if t_min > 1e-6 else t_max
    if t_hit <= 0 or t_hit > max_range:
        return None
    return float(t_hit)


def _ray_circle_intersection(
    origin: np.ndarray,
    direction: np.ndarray,
    center: Tuple[float, float],
    radius: float,
    max_range: float = 30.0,
) -> Optional[float]:
    """Return distance to intersection of ray with circle."""
    cx, cy = center
    oc = np.array([cx - origin[0], cy - origin[1]])
    t_ca = np.dot(oc, direction)
    if t_ca < 0:
        return None
    d2 = np.dot(oc, oc) - t_ca * t_ca
    r2 = radius * radius
    if d2 > r2:
        return None
    t_hc = math.sqrt(r2 - d2)
    t = t_ca - t_hc
    if t < 1e-6:
        t = t_ca + t_hc
    if t > 1e-6 and t <= max_range:
        return float(t)
    return None


# ======================================================================
#  Environment
# ======================================================================

class LeKiwiScanEnv(gym.Env):
    """Simplified omniwheel AGV navigation environment.

    Action
    ------
    [vx, vy, omega_deg_s]  (3-D continuous)
        vx   : forward velocity  (m/s)        [-0.3, 0.3]
        vy   : lateral velocity  (m/s)        [-0.3, 0.3]
        omega: yaw rate          (deg/s)      [-90,  90]

    Observation  (73-D)
    -------------------
    scan64          (64,)  range readings (m) in equally-spaced angular bins.
    goal_vec_3d     (3,)   [dx, dy, dtheta] to goal in agent frame.
    velocity_3d     (3,)   [vx, vy, omega_rad_s].
    last_action_3d  (3,)   previous action.

    Worlds
    ------
    lab_empty        – large empty room, walls only.
    warehouse_aisle  – parallel shelves creating aisles.
    pallet_pickup    – approach a pallet zone.
    cluttered_lab    – random boxes and cylinders.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    # ------------------------------------------------------------------
    def __init__(
        self,
        world: str = "lab_empty",
        max_steps: int = 500,
        scan_dim: int = 64,
        scan_range: float = 8.0,
        collision_threshold: float = 0.15,
        action_limits: Optional[Dict[str, Tuple[float, float]]] = None,
        reward_weights: Optional[Dict[str, float]] = None,
    ):
        super().__init__()

        if world not in WORLD_BUILDERS:
            raise ValueError(
                f"Unknown world '{world}'.  Known: {list(WORLD_BUILDERS.keys())}")
        self._world_name = world
        self._world_builder = WORLD_BUILDERS[world]
        self._max_steps = max_steps
        self._scan_dim = scan_dim
        self._scan_range = scan_range
        self._collision_threshold = collision_threshold

        # Action limits
        if action_limits is None:
            action_limits = {
                "vx": (-0.3, 0.3),
                "vy": (-0.3, 0.3),
                "omega": (-90.0, 90.0),
            }
        self._action_limits = action_limits
        vx_low, vx_high = action_limits["vx"]
        vy_low, vy_high = action_limits["vy"]
        om_low, om_high = action_limits["omega"]

        self.action_space = gym.spaces.Box(
            low=np.array([vx_low, vy_low, om_low], dtype=np.float32),
            high=np.array([vx_high, vy_high, om_high], dtype=np.float32),
            dtype=np.float32,
        )

        obs_dim = scan_dim + 3 + 3 + 3
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32,
        )

        # Reward weights
        if reward_weights is None:
            reward_weights = {
                "progress": 1.0,
                "collision": -10.0,
                "clearance": 0.1,
                "smoothness": -0.05,
                "spin": -0.02,
                "goal": 50.0,
            }
        self._rw = reward_weights

        # Internal state
        self._world_state: dict = {}
        self._pose: np.ndarray = np.zeros(3, dtype=np.float32)  # [x, y, theta]
        self._velocity: np.ndarray = np.zeros(3, dtype=np.float32)
        self._last_action: np.ndarray = np.zeros(3, dtype=np.float32)
        self._goal_pose: np.ndarray = np.zeros(3, dtype=np.float32)
        self._step_count: int = 0
        self._rng: np.random.Generator = np.random.default_rng()

    # ------------------------------------------------------------------
    #  gym.Env interface
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._world_state = self._world_builder()
        self._step_count = 0

        # Random start within start zone
        sz = self._world_state["start_zone"]
        angle = self._rng.uniform(0, 2 * np.pi)
        dist = self._rng.uniform(0, sz["radius"])
        sx = sz["center"][0] + dist * math.cos(angle)
        sy = sz["center"][1] + dist * math.sin(angle)
        stheta = self._rng.uniform(-math.pi, math.pi)

        self._pose = np.array([sx, sy, stheta], dtype=np.float32)

        # Random goal within goal zone
        gz = self._world_state["goal_zone"]
        angle = self._rng.uniform(0, 2 * np.pi)
        dist = self._rng.uniform(0, gz["radius"])
        gx = gz["center"][0] + dist * math.cos(angle)
        gy = gz["center"][1] + dist * math.sin(angle)
        gtheta = self._rng.uniform(-math.pi, math.pi)
        self._goal_pose = np.array([gx, gy, gtheta], dtype=np.float32)

        self._velocity = np.zeros(3, dtype=np.float32)
        self._last_action = np.zeros(3, dtype=np.float32)

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        self._step_count += 1

        # Clip action
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Dynamics
        prev_pose = self._pose.copy()
        prev_dist_to_goal = float(np.linalg.norm(
            self._pose[:2] - self._goal_pose[:2]))

        self._apply_action(action)

        # Scan
        scan = self._generate_scan(self._pose, self._world_state)

        # Reward
        reward = self._compute_reward(
            action, scan, prev_pose, prev_dist_to_goal)

        # Termination
        terminated = False
        truncated = self._step_count >= self._max_steps

        # Collision check
        min_scan = float(np.min(scan))
        if min_scan < self._collision_threshold:
            terminated = True
            reward += self._rw.get("collision", -10.0)

        # Goal reached
        dist_to_goal = float(np.linalg.norm(
            self._pose[:2] - self._goal_pose[:2]))
        if dist_to_goal < 0.3:
            terminated = True
            reward += self._rw.get("goal", 50.0)

        self._last_action = action.copy()

        obs = self._get_obs()
        info = self._get_info()
        info["dist_to_goal"] = dist_to_goal
        info["min_scan"] = min_scan

        return obs, reward, terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        """Return a simple top-down render as an RGB array, or print to console."""
        mode = self.metadata.get("render_mode", "human")
        if mode == "human":
            pose = self._pose
            print(f"\rStep {self._step_count:4d}  "
                  f"pose=({pose[0]:.2f}, {pose[1]:.2f}, {math.degrees(pose[2]):.0f}deg)  "
                  f"goal=({self._goal_pose[0]:.2f}, {self._goal_pose[1]:.2f})", end="")
            return None

        # RGB array render
        import cv2
        img_size = 400
        world_range = 12.0
        scale = img_size / (2 * world_range)
        img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 240

        def to_px(x, y):
            px = int((x + world_range) * scale)
            py = int((world_range - y) * scale)  # flip Y
            return np.clip(px, 0, img_size - 1), np.clip(py, 0, img_size - 1)

        # Walls
        for (ax, ay), (bx, by) in self._world_state.get("walls", []):
            p1 = to_px(ax, ay)
            p2 = to_px(bx, by)
            cv2.line(img, p1, p2, (0, 0, 0), 2)

        # Boxes
        for (cx, cy, w, h) in self._world_state.get("boxes", []):
            x1, y1 = to_px(cx - w / 2, cy + h / 2)
            x2, y2 = to_px(cx + w / 2, cy - h / 2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (100, 100, 100), -1)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), 1)

        # Cylinders
        for (cx, cy, r) in self._world_state.get("cylinders", []):
            pc = to_px(cx, cy)
            pr = int(r * scale)
            cv2.circle(img, pc, pr, (150, 100, 100), -1)
            cv2.circle(img, pc, pr, (0, 0, 0), 1)

        # Goal
        gz = self._world_state["goal_zone"]
        gc = to_px(gz["center"][0], gz["center"][1])
        gr = int(gz["radius"] * scale)
        cv2.circle(img, gc, gr, (0, 200, 0), 2)

        # Agent
        ax, ay = to_px(self._pose[0], self._pose[1])
        cv2.circle(img, (ax, ay), 6, (0, 0, 255), -1)
        # heading line
        heading_len = 0.5
        hx = ax + int(heading_len * scale * math.cos(self._pose[2]))
        hy = ay - int(heading_len * scale * math.sin(self._pose[2]))
        cv2.line(img, (ax, ay), (hx, hy), (255, 0, 0), 2)

        # Scan rays (subset for visibility)
        scan = self._generate_scan(self._pose, self._world_state)
        for i in range(0, self._scan_dim, 4):
            angle = self._pose[2] + i * 2 * math.pi / self._scan_dim
            dist = scan[i]
            if dist < self._scan_range:
                sx = ax + int(dist * scale * math.cos(angle))
                sy = ay - int(dist * scale * math.sin(angle))
                cv2.line(img, (ax, ay), (sx, sy), (255, 200, 0), 1)

        return img

    def close(self):
        pass

    # ------------------------------------------------------------------
    #  Internal helpers
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        scan = self._generate_scan(self._pose, self._world_state)

        # Goal vector in agent frame
        dx = self._goal_pose[0] - self._pose[0]
        dy = self._goal_pose[1] - self._pose[1]
        dtheta = self._goal_pose[2] - self._pose[2]
        # Rotate into agent frame
        c = math.cos(-self._pose[2])
        s = math.sin(-self._pose[2])
        gx_local = c * dx - s * dy
        gy_local = s * dx + c * dy
        # Wrap angle
        dtheta = math.atan2(math.sin(dtheta), math.cos(dtheta))
        goal_vec = np.array([gx_local, gy_local, dtheta], dtype=np.float32)

        obs = np.concatenate([
            scan.astype(np.float32),
            goal_vec,
            self._velocity.astype(np.float32),
            self._last_action.astype(np.float32),
        ])
        return obs

    def _get_info(self) -> dict:
        return {
            "pose": self._pose.copy(),
            "goal": self._goal_pose.copy(),
            "step": self._step_count,
            "world": self._world_state.get("name", self._world_name),
        }

    def _apply_action(self, action: np.ndarray, dt: float = 0.1):
        """Omniwheel kinematics."""
        vx, vy, omega_deg = action
        omega_rad = math.radians(omega_deg)

        # World-frame displacement
        c = math.cos(self._pose[2])
        s = math.sin(self._pose[2])
        dx_world = vx * c - vy * s
        dy_world = vx * s + vy * c

        self._pose[0] += dx_world * dt
        self._pose[1] += dy_world * dt
        self._pose[2] += omega_rad * dt
        # Wrap angle
        self._pose[2] = math.atan2(math.sin(self._pose[2]),
                                   math.cos(self._pose[2]))

        self._velocity = np.array([vx, vy, omega_rad], dtype=np.float32)

    def _generate_scan(self, pose: np.ndarray,
                       world_state: dict) -> np.ndarray:
        """Simulate 64-D lidar-like scan.

        For each of scan_dim rays evenly spaced in angle, compute the
        minimum distance to any obstacle (wall, box, cylinder).

        Returns
        -------
        scan : ndarray (scan_dim,) float32  range values (metres).
        """
        origin = pose[:2].astype(np.float64)
        theta0 = float(pose[2])
        scan = np.full(self._scan_dim, self._scan_range, dtype=np.float32)

        walls = world_state.get("walls", [])
        boxes = world_state.get("boxes", [])
        cylinders = world_state.get("cylinders", [])

        for i in range(self._scan_dim):
            angle = theta0 + 2.0 * math.pi * i / self._scan_dim
            direction = np.array([math.cos(angle), math.sin(angle)])

            min_dist = self._scan_range

            # Walls
            for (ax, ay), (bx, by) in walls:
                d = _ray_segment_intersection(
                    origin, direction,
                    np.array([ax, ay]), np.array([bx, by]),
                    max_range=min_dist,
                )
                if d is not None and d < min_dist:
                    min_dist = d

            # Boxes
            for box in boxes:
                d = _ray_box_intersection(
                    origin, direction, box, max_range=min_dist)
                if d is not None and d < min_dist:
                    min_dist = d

            # Cylinders
            for cyl in cylinders:
                cx, cy, r = cyl
                d = _ray_circle_intersection(
                    origin, direction, (cx, cy), r, max_range=min_dist)
                if d is not None and d < min_dist:
                    min_dist = d

            scan[i] = min_dist

        return scan

    def _compute_reward(
        self,
        action: np.ndarray,
        scan: np.ndarray,
        prev_pose: np.ndarray,
        prev_dist_to_goal: float,
    ) -> float:
        """Compute scalar reward."""
        r = 0.0

        # Progress: negative distance to goal decreased
        dist = float(np.linalg.norm(self._pose[:2] - self._goal_pose[:2]))
        r += self._rw.get("progress", 1.0) * (prev_dist_to_goal - dist)

        # Clearance: proportional to min scan
        min_s = float(np.min(scan))
        r += self._rw.get("clearance", 0.1) * min_s

        # Smoothness: penalty for large action change
        action_diff = np.linalg.norm(action - self._last_action)
        r += self._rw.get("smoothness", -0.05) * action_diff

        # Spin penalty: penalise rotating while nearly stationary
        speed = float(np.linalg.norm(self._velocity[:2]))
        omega_abs = abs(float(self._velocity[2]))
        r += self._rw.get("spin", -0.02) * omega_abs * float(speed < 0.02)

        return float(r)

    # ------------------------------------------------------------------
    #  Properties
    # ------------------------------------------------------------------

    @property
    def world_name(self) -> str:
        return self._world_state.get("name", self._world_name)

    @property
    def pose(self) -> np.ndarray:
        return self._pose.copy()

    @property
    def goal_pose(self) -> np.ndarray:
        return self._goal_pose.copy()
