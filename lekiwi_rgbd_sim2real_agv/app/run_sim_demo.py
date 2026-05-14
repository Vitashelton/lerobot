"""LeKiwi 2D navigation simulation demo.

This is the main simulation demo:
- real 2D robot pose update
- holonomic LeKiwi action [vx, vy, omega]
- ray-cast scan
- DWA / rule / safety shield
- collision / success / timeout metrics
- top-down visualization
"""

import argparse
import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


# -----------------------------
# CLI
# -----------------------------

def str2bool(v):
    if isinstance(v, bool):
        return v
    return str(v).lower() in ("true", "1", "yes", "y")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scene_type", type=str, default="warehouse_like",
                   choices=["empty", "single_obstacle", "narrow_gap", "cluttered_lab", "warehouse_like"])
    p.add_argument("--policy", type=str, default="dwa_shield",
                   choices=["rule", "dwa", "dwa_shield", "mock_logoplanner", "full"])
    p.add_argument("--num_episodes", type=int, default=5)
    p.add_argument("--max_steps", type=int, default=300)
    p.add_argument("--display", type=str2bool, default=True)
    p.add_argument("--record_video", type=str2bool, default=False)
    p.add_argument("--output_dir", type=str, default="logs/sim_demo")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# -----------------------------
# Geometry
# -----------------------------

@dataclass
class RectObstacle:
    x: float
    y: float
    w: float
    h: float
    name: str = "obstacle"

    @property
    def xmin(self):
        return self.x - self.w / 2

    @property
    def xmax(self):
        return self.x + self.w / 2

    @property
    def ymin(self):
        return self.y - self.h / 2

    @property
    def ymax(self):
        return self.y + self.h / 2

    def contains_point(self, px, py):
        return self.xmin <= px <= self.xmax and self.ymin <= py <= self.ymax

    def circle_collision(self, cx, cy, r):
        nearest_x = np.clip(cx, self.xmin, self.xmax)
        nearest_y = np.clip(cy, self.ymin, self.ymax)
        return (cx - nearest_x) ** 2 + (cy - nearest_y) ** 2 <= r ** 2

    def distance_to_point(self, px, py):
        dx = max(self.xmin - px, 0, px - self.xmax)
        dy = max(self.ymin - py, 0, py - self.ymax)
        return math.hypot(dx, dy)


# -----------------------------
# 2D LeKiwi environment
# -----------------------------

class LeKiwi2DEnv:
    def __init__(self, scene_type="warehouse_like", seed=42, max_steps=300):
        self.scene_type = scene_type
        self.rng = np.random.RandomState(seed)
        self.max_steps = max_steps

        self.world_w = 5.0
        self.world_h = 6.0
        self.robot_radius = 0.18
        self.dt = 0.08

        self.scan_dim = 64
        self.scan_fov = math.radians(100)
        self.scan_max = 5.0
        self.scan_min = 0.15

        self.max_vx = 0.35
        self.max_vy = 0.25
        self.max_w = 1.5

        self.obstacles = []
        self.pose = None
        self.vel = np.zeros(3, dtype=np.float32)
        self.last_action = np.zeros(3, dtype=np.float32)
        self.goal = None
        self.step_count = 0
        self.trajectory = []
        self.smoothness_acc = 0.0
        self.spin_count = 0
        self.path_length = 0.0
        self.prev_dist_goal = None
        self.min_clearance_seen = float("inf")

    def build_scene(self):
        obs = []

        if self.scene_type == "empty":
            pass

        elif self.scene_type == "single_obstacle":
            obs.append(RectObstacle(2.5, 3.0, 0.55, 0.8, "box"))

        elif self.scene_type == "narrow_gap":
            obs.append(RectObstacle(2.5, 2.35, 1.1, 0.55, "left_box"))
            obs.append(RectObstacle(2.5, 3.65, 1.1, 0.55, "right_box"))

        elif self.scene_type == "cluttered_lab":
            obs.extend([
                RectObstacle(1.8, 2.0, 1.2, 0.6, "table"),
                RectObstacle(2.9, 4.1, 0.6, 0.6, "chair"),
                RectObstacle(3.6, 2.6, 0.7, 1.0, "box"),
                RectObstacle(2.2, 3.4, 0.45, 0.45, "chair"),
            ])

        elif self.scene_type == "warehouse_like":
            # two long shelves + one pallet
            obs.extend([
                RectObstacle(2.6, 0.75, 3.7, 0.45, "lower_shelf"),
                RectObstacle(2.6, 5.25, 3.7, 0.45, "upper_shelf"),
                RectObstacle(2.7, 3.0, 0.55, 0.85, "pallet"),
            ])

        return obs

    def reset(self):
        self.obstacles = self.build_scene()
        self.pose = np.array([0.65, 3.0, 0.0], dtype=np.float32)

        if self.scene_type == "narrow_gap":
            self.goal = np.array([4.35, 3.0], dtype=np.float32)
        elif self.scene_type == "cluttered_lab":
            self.goal = np.array([4.4, 4.7], dtype=np.float32)
        else:
            self.goal = np.array([4.35, 3.0], dtype=np.float32)

        self.vel[:] = 0
        self.last_action[:] = 0
        self.step_count = 0
        self.trajectory = [self.pose[:2].copy()]
        self.smoothness_acc = 0.0
        self.spin_count = 0
        self.path_length = 0.0
        self.prev_dist_goal = self.distance_to_goal()
        self.min_clearance_seen = float("inf")

        return self.get_obs(), self.get_info(False, False, False)

    def distance_to_goal(self):
        return float(np.linalg.norm(self.goal - self.pose[:2]))

    def in_bounds(self, x, y):
        return self.robot_radius <= x <= self.world_w - self.robot_radius and self.robot_radius <= y <= self.world_h - self.robot_radius

    def collision_at(self, x, y):
        if not self.in_bounds(x, y):
            return True
        for ob in self.obstacles:
            if ob.circle_collision(x, y, self.robot_radius):
                return True
        return False

    def point_occupied(self, x, y):
        if x < 0 or x > self.world_w or y < 0 or y > self.world_h:
            return True
        for ob in self.obstacles:
            if ob.contains_point(x, y):
                return True
        return False

    def raycast_scan(self, pose=None):
        if pose is None:
            pose = self.pose
        x, y, th = float(pose[0]), float(pose[1]), float(pose[2])

        scan = np.full(self.scan_dim, self.scan_max, dtype=np.float32)
        angles = np.linspace(-self.scan_fov / 2, self.scan_fov / 2, self.scan_dim)

        step = 0.025
        for i, a in enumerate(angles):
            global_a = th + a
            d = self.scan_min
            while d <= self.scan_max:
                px = x + d * math.cos(global_a)
                py = y + d * math.sin(global_a)
                if self.point_occupied(px, py):
                    scan[i] = d
                    break
                d += step

        # small sensor noise
        scan += self.rng.normal(0, 0.01, size=scan.shape).astype(np.float32)
        scan = np.clip(scan, self.scan_min, self.scan_max)
        return scan

    def min_clearance(self):
        x, y = self.pose[:2]
        d = min(x, y, self.world_w - x, self.world_h - y)
        for ob in self.obstacles:
            d = min(d, ob.distance_to_point(x, y))
        return float(max(0.0, d - self.robot_radius))

    def get_obs(self):
        scan = self.raycast_scan()
        rel_goal_world = self.goal - self.pose[:2]

        c, s = math.cos(-self.pose[2]), math.sin(-self.pose[2])
        gx = c * rel_goal_world[0] - s * rel_goal_world[1]
        gy = s * rel_goal_world[0] + c * rel_goal_world[1]

        return {
            "scan": scan,
            "goal": np.array([gx, gy], dtype=np.float32),
            "velocity": self.vel.copy(),
            "last_action": self.last_action.copy(),
            "pose": self.pose.copy(),
        }

    def step(self, action):
        self.step_count += 1

        raw = np.asarray(action, dtype=np.float32)
        raw[0] = np.clip(raw[0], -self.max_vx, self.max_vx)
        raw[1] = np.clip(raw[1], -self.max_vy, self.max_vy)
        raw[2] = np.clip(raw[2], -self.max_w, self.max_w)

        # action smoothing
        alpha = 0.55
        act = alpha * self.last_action + (1 - alpha) * raw

        old_pose = self.pose.copy()

        # robot-frame velocity to world-frame velocity
        th = float(self.pose[2])
        c, s = math.cos(th), math.sin(th)
        vx_world = c * act[0] - s * act[1]
        vy_world = s * act[0] + c * act[1]

        new_pose = self.pose.copy()
        new_pose[0] += vx_world * self.dt
        new_pose[1] += vy_world * self.dt
        new_pose[2] += act[2] * self.dt
        new_pose[2] = (new_pose[2] + math.pi) % (2 * math.pi) - math.pi

        collision = self.collision_at(new_pose[0], new_pose[1])
        if not collision:
            self.pose = new_pose
            self.path_length += float(np.linalg.norm(self.pose[:2] - old_pose[:2]))

        self.vel = act.copy()
        self.smoothness_acc += float(np.linalg.norm(act - self.last_action))
        self.last_action = act.copy()
        self.trajectory.append(self.pose[:2].copy())

        dist_goal = self.distance_to_goal()
        progress = self.prev_dist_goal - dist_goal
        self.prev_dist_goal = dist_goal

        clearance = self.min_clearance()
        self.min_clearance_seen = min(self.min_clearance_seen, clearance)

        if abs(act[2]) > 0.8 and np.linalg.norm(act[:2]) < 0.03:
            self.spin_count += 1

        success = dist_goal < 0.28
        timeout = self.step_count >= self.max_steps

        reward = 4.0 * progress
        reward += 10.0 if success else 0.0
        reward -= 10.0 if collision else 0.0
        reward -= 0.05 * np.linalg.norm(act - self.last_action)
        reward -= 0.02 if clearance < 0.25 else 0.0
        reward -= 0.02 if self.spin_count > 0 else 0.0

        done = success or collision or timeout
        return self.get_obs(), reward, done, False, self.get_info(success, collision, timeout)

    def get_info(self, success, collision, timeout):
        return {
            "success": bool(success),
            "collision": bool(collision),
            "timeout": bool(timeout),
            "min_clearance": float(self.min_clearance()),
            "min_clearance_seen": float(self.min_clearance_seen),
            "path_length": float(self.path_length),
            "smoothness": float(self.smoothness_acc),
            "spin_count": int(self.spin_count),
            "step": int(self.step_count),
            "dist_goal": float(self.distance_to_goal()),
        }


# -----------------------------
# Policies
# -----------------------------

def sector_mins(scan):
    n = len(scan)
    right = np.min(scan[: n // 3])
    front = np.min(scan[n // 3: 2 * n // 3])
    left = np.min(scan[2 * n // 3:])
    return {"right": float(right), "front": float(front), "left": float(left)}


def rule_policy(obs):
    scan = obs["scan"]
    goal = obs["goal"]
    sectors = sector_mins(scan)

    angle = math.atan2(goal[1], goal[0])
    dist = np.linalg.norm(goal)

    vx = 0.22 if dist > 0.5 else 0.12
    vy = 0.0
    omega = np.clip(1.2 * angle, -1.0, 1.0)

    if sectors["front"] < 0.45:
        vx = 0.0
        omega = 0.9 if sectors["left"] > sectors["right"] else -0.9

    return np.array([vx, vy, omega], dtype=np.float32)


def rollout_score(env, obs, action):
    pose = obs["pose"].copy()
    goal_world = env.goal.copy()
    last = obs["last_action"]

    score = 0.0
    min_clear = float("inf")
    collision = False

    act = np.asarray(action, dtype=np.float32)

    for _ in range(8):
        th = float(pose[2])
        c, s = math.cos(th), math.sin(th)
        vxw = c * act[0] - s * act[1]
        vyw = s * act[0] + c * act[1]

        pose[0] += vxw * env.dt
        pose[1] += vyw * env.dt
        pose[2] += act[2] * env.dt

        if env.collision_at(pose[0], pose[1]):
            collision = True
            break

        # Fast approximate clearance:
        # Do not raycast inside every DWA rollout step.
        # Using geometric distance is much faster for demo.
        clearance = min(
            [pose[0], pose[1], env.world_w - pose[0], env.world_h - pose[1]]
            + [ob.distance_to_point(pose[0], pose[1]) for ob in env.obstacles]
        ) - env.robot_radius
        min_clear = min(min_clear, float(clearance))

    if collision:
        return -1e6

    goal_dist = float(np.linalg.norm(goal_world - pose[:2]))
    clear_score = min(min_clear, 1.0)
    speed_score = float(np.linalg.norm(act[:2]))
    smooth_penalty = float(np.linalg.norm(act - last))
    spin_penalty = 0.3 if abs(act[2]) > 1.0 and np.linalg.norm(act[:2]) < 0.04 else 0.0

    score = -2.0 * goal_dist + 1.2 * clear_score + 0.2 * speed_score - 0.3 * smooth_penalty - spin_penalty
    return score


def dwa_policy(env, obs):
    candidates = []

    # Fast DWA sampling for real-time demo.
    vx_set = np.linspace(0.00, 0.30, 5)
    vy_set = np.linspace(-0.12, 0.12, 3)
    w_set = np.linspace(-1.0, 1.0, 5)

    best_score = -1e9
    best = np.zeros(3, dtype=np.float32)

    for vx in vx_set:
        for vy in vy_set:
            for w in w_set:
                a = np.array([vx, vy, w], dtype=np.float32)
                sc = rollout_score(env, obs, a)
                if sc > best_score:
                    best_score = sc
                    best = a

    return best


def mock_logoplanner_policy(obs):
    # pretend a pretrained policy: goal-seeking but not very safe
    goal = obs["goal"]
    angle = math.atan2(goal[1], goal[0])
    vx = 0.25
    vy = 0.08 * np.tanh(goal[1])
    w = np.clip(0.8 * angle, -1.0, 1.0)
    return np.array([vx, vy, w], dtype=np.float32)


def emergency_shield(obs, action):
    scan = obs["scan"]
    sectors = sector_mins(scan)
    a = np.asarray(action, dtype=np.float32).copy()
    active = False
    reason = "normal"

    if sectors["front"] < 0.35 and a[0] > 0:
        a[0] = 0.0
        active = True
        reason = "front_stop"

        if sectors["left"] > sectors["right"]:
            a[2] = 0.7
        else:
            a[2] = -0.7

    if sectors["left"] < 0.25 and a[1] > 0:
        a[1] = 0.0
        active = True
        reason = "left_block"

    if sectors["right"] < 0.25 and a[1] < 0:
        a[1] = 0.0
        active = True
        reason = "right_block"

    return a, {"active": active, "reason": reason, **sectors}


def choose_action(env, obs, policy_name):
    if policy_name == "rule":
        raw = rule_policy(obs)
        final = raw
        shield = {"active": False, "reason": "none"}
    elif policy_name == "dwa":
        raw = dwa_policy(env, obs)
        final = raw
        shield = {"active": False, "reason": "none"}
    elif policy_name == "dwa_shield":
        raw = dwa_policy(env, obs)
        final, shield = emergency_shield(obs, raw)
    elif policy_name == "mock_logoplanner":
        raw = mock_logoplanner_policy(obs)
        final = raw
        shield = {"active": False, "reason": "none"}
    elif policy_name == "full":
        # For now: mock LoGoPlanner + shield.
        # Later replace this with residual model + shield.
        raw = mock_logoplanner_policy(obs)
        final, shield = emergency_shield(obs, raw)
    else:
        raw = rule_policy(obs)
        final = raw
        shield = {"active": False, "reason": "none"}

    return final, raw, shield


# -----------------------------
# Visualization
# -----------------------------

class Renderer:
    def __init__(self, display=True, record_video=False, output_dir="logs/sim_demo"):
        self.display = display
        self.record_video = record_video
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        import matplotlib
        if not display:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        self.plt = plt
        self.fig, self.ax = plt.subplots(figsize=(7, 8))

        self.frames = []

    def render(self, env, obs, action, raw_action, shield, episode, step, policy):
        plt = self.plt
        ax = self.ax
        ax.clear()

        ax.set_xlim(0, env.world_w)
        ax.set_ylim(0, env.world_h)
        ax.set_aspect("equal")
        ax.set_title(f"LeKiwi 2D Navigation | {env.scene_type} | {policy}")
        ax.grid(True, alpha=0.25)

        # obstacles
        for ob in env.obstacles:
            rect = plt.Rectangle((ob.xmin, ob.ymin), ob.w, ob.h, color="gray", alpha=0.8)
            ax.add_patch(rect)
            ax.text(ob.x, ob.y, ob.name, ha="center", va="center", fontsize=8, color="white")

        # trajectory
        traj = np.array(env.trajectory)
        if len(traj) > 1:
            ax.plot(traj[:, 0], traj[:, 1], "b-", linewidth=2, label="trajectory")

        # scan rays
        pose = env.pose
        scan = obs["scan"]
        angles = np.linspace(-env.scan_fov / 2, env.scan_fov / 2, env.scan_dim)
        for i in range(0, env.scan_dim, 4):
            a = pose[2] + angles[i]
            d = scan[i]
            x2 = pose[0] + d * math.cos(a)
            y2 = pose[1] + d * math.sin(a)
            ax.plot([pose[0], x2], [pose[1], y2], color="orange", alpha=0.25, linewidth=0.8)

        # robot
        robot = plt.Circle((pose[0], pose[1]), env.robot_radius, color="red", alpha=0.85)
        ax.add_patch(robot)
        ax.arrow(
            pose[0], pose[1],
            0.35 * math.cos(pose[2]),
            0.35 * math.sin(pose[2]),
            head_width=0.08,
            color="black",
        )

        # goal
        ax.plot(env.goal[0], env.goal[1], "g*", markersize=18, label="goal")

        # action arrow in world frame
        th = pose[2]
        c, s = math.cos(th), math.sin(th)
        vxw = c * action[0] - s * action[1]
        vyw = s * action[0] + c * action[1]
        ax.arrow(pose[0], pose[1], vxw, vyw, color="purple", head_width=0.06, length_includes_head=True)

        info = env.get_info(False, False, False)
        txt = (
            f"Episode: {episode}  Step: {step}\n"
            f"Action final: [{action[0]:.2f}, {action[1]:.2f}, {action[2]:.2f}]\n"
            f"Action raw:   [{raw_action[0]:.2f}, {raw_action[1]:.2f}, {raw_action[2]:.2f}]\n"
            f"Shield: {shield.get('active')} | {shield.get('reason')}\n"
            f"Front/Left/Right: {shield.get('front', 0):.2f} / {shield.get('left', 0):.2f} / {shield.get('right', 0):.2f}\n"
            f"Dist goal: {info['dist_goal']:.2f} m\n"
            f"Min clearance: {info['min_clearance']:.2f} m"
        )
        ax.text(0.03, 0.98, txt, transform=ax.transAxes, va="top",
                bbox=dict(facecolor="white", alpha=0.85), fontsize=9)

        ax.legend(loc="lower right")

        if self.display:
            plt.pause(0.001)

    def save_trajectory(self, env, path):
        plt = self.plt
        fig, ax = plt.subplots(figsize=(7, 8))
        ax.set_xlim(0, env.world_w)
        ax.set_ylim(0, env.world_h)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.25)

        for ob in env.obstacles:
            ax.add_patch(plt.Rectangle((ob.xmin, ob.ymin), ob.w, ob.h, color="gray", alpha=0.8))

        traj = np.array(env.trajectory)
        if len(traj) > 1:
            ax.plot(traj[:, 0], traj[:, 1], "b-", linewidth=2)
        ax.plot(env.goal[0], env.goal[1], "g*", markersize=18)
        ax.plot(traj[0, 0], traj[0, 1], "ro", markersize=8)
        fig.savefig(path, dpi=160)
        plt.close(fig)


# -----------------------------
# Main
# -----------------------------

def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(args.seed)

    renderer = Renderer(args.display, args.record_video, args.output_dir)

    metrics = []

    for ep in range(args.num_episodes):
        env = LeKiwi2DEnv(scene_type=args.scene_type, seed=args.seed + ep, max_steps=args.max_steps)
        obs, info = env.reset()

        shield_count = 0
        raw_smooth = 0.0
        last_raw = np.zeros(3, dtype=np.float32)

        done = False
        final_info = None

        for step in range(args.max_steps):
            action, raw_action, shield = choose_action(env, obs, args.policy)
            if shield.get("active"):
                shield_count += 1

            raw_smooth += float(np.linalg.norm(raw_action - last_raw))
            last_raw = raw_action.copy()

            obs, reward, done, truncated, info = env.step(action)

            if args.display or args.record_video:
                renderer.render(env, obs, action, raw_action, shield, ep + 1, step, args.policy)

            if done:
                final_info = info
                break

        if final_info is None:
            final_info = info

        row = {
            "episode": ep,
            "scene": args.scene_type,
            "policy": args.policy,
            "success": int(final_info["success"]),
            "collision": int(final_info["collision"]),
            "timeout": int(final_info["timeout"]),
            "steps": final_info["step"],
            "path_length": final_info["path_length"],
            "min_clearance_seen": final_info["min_clearance_seen"],
            "smoothness": final_info["smoothness"],
            "raw_smoothness": raw_smooth,
            "spin_count": final_info["spin_count"],
            "shield_count": shield_count,
            "dist_goal": final_info["dist_goal"],
        }
        metrics.append(row)

        print(
            f"[ep {ep+1}/{args.num_episodes}] "
            f"success={row['success']} collision={row['collision']} timeout={row['timeout']} "
            f"steps={row['steps']} min_clear={row['min_clearance_seen']:.2f} "
            f"shield={row['shield_count']}"
        )

        if ep == 0:
            renderer.save_trajectory(env, out / "sim_demo_trajectory.png")

    # Save metrics
    csv_path = out / "sim_demo_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics[0].keys()))
        writer.writeheader()
        writer.writerows(metrics)

    # Print summary
    success_rate = np.mean([m["success"] for m in metrics])
    collision_rate = np.mean([m["collision"] for m in metrics])
    timeout_rate = np.mean([m["timeout"] for m in metrics])
    avg_min_clear = np.mean([m["min_clearance_seen"] for m in metrics])
    avg_smooth = np.mean([m["smoothness"] for m in metrics])

    print("\n=== Summary ===")
    print(f"success_rate:  {success_rate:.2f}")
    print(f"collision_rate:{collision_rate:.2f}")
    print(f"timeout_rate:  {timeout_rate:.2f}")
    print(f"avg_min_clear: {avg_min_clear:.2f} m")
    print(f"avg_smooth:    {avg_smooth:.2f}")
    print(f"metrics saved: {csv_path}")

    # Save bar chart
    import matplotlib
    if not args.display:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    names = ["success", "collision", "timeout"]
    vals = [success_rate, collision_rate, timeout_rate]
    ax.bar(names, vals)
    ax.set_ylim(0, 1.0)
    ax.set_title("Navigation Metrics")
    ax.set_ylabel("Rate")
    fig.tight_layout()
    fig.savefig(out / "sim_demo_bar_chart.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
