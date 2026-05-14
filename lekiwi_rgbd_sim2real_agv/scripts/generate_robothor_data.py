#!/usr/bin/env python
"""Generate RoboTHOR navigation dataset via ai2thor simulator.

RoboTHOR provides:
    - 75 indoor scenes (kitchens, living rooms, bedrooms, bathrooms)
    - RGB-D observations (300x300 → resized to 224x224)
    - Agent pose (x, y, z, rotation_y)
    - Discrete actions: MoveAhead, RotateLeft, RotateRight, LookUp, LookDown, Done
    - ObjectNav: navigate to find target object category

The discrete actions are mapped to continuous LeKiwi actions:
    MoveAhead   → [v, 0.0, 0.0]    (forward 0.25 m/s)
    RotateLeft  → [0.0, 0.0, +ω]   (turn +30 deg/s)
    RotateRight → [0.0, 0.0, -ω]   (turn -30 deg/s)
    Done/Stop   → [0.0, 0.0, 0.0]

Usage:
    # Generate 100 navigation episodes
    python scripts/generate_robothor_data.py \
        --num-episodes 100 \
        --max-steps 200 \
        --output-dir data/robothor \
        --seed 42

    # Generate with specific scene types
    python scripts/generate_robothor_data.py \
        --scenes FloorPlan_Train1_1 FloorPlan_Train1_2 \
        --num-episodes 50 \
        --output-dir data/robothor

    # Then convert to LeRobot format
    python scripts/convert_to_lerobot.py \
        --dataset robothor \
        --data-dir data/robothor \
        --output-dir data/lerobot/robothor_nav
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import cv2

# ai2thor imports
try:
    import ai2thor.controller
    AI2THOR_AVAILABLE = True
except ImportError:
    AI2THOR_AVAILABLE = False
    print("[WARN] ai2thor not installed. Run: pip install ai2thor")


# ======================================================================
# RoboTHOR scene list
# ======================================================================

ROBOTHOR_TRAIN_SCENES = [
    "FloorPlan_Train1_1", "FloorPlan_Train1_2", "FloorPlan_Train1_3",
    "FloorPlan_Train1_4", "FloorPlan_Train1_5",
    "FloorPlan_Train2_1", "FloorPlan_Train2_2", "FloorPlan_Train2_3",
    "FloorPlan_Train2_4", "FloorPlan_Train2_5",
    "FloorPlan_Train3_1", "FloorPlan_Train3_2", "FloorPlan_Train3_3",
    "FloorPlan_Train3_4", "FloorPlan_Train3_5",
    "FloorPlan_Train4_1", "FloorPlan_Train4_2", "FloorPlan_Train4_3",
    "FloorPlan_Train4_4", "FloorPlan_Train4_5",
    "FloorPlan_Train5_1", "FloorPlan_Train5_2", "FloorPlan_Train5_3",
    "FloorPlan_Train5_4", "FloorPlan_Train5_5",
    "FloorPlan_Train6_1", "FloorPlan_Train6_2", "FloorPlan_Train6_3",
    "FloorPlan_Train6_4", "FloorPlan_Train6_5",
    "FloorPlan_Train7_1", "FloorPlan_Train7_2", "FloorPlan_Train7_3",
    "FloorPlan_Train7_4", "FloorPlan_Train7_5",
]

ROBOTHOR_VAL_SCENES = [
    "FloorPlan_Val1_1", "FloorPlan_Val1_2", "FloorPlan_Val1_3",
    "FloorPlan_Val1_4", "FloorPlan_Val1_5",
    "FloorPlan_Val2_1", "FloorPlan_Val2_2", "FloorPlan_Val2_3",
    "FloorPlan_Val2_4", "FloorPlan_Val2_5",
    "FloorPlan_Val3_1", "FloorPlan_Val3_2", "FloorPlan_Val3_3",
    "FloorPlan_Val3_4", "FloorPlan_Val3_5",
]

ROBOTHOR_TARGET_OBJECTS = [
    "Apple", "BaseballBat", "BasketBall", "Bowl", "Candle",
    "Chair", "CoffeeTable", "Cup", "FloorLamp", "GarbageCan",
    "Laptop", "Microwave", "Mug", "Pillow", "Pot",
    "RemoteControl", "Sofa", "Television", "Toaster", "Vase",
]

# Object categories for ObjectNav-style tasks
ROBOTHOR_OBJECT_CATEGORIES = {
    "kitchen": ["Microwave", "Toaster", "Pot", "Mug", "Cup", "Bowl", "Apple"],
    "living_room": ["Television", "Sofa", "Chair", "CoffeeTable", "RemoteControl",
                     "Pillow", "Vase", "FloorLamp", "Candle"],
    "bedroom": ["Pillow", "Laptop", "Book", "AlarmClock", "BaseballBat", "BasketBall"],
    "bathroom": ["SoapBar", "ToiletPaper", "Towel", "Candle", "ScrubBrush"],
}

IMAGE_SIZE = (224, 224)
SCAN_DIM = 64

# ----------------------------------------------------------------------
# Action space
# ----------------------------------------------------------------------

FORWARD_SPEED = 0.25   # m/s
TURN_SPEED = 30.0      # deg/s
STEP_SIZE = 0.25       # meters per MoveAhead
TURN_ANGLE = 30.0      # degrees per RotateLeft/Right
DT = 0.1               # approximate time per step

DISCRETE_TO_CONTINUOUS = {
    "MoveAhead":  np.array([FORWARD_SPEED, 0.0, 0.0], dtype=np.float32),
    "MoveBack":   np.array([-FORWARD_SPEED, 0.0, 0.0], dtype=np.float32),
    "RotateLeft": np.array([0.0, 0.0, TURN_SPEED], dtype=np.float32),
    "RotateRight":np.array([0.0, 0.0, -TURN_SPEED], dtype=np.float32),
    "Done":       np.array([0.0, 0.0, 0.0], dtype=np.float32),
}

DISCRETE_ACTIONS = ["MoveAhead", "RotateLeft", "RotateRight", "Done"]


# ======================================================================
# Depth → Scan64 (same as in perception/depth_to_scan.py)
# ======================================================================

def depth_to_scan64(depth_m: np.ndarray, scan_dim: int = 64) -> np.ndarray:
    """Convert depth image to 64-beam polar scan."""
    h, w = depth_m.shape
    if h < 3 or w < 3:
        return np.full(scan_dim, np.nan, dtype=np.float32)

    slice_frac = 0.3
    band_half = max(1, int(h * slice_frac / 2.0))
    row_center = h // 2
    band = depth_m[row_center - band_half : row_center + band_half, :]

    bin_edges = np.linspace(0, w, scan_dim + 1, dtype=np.int32)
    meter_scan = np.full(scan_dim, np.nan, dtype=np.float32)

    for i in range(scan_dim):
        col_start, col_end = bin_edges[i], bin_edges[i + 1]
        if col_end <= col_start:
            continue
        bin_pixels = band[:, col_start:col_end].ravel()
        valid = bin_pixels[~np.isnan(bin_pixels)]
        if len(valid) > 0:
            meter_scan[i] = float(np.percentile(valid, 10.0))

    return meter_scan


# ======================================================================
# Episode generator
# ======================================================================

class RoboTHORGenerator:
    """Generate navigation episodes using ai2thor RoboTHOR scenes.

    Parameters
    ----------
    scenes : list[str]
        RoboTHOR scene names.
    target_objects : list[str]
        Object categories for ObjectNav goal specification.
    image_size : tuple[int, int]
        Output image (H, W).
    headless : bool
        Run without rendering window.
    """

    def __init__(
        self,
        scenes: List[str] | None = None,
        target_objects: List[str] | None = None,
        image_size: Tuple[int, int] = (224, 224),
        headless: bool = True,
        field_of_view: float = 90.0,
    ) -> None:
        if not AI2THOR_AVAILABLE:
            raise RuntimeError(
                "ai2thor not installed. Run: pip install ai2thor"
            )

        self.scenes = scenes or ROBOTHOR_TRAIN_SCENES[:5]
        self.target_objects = target_objects or ROBOTHOR_TARGET_OBJECTS
        self.image_size = image_size
        self.headless = headless
        self.field_of_view = field_of_view

        self.controller: Optional[ai2thor.controller.Controller] = None

    # ------------------------------------------------------------------
    # Controller lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Initialize ai2thor controller."""
        self.controller = ai2thor.controller.Controller(
            quality="Medium",
            fullscreen=False,
            headless=self.headless,
            fieldOfView=self.field_of_view,
            width=self.image_size[1],
            height=self.image_size[0],
        )
        print(f"[RoboTHOR] Controller initialized, {len(self.scenes)} scenes")

    def stop(self) -> None:
        """Stop ai2thor controller."""
        if self.controller is not None:
            self.controller.stop()
            self.controller = None

    # ------------------------------------------------------------------
    # Episode generation
    # ------------------------------------------------------------------

    def generate_episodes(
        self,
        num_episodes: int = 100,
        max_steps: int = 200,
        seed: int = 42,
        output_dir: str = "data/robothor",
    ) -> str:
        """Generate navigation episodes and save to disk.

        Parameters
        ----------
        num_episodes : int
            Total episodes to generate.
        max_steps : int
            Maximum steps per episode.
        seed : int
            Random seed.
        output_dir : str
            Output directory for episode data.

        Returns
        -------
        str
            Output directory path.
        """
        if self.controller is None:
            self.start()

        rng = random.Random(seed)
        np_rng = np.random.RandomState(seed)
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        # Episode metadata
        all_metadata: List[Dict[str, Any]] = []

        for ep_idx in range(num_episodes):
            print(f"\n[Episode {ep_idx + 1}/{num_episodes}]")

            # Pick random scene and target
            scene = rng.choice(self.scenes)
            target_obj = rng.choice(self.target_objects)

            episode_data = self._run_single_episode(
                scene, target_obj, max_steps, rng, np_rng, ep_idx
            )

            if episode_data is not None:
                self._save_episode(episode_data, out_path, ep_idx)
                all_metadata.append({
                    "episode_id": ep_idx,
                    "scene": scene,
                    "target_object": target_obj,
                    "num_steps": episode_data["num_steps"],
                    "goal_reached": episode_data["goal_reached"],
                    "collision": episode_data["collision"],
                })

        # Save metadata
        with open(out_path / "episodes_metadata.json", "w") as f:
            json.dump(all_metadata, f, indent=2)

        # Print summary
        success_count = sum(1 for m in all_metadata if m["goal_reached"])
        collision_count = sum(1 for m in all_metadata if m["collision"])
        avg_steps = np.mean([m["num_steps"] for m in all_metadata])
        print(f"\n{'='*50}")
        print(f"  RoboTHOR Generation Summary")
        print(f"  Total episodes:    {len(all_metadata)}")
        print(f"  Success rate:      {success_count}/{len(all_metadata)} ({success_count/max(len(all_metadata),1):.1%})")
        print(f"  Collision rate:    {collision_count}/{len(all_metadata)} ({collision_count/max(len(all_metadata),1):.1%})")
        print(f"  Avg episode steps: {avg_steps:.1f}")
        print(f"  Output:            {out_path}")
        print(f"{'='*50}")

        return str(out_path)

    def _run_single_episode(
        self,
        scene: str,
        target_object: str,
        max_steps: int,
        rng: random.Random,
        np_rng: np.random.RandomState,
        ep_idx: int,
    ) -> Dict[str, Any] | None:
        """Run a single navigation episode.

        Uses a simple exploration policy (random + bias toward target)
        to collect trajectory data.
        """
        controller = self.controller

        # Initialize scene
        try:
            controller.reset(scene=scene)
            # Randomize agent starting position
            event = controller.step(action="GetReachablePositions")
            reachable = event.metadata["actionReturn"]
            if not reachable:
                print(f"  Scene {scene}: no reachable positions, skipping")
                return None

            # Pick random start
            start_pos = rng.choice(reachable)
            controller.step(
                action="Teleport",
                position=start_pos,
                rotation=dict(x=0, y=rng.uniform(0, 360), z=0),
            )
        except Exception as e:
            print(f"  Scene {scene} init failed: {e}, skipping")
            return None

        # Find target object positions
        target_positions = self._find_object_positions(target_object)
        if not target_positions:
            print(f"  Target '{target_object}' not found in {scene}, skipping")
            return None

        target_pos = rng.choice(target_positions)
        agent_pos = self._get_agent_position()

        print(f"  Scene: {scene}, Target: {target_object} at "
              f"({target_pos['x']:.1f}, {target_pos['z']:.1f}), "
              f"Start: ({agent_pos['x']:.1f}, {agent_pos['z']:.1f})")

        # Data buffers
        rgb_frames = []
        depth_frames = []
        scan64_frames = []
        actions = []
        states = []
        goals = []
        prev_actions = []
        rewards = []
        dones = []
        collisions = []
        goal_reached_flags = []
        robot_positions = []
        goal_positions_arr = []

        goal_reached = False
        had_collision = False
        last_action = np.zeros(3, dtype=np.float32)

        for step in range(max_steps):
            # Get observation
            event = controller.last_event
            rgb = event.frame.copy()  # (H, W, 3) RGB
            depth = event.depth_frame.copy()  # (H, W) float32 meters

            rgb_frames.append(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
            depth_frames.append(depth)

            # Compute Scan64 from depth
            scan64_frames.append(depth_to_scan64(depth, SCAN_DIM))

            # Agent state
            agent_pos = self._get_agent_position()
            agent_rot = self._get_agent_rotation()
            robot_positions.append([agent_pos["x"], agent_pos["z"]])

            # Goal in robot frame
            dx = target_pos["x"] - agent_pos["x"]
            dz = target_pos["z"] - agent_pos["z"]
            # Transform to robot frame (robot faces -z in ai2thor)
            heading_rad = np.deg2rad(agent_rot)
            dx_robot = dx * np.cos(-heading_rad) - dz * np.sin(-heading_rad)
            dz_robot = dx * np.sin(-heading_rad) + dz * np.cos(-heading_rad)
            goal_vec = np.array([dx_robot, dz_robot, 0.0], dtype=np.float32)
            goals.append(goal_vec)
            goal_positions_arr.append([target_pos["x"], target_pos["z"]])

            # Velocity state (approximate from last action)
            state = np.array(last_action, dtype=np.float32)
            states.append(state)

            # Choose action (exploration policy)
            action_name = self._exploration_policy(
                goal_vec, rng, step, max_steps
            )
            action = DISCRETE_TO_CONTINUOUS[action_name].copy()
            actions.append(action)
            prev_actions.append(last_action.copy())

            # Execute action
            event = controller.step(action=action_name)

            # Check outcome
            is_done = False
            is_collision = False
            is_success = False

            # Collision detection from action result
            if not event.metadata["lastActionSuccess"]:
                is_collision = True
                had_collision = True

            # Goal check
            dist_to_target = np.linalg.norm(goal_vec[:2])
            if dist_to_target < 0.5:
                is_success = True
                is_done = True
                goal_reached = True

            # Timeout
            if step >= max_steps - 1:
                is_done = True

            dones.append(is_done)
            collisions.append(is_collision)
            goal_reached_flags.append(is_success)
            rewards.append(0.0)  # Will be relabeled

            last_action = action.copy()

            if is_done:
                break

        T = len(actions)
        print(f"    Steps: {T}, Goal reached: {goal_reached}, Collision: {had_collision}")

        return {
            "episode_idx": ep_idx,
            "scene": scene,
            "target_object": target_object,
            "num_steps": T,
            "goal_reached": goal_reached,
            "collision": had_collision,
            "observations": {
                "rgb": np.array(rgb_frames, dtype=np.uint8),
                "depth": np.array(depth_frames, dtype=np.float32),
                "scan64": np.array(scan64_frames, dtype=np.float32),
                "state": np.array(states, dtype=np.float32),
                "goal": np.array(goals, dtype=np.float32),
                "prev_action": np.array(prev_actions, dtype=np.float32),
            },
            "actions": np.array(actions, dtype=np.float32),
            "rewards": np.array(rewards, dtype=np.float32),
            "dones": np.array(dones, dtype=bool),
            "info": {
                "collision": np.array(collisions, dtype=bool),
                "intervention": np.zeros(T, dtype=bool),
                "goal_reached": np.array(goal_reached_flags, dtype=bool),
                "robot_position": np.array(robot_positions, dtype=np.float32),
                "goal_position": np.array(goal_positions_arr, dtype=np.float32),
            },
        }

    def _exploration_policy(
        self,
        goal_vec: np.ndarray,
        rng: random.Random,
        step: int,
        max_steps: int,
    ) -> str:
        """Simple exploration policy with goal bias.

        Mix: 60% goal-directed, 20% random turn, 20% forward.
        """
        dist = np.linalg.norm(goal_vec[:2])

        if dist < 0.5:
            return "Done"

        # Goal-directed: align heading with goal
        goal_angle = np.rad2deg(np.arctan2(goal_vec[1], goal_vec[0]))

        if abs(goal_angle) < 15.0:
            # Facing goal → move forward
            if rng.random() < 0.7:
                return "MoveAhead"
            else:
                return rng.choice(["RotateLeft", "RotateRight"])
        elif goal_angle > 0:
            # Goal is to the left
            if rng.random() < 0.6:
                return "RotateLeft"
            else:
                return "MoveAhead"
        else:
            # Goal is to the right
            if rng.random() < 0.6:
                return "RotateRight"
            else:
                return "MoveAhead"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _find_object_positions(self, object_type: str) -> List[Dict]:
        """Find all instances of an object in the current scene."""
        controller = self.controller
        positions = []
        event = controller.step(action="GetReachablePositions")
        all_objects = controller.last_event.metadata.get("objects", [])

        for obj in all_objects:
            if obj.get("objectType", "").lower() == object_type.lower():
                positions.append(obj["position"])
            # Also check by name substring
            elif object_type.lower() in obj.get("objectType", "").lower():
                positions.append(obj["position"])

        # Fallback: use visible objects
        if not positions:
            visible = [
                o for o in all_objects
                if o.get("visible", False)
                and object_type.lower() in o.get("objectType", "").lower()
            ]
            positions = [o["position"] for o in visible]

        return positions

    def _get_agent_position(self) -> Dict[str, float]:
        """Get current agent position."""
        event = self.controller.last_event
        meta = event.metadata
        return meta.get("agent", meta).get("position", {"x": 0, "y": 0, "z": 0})

    def _get_agent_rotation(self) -> float:
        """Get current agent rotation (degrees around Y axis)."""
        event = self.controller.last_event
        meta = event.metadata
        return meta.get("agent", meta).get("rotation", {}).get("y", 0.0)

    def _is_visible(self, obj: Dict) -> bool:
        """Check if object is visible from current view."""
        return obj.get("visible", False)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def _save_episode(
        self,
        episode: Dict[str, Any],
        output_dir: Path,
        ep_idx: int,
    ) -> None:
        """Save a single episode to disk in unified format."""
        ep_dir = output_dir / f"episode_{ep_idx:05d}"
        ep_dir.mkdir(parents=True, exist_ok=True)

        rgb_dir = ep_dir / "rgb"
        depth_dir = ep_dir / "depth"
        rgb_dir.mkdir(exist_ok=True)
        depth_dir.mkdir(exist_ok=True)

        obs = episode["observations"]
        T = episode["num_steps"]

        # Save RGB frames
        for t in range(T):
            rgb = obs["rgb"][t]
            cv2.imwrite(str(rgb_dir / f"frame_{t:06d}.png"),
                        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

        # Save depth frames
        for t in range(T):
            depth = obs["depth"][t]
            np.save(str(depth_dir / f"frame_{t:06d}.npy"), depth)

        # Save metadata
        metadata = {
            "actions": episode["actions"],
            "rewards": episode["rewards"],
            "dones": episode["dones"],
            "goal": obs["goal"],
            "state": obs["state"],
            "scan64": obs["scan64"],
            "prev_action": obs["prev_action"],
            "collision": episode["info"]["collision"],
            "intervention": episode["info"]["intervention"],
            "goal_reached": episode["info"]["goal_reached"],
            "robot_position": episode["info"]["robot_position"],
            "goal_position": episode["info"]["goal_position"],
            "scene": episode["scene"],
            "target_object": episode["target_object"],
            "episode_idx": ep_idx,
        }
        np.savez_compressed(str(ep_dir / "metadata.npz"), **metadata)


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate RoboTHOR navigation dataset"
    )
    parser.add_argument("--num-episodes", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--scenes", type=str, nargs="*", default=None,
                        help="Specific scene names (default: first 5 training scenes)")
    parser.add_argument("--target-objects", type=str, nargs="*", default=None,
                        help="Specific target objects")
    parser.add_argument("--output-dir", type=str, default="data/robothor")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--no-headless", dest="headless", action="store_false",
                        help="Show rendering window")
    parser.add_argument("--image-size", type=int, nargs=2, default=[224, 224])
    parser.add_argument("--only-val", action="store_true",
                        help="Use validation scenes instead of training")
    args = parser.parse_args()

    if not AI2THOR_AVAILABLE:
        print("ERROR: ai2thor not installed.")
        print("Run: pip install ai2thor")
        print("The first run will download ~500MB of Unity environment automatically.")
        return 1

    scenes = args.scenes
    if scenes is None:
        scenes = ROBOTHOR_VAL_SCENES if args.only_val else ROBOTHOR_TRAIN_SCENES[:5]

    target_objects = args.target_objects
    if target_objects is None:
        target_objects = ROBOTHOR_TARGET_OBJECTS

    generator = RoboTHORGenerator(
        scenes=scenes,
        target_objects=target_objects,
        image_size=tuple(args.image_size),
        headless=args.headless,
    )

    try:
        generator.start()
        output_path = generator.generate_episodes(
            num_episodes=args.num_episodes,
            max_steps=args.max_steps,
            seed=args.seed,
            output_dir=args.output_dir,
        )
        print(f"\nData saved to: {output_path}")
        print(f"\nNext step:")
        print(f"  python scripts/convert_to_lerobot.py \\")
        print(f"      --dataset robothor \\")
        print(f"      --data-dir {output_path} \\")
        print(f"      --output-dir data/lerobot/robothor_nav")
    finally:
        generator.stop()

    return 0


if __name__ == "__main__":
    exit(main())
