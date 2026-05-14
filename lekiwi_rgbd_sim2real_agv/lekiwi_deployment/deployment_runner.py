"""Main deployment loop on LeKiwi real robot.

Connects RL policy inference with real sensor streams and
safety filtering, sending commands via ZMQ to the LeKiwi base.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

import numpy as np
import torch
import cv2

from lekiwi_deployment.observation_assembler import ObservationAssembler
from safety.safety_filter import SafetyFilter, SafetyConfig
from control.action_adapter import ActionAdapter


class DeploymentRunner:
    """Run RL policy on real LeKiwi robot.

    Parameters
    ----------
    trainer : TD3BC or IQL or BehaviorCloning
        Trained policy with ``predict`` method.
    config : dict
        Deployment configuration.
    device : str
        Inference device.
    """

    def __init__(
        self,
        trainer,
        config: Dict[str, Any],
        device: str = "cuda",
    ) -> None:
        self.trainer = trainer
        self.config = config
        self.device = device

        deploy_cfg = config.get("deployment", {})
        lekiwi_cfg = deploy_cfg.get("lekiwi", {})
        camera_cfg = deploy_cfg.get("camera", {})
        safety_cfg = config.get("safety", {})

        # Observation assembler
        self.obs_assembler = ObservationAssembler(
            image_size=(
                camera_cfg.get("height", 480),
                camera_cfg.get("width", 640),
            ),
            scan_dim=safety_cfg.get("stop_distance", 0.15),  # placeholder, will be corrected
        )

        # Safety filter
        self.safety_filter = SafetyFilter(
            SafetyConfig(
                stop_distance=safety_cfg.get("stop_distance", 0.15),
                slow_distance=safety_cfg.get("slow_distance", 0.5),
                lateral_inhibit_distance=safety_cfg.get("lateral_inhibit_distance", 0.3),
                max_velocity=tuple(safety_cfg.get("max_velocity", [0.3, 0.3, 90.0])),
                max_acceleration=tuple(safety_cfg.get("max_acceleration", [0.5, 0.5, 180.0])),
                smoothing_alpha=safety_cfg.get("smoothing_alpha", 0.3),
                dt=safety_cfg.get("dt", 0.1),
            )
        )

        # Action adapter
        action_cfg = config.get("action", {})
        self.action_adapter = ActionAdapter(
            vx_limits=tuple(action_cfg.get("vx_limits", [-0.3, 0.3])),
            vy_limits=tuple(action_cfg.get("vy_limits", [-0.3, 0.3])),
            omega_limits=tuple(action_cfg.get("omega_limits", [-90.0, 90.0])),
            smoothing_alpha=safety_cfg.get("smoothing_alpha", 0.3),
            dt=safety_cfg.get("dt", 0.1),
        )

        # State
        self.mock_mode = deploy_cfg.get("mock_mode", False)
        self.max_episode_steps = deploy_cfg.get("max_episode_steps", 1000)
        self._step = 0
        self._episode_log: list[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run_episode(
        self,
        goal_vector: np.ndarray,
        rgb_source: str = "d435i",
        depth_source: str = "d435i",
    ) -> Dict[str, Any]:
        """Run a single navigation episode.

        Parameters
        ----------
        goal_vector : np.ndarray, shape (3,)
            Goal in robot frame [dx, dy, dtheta].
        rgb_source : str
            "d435i" or "mock".
        depth_source : str
            "d435i" or "mock".

        Returns
        -------
        dict with episode log entries.
        """
        self._episode_log = []
        self._step = 0
        self.safety_filter.reset_last_action()
        self.safety_filter.reset_stats()

        # Camera setup
        rgb_cap, depth_cap = self._setup_cameras(rgb_source, depth_source)

        start_time = time.time()

        for step in range(self.max_episode_steps):
            loop_start = time.time()

            # 1. Capture sensor data
            rgb_frame = self._capture_rgb(rgb_cap)
            depth_frame = self._capture_depth(depth_cap)

            depth_valid = depth_frame is not None and np.any(
                (depth_frame > 0.1) & (depth_frame < 5.0)
            )

            # 2. Get robot velocity (from odometry or mock)
            velocity = self._get_velocity()

            # 3. Assemble observation
            obs = self.obs_assembler.assemble(
                rgb_frame, depth_frame, velocity, goal_vector
            )

            # 4. RL policy inference
            inference_start = time.time()
            rl_action = self._predict_action(obs)
            inference_time = (time.time() - inference_start) * 1000  # ms

            # 5. Safety filter
            scan64 = obs["scan64"]
            safe_action, safety_info = self.safety_filter.filter(
                rl_action, scan64, depth_valid=depth_valid
            )

            # 6. Action adapter (clipping + smoothing)
            adapted_action = self.action_adapter.adapt_from_arrays(
                safe_action, self.action_adapter.action_to_array(
                    {"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0}
                ) if step == 0 else None
            )

            # 7. Send action to robot
            if not self.mock_mode:
                self._send_action(adapted_action)

            self.obs_assembler.update_prev_action(safe_action)

            # 8. Log
            loop_time = (time.time() - loop_start) * 1000
            self._episode_log.append({
                "step": step,
                "timestamp": time.time() - start_time,
                "rl_action": rl_action.tolist(),
                "safe_action": safe_action.tolist(),
                "adapted_action": {k: float(v) for k, v in adapted_action.items()},
                "safety_info": safety_info,
                "inference_ms": inference_time,
                "loop_ms": loop_time,
                "scan_front_min": float(np.nanmin(scan64[28:36])),
                "depth_valid": depth_valid,
            })

            # 9. Check termination
            dist_to_goal = np.linalg.norm(goal_vector[:2])
            if dist_to_goal < 0.5:
                print(f"[Deployment] Goal reached at step {step}")
                break
            if safety_info["emergency_stop"]:
                print(f"[Deployment] Emergency stop at step {step}")

            self._step += 1

        # Cleanup
        if rgb_cap is not None:
            rgb_cap.release()
        if depth_cap is not None:
            depth_cap.release()

        return {
            "episode_length": self._step,
            "log": self._episode_log,
            "safety_stats": self.safety_filter.get_stats(),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _setup_cameras(self, rgb_source: str, depth_source: str):
        """Setup camera capture objects."""
        rgb_cap = None
        depth_cap = None
        if not self.mock_mode:
            if rgb_source == "d435i":
                rgb_cap = cv2.VideoCapture(0)  # Adjust device index
            if depth_source == "d435i":
                depth_cap = cv2.VideoCapture(1)
        return rgb_cap, depth_cap

    def _capture_rgb(self, cap) -> np.ndarray | None:
        if cap is None or self.mock_mode:
            return np.zeros((480, 640, 3), dtype=np.uint8)
        ret, frame = cap.read()
        if ret:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return None

    def _capture_depth(self, cap) -> np.ndarray | None:
        if cap is None or self.mock_mode:
            return np.ones((480, 640), dtype=np.float32) * 2.0
        ret, frame = cap.read()
        if ret:
            return frame.astype(np.float32) / 1000.0  # mm → m
        return None

    def _get_velocity(self) -> np.ndarray:
        """Get current robot velocity. Override with real odometry."""
        return np.zeros(3, dtype=np.float32)

    def _predict_action(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """Run RL policy inference."""
        rgb_t = None
        if obs.get("rgb") is not None:
            rgb_t = torch.from_numpy(obs["rgb"]).float().unsqueeze(0) / 255.0
            if rgb_t.shape[1] != 3:
                rgb_t = rgb_t.permute(0, 3, 1, 2)
            rgb_t = rgb_t.to(self.device)

        scan_t = None
        if obs.get("scan64") is not None:
            s = np.nan_to_num(obs["scan64"], nan=5.0)
            scan_t = torch.from_numpy(s).float().unsqueeze(0).to(self.device)

        state_t = None
        if obs.get("state") is not None:
            state_t = torch.from_numpy(obs["state"]).float().unsqueeze(0).to(self.device)

        goal_t = None
        if obs.get("goal") is not None:
            goal_t = torch.from_numpy(obs["goal"]).float().unsqueeze(0).to(self.device)

        action = self.trainer.predict(rgb_t, scan_t, state_t, goal_t)
        return action[0]  # (3,)

    def _send_action(self, action_dict: Dict[str, float]) -> None:
        """Send action to LeKiwi via ZMQ. Override with actual ZMQ client."""
        pass  # TODO: Integrate with communication/client.py
