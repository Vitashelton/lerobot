"""
Complete control pipeline.

raw_policy (DWA/teleop) -> residual correction -> emergency shield -> action adapter

This is the main controller used in demos and evaluation runs.
"""

from __future__ import annotations

import copy
import time
from typing import Optional

import numpy as np

# Local imports
try:
    from lekiwi_rgbd_sim2real_agv.control.dwa_policy import DWAPolicy
    from lekiwi_rgbd_sim2real_agv.control.emergency_shield import EmergencyShield
    from lekiwi_rgbd_sim2real_agv.control.action_adapter import ActionAdapter
    from lekiwi_rgbd_sim2real_agv.learning.residual_model import ResidualSafetyModel
except ImportError:
    import sys
    import os

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from control.dwa_policy import DWAPolicy  # type: ignore[import-not-found]
    from control.emergency_shield import EmergencyShield  # type: ignore[import-not-found]
    from control.action_adapter import ActionAdapter  # type: ignore[import-not-found]
    from learning.residual_model import ResidualSafetyModel  # type: ignore[import-not-found]


class ResidualController:
    """Complete control pipeline with residual safety correction.

    Pipeline (executed in order):

    1. **Raw policy** -- DWA or external (e.g. teleop) produces a
       candidate action ``(vx, vy, omega)``.
    2. **Residual correction** (optional) -- a learned
       :class:`ResidualSafetyModel` predicts a delta correction
       that is added to the raw action.
    3. **Emergency shield** -- hard safety rules override the action
       when obstacles are extremely close.
    4. **Action adapter** -- clips to limits, limits acceleration,
       and smooths the output.

    Parameters
    ----------
    dwa_policy : DWAPolicy or None
        The DWA planner for computing raw actions.
    residual_model : ResidualSafetyModel or None
        Learned residual safety model (PyTorch).
    emergency_shield : EmergencyShield or None
        Hard safety shield.
    action_adapter : ActionAdapter or None
        Post-processing adapter (clip + smooth).
    use_residual : bool
        Whether to apply residual correction (requires model).
    use_shield : bool
        Whether to apply the emergency shield.
    device : str
        Torch device for residual model inference.
    """

    def __init__(
        self,
        dwa_policy: Optional[DWAPolicy] = None,
        residual_model: Optional[ResidualSafetyModel] = None,
        emergency_shield: Optional[EmergencyShield] = None,
        action_adapter: Optional[ActionAdapter] = None,
        use_residual: bool = True,
        use_shield: bool = True,
        device: str = "cpu",
    ) -> None:
        self.dwa_policy = dwa_policy or DWAPolicy()
        self.residual_model = residual_model
        self.emergency_shield = emergency_shield or EmergencyShield()
        self.action_adapter = action_adapter or ActionAdapter()

        self.use_residual = use_residual and self.residual_model is not None
        self.use_shield = use_shield
        self.device = device

        # State
        self.last_action: Optional[dict] = None
        self.last_raw_action: Optional[np.ndarray] = None
        self.last_residual: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------

    def compute(
        self,
        scan_m: np.ndarray,
        goal_position: np.ndarray,
        current_velocity: np.ndarray,
        front_min: Optional[float] = None,
        left_min: Optional[float] = None,
        right_min: Optional[float] = None,
        external_raw_action: Optional[dict] = None,
    ) -> dict:
        """Run the full control pipeline.

        Parameters
        ----------
        scan_m : np.ndarray, shape (N,)
            LiDAR scan in metres (typically 64 beams, 360 deg).
        goal_position : np.ndarray, shape (3,)
            Goal (dx, dy, dtheta) in robot frame.
        current_velocity : np.ndarray, shape (3,)
            Current velocity [vx, vy, omega_deg_s].
        front_min : float, optional
            Pre-computed minimum range in the front sector (saves
            re-computation if already known).
        left_min : float, optional
            Pre-computed minimum range in the left sector.
        right_min : float, optional
            Pre-computed minimum range in the right sector.
        external_raw_action : dict, optional
            If provided, use this as the raw action instead of running DWA.
            Useful for teleoperation or external planning modules.

        Returns
        -------
        dict
            ``{
                "action": {"x.vel": ..., "y.vel": ..., "theta.vel": ...},
                "raw_action": dict,
                "residual": {"delta": np.ndarray, "safe_action_raw": np.ndarray} or None,
                "shielded": bool,
                "shield_reason": str or None,
                "diagnostics": {...}
            }``
        """
        diagnostics: dict = {}

        # --------------------------------------------------------------
        # Step 1: Raw action from DWA or external
        # --------------------------------------------------------------
        if external_raw_action is not None:
            raw = copy.deepcopy(external_raw_action)
            diagnostics["policy"] = "external"
        else:
            t0 = time.perf_counter()
            raw = self.dwa_policy.compute_action(scan_m, goal_position, current_velocity)
            diagnostics["dwa_time_ms"] = (time.perf_counter() - t0) * 1000.0
            diagnostics["policy"] = "dwa"

        raw_array = np.array(
            [raw["x.vel"], raw["y.vel"], raw["theta.vel"]], dtype=np.float32
        )
        self.last_raw_action = raw_array

        # --------------------------------------------------------------
        # Step 2: Residual correction
        # --------------------------------------------------------------
        residual_info: Optional[dict] = None

        if self.use_residual and self.residual_model is not None:
            t0 = time.perf_counter()
            result = self.residual_model.predict_numpy(
                scan=scan_m,
                raw_action=raw_array,
                goal=goal_position,
                velocity=current_velocity,
                last_action=self.last_raw_action,
                delta_scale=1.0,
                device=self.device,
            )
            diagnostics["residual_time_ms"] = (time.perf_counter() - t0) * 1000.0

            delta = result["delta"]
            safe_array = result["safe_action"]

            self.last_residual = delta
            residual_info = {
                "delta": delta,
                "safe_action_raw": safe_array,
            }

            # Update the action dict for the next stages
            raw = {
                "x.vel": float(safe_array[0]),
                "y.vel": float(safe_array[1]),
                "theta.vel": float(safe_array[2]),
            }
        else:
            residual_info = None

        # --------------------------------------------------------------
        # Step 3: Emergency shield
        # --------------------------------------------------------------
        shielded = False
        shield_reason: Optional[str] = None

        if self.use_shield:
            t0 = time.perf_counter()
            shielded_result = self.emergency_shield.apply(
                raw, scan_m, front_min, left_min, right_min
            )
            diagnostics["shield_time_ms"] = (time.perf_counter() - t0) * 1000.0
            raw = shielded_result["action"]
            shielded = shielded_result["shielded"]
            shield_reason = shielded_result["trigger_reason"]

        # --------------------------------------------------------------
        # Step 4: Action adapter (clip + smooth)
        # --------------------------------------------------------------
        t0 = time.perf_counter()
        adapted = self.action_adapter.adapt(raw, self.last_action)
        diagnostics["adapter_time_ms"] = (time.perf_counter() - t0) * 1000.0

        self.last_action = adapted

        return {
            "action": adapted,
            "raw_action": {
                "x.vel": float(self.last_raw_action[0]),
                "y.vel": float(self.last_raw_action[1]),
                "theta.vel": float(self.last_raw_action[2]),
            },
            "residual": residual_info,
            "shielded": shielded,
            "shield_reason": shield_reason,
            "diagnostics": diagnostics,
        }

    # ------------------------------------------------------------------
    # Convenience: quick teleop integration
    # ------------------------------------------------------------------

    def compute_with_teleop(
        self,
        teleop_action: dict,
        scan_m: np.ndarray,
        current_velocity: np.ndarray,
        goal_position: Optional[np.ndarray] = None,
        front_min: Optional[float] = None,
        left_min: Optional[float] = None,
        right_min: Optional[float] = None,
    ) -> dict:
        """Run pipeline with an externally-supplied teleop action.

        This is the typical usage during human teleoperation demos:
        the human provides a raw command, and the controller adds
        residual correction + shielding + adaptation.

        Parameters
        ----------
        teleop_action : dict
            Human-provided action with keys ``"x.vel"``, ``"y.vel"``,
            ``"theta.vel"``.
        scan_m : np.ndarray
            Current LiDAR scan.
        current_velocity : np.ndarray
            Current robot velocity.
        goal_position : np.ndarray, optional
            Goal for residual correction context.  Defaults to zeros
            (no explicit goal).
        front_min, left_min, right_min : float, optional
            Pre-computed sector minimums for the shield.

        Returns
        -------
        dict
            Same structure as :meth:`compute`.
        """
        if goal_position is None:
            goal_position = np.zeros(3, dtype=np.float32)

        return self.compute(
            scan_m=scan_m,
            goal_position=goal_position,
            current_velocity=current_velocity,
            front_min=front_min,
            left_min=left_min,
            right_min=right_min,
            external_raw_action=teleop_action,
        )

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset internal state (last actions, residuals)."""
        self.last_action = None
        self.last_raw_action = None
        self.last_residual = None
