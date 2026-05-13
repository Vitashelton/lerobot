"""
Small MLP that predicts safety residual corrections to raw actions.

Input: [scan64, raw_action(3), goal(3), velocity(3), last_action(3)] = 76D
Output: delta_action [dvx, dvy, domega] = 3D

final_action = raw_action + delta_action
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn


class ResidualSafetyModel(nn.Module):
    """Predicts delta_action to correct raw actions for safety.

    The model takes a concatenated feature vector (LiDAR scan, proposed raw action,
    goal position, current velocity, and previous action) and outputs a small
    delta correction that is added to the raw action.  The final layer is
    initialised with near-zero weights so that the model starts as a near-identity
    pass-through (delta ~ 0) and learns corrections from data.

    Parameters
    ----------
    scan_dim : int
        Number of LiDAR scan beams (default 64).
    action_dim : int
        Dimensionality of action, typically (vx, vy, omega) = 3.
    goal_dim : int
        Dimensionality of goal representation, typically (dx, dy, dtheta) = 3.
    velocity_dim : int
        Dimensionality of current velocity vector, typically (vx, vy, omega) = 3.
    hidden_dims : list[int]
        Sizes of hidden layers.
    dropout : float
        Dropout probability applied after each hidden layer.
    activation : str
        Activation function name: ``"relu"`` or ``"leaky_relu"``.
    """

    def __init__(
        self,
        scan_dim: int = 64,
        action_dim: int = 3,
        goal_dim: int = 3,
        velocity_dim: int = 3,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.1,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64, 32]

        input_dim = scan_dim + action_dim + goal_dim + velocity_dim + action_dim

        act_cls: type[nn.Module]
        if activation == "relu":
            act_cls = nn.ReLU
        elif activation == "leaky_relu":
            act_cls = lambda: nn.LeakyReLU(0.1)  # noqa: E731
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        layers: list[nn.Module] = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(act_cls())
            layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, action_dim))

        self.net = nn.Sequential(*layers)

        # Initialise last layer with small weights for near-zero initial output
        # so the model starts as a near-identity module.
        self._init_last_layer()

    def _init_last_layer(self) -> None:
        """Zero-mean, tiny-std init for the final Linear layer."""
        last_linear = self.net[-1]
        if isinstance(last_linear, nn.Linear):
            nn.init.normal_(last_linear.weight, mean=0.0, std=1e-4)
            nn.init.zeros_(last_linear.bias)

    def forward(
        self,
        scan: torch.Tensor,
        raw_action: torch.Tensor,
        goal: torch.Tensor,
        velocity: torch.Tensor,
        last_action: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute the delta correction.

        Parameters
        ----------
        scan : (B, scan_dim)
            Current LiDAR scan (metres).
        raw_action : (B, action_dim)
            Proposed raw action from DWA or teleop.
        goal : (B, goal_dim)
            Goal / pallet position in robot frame.
        velocity : (B, velocity_dim)
            Current robot velocity.
        last_action : (B, action_dim), optional
            Previous action.  Defaults to *raw_action* when ``None``.

        Returns
        -------
        delta_action : (B, action_dim)
            Safety correction to add to the raw action.
        """
        if last_action is None:
            last_action = raw_action
        x = torch.cat([scan, raw_action, goal, velocity, last_action], dim=-1)
        return self.net(x)

    def predict_safe_action(
        self,
        scan: torch.Tensor,
        raw_action: torch.Tensor,
        goal: torch.Tensor,
        velocity: torch.Tensor,
        last_action: Optional[torch.Tensor] = None,
        delta_scale: float = 1.0,
    ) -> torch.Tensor:
        """Predict the final safe action.

        ``safe_action = raw_action + delta_scale * delta_action``
        """
        delta = self.forward(scan, raw_action, goal, velocity, last_action)
        return raw_action + delta_scale * delta

    def predict_numpy(
        self,
        scan: np.ndarray,
        raw_action: np.ndarray,
        goal: np.ndarray,
        velocity: np.ndarray,
        last_action: Optional[np.ndarray] = None,
        delta_scale: float = 1.0,
        device: str = "cpu",
    ) -> dict[str, np.ndarray]:
        """Convenience wrapper that accepts / returns numpy arrays.

        Returns
        -------
        dict with keys ``"delta"`` and ``"safe_action"``.
        """
        self.eval()
        tensors = {}
        for name, arr in [
            ("scan", scan),
            ("raw_action", raw_action),
            ("goal", goal),
            ("velocity", velocity),
        ]:
            t = torch.as_tensor(arr, dtype=torch.float32, device=device)
            if t.ndim == 1:
                t = t.unsqueeze(0)
            tensors[name] = t

        if last_action is not None:
            last_action_t = torch.as_tensor(last_action, dtype=torch.float32, device=device)
            if last_action_t.ndim == 1:
                last_action_t = last_action_t.unsqueeze(0)
        else:
            last_action_t = None

        with torch.no_grad():
            delta = self.forward(
                tensors["scan"],
                tensors["raw_action"],
                tensors["goal"],
                tensors["velocity"],
                last_action_t,
            )
            safe = tensors["raw_action"] + delta_scale * delta

        return {
            "delta": delta.squeeze(0).cpu().numpy(),
            "safe_action": safe.squeeze(0).cpu().numpy(),
        }
