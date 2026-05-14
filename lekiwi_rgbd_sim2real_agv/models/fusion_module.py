"""Multimodal fusion module.

Combines features from RGB, Scan64, state, and goal encoders
into a unified representation for the actor and critic.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FusionModule(nn.Module):
    """Fuse multimodal features via concatenation + MLP.

    Parameters
    ----------
    rgb_dim : int
        RGB feature dimension.
    scan_dim : int
        Scan64 feature dimension.
    state_dim : int
        State feature dimension.
    goal_dim : int
        Goal feature dimension.
    hidden_dims : list[int]
        Hidden layer sizes after concatenation.
    output_dim : int
        Fused feature dimension.
    dropout : float
        Dropout rate.
    use_rgb : bool
        Whether RGB encoder is enabled.
    use_scan : bool
        Whether Scan64 encoder is enabled.
    use_state : bool
        Whether state encoder is enabled.
    use_goal : bool
        Whether goal encoder is enabled.
    """

    def __init__(
        self,
        rgb_dim: int = 256,
        scan_dim: int = 128,
        state_dim: int = 64,
        goal_dim: int = 64,
        hidden_dims: list[int] | None = None,
        output_dim: int = 256,
        dropout: float = 0.1,
        use_rgb: bool = True,
        use_scan: bool = True,
        use_state: bool = True,
        use_goal: bool = True,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256]

        self.use_rgb = use_rgb
        self.use_scan = use_scan
        self.use_state = use_state
        self.use_goal = use_goal

        input_dim = 0
        if use_rgb:
            input_dim += rgb_dim
        if use_scan:
            input_dim += scan_dim
        if use_state:
            input_dim += state_dim
        if use_goal:
            input_dim += goal_dim

        assert input_dim > 0, "At least one modality must be enabled"

        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        layers.append(nn.LayerNorm(output_dim))
        layers.append(nn.ReLU(inplace=True))

        self.net = nn.Sequential(*layers)
        self.output_dim = output_dim

    def forward(
        self,
        rgb_feat: torch.Tensor | None = None,
        scan_feat: torch.Tensor | None = None,
        state_feat: torch.Tensor | None = None,
        goal_feat: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Fuse multimodal features.

        Parameters
        ----------
        rgb_feat : (B, rgb_dim) or None
        scan_feat : (B, scan_dim) or None
        state_feat : (B, state_dim) or None
        goal_feat : (B, goal_dim) or None

        Returns
        -------
        fused : (B, output_dim)
        """
        features: list[torch.Tensor] = []

        if self.use_rgb:
            assert rgb_feat is not None, "RGB enabled but rgb_feat is None"
            features.append(rgb_feat)
        if self.use_scan:
            assert scan_feat is not None, "Scan enabled but scan_feat is None"
            features.append(scan_feat)
        if self.use_state:
            assert state_feat is not None, "State enabled but state_feat is None"
            features.append(state_feat)
        if self.use_goal:
            assert goal_feat is not None, "Goal enabled but goal_feat is None"
            features.append(goal_feat)

        x = torch.cat(features, dim=-1)
        return self.net(x)
