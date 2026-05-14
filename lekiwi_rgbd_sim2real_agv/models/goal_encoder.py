"""Goal vector encoder."""

from __future__ import annotations

import torch
import torch.nn as nn


class GoalEncoder(nn.Module):
    """Encode goal vector [dx, dy, dtheta] into feature representation.

    Parameters
    ----------
    goal_dim : int
        Goal dimension (default 3).
    hidden_dims : list[int]
        Hidden layer sizes.
    output_dim : int
        Output feature dimension.
    dropout : float
        Dropout rate.
    """

    def __init__(
        self,
        goal_dim: int = 3,
        hidden_dims: list[int] | None = None,
        output_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 64]

        layers: list[nn.Module] = []
        prev = goal_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, goal: torch.Tensor) -> torch.Tensor:
        """Encode goal.

        Parameters
        ----------
        goal : (B, goal_dim)

        Returns
        -------
        feat : (B, output_dim)
        """
        return self.net(goal)
