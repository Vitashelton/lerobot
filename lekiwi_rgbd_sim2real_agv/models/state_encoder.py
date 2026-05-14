"""Robot state encoder (velocity, etc.)."""

from __future__ import annotations

import torch
import torch.nn as nn


class StateEncoder(nn.Module):
    """Encode robot state vector into a feature representation.

    Parameters
    ----------
    state_dim : int
        Dimensionality of state input (default 3: vx, vy, omega).
    hidden_dims : list[int]
        Hidden layer sizes.
    output_dim : int
        Output feature dimension.
    dropout : float
        Dropout rate.
    """

    def __init__(
        self,
        state_dim: int = 3,
        hidden_dims: list[int] | None = None,
        output_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 64]

        layers: list[nn.Module] = []
        prev = state_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Encode state.

        Parameters
        ----------
        state : (B, state_dim)

        Returns
        -------
        feat : (B, output_dim)
        """
        return self.net(state)
