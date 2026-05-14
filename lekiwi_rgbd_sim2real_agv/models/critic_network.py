"""Twin critic networks for offline RL (TD3BC, IQL, CQL)."""

from __future__ import annotations

import torch
import torch.nn as nn


class CriticNetwork(nn.Module):
    """Single Q-network: fused_feat + action → Q-value.

    Parameters
    ----------
    feat_dim : int
        Fused feature dimension.
    action_dim : int
        Action dimension.
    hidden_dims : list[int]
        Hidden layer sizes.
    activation : str
        "relu" or "leaky_relu".
    """

    def __init__(
        self,
        feat_dim: int = 256,
        action_dim: int = 3,
        hidden_dims: list[int] | None = None,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256]

        act_fn: type[nn.Module] = (
            nn.ReLU if activation == "relu" else nn.LeakyReLU
        )

        layers: list[nn.Module] = []
        prev = feat_dim + action_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.LayerNorm(h))
            layers.append(act_fn(inplace=True))
            prev = h
        layers.append(nn.Linear(prev, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, feat: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Compute Q-value.

        Parameters
        ----------
        feat : (B, feat_dim)
        action : (B, action_dim)

        Returns
        -------
        q_value : (B, 1)
        """
        x = torch.cat([feat, action], dim=-1)
        return self.net(x)


class TwinCritic(nn.Module):
    """Two Q-networks for clipped double Q-learning.

    Parameters
    ----------
    feat_dim : int
        Fused feature dimension.
    action_dim : int
        Action dimension.
    hidden_dims : list[int]
        Hidden layer sizes.
    activation : str
        "relu" or "leaky_relu".
    """

    def __init__(
        self,
        feat_dim: int = 256,
        action_dim: int = 3,
        hidden_dims: list[int] | None = None,
        activation: str = "relu",
    ) -> None:
        super().__init__()

        self.q1 = CriticNetwork(feat_dim, action_dim, hidden_dims, activation)
        self.q2 = CriticNetwork(feat_dim, action_dim, hidden_dims, activation)

    def forward(
        self, feat: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute both Q-values.

        Returns
        -------
        q1, q2 : (B, 1), (B, 1)
        """
        return self.q1(feat, action), self.q2(feat, action)

    def q_min(self, feat: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Minimum of the two Q-values (for conservative updates)."""
        q1, q2 = self(feat, action)
        return torch.min(q1, q2)
