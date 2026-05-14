"""Actor (policy) network: fused features → action."""

from __future__ import annotations

import torch
import torch.nn as nn


class ActorNetwork(nn.Module):
    """Policy network that outputs continuous navigation actions.

    Parameters
    ----------
    input_dim : int
        Fused feature dimension.
    action_dim : int
        Action dimension (3: vx, vy, omega).
    hidden_dims : list[int]
        Hidden layer sizes.
    activation : str
        "relu" or "leaky_relu".
    output_activation : str
        "tanh" or "none". Tanh limits output to [-1, 1].
    init_std : float
        Standard deviation for final layer weight initialization.
    """

    def __init__(
        self,
        input_dim: int = 256,
        action_dim: int = 3,
        hidden_dims: list[int] | None = None,
        activation: str = "relu",
        output_activation: str = "tanh",
        init_std: float = 0.01,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256]

        act_fn: type[nn.Module] = (
            nn.ReLU if activation == "relu" else nn.LeakyReLU
        )

        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.LayerNorm(h))
            layers.append(act_fn(inplace=True))
            prev = h
        layers.append(nn.Linear(prev, action_dim))

        if output_activation == "tanh":
            layers.append(nn.Tanh())

        self.net = nn.Sequential(*layers)
        self.action_dim = action_dim
        self._output_tanh = output_activation == "tanh"

        # Small init for final layer
        self._init_last_layer(init_std)

    def _init_last_layer(self, std: float) -> None:
        last_linear = None
        for m in reversed(self.net):
            if isinstance(m, nn.Linear):
                last_linear = m
                break
        if last_linear is not None:
            nn.init.normal_(last_linear.weight, mean=0.0, std=std)
            nn.init.zeros_(last_linear.bias)

    def forward(self, fused_feat: torch.Tensor) -> torch.Tensor:
        """Predict action.

        Parameters
        ----------
        fused_feat : (B, input_dim)

        Returns
        -------
        action : (B, action_dim)
        """
        return self.net(fused_feat)
