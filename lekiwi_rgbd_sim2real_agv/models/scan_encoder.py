"""Scan64 encoder: MLP or 1D-CNN for polar scan representation."""

from __future__ import annotations

import torch
import torch.nn as nn


class ScanEncoder(nn.Module):
    """Encode Scan64 (64-beam polar scan) into a feature vector.

    Supports two architectures:
    - "mlp": Simple MLP over the 64 beams
    - "cnn1d": 1D convolution treating the scan as a 1D signal

    Parameters
    ----------
    scan_dim : int
        Number of scan beams (default 64).
    hidden_dims : list[int]
        Hidden layer sizes.
    output_dim : int
        Output feature dimension.
    dropout : float
        Dropout rate.
    encoder_type : str
        "mlp" or "cnn1d".
    """

    def __init__(
        self,
        scan_dim: int = 64,
        hidden_dims: list[int] | None = None,
        output_dim: int = 128,
        dropout: float = 0.1,
        encoder_type: str = "mlp",
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 128, 128]
        self.encoder_type = encoder_type

        if encoder_type == "cnn1d":
            self.net = nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=5, stride=2, padding=2),
                nn.ReLU(inplace=True),
                nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(128, output_dim),
                nn.LayerNorm(output_dim),
                nn.ReLU(inplace=True),
            )
        else:
            layers: list[nn.Module] = []
            prev = scan_dim
            for h in hidden_dims:
                layers.append(nn.Linear(prev, h))
                layers.append(nn.LayerNorm(h))
                layers.append(nn.ReLU(inplace=True))
                layers.append(nn.Dropout(dropout))
                prev = h
            layers.append(nn.Linear(prev, output_dim))
            self.net = nn.Sequential(*layers)

    def forward(self, scan64: torch.Tensor) -> torch.Tensor:
        """Encode Scan64.

        Parameters
        ----------
        scan64 : (B, scan_dim) float32
            Scan distances in meters. NaN replaced with max_range.

        Returns
        -------
        feat : (B, output_dim)
        """
        if self.encoder_type == "cnn1d":
            scan = scan64.unsqueeze(1)  # (B, 1, scan_dim)
        else:
            scan = scan64
        return self.net(scan)
