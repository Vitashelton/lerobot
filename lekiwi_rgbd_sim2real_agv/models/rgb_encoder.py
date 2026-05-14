"""RGB image encoder based on ResNet backbone."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torchvision.models as tv_models


class RGBEncoder(nn.Module):
    """Encode RGB images into a feature vector.

    Parameters
    ----------
    backbone : str
        ResNet variant: "resnet18", "resnet34", "resnet50".
    pretrained : bool
        Use ImageNet pretrained weights.
    output_dim : int
        Output feature dimension.
    freeze_bn : bool
        Freeze batch normalization layers.
    """

    BACKBONE_DIMS = {
        "resnet18": 512,
        "resnet34": 512,
        "resnet50": 2048,
    }

    def __init__(
        self,
        backbone: str = "resnet18",
        pretrained: bool = True,
        output_dim: int = 256,
        freeze_bn: bool = True,
    ) -> None:
        super().__init__()

        backbone_dim = self.BACKBONE_DIMS.get(backbone, 512)

        weights = "IMAGENET1K_V1" if pretrained else None
        if backbone == "resnet18":
            self.backbone = tv_models.resnet18(weights=weights)
        elif backbone == "resnet34":
            self.backbone = tv_models.resnet34(weights=weights)
        elif backbone == "resnet50":
            self.backbone = tv_models.resnet50(weights=weights)
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        # Remove classification head
        self.backbone.fc = nn.Identity()

        # Projection head
        self.projection = nn.Sequential(
            nn.Linear(backbone_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(inplace=True),
        )

        if freeze_bn:
            self._freeze_bn()

    def _freeze_bn(self) -> None:
        for m in self.backbone.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                for p in m.parameters():
                    p.requires_grad = False

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        """Encode RGB images.

        Parameters
        ----------
        rgb : (B, 3, H, W) float32
            Normalized RGB images in [0, 1].

        Returns
        -------
        feat : (B, output_dim)
        """
        feat = self.backbone(rgb)
        return self.projection(feat)
