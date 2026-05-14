"""Behavior Cloning baseline: supervised learning on offline actions."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from models.model_factory import MultimodalNavModel
from rl.replay_buffer import OfflineReplayBuffer


class BehaviorCloning:
    """Simple BC: minimize MSE between predicted and dataset actions.

    Parameters
    ----------
    model : MultimodalNavModel
        Only the encoder + actor are used; critic is ignored.
    config : Dict
        Training config.
    device : str
    """

    def __init__(
        self,
        model: MultimodalNavModel,
        config: Dict[str, Any],
        device: str = "cuda",
    ) -> None:
        self.model = model.to(device)
        self.device = device

        train_cfg = config.get("training", config)
        lr = train_cfg.get("actor_lr", 3e-4)
        weight_decay = train_cfg.get("weight_decay", 1e-4)

        params = (
            list(model.rgb_encoder.parameters())
            + list(model.scan_encoder.parameters())
            + list(model.state_encoder.parameters())
            + list(model.goal_encoder.parameters())
            + list(model.fusion.parameters())
            + list(model.actor.parameters())
        )
        self.optimizer = Adam(params, lr=lr, weight_decay=weight_decay)

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single BC training step."""
        self.model.train()

        rgb = batch.get("rgb")
        if rgb is not None:
            if rgb.dim() == 5:
                rgb = rgb.squeeze(1)
            if rgb.dtype == torch.uint8:
                rgb = rgb.float() / 255.0
            if rgb.shape[1] != 3:
                rgb = rgb.permute(0, 3, 1, 2)

        scan64 = batch.get("scan64")
        if scan64 is not None:
            scan64 = torch.nan_to_num(scan64, nan=5.0)

        state = batch.get("state")
        goal = batch.get("goal")
        action = batch["action"]

        fused = self.model.encode(rgb, scan64, state, goal)
        pred_action = self.model.actor(fused)

        loss = F.mse_loss(pred_action, action)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"bc_loss": loss.item()}

    @torch.no_grad()
    def predict(
        self,
        rgb: torch.Tensor | None = None,
        scan64: torch.Tensor | None = None,
        state: torch.Tensor | None = None,
        goal: torch.Tensor | None = None,
    ) -> np.ndarray:
        """Predict action."""
        self.model.eval()
        if rgb is not None and rgb.dtype == torch.uint8:
            rgb = rgb.float() / 255.0
        if scan64 is not None:
            scan64 = torch.nan_to_num(scan64, nan=5.0)

        fused = self.model.encode(rgb, scan64, state, goal)
        action = self.model.actor(fused)
        return action.cpu().numpy()

    def state_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(self, ckpt: Dict[str, Any]) -> None:
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
