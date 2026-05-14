"""IQL: Implicit Q-Learning for offline RL.

Reference:
    "Offline Reinforcement Learning with Implicit Q-Learning"
    Kostrikov, Nair & Levine, ICLR 2022

Key formulas:
    V(s) loss (expectile regression):
        L_V = E[L2^τ(Q_target(s,a) - V(s))]
        where L2^τ(u) = |τ - 1_{u < 0}| * u²

    Q(s,a) loss:
        L_Q = E[(Q(s,a) - (r + γ * V(s'))²]

    Policy loss (AWR):
        L_π = E[exp((Q(s,a) - V(s)) / β) * (a - π(s))²]

Advantages for navigation:
    - No OOD action sampling → naturally conservative
    - Expectile regression → can control conservatism via τ
    - AWR policy extraction → stays close to data distribution
"""

from __future__ import annotations

import copy
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from models.model_factory import MultimodalNavModel
from rl.replay_buffer import OfflineReplayBuffer


class IQL:
    """IQL trainer for multimodal navigation.

    Parameters
    ----------
    model : MultimodalNavModel
    config : Dict
        Training config with algorithm and training sections.
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

        # Target networks (for Q and V)
        self.target_model = copy.deepcopy(model).to(device)
        self.target_model.eval()

        # Config
        train_cfg = config.get("training", config)
        algo_cfg = config.get("algorithm", config)
        self.gamma = algo_cfg.get("gamma", 0.99)
        self.tau = algo_cfg.get("tau", 0.005)
        self.expectile = algo_cfg.get("expectile", 0.8)
        self.temperature = algo_cfg.get("temperature", 3.0)

        self.actor_lr = train_cfg.get("actor_lr", 3e-4)
        self.critic_lr = train_cfg.get("critic_lr", 3e-4)
        self.value_lr = train_cfg.get("value_lr", 3e-4)
        self.weight_decay = train_cfg.get("weight_decay", 1e-4)
        self.grad_clip_norm = train_cfg.get("grad_clip_norm", 10.0)
        self.actor_delay = train_cfg.get("actor_delay", 2)

        # Separate V-network
        fusion_dim = model.fusion.output_dim
        self.v_net = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        ).to(device)

        # Optimizers
        self.actor_optim = Adam(model.actor.parameters(), lr=self.actor_lr)
        self.critic_optim = Adam(model.critic.parameters(), lr=self.critic_lr)
        self.value_optim = Adam(
            list(self.v_net.parameters()), lr=self.value_lr
        )

        encoder_params = (
            list(model.rgb_encoder.parameters())
            + list(model.scan_encoder.parameters())
            + list(model.state_encoder.parameters())
            + list(model.goal_encoder.parameters())
            + list(model.fusion.parameters())
        )
        self.encoder_optim = Adam(encoder_params, lr=self.actor_lr)

        self.total_steps = 0

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        self.total_steps += 1

        rgb, scan64, state, goal = self._preprocess_obs(batch, current=True)
        action = batch["action"]
        reward = batch["reward"].unsqueeze(-1)
        done = batch["done"].unsqueeze(-1).float()
        next_rgb, next_scan64, next_state, next_goal = self._preprocess_obs(
            batch, current=False
        )

        # 1. Value update
        value_loss = self._update_value(
            rgb, scan64, state, goal, action,
            next_rgb, next_scan64, next_state, next_goal,
        )

        # 2. Critic update
        critic_loss = self._update_critic(
            rgb, scan64, state, goal, action, reward, done,
            next_rgb, next_scan64, next_state, next_goal,
        )

        # 3. Actor + encoder update (delayed)
        actor_loss = torch.tensor(0.0)
        if self.total_steps % self.actor_delay == 0:
            actor_loss = self._update_actor(
                rgb, scan64, state, goal, action,
            )

        # 4. Soft update targets
        self._soft_update()

        return {
            "value_loss": value_loss.item() if isinstance(value_loss, torch.Tensor) else value_loss,
            "critic_loss": critic_loss.item() if isinstance(critic_loss, torch.Tensor) else critic_loss,
            "actor_loss": actor_loss.item() if isinstance(actor_loss, torch.Tensor) else actor_loss,
        }

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _preprocess_obs(
        self, batch: Dict[str, torch.Tensor], current: bool = True
    ) -> tuple:
        prefix = "" if current else "next_"
        rgb_key = f"{prefix}rgb"
        scan_key = f"{prefix}scan64"
        state_key = f"{prefix}state"
        goal_key = f"{prefix}goal"

        rgb = batch.get(rgb_key)
        if rgb is not None:
            if rgb.dim() == 5:
                rgb = rgb.squeeze(1)
            if rgb.dtype == torch.uint8:
                rgb = rgb.float() / 255.0
            if rgb.shape[1] != 3:
                rgb = rgb.permute(0, 3, 1, 2)

        scan64 = batch.get(scan_key)
        if scan64 is not None:
            scan64 = torch.nan_to_num(scan64, nan=5.0)

        state = batch.get(state_key)
        goal = batch.get(goal_key)

        return rgb, scan64, state, goal

    # ------------------------------------------------------------------
    # Value update (expectile regression)
    # ------------------------------------------------------------------

    def _update_value(
        self,
        rgb, scan64, state, goal, action,
        next_rgb, next_scan64, next_state, next_goal,
    ) -> torch.Tensor:
        with torch.no_grad():
            # Use target model for Q
            fused = self.target_model.encode(rgb, scan64, state, goal)
            q1, q2 = self.target_model.critic(fused, action)
            q = torch.min(q1, q2)

        fused_detach = self.model.encode(rgb, scan64, state, goal).detach()
        v = self.v_net(fused_detach)
        diff = q - v
        weight = torch.where(diff > 0, self.expectile, 1.0 - self.expectile)
        value_loss = (weight * (diff ** 2)).mean()

        self.value_optim.zero_grad()
        value_loss.backward()
        nn.utils.clip_grad_norm_(self.v_net.parameters(), self.grad_clip_norm)
        self.value_optim.step()

        return value_loss

    # ------------------------------------------------------------------
    # Critic update (standard Bellman)
    # ------------------------------------------------------------------

    def _update_critic(
        self,
        rgb, scan64, state, goal, action, reward, done,
        next_rgb, next_scan64, next_state, next_goal,
    ) -> torch.Tensor:
        with torch.no_grad():
            next_fused = self.model.encode(
                next_rgb, next_scan64, next_state, next_goal
            )
            next_v = self.v_net(next_fused)
            target = reward + (1.0 - done) * self.gamma * next_v

        fused = self.model.encode(rgb, scan64, state, goal)
        q1, q2 = self.model.critic(fused, action)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.model.critic.parameters(), self.grad_clip_norm)
        self.critic_optim.step()

        return critic_loss

    # ------------------------------------------------------------------
    # Actor update (AWR)
    # ------------------------------------------------------------------

    def _update_actor(
        self,
        rgb, scan64, state, goal, action,
    ) -> torch.Tensor:
        fused = self.model.encode(rgb, scan64, state, goal)
        pred_action = self.model.actor(fused)

        with torch.no_grad():
            q1, q2 = self.model.critic(fused, action)
            q = torch.min(q1, q2)
            v = self.v_net(fused)
            advantage = q - v
            # Advantage-weighted regression weights
            weight = torch.exp(advantage / self.temperature).clamp(max=100.0)

        actor_loss = (weight * F.mse_loss(pred_action, action, reduction="none").sum(-1)).mean()

        self.actor_optim.zero_grad()
        self.encoder_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        self.encoder_optim.step()

        return actor_loss

    # ------------------------------------------------------------------
    # Target update
    # ------------------------------------------------------------------

    def _soft_update(self) -> None:
        for target_param, param in zip(
            self.target_model.parameters(), self.model.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

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

    @torch.no_grad()
    def get_q_value(
        self,
        rgb, scan64, state, goal, action,
    ) -> float:
        """Get Q-value for diagnostic purposes."""
        self.model.eval()
        fused = self.model.encode(rgb, scan64, state, goal)
        q1, q2 = self.model.critic(fused, action)
        return torch.min(q1, q2).mean().item()

    def state_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model.state_dict(),
            "target_model": self.target_model.state_dict(),
            "v_net": self.v_net.state_dict(),
            "actor_optim": self.actor_optim.state_dict(),
            "critic_optim": self.critic_optim.state_dict(),
            "value_optim": self.value_optim.state_dict(),
            "encoder_optim": self.encoder_optim.state_dict(),
            "total_steps": self.total_steps,
        }

    def load_state_dict(self, ckpt: Dict[str, Any]) -> None:
        self.model.load_state_dict(ckpt["model"])
        self.target_model.load_state_dict(ckpt["target_model"])
        self.v_net.load_state_dict(ckpt["v_net"])
        self.actor_optim.load_state_dict(ckpt["actor_optim"])
        self.critic_optim.load_state_dict(ckpt["critic_optim"])
        self.value_optim.load_state_dict(ckpt["value_optim"])
        if "encoder_optim" in ckpt:
            self.encoder_optim.load_state_dict(ckpt["encoder_optim"])
        self.total_steps = ckpt.get("total_steps", 0)
