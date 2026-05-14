"""TD3+BC: Twin Delayed DDPG with Behavior Cloning regularization.

Reference:
    "A Minimalist Approach to Offline Reinforcement Learning"
    Fujimoto & Gu, NeurIPS 2021

Key formulas:
    L_critic = E[(Q(s,a) - y)²]
        where y = r + γ * min(Q1, Q2)(s', a' + noise)

    L_actor = -λ * E[Q(s, π(s))] + E[(π(s) - a)²]
        where λ = α / E[|Q|] normalizes Q and BC terms
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


class TD3BC:
    """TD3+BC trainer.

    Parameters
    ----------
    model : MultimodalNavModel
    config : dict
        Training configuration.
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

        # Create target networks
        self.target_model = copy.deepcopy(model).to(device)
        self.target_model.eval()

        # Config
        train_cfg = config.get("training", config)
        algo_cfg = config.get("algorithm", config)
        self.gamma = algo_cfg.get("gamma", 0.99)
        self.tau = algo_cfg.get("tau", 0.005)
        self.alpha = algo_cfg.get("alpha", 2.5)
        self.policy_noise = algo_cfg.get("policy_noise", 0.2)
        self.noise_clip = algo_cfg.get("noise_clip", 0.5)
        self.policy_delay = algo_cfg.get("policy_delay", 2)

        self.actor_lr = train_cfg.get("actor_lr", 3e-4)
        self.critic_lr = train_cfg.get("critic_lr", 3e-4)
        self.weight_decay = train_cfg.get("weight_decay", 1e-4)
        self.grad_clip_norm = train_cfg.get("grad_clip_norm", 10.0)
        self.max_action = 1.0  # tanh output space

        # Optimizers
        self.actor_optim = Adam(
            self.model.actor.parameters(),
            lr=self.actor_lr,
            weight_decay=self.weight_decay,
        )
        self.critic_optim = Adam(
            self.model.critic.parameters(),
            lr=self.critic_lr,
            weight_decay=self.weight_decay,
        )

        # Encoder optimizer (shared for all encoders + fusion)
        encoder_params = (
            list(self.model.rgb_encoder.parameters())
            + list(self.model.scan_encoder.parameters())
            + list(self.model.state_encoder.parameters())
            + list(self.model.goal_encoder.parameters())
            + list(self.model.fusion.parameters())
        )
        self.encoder_optim = Adam(encoder_params, lr=self.actor_lr)

        self.total_steps = 0

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step.

        Parameters
        ----------
        batch : dict
            Batch from OfflineReplayBuffer.sample().

        Returns
        -------
        dict with loss values.
        """
        self.total_steps += 1

        # Preprocess observations
        rgb = batch["rgb"]
        if rgb is not None and rgb.dim() == 5:
            rgb = rgb.squeeze(1)
        if rgb is not None and rgb.dtype == torch.uint8:
            rgb = rgb.float() / 255.0
        if rgb is not None and rgb.shape[1] != 3:
            # (B, H, W, C) → (B, C, H, W)
            rgb = rgb.permute(0, 3, 1, 2)

        scan64 = batch["scan64"]
        if scan64 is not None:
            scan64 = torch.nan_to_num(scan64, nan=5.0)

        state = batch["state"]
        goal = batch["goal"]
        action = batch["action"]
        reward = batch["reward"].unsqueeze(-1)
        done = batch["done"].unsqueeze(-1).float()

        next_rgb = batch["next_rgb"]
        if next_rgb is not None and next_rgb.dim() == 5:
            next_rgb = next_rgb.squeeze(1)
        if next_rgb is not None and next_rgb.dtype == torch.uint8:
            next_rgb = next_rgb.float() / 255.0
        if next_rgb is not None and next_rgb.shape[1] != 3:
            next_rgb = next_rgb.permute(0, 3, 1, 2)

        next_scan64 = batch["next_scan64"]
        if next_scan64 is not None:
            next_scan64 = torch.nan_to_num(next_scan64, nan=5.0)

        next_state = batch["next_state"]
        next_goal = batch["next_goal"]

        # ---- Critic update ----
        critic_loss = self._update_critic(
            rgb, scan64, state, goal, action, reward, done,
            next_rgb, next_scan64, next_state, next_goal,
        )

        # ---- Actor + Encoder update (delayed) ----
        actor_loss = torch.tensor(0.0)
        if self.total_steps % self.policy_delay == 0:
            actor_loss = self._update_actor(
                rgb, scan64, state, goal, action,
            )

        # ---- Target network soft update ----
        self._soft_update()

        return {
            "critic_loss": critic_loss.item() if isinstance(critic_loss, torch.Tensor) else critic_loss,
            "actor_loss": actor_loss.item() if isinstance(actor_loss, torch.Tensor) else actor_loss,
        }

    # ------------------------------------------------------------------
    # Critic update
    # ------------------------------------------------------------------

    def _update_critic(
        self,
        rgb, scan64, state, goal, action, reward, done,
        next_rgb, next_scan64, next_state, next_goal,
    ) -> torch.Tensor:
        with torch.no_grad():
            # Target action with noise
            next_fused = self.target_model.encode(
                next_rgb, next_scan64, next_state, next_goal
            )
            noise = (
                torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)
            next_action = (
                self.target_model.actor(next_fused) + noise
            ).clamp(-self.max_action, self.max_action)

            target_q1, target_q2 = self.target_model.critic(next_fused, next_action)
            target_q = torch.min(target_q1, target_q2)
            target = reward + (1.0 - done) * self.gamma * target_q

        fused = self.model.encode(rgb, scan64, state, goal)
        q1, q2 = self.model.critic(fused, action)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.model.critic.parameters(), self.grad_clip_norm)
        self.critic_optim.step()

        return critic_loss

    # ------------------------------------------------------------------
    # Actor update (TD3+BC style)
    # ------------------------------------------------------------------

    def _update_actor(
        self,
        rgb, scan64, state, goal, action,
    ) -> torch.Tensor:
        fused = self.model.encode(rgb, scan64, state, goal)
        pred_action = self.model.actor(fused)

        # Q-value term
        q = self.model.critic.q1(fused, pred_action)
        q_mean = q.abs().mean().detach()

        # BC regularization
        lmbda = self.alpha / max(q_mean.item(), 1e-8)
        l2_reg = F.mse_loss(pred_action, action)

        actor_loss = -lmbda * q.mean() + l2_reg

        self.actor_optim.zero_grad()
        self.encoder_optim.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.model.actor.parameters(), self.grad_clip_norm)
        nn.utils.clip_grad_norm_(self.model.fusion.parameters(), self.grad_clip_norm)
        self.actor_optim.step()
        self.encoder_optim.step()

        return actor_loss

    # ------------------------------------------------------------------
    # Target network update
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
        """Predict action.

        Parameters
        ----------
        rgb, scan64, state, goal : tensors or None, each (B, ...)

        Returns
        -------
        action : np.ndarray, shape (B, 3)
        """
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
            "target_model": self.target_model.state_dict(),
            "actor_optim": self.actor_optim.state_dict(),
            "critic_optim": self.critic_optim.state_dict(),
            "encoder_optim": self.encoder_optim.state_dict(),
            "total_steps": self.total_steps,
        }

    def load_state_dict(self, ckpt: Dict[str, Any]) -> None:
        self.model.load_state_dict(ckpt["model"])
        self.target_model.load_state_dict(ckpt["target_model"])
        self.actor_optim.load_state_dict(ckpt["actor_optim"])
        self.critic_optim.load_state_dict(ckpt["critic_optim"])
        if "encoder_optim" in ckpt:
            self.encoder_optim.load_state_dict(ckpt["encoder_optim"])
        self.total_steps = ckpt.get("total_steps", 0)
