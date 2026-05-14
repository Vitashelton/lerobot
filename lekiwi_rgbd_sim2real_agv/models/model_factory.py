"""Build the full multimodal model from configuration."""

from __future__ import annotations

from typing import Any, Dict

import torch.nn as nn

from models.rgb_encoder import RGBEncoder
from models.scan_encoder import ScanEncoder
from models.state_encoder import StateEncoder
from models.goal_encoder import GoalEncoder
from models.fusion_module import FusionModule
from models.actor_network import ActorNetwork
from models.critic_network import TwinCritic


class MultimodalNavModel(nn.Module):
    """Full multimodal navigation model.

    Contains all encoders, fusion module, actor, and critic.
    """

    def __init__(
        self,
        rgb_encoder: RGBEncoder,
        scan_encoder: ScanEncoder,
        state_encoder: StateEncoder,
        goal_encoder: GoalEncoder,
        fusion: FusionModule,
        actor: ActorNetwork,
        critic: TwinCritic,
    ) -> None:
        super().__init__()
        self.rgb_encoder = rgb_encoder
        self.scan_encoder = scan_encoder
        self.state_encoder = state_encoder
        self.goal_encoder = goal_encoder
        self.fusion = fusion
        self.actor = actor
        self.critic = critic

    def encode(
        self,
        rgb: torch.Tensor | None = None,
        scan64: torch.Tensor | None = None,
        state: torch.Tensor | None = None,
        goal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode all available modalities and fuse.

        Returns
        -------
        fused_feat : (B, fusion_dim)
        """
        rgb_feat = self.rgb_encoder(rgb) if rgb is not None else None
        scan_feat = self.scan_encoder(scan64) if scan64 is not None else None
        state_feat = self.state_encoder(state) if state is not None else None
        goal_feat = self.goal_encoder(goal) if goal is not None else None

        return self.fusion(rgb_feat, scan_feat, state_feat, goal_feat)

    def forward(
        self,
        rgb: torch.Tensor | None = None,
        scan64: torch.Tensor | None = None,
        state: torch.Tensor | None = None,
        goal: torch.Tensor | None = None,
        action: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass: encode → actor action + critic Q-values.

        Returns
        -------
        dict with keys "action", "q1", "q2".
        """
        fused = self.encode(rgb, scan64, state, goal)
        pred_action = self.actor(fused)

        result = {"action": pred_action}
        if action is not None:
            q1, q2 = self.critic(fused, action)
            result["q1"] = q1
            result["q2"] = q2

        return result


class ModelFactory:
    """Create MultimodalNavModel from a config dict.

    Usage::

        model = ModelFactory.create(config)
    """

    @staticmethod
    def create(config: Dict[str, Any]) -> MultimodalNavModel:
        """Build full model from config.

        Parameters
        ----------
        config : dict
            Configuration dict with "model" section.

        Returns
        -------
        MultimodalNavModel
        """
        cfg = config.get("model", config)
        obs_cfg = config.get("observation", {})

        use_rgb = obs_cfg.get("rgb", {}).get("enabled", True)
        use_scan = obs_cfg.get("scan64", {}).get("enabled", True)
        use_state = obs_cfg.get("state", {}).get("enabled", True)
        use_goal = obs_cfg.get("goal", {}).get("enabled", True)

        # RGB encoder
        rgb_cfg = cfg.get("rgb_encoder", {})
        rgb_encoder = RGBEncoder(
            backbone=rgb_cfg.get("backbone", "resnet18"),
            pretrained=rgb_cfg.get("pretrained", True),
            output_dim=rgb_cfg.get("output_dim", 256),
            freeze_bn=rgb_cfg.get("freeze_bn", True),
        )

        # Scan64 encoder
        scan_cfg = cfg.get("scan_encoder", {})
        scan_encoder = ScanEncoder(
            scan_dim=obs_cfg.get("scan64", {}).get("dim", 64),
            hidden_dims=scan_cfg.get("hidden_dims", [128, 128, 128]),
            output_dim=scan_cfg.get("output_dim", 128),
            dropout=scan_cfg.get("dropout", 0.1),
            encoder_type=scan_cfg.get("type", "mlp"),
        )

        # State encoder
        state_cfg = cfg.get("state_encoder", {})
        state_encoder = StateEncoder(
            state_dim=obs_cfg.get("state", {}).get("dim", 3),
            hidden_dims=state_cfg.get("hidden_dims", [64, 64]),
            output_dim=state_cfg.get("output_dim", 64),
            dropout=state_cfg.get("dropout", 0.1),
        )

        # Goal encoder
        goal_cfg = cfg.get("goal_encoder", {})
        goal_encoder = GoalEncoder(
            goal_dim=obs_cfg.get("goal", {}).get("dim", 3),
            hidden_dims=goal_cfg.get("hidden_dims", [64, 64]),
            output_dim=goal_cfg.get("output_dim", 64),
            dropout=goal_cfg.get("dropout", 0.1),
        )

        # Fusion
        fusion_cfg = cfg.get("fusion", {})
        fusion = FusionModule(
            rgb_dim=rgb_cfg.get("output_dim", 256),
            scan_dim=scan_cfg.get("output_dim", 128),
            state_dim=state_cfg.get("output_dim", 64),
            goal_dim=goal_cfg.get("output_dim", 64),
            hidden_dims=fusion_cfg.get("hidden_dims", [512, 256]),
            output_dim=fusion_cfg.get("output_dim", 256),
            dropout=fusion_cfg.get("dropout", 0.1),
            use_rgb=use_rgb,
            use_scan=use_scan,
            use_state=use_state,
            use_goal=use_goal,
        )

        # Actor
        actor_cfg = cfg.get("actor", {})
        actor = ActorNetwork(
            input_dim=fusion_cfg.get("output_dim", 256),
            action_dim=config.get("action", {}).get("dim", 3),
            hidden_dims=actor_cfg.get("hidden_dims", [256, 256]),
            activation=actor_cfg.get("activation", "relu"),
            output_activation=actor_cfg.get("output_activation", "tanh"),
            init_std=actor_cfg.get("init_std", 0.01),
        )

        # Critic
        critic_cfg = cfg.get("critic", {})
        critic = TwinCritic(
            feat_dim=fusion_cfg.get("output_dim", 256),
            action_dim=config.get("action", {}).get("dim", 3),
            hidden_dims=critic_cfg.get("hidden_dims", [256, 256]),
            activation=critic_cfg.get("activation", "relu"),
        )

        return MultimodalNavModel(
            rgb_encoder=rgb_encoder,
            scan_encoder=scan_encoder,
            state_encoder=state_encoder,
            goal_encoder=goal_encoder,
            fusion=fusion,
            actor=actor,
            critic=critic,
        )
