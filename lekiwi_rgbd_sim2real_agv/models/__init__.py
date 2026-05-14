"""Multimodal neural network modules for offline RL navigation.

Architecture:
    RGB → ResNet-18 → rgb_feat (256,)
    Scan64 → MLP/1D-CNN → scan_feat (128,)
    State → MLP → state_feat (64,)
    Goal → MLP → goal_feat (64,)
    Fusion → concat + MLP → fused_feat (256,)
    Actor → MLP → action [vx, vy, omega]
    Critic → MLP → Q-value
"""

from models.rgb_encoder import RGBEncoder
from models.scan_encoder import ScanEncoder
from models.state_encoder import StateEncoder
from models.goal_encoder import GoalEncoder
from models.fusion_module import FusionModule
from models.actor_network import ActorNetwork
from models.critic_network import TwinCritic
from models.model_factory import ModelFactory

__all__ = [
    "RGBEncoder",
    "ScanEncoder",
    "StateEncoder",
    "GoalEncoder",
    "FusionModule",
    "ActorNetwork",
    "TwinCritic",
    "ModelFactory",
]
