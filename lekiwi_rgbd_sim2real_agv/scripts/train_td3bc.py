#!/usr/bin/env python
"""Train TD3+BC on LeRobot-compatible navigation dataset.

Usage:
    python scripts/train_td3bc.py \
        --data-dir data/lerobot/habitat_nav \
        --config configs/new/td3bc_config.yaml \
        --output-dir checkpoints/td3bc_habitat \
        --device cuda
"""

from __future__ import annotations

import argparse
import os

import yaml
import numpy as np
import torch

from models.model_factory import ModelFactory
from rl.td3bc import TD3BC
from rl.replay_buffer import OfflineReplayBuffer
from lerobot_conversion.unified_to_lerobot import UnifiedToLeRobotConverter


def main():
    parser = argparse.ArgumentParser(description="Train TD3+BC")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/new/td3bc_config.yaml")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    # Load configs
    with open("configs/new/default.yaml") as f:
        default_cfg = yaml.safe_load(f)
    with open(args.config) as f:
        algo_cfg = yaml.safe_load(f)
    config = {**default_cfg, **algo_cfg}

    os.makedirs(args.output_dir, exist_ok=True)

    # Load dataset
    print(f"Loading dataset from {args.data_dir}...")
    frames = UnifiedToLeRobotConverter.load_lerobot_dataset(args.data_dir)

    T = len(frames)
    obs = {"rgb": None, "scan64": None, "state": None, "goal": None, "prev_action": None}
    actions = np.zeros((T, 3), dtype=np.float32)
    rewards = np.zeros(T, dtype=np.float32)
    dones = np.zeros(T, dtype=bool)
    episode_ids = np.zeros(T, dtype=np.int32)

    for t in range(T):
        f = frames[t]
        actions[t] = f["action"].numpy()
        rewards[t] = f["reward"].item()
        dones[t] = f["done"].item()
        episode_ids[t] = f["episode_index"].item()

        if "observation.images.rgb" in f:
            rgb = f["observation.images.rgb"]
            rgb_np = (rgb.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            if obs["rgb"] is None:
                obs["rgb"] = np.zeros((T, *rgb_np.shape), dtype=np.uint8)
            obs["rgb"][t] = rgb_np

        for key in ["scan64", "state", "goal"]:
            obs_key = f"observation.{key}"
            if obs_key in f:
                arr = f[obs_key].numpy()
                if obs[key] is None:
                    obs[key] = np.zeros((T, *arr.shape), dtype=np.float32)
                obs[key][t] = arr

    dataset = {
        "observations": obs,
        "actions": actions,
        "rewards": rewards,
        "dones": dones,
        "episode_ids": episode_ids,
    }

    # Build model and trainer
    model = ModelFactory.create(config)
    td3bc = TD3BC(model, config, device=args.device)
    buffer = OfflineReplayBuffer(dataset, device=args.device)

    if args.resume:
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=args.device)
        td3bc.load_state_dict(ckpt)

    # Training loop
    train_cfg = config.get("training", {})
    num_epochs = train_cfg.get("num_epochs", 500)
    batch_size = train_cfg.get("batch_size", 256)
    steps_per_epoch = max(1, len(buffer) // batch_size)

    print(f"Training for {num_epochs} epochs, {steps_per_epoch} steps/epoch")
    best_loss = float("inf")

    for epoch in range(num_epochs):
        epoch_losses = {"critic_loss": 0.0, "actor_loss": 0.0}

        for _ in range(steps_per_epoch):
            batch = buffer.sample(batch_size)
            losses = td3bc.train_step(batch)
            for k, v in losses.items():
                epoch_losses[k] += v

        for k in epoch_losses:
            epoch_losses[k] /= max(steps_per_epoch, 1)

        total_loss = sum(epoch_losses.values())

        print(f"Epoch {epoch + 1}/{num_epochs}: "
              f"Q={epoch_losses['critic_loss']:.4f} "
              f"Actor={epoch_losses['actor_loss']:.4f}")

        if total_loss < best_loss:
            best_loss = total_loss
            torch.save(td3bc.state_dict(), os.path.join(args.output_dir, "best.pt"))
            print(f"  → Saved best checkpoint (loss={best_loss:.4f})")

        if (epoch + 1) % 50 == 0:
            torch.save(td3bc.state_dict(),
                       os.path.join(args.output_dir, f"epoch_{epoch + 1}.pt"))

    torch.save(td3bc.state_dict(), os.path.join(args.output_dir, "final.pt"))
    print(f"Training complete! Best model saved to {args.output_dir}/best.pt")


if __name__ == "__main__":
    main()
