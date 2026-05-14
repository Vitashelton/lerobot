#!/usr/bin/env python
"""Train Behavior Cloning baseline.

Usage:
    python scripts/train_bc.py \
        --data-dir data/lerobot/habitat_nav \
        --config configs/new/default.yaml \
        --output-dir checkpoints/bc_habitat \
        --device cuda
"""

from __future__ import annotations

import argparse
import os

import yaml
import numpy as np
import torch

from models.model_factory import ModelFactory
from baselines.behavior_cloning import BehaviorCloning
from rl.replay_buffer import OfflineReplayBuffer
from lerobot_conversion.unified_to_lerobot import UnifiedToLeRobotConverter


def main():
    parser = argparse.ArgumentParser(description="Train Behavior Cloning")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/new/default.yaml")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    os.makedirs(args.output_dir, exist_ok=True)

    # Load dataset
    print(f"Loading dataset from {args.data_dir}...")
    frames = UnifiedToLeRobotConverter.load_lerobot_dataset(args.data_dir)

    T = len(frames)
    obs = {"rgb": None, "scan64": None, "state": None, "goal": None}
    actions = np.zeros((T, 3), dtype=np.float32)
    episode_ids = np.zeros(T, dtype=np.int32)

    for t in range(T):
        f = frames[t]
        actions[t] = f["action"].numpy()
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
        "episode_ids": episode_ids,
    }

    # Build model
    model = ModelFactory.create(config)
    bc_config = {"training": {"actor_lr": args.lr, "weight_decay": 1e-4}}
    bc = BehaviorCloning(model, bc_config, device=args.device)

    # Replay buffer
    dataset["rewards"] = np.zeros(T, dtype=np.float32)
    dataset["dones"] = np.zeros(T, dtype=bool)
    buffer = OfflineReplayBuffer(dataset, device=args.device)

    # Training
    steps_per_epoch = max(1, len(buffer) // args.batch_size)
    best_loss = float("inf")

    print(f"Training BC for {args.num_epochs} epochs")
    for epoch in range(args.num_epochs):
        epoch_loss = 0.0
        for _ in range(steps_per_epoch):
            batch = buffer.sample(args.batch_size)
            losses = bc.train_step(batch)
            epoch_loss += losses["bc_loss"]

        epoch_loss /= max(steps_per_epoch, 1)
        print(f"Epoch {epoch + 1}/{args.num_epochs}: BC Loss={epoch_loss:.6f}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(bc.state_dict(), os.path.join(args.output_dir, "best.pt"))

        if (epoch + 1) % 50 == 0:
            torch.save(bc.state_dict(), os.path.join(args.output_dir, f"epoch_{epoch + 1}.pt"))

    torch.save(bc.state_dict(), os.path.join(args.output_dir, "final.pt"))
    print(f"Training complete! Best: {args.output_dir}/best.pt")


if __name__ == "__main__":
    main()
