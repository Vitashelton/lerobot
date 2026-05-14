#!/usr/bin/env python
"""Offline evaluation of trained policies.

Usage:
    python scripts/evaluate_offline.py \
        --checkpoint checkpoints/iql_habitat/best.pt \
        --data-dir data/lerobot/habitat_nav_test \
        --method iql \
        --config configs/new/default.yaml \
        --output results/offline/iql.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import yaml
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.model_factory import ModelFactory
from rl.iql import IQL
from rl.td3bc import TD3BC
from baselines.behavior_cloning import BehaviorCloning
from eval.offline_evaluator import OfflineEvaluator
from safety.safety_filter import SafetyFilter, SafetyConfig
from scripts.convert_to_lerobot import _load_split_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--method", type=str, required=True,
                        choices=["iql", "td3bc", "bc"])
    parser.add_argument("--config", type=str, default="configs/new/default.yaml")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--with-safety", action="store_true")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Load test data (supports both old and new format)
    test_data = _load_split_data(args.data_dir + ".npz")

    # Build model
    model = ModelFactory.create(config)

    # Create trainer
    if args.method == "iql":
        trainer = IQL(model, config, device=args.device)
    elif args.method == "td3bc":
        trainer = TD3BC(model, config, device=args.device)
    elif args.method == "bc":
        bc_config = {"training": {"actor_lr": 3e-4}}
        trainer = BehaviorCloning(model, bc_config, device=args.device)

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=args.device)
    trainer.load_state_dict(ckpt)

    # Safety filter
    safety = None
    if args.with_safety:
        safety = SafetyFilter(SafetyConfig())

    # Evaluate
    evaluator = OfflineEvaluator(trainer, safety_filter=safety, device=args.device)
    metrics = evaluator.evaluate(
        test_data,
        batch_size=args.batch_size,
        use_safety_filter=args.with_safety,
    )

    # Print results
    print("\n" + "=" * 50)
    print(f"  Offline Evaluation: {args.method.upper()}")
    print("=" * 50)
    for k, v in sorted(metrics.items()):
        print(f"  {k:30s}: {v:.6f}")
    print("=" * 50)

    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    results = {
        "method": args.method,
        "checkpoint": args.checkpoint,
        "safety_filter": args.with_safety,
        "metrics": metrics,
    }
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
