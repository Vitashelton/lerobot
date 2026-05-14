# LeKiwi Public Offline Nav RL

**Multimodal Offline Reinforcement Learning for Safe Navigation of Low-Cost Mobile Robots from Public Visual Navigation Datasets**

[![Python >= 3.10](https://img.shields.io/badge/python-%3E%3D3.10-blue)](https://python.org)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)

## Overview

This project learns safe visual navigation policies for the LeKiwi omnidirectional mobile robot **entirely from public datasets** — no large-scale real-robot data collection required.

**Method**: Convert public visual navigation datasets (Habitat, GNM/ViNT) into LeRobot-compatible format → relabel rewards → train conservative offline RL (IQL, TD3+BC) with multimodal fusion (RGB + Scan64 + state + goal) → deploy with a rule-based safety filter on LeKiwi + Intel RealSense D435i.

```
Public Dataset → LeRobot Format → Reward Relabeling → Multimodal Offline RL → Safety Filter → LeKiwi Deploy
```

## Motivation

Training safe visual navigation policies typically requires:
- Large-scale real-robot data collection (**expensive, time-consuming**)
- High-fidelity simulation environments (**complex to build**)
- Expert demonstrations (**not always available**)

**This project asks**: Can we learn safe navigation policies from existing public datasets and deploy them on a low-cost robot (LeKiwi) with minimal real-robot data?

## What This Project Is

- A pipeline that converts public visual navigation datasets to **LeRobot-compatible** multimodal format
- A **multimodal offline RL** framework (TD3+BC / IQL) incorporating RGB, Scan64, robot state, and goal
- A **rule-based safety filter** as deployment safeguard
- A **deployment stack** for LeKiwi with Intel RealSense D435i
- **Small-scale real-robot validation** (5–10 scenarios)

## What This Project Is NOT

- NOT a diffusion policy project (data quality insufficient)
- NOT a large-scale real-robot data collection project
- NOT a full sim-to-real transfer framework
- NOT an end-to-end learned safety approach (safety filter is rule-based)
- NOT a claim that offline RL alone solves all navigation problems

## Relation to Previous Work

This project builds upon the `lekiwi_rgbd_sim2real_agv` codebase, which provides the **real-robot deployment stack**:

| Module | Source | Role |
|--------|--------|------|
| D435i RGB-D perception | `perception/` | Camera interface |
| Depth → Scan64 projection | `perception/depth_to_scan.py` | Virtual LiDAR |
| ZMQ communication | `communication/` | Robot-server bridge |
| Emergency safety shield | `control/emergency_shield.py` | Hard safety layer |
| Action adapter | `control/action_adapter.py` | Velocity clipping/smoothing |
| DWA policy | `control/dwa_policy.py` | Traditional baseline |

The old deployment stack is reused as the **perception and execution layer** — not the main algorithmic contribution.

## Directory Structure

```
├── configs/                    # YAML configuration files
│   └── new/                    # New thesis project configs
│       ├── default.yaml        #   Default training config
│       ├── iql_config.yaml     #   IQL hyperparameters
│       └── td3bc_config.yaml   #   TD3+BC hyperparameters
│
├── data_adapters/              # Public dataset → unified format
│   ├── base_adapter.py         #   Abstract base class
│   ├── habitat_adapter.py      #   Habitat/HM3D adapter
│   ├── gnm_adapter.py          #   GNM/ViNT adapter
│   ├── data_splitter.py        #   Train/val/test split
│   └── observation_normalizer.py  # Fit & apply normalization
│
├── lerobot_conversion/         # Unified format → LeRobotDataset
│   ├── unified_to_lerobot.py   #   Main converter
│   ├── lerobot_schema.py       #   Observation/action space schema
│   └── validate_dataset.py     #   Integrity validation
│
├── reward/                     # Reward relabeling
│   ├── reward_calculator.py    #   Core reward function
│   ├── progress_estimator.py   #   Goal progress estimation
│   ├── collision_detector.py   #   Collision detection
│   └── intervention_labeler.py #   Intervention labeling
│
├── models/                     # Neural network modules
│   ├── rgb_encoder.py          #   ResNet-18 RGB encoder
│   ├── scan_encoder.py         #   1D-CNN/MLP Scan64 encoder
│   ├── state_encoder.py        #   State vector encoder
│   ├── goal_encoder.py         #   Goal vector encoder
│   ├── fusion_module.py        #   Multimodal fusion
│   ├── actor_network.py        #   Policy network
│   ├── critic_network.py       #   Twin Q-networks
│   └── model_factory.py        #   Build model from config
│
├── rl/                         # Offline RL algorithms
│   ├── replay_buffer.py        #   Offline replay buffer
│   ├── td3bc.py                #   TD3+BC implementation
│   └── iql.py                  #   IQL implementation
│
├── safety/                     # Safety filter
│   └── safety_filter.py        #   9-layer safety pipeline
│
├── baselines/                  # Baseline methods
│   ├── behavior_cloning.py     #   BC baseline
│   └── dwa_baseline.py         #   DWA wrapper
│
├── eval/                       # Evaluation
│   ├── metrics.py              #   All metrics
│   └── offline_evaluator.py    #   Offline evaluation loop
│
├── lekiwi_deployment/          # Real-robot deployment
│   ├── observation_assembler.py  # Sensor → observation dict
│   └── deployment_runner.py    #   Main deployment loop
│
├── scripts/                    # Entry-point scripts
│   ├── convert_to_lerobot.py   #   Dataset conversion
│   ├── train_iql.py            #   IQL training
│   ├── train_td3bc.py          #   TD3+BC training
│   ├── train_bc.py             #   BC training
│   └── evaluate_offline.py     #   Offline evaluation
│
├── control/                    # (Reused) DWA, shield, adapter
├── communication/              # (Reused) ZMQ host/client
├── perception/                 # (Reused) D435i pipeline
├── paper/                      # Thesis documents
│   └── THESIS_PLAN.md          #   Full project plan
│
└── tests/                      # Unit tests
```

## Installation

Requires the `lerobot` conda environment:

```bash
conda activate lerobot
cd lekiwi_rgbd_sim2real_agv
pip install -e ".[all]"
```

Dependencies: `torch >= 2.0`, `torchvision`, `numpy`, `opencv-python`, `pyzmq`, `draccus`, `scipy`, `tqdm`, `pyyaml`, `h5py`.

## Quick Start

### 1. Prepare Dataset

```bash
# Convert a Habitat dataset to LeRobot format
python scripts/convert_to_lerobot.py \
    --dataset habitat \
    --data-dir data/raw/habitat \
    --output-dir data/lerobot/habitat_nav

# Or convert GNM/ViNT dataset
python scripts/convert_to_lerobot.py \
    --dataset gnm \
    --format h5 \
    --data-dir data/raw/gnm \
    --output-dir data/lerobot/gnm_nav
```

### 2. Train

```bash
# Train IQL (recommended — naturally conservative)
python scripts/train_iql.py \
    --data-dir data/lerobot/habitat_nav \
    --config configs/new/iql_config.yaml \
    --output-dir checkpoints/iql_habitat \
    --device cuda

# Train TD3+BC (comparison)
python scripts/train_td3bc.py \
    --data-dir data/lerobot/habitat_nav \
    --config configs/new/td3bc_config.yaml \
    --output-dir checkpoints/td3bc_habitat \
    --device cuda

# Train BC baseline
python scripts/train_bc.py \
    --data-dir data/lerobot/habitat_nav \
    --output-dir checkpoints/bc_habitat \
    --device cuda
```

### 3. Evaluate

```bash
# Offline evaluation
python scripts/evaluate_offline.py \
    --checkpoint checkpoints/iql_habitat/best.pt \
    --data-dir data/lerobot/habitat_nav_test \
    --method iql \
    --with-safety \
    --output results/iql_offline.json
```

### 4. Deploy on LeKiwi (Real Robot)

```bash
# On the LeKiwi host (Jetson / Raspberry Pi):
python communication/host.py robot:@lekiwi_d435i

# On the control laptop with GPU:
python -c "
from lekiwi_deployment.deployment_runner import DeploymentRunner
from rl.iql import IQL
from models.model_factory import ModelFactory
import torch, yaml

# Load model
with open('configs/new/default.yaml') as f:
    config = yaml.safe_load(f)
model = ModelFactory.create(config)
trainer = IQL(model, config, device='cuda')
trainer.load_state_dict(torch.load('checkpoints/iql_habitat/best.pt'))

# Run deployment
runner = DeploymentRunner(trainer, config)
result = runner.run_episode(goal_vector=np.array([3.0, 0.0, 0.0]))
print(result['safety_stats'])
"
```

## Method Overview

### Multimodal Architecture

```
RGB (3,224,224) ─→ ResNet-18 ──→ rgb_feat (256,)  ─┐
Scan64 (64,)  ──→ 1D-CNN/MLP ─→ scan_feat (128,)  ─┤
State (3,)    ──→ MLP ────────→ state_feat (64,)   ─┼─→ Fusion → Actor → [vx, vy, ω]
Goal (3,)     ──→ MLP ────────→ goal_feat (64,)    ─┤
                                                     │
                                   Twin Critic ←────┘
```

### Offline RL: IQL (Implicit Q-Learning)

Naturally conservative — no OOD action sampling needed:

- **V(s)**: expectile regression `L2^τ(Q - V)` — controls conservatism via τ
- **Q(s,a)**: standard Bellman backup with target V
- **π(s)**: advantage-weighted regression — stays close to dataset

### Safety Filter Pipeline

```
RL Action → Invalid Depth? → Emergency Stop? → Lateral Inhibit?
→ Velocity Scaling → Rotation Stop → Clipping → Accel Limit → Smoothing
→ Safe Action
```

## Baselines

1. **BC** — Behavior Cloning (MSE on actions)
2. **TD3+BC w/o Fusion** — No RGB, no goal encoder
3. **IQL w/o Safety** — Full model without safety filter
4. **DWA** — Dynamic Window Approach (traditional planner)
5. **Ours w/o Scan64** — RGB only (no depth/scan)
6. **Ours w/o Safety Filter** — Full model, no safety
7. **Ours (Full)** — IQL + multimodal + safety filter

## Evaluation Metrics

### Offline
- Return (Σ γ^t r_t), success proxy, collision risk, unsafe action rate
- Action smoothness, OOD action deviation, Q-value distribution

### Real-Robot (Small-Scale)
- Success rate, collision rate, min obstacle distance
- Emergency stop count, intervention count, trajectory smoothness
- Inference latency, CPU/GPU usage

## Citation

```bibtex
@mastersthesis{lekiwi_public_offline_nav_rl2026,
  title  = {Multimodal Offline Reinforcement Learning for Safe Navigation
            of Low-Cost Mobile Robots from Public Visual Navigation Datasets},
  author = {[TBD]},
  year   = {2026},
  school = {[TBD]},
}

@misc{lerobot2024,
  title  = {LeRobot: State-of-the-art AI for Real-World Robotics},
  author = {Cadene, Remi and Alibert, Simon and So, Alexander and others},
  year   = {2024},
  publisher = {Hugging Face}
}
```

## License

Apache 2.0 (same as LeRobot).
