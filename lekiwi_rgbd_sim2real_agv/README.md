# LeKiwi RGB-D Sim-to-Real AGV

AGV perception and safe navigation system extending [Hugging Face LeRobot](https://github.com/huggingface/lerobot) LeKiwi with Intel RealSense D435i RGB-D perception.

## Overview

This project builds a complete "simulation + real" pipeline for AGV safe navigation:

```
Synthetic RGB-D → MuJoCo/Gym LeKiwi → LeRobotDataset → residual safety model → real LeKiwi + D435i
```

### Why this project?

- **LeRobot** already supports LeKiwi (omniwheel mobile manipulator) and RealSense cameras.
- **LeKiwi's default camera config** uses `OpenCVCamera`, which does NOT provide depth.
- **LeRobot's `RealSenseCameraConfig`** supports `use_depth=True`, but LeKiwi doesn't combine them out of the box.
- This project extends LeKiwi + D435i with a full perception-to-control stack for AGV safety.

### Key Features

| Module                        | Description                                                        |
| ----------------------------- | ------------------------------------------------------------------ |
| **D435i Depth-to-Scan**       | Convert RealSense depth to 64-D polar scan with percentile pooling |
| **Obstacle Detection**        | Left/front/right sector risk assessment from depth scan            |
| **ArUco Pallet Localization** | 3D pallet pose estimation via ArUco markers + depth                |
| **YOLO Detection**            | Object detection (person, box, chair) with depth localization      |
| **Synthetic RGB-D**           | Procedural warehouse/lab scene generator with depth noise          |
| **MuJoCo LeKiwi Env**         | Gymnasium env for omniwheel AGV with scan observation              |
| **Residual Safety Model**     | Small MLP that predicts safety corrections to raw actions          |
| **DWA Policy**                | Dynamic Window Approach for omnidirectional navigation             |
| **Emergency Shield**          | Hard safety layer: stop, slow-down, lateral inhibit                |
| **Web Dashboard**             | Flask-based real-time monitoring (RGB, depth, scan, FPS)           |
| **LeRobotDataset Writer**     | Record data in LeRobot-compatible format                           |

## Project Structure

```
lekiwi_rgbd_sim2real_agv/
  configs/             # YAML configuration files
  lerobot_ext/         # LeKiwi + D435i config, host, client, dataset writer
  camera/              # D435i depth processing pipeline
  perception/          # Obstacle, ArUco, YOLO, tracker, safety zone
  sim/                 # Synthetic RGB-D + MuJoCo envs + Gazebo placeholder
  learning/            # Residual safety model training
  control/             # DWA, emergency shield, residual controller
  app/                 # Demo scripts + web dashboard
  tools/               # Data collection, evaluation, comparison
```

## Installation

Requires the `lerobot` conda environment (with LeRobot and its dependencies already installed).

```bash
conda activate lerobot
cd lekiwi_rgbd_sim2real_agv
pip install -e .

# With optional dependencies
pip install -e ".[camera]"       # pyrealsense2 (for real D435i)
pip install -e ".[sim]"          # gymnasium + mujoco
pip install -e ".[learning]"     # torch
pip install -e ".[detection]"    # ultralytics YOLO
pip install -e ".[dash]"         # flask web dashboard
pip install -e ".[all]"          # everything
```

### Dependencies

| Extra         | Packages                                                      | Purpose                        |
| ------------- | ------------------------------------------------------------- | ------------------------------ |
| (core)        | `numpy`, `opencv-python`, `pyzmq`, `draccus`, `scipy`, `tqdm` | Always required                |
| `[camera]`    | `pyrealsense2`                                                | Real D435i depth streaming     |
| `[sim]`       | `gymnasium`, `mujoco`                                         | Simulation environments        |
| `[learning]`  | `torch`                                                       | Residual safety model training |
| `[detection]` | `ultralytics`                                                 | YOLO object detection          |
| `[dash]`      | `flask`                                                       | Web monitoring dashboard       |

LeRobot itself (`lerobot`) must be installed in the environment — this project extends it, not replaces it.

## Usage

### 1. Synthetic RGB-D Generation

```bash
python sim/synthetic_rgbd/render_dataset.py \
    --output-dir data/synthetic \
    --num-scenes 1000 \
    --seed 42
```

### 2. Sim Demo (Synthetic + DWA)

```bash
python app/run_sim_demo.py \
    --scene-type warehouse_aisle \
    --num-episodes 5 \
    --display
```

### 3. Real Demo (LeKiwi + D435i)

```bash
# On the LeKiwi host (Raspberry Pi / Jetson):
python lerobot_ext/lekiwi_d435i_host.py \
    robot:@lekiwi_d435i \
    host:scan_dim=64 host:enable_aruco=true

# On the control laptop:
python app/run_real_demo.py \
    --remote-ip 192.168.1.100 \
    --safe-mode \
    --display
```

### 4. Collect Real Dataset

```bash
python tools/collect_real_dataset.py \
    --remote-ip 192.168.1.100 \
    --output-dir data/real_logs \
    --num-episodes 10 \
    --teleop
```

### 5. Build Training Dataset

```bash
python learning/build_dataset.py \
    --sim-data-dir data/synthetic \
    --real-data-dir data/real_logs \
    --output-dir data/training
```

### 6. Train Residual Safety Model

```bash
python learning/train_residual.py \
    --data-dir data/training \
    --output-dir checkpoints \
    --epochs 100 \
    --device cuda
```

### 7. Evaluate Navigation

```bash
python tools/evaluate_navigation.py --results-dir demo_output/sim
```

### 8. Sim-to-Real Comparison

```bash
python tools/sim_to_real_compare.py \
    --sim-data data/sim_logs \
    --real-data data/real_logs \
    --output report/sim2real
```

### 9. Offline Replay

```bash
python app/run_offline_replay.py --data-dir data/real_logs/episode_000
```

### 10. Export Demo Video

```bash
python tools/export_demo_video.py \
    --input-dir demo_output/real \
    --output demo.mp4
```

## Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│  Phase 1: Perception & Data Pipeline                        │
│                                                             │
│  D435i ──→ depth_preprocess ──→ depth_to_scan ──→ scan64   │
│    │                          │                             │
│    ├─→ RGB ──→ ArUco/YOLO ──→ detections                   │
│    │              │                                         │
│    │              └─→ depth_localizer ──→ 3D positions      │
│    │                                                        │
│    └─→ LeKiwiDatasetWriter ──→ LeRobotDataset               │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│  Phase 2: Learning                                          │
│                                                             │
│  Synthetic ──→ Training Data ──→ ResidualSafetyModel        │
│  Real Logs ──┘               ──→ risk_scorer labels         │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│  Phase 3: Control                                           │
│                                                             │
│  scan ──→ DWA Policy ──→ raw_action                         │
│                           │                                 │
│                           └─→ Residual Controller           │
│                               │ (raw + residual_delta)      │
│                               └─→ Emergency Shield          │
│                                   │ (stop/slow/lateral)     │
│                                   └─→ Action Adapter        │
│                                       │ (clip/smooth)       │
│                                       └─→ LeKiwi Base       │
└─────────────────────────────────────────────────────────────┘
```

## Safety Architecture

The safety stack has three layers:

1. **Emergency Shield** (hard, non-learned): Stops forward motion if `front_min < 0.15m`. Inhibits lateral motion if side clearance insufficient. Overrides everything.

2. **Residual Safety Model** (learned): MLP that predicts `delta_action` corrections to the raw DWA output. Trained to avoid collisions while preserving goal-directed behavior.

3. **Action Adapter** (rule-based): Clips to velocity limits, enforces acceleration limits, applies low-pass temporal smoothing.

## Phase Priority

### Phase 1 (current)
- [x] D435i depth-to-scan
- [x] Obstacle detection
- [x] ArUco pallet localization
- [x] LeKiwi + D435i host/client
- [x] Synthetic RGB-D generation
- [x] Web dashboard
- [x] Logging/replay

### Phase 2 (planned)
- [ ] MuJoCo LeKiwi env refinement
- [ ] Residual safety model training & validation
- [ ] Sim-to-real comparison & domain gap analysis
- [ ] Real LeKiwi low-speed validation

### Phase 3 (planned)
- [ ] Gazebo/ROS2 system-level simulation
- [ ] YOLO fine-tuning on warehouse objects
- [ ] Complete AGV pallet approach demo

## Key Design Decisions

1. **Not modifying LeRobot main repo**: This project lives alongside LeRobot, reusing its abstractions without forking.

2. **Depth stays local**: Raw depth images (640x480 uint16) are large. They are processed on-board and only the 64-D scan (bytes) and safety distances (floats) are sent over ZMQ.

3. **Percentile pooling, not min**: The `depth_to_scan` uses 10th percentile, not minimum, to be robust to noise while still detecting thin obstacles.

4. **Residual learning, not end-to-end**: The residual model corrects a DWA baseline rather than learning navigation from scratch. This is safer and requires less data.

5. **Simulation-first training**: Synthetic data + MuJoCo provide large-scale training data before real-world validation.

## Addition
If TypeError: must be called with a dataclass type or instance
```bash
PYTHONPATH=$PWD python -m sim.synthetic_rgbd.render_dataset \
    --output_dir data/synthetic \
    --num_scenes 1000 \
    --seed 42
```
## License

Apache 2.0 (same as LeRobot)

## Citation

If you use this work, please cite both this project and LeRobot:

```bibtex
@misc{lerobot2024,
  title = {LeRobot: State-of-the-art AI for real-world robotics},
  author = {Cadene, Remi and Alibert, Simon and So, Alexander and others},
  year = {2024},
  publisher = {Hugging Face}
}
```
