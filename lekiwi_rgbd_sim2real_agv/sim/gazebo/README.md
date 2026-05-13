# Gazebo Simulation (Phase 3)

Gazebo + ROS2 integration for system-level simulation and demonstration.

## Planned Features

- Gazebo world with textured warehouse/lab models
- ROS2 bridge for LeKiwi base control (`/cmd_vel` <-> x.vel, y.vel, theta.vel)
- D435i depth camera plugin producing point clouds
- ArUco pallet models for detection testing
- Multi-sensor recording (RGB, depth, IMU, odometry)

## Status

**Not yet implemented.** This is a Phase 3 deliverable.

Phase 1-2 focus on Synthetic RGB-D + MuJoCo for training data generation.
Gazebo is reserved for system-level demonstration with sensor noise models.

## Directory layout (future)

```
sim/gazebo/
    worlds/          # .sdf world files
    models/          # custom model meshes and URDFs
    ros_nodes/       # ROS 2 Python nodes (camera driver, bridge)
    launch/          # .launch.py files
```

## Quick-start (future)

```bash
# Terminal 1: Gazebo
ign gazebo sim/gazebo/worlds/warehouse_aisle.sdf

# Terminal 2: ROS 2 bridge
ros2 launch lekiwi_sim bringup.launch.py

# Terminal 3: Run policy
python -m sim.gazebo.run_policy --config configs/robot_lekiwi.yaml
```

## References

- [LeKiwi MuJoCo assets](../mujoco_lekiwi/assets/) for equivalent MuJoCo models
- RealSense D435i [ROS2 wrapper](https://github.com/IntelRealSense/realsense-ros)
