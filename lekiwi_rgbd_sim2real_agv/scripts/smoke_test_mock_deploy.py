import numpy as np

from lekiwi_deployment.deployment_runner import DeploymentRunner


class DummyTrainer:
    def predict(self, rgb, scan, state, goal):
        batch = scan.shape[0]
        return np.tile(
            np.array([[0.1, 0.0, 0.0]], dtype=np.float32),
            (batch, 1),
        )


def main():
    config = {
        "deployment": {
            "mock_mode": True,
            "max_episode_steps": 5,
            "camera": {
                "height": 224,
                "width": 224,
            },
        },
        "observation": {
            "scan64": {
                "dim": 64,
                "max_range": 5.0,
            }
        },
        "action": {
            "vx_limits": [-0.3, 0.3],
            "vy_limits": [-0.3, 0.3],
            "omega_limits": [-90.0, 90.0],
        },
        "safety": {
            "stop_distance": 0.15,
            "slow_distance": 0.5,
            "lateral_inhibit_distance": 0.3,
            "rotation_stop_multiplier": 1.5,
            "max_velocity": [0.3, 0.3, 90.0],
            "max_acceleration": [0.5, 0.5, 180.0],
            "smoothing_alpha": 0.3,
            "dt": 0.1,
        },
    }

    runner = DeploymentRunner(
        trainer=DummyTrainer(),
        config=config,
        device="cpu",
    )

    result = runner.run_episode(
        goal_vector=np.array([3.0, 0.0, 0.0], dtype=np.float32)
    )

    print("Smoke test passed.")
    print("Episode length:", result["episode_length"])
    print("Safety stats:", result["safety_stats"])
    print("First action:", result["log"][0]["adapted_action"])


if __name__ == "__main__":
    main()