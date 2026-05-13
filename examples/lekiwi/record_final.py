import argparse
import time
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.processor import make_default_processors
from lerobot.robots.lekiwi.config_lekiwi import LeKiwiClientConfig
from lerobot.robots.lekiwi.lekiwi_client import LeKiwiClient
from lerobot.scripts.lerobot_record import record_loop
from lerobot.teleoperators.keyboard import KeyboardTeleop, KeyboardTeleopConfig
from lerobot.teleoperators.so_leader import SO100Leader, SO100LeaderConfig
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.visualization_utils import init_rerun

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", type=str, required=True)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--num_episodes", type=int, default=5)
    parser.add_argument("--episode_time_s", type=int, default=40)
    parser.add_argument("--reset_time_s", type=int, default=10)
    args = parser.parse_args()

    # 1. 老老实实把所有硬件都配上，满足底层自检
    robot = LeKiwiClient(LeKiwiClientConfig(remote_ip="192.168.3.215", id="lekiwi"))
    leader_arm = SO100Leader(SO100LeaderConfig(port="/dev/ttyACM0", id="leader_arm"))
    keyboard = KeyboardTeleop(KeyboardTeleopConfig())

    tele_proc, rob_act_proc, rob_obs_proc = make_default_processors()
    dataset_features = {
        **hw_to_dataset_features(robot.action_features, ACTION),
        **hw_to_dataset_features(robot.observation_features, OBS_STR)
    }

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id, fps=args.fps, features=dataset_features, use_videos=True, image_writer_threads=8
    )

    print("[INFO] 连接设备...")
    robot.connect()
    leader_arm.connect()
    keyboard.connect()

    listener, events = init_keyboard_listener()
    init_rerun(session_name="lekiwi_nav")

    recorded_episodes = 0
    while recorded_episodes < args.num_episodes and not events["stop_recording"]:
        print(f"\n🎬 第 {recorded_episodes + 1}/{args.num_episodes} 个 Episode... (只用键盘控制底盘)")
        record_loop(
            robot=robot, events=events, fps=args.fps, dataset=dataset,
            teleop=[leader_arm, keyboard], # 老老实实双遥控器，机械臂放桌上别动
            control_time_s=args.episode_time_s, single_task="navigation",
            display_data=True, teleop_action_processor=tele_proc,
            robot_action_processor=rob_act_proc, robot_observation_processor=rob_obs_proc,
        )
        dataset.save_episode()
        recorded_episodes += 1
        if recorded_episodes < args.num_episodes and not events["stop_recording"]:
            time.sleep(args.reset_time_s)

    dataset.finalize()
    print("🎉 录制完成！这是一个合法的 9 维数据集。")

if __name__ == "__main__":
    main()