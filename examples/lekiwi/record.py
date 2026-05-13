#!/usr/bin/env python

import argparse
import time
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.processor import make_default_processors
from lerobot.robots.lekiwi.config_lekiwi import LeKiwiClientConfig
from lerobot.robots.lekiwi.lekiwi_client import LeKiwiClient
from lerobot.scripts.lerobot_record import record_loop
from lerobot.teleoperators.keyboard import KeyboardTeleop, KeyboardTeleopConfig
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--control.repo_id", dest="repo_id", type=str, default="akzhao1238/pure_nav_test")
    parser.add_argument("--control.num_episodes", dest="num_episodes", type=int, default=30)
    parser.add_argument("--control.episode_time_s", dest="episode_time_s", type=int, default=60)
    parser.add_argument("--control.reset_time_s", dest="reset_time_s", type=int, default=10)
    parser.add_argument("--control.single_task", dest="single_task", type=str, default="navigate")
    parser.add_argument("--control.fps", dest="fps", type=int, default=10) 
    parser.add_argument("--control.tags", dest="tags", type=str, default='["nav"]')
    return parser.parse_args()

def main():
    args = parse_args()
    HF_REPO_ID = args.repo_id

    # 1. 配置小车
    robot_config = LeKiwiClientConfig(remote_ip="192.168.3.215", id="lekiwi")
    
    # 2. 配置遥控设备（只留键盘！）
    keyboard_config = KeyboardTeleopConfig()

    # 3. 实例化设备（只留小车和键盘！）
    robot = LeKiwiClient(robot_config)
    keyboard = KeyboardTeleop(keyboard_config)

    # 4. 预处理器逻辑
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    action_features = hw_to_dataset_features(robot.action_features, ACTION)
    obs_features = hw_to_dataset_features(robot.observation_features, OBS_STR)
    dataset_features = {**action_features, **obs_features}

    # 5. 创建数据集
    dataset = LeRobotDataset.create(
        repo_id=HF_REPO_ID,
        fps=args.fps,
        features=dataset_features,
        robot_type=robot.name,
        use_videos=True,
        image_writer_threads=8,
    )

    print("\n[INFO] 正在连接设备...")
    robot.connect()
    keyboard.connect()  # 只连键盘！

    listener, events = init_keyboard_listener()

    recorded_episodes = 0
    while recorded_episodes < args.num_episodes and not events["stop_recording"]:
        log_say(f"Recording episode {recorded_episodes}")

        record_loop(
            robot=robot,
            events=events,
            fps=args.fps,
            dataset=dataset,
            teleop=[keyboard],  # 只传键盘！
            control_time_s=args.episode_time_s,
            single_task=args.single_task,
            display_data=False,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
        )

        if events["rerecord_episode"]:
            dataset.clear_episode_buffer()
            events["rerecord_episode"] = False
            continue

        dataset.save_episode()
        recorded_episodes += 1

    dataset.finalize()
    dataset.push_to_hub()
    print("🎉 录制结束，数据集已上传！")

if __name__ == "__main__":
    main()

