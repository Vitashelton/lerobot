# !/usr/bin/env python

import time
import sys

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.robots.lekiwi.config_lekiwi import LeKiwiClientConfig
from lerobot.robots.lekiwi.lekiwi_client import LeKiwiClient
from lerobot.utils.constants import ACTION
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import log_say


def get_arg(prefix, default):
    for arg in sys.argv:
        if arg.startswith(prefix + "="):
            return arg.split("=", 1)[1]
    return default


def main():
    # ===== 从命令行读取参数（没有就用默认）=====
    EPISODE_IDX = int(get_arg("--control.episode", 0))
    REPO_ID = get_arg("--control.repo_id", "akzhao1238/move_1")
    FPS = int(get_arg("--control.fps", 30))
    REMOTE_IP = get_arg("--robot.ip", "192.168.3.215")

    print(f"[INFO] repo_id = {REPO_ID}")
    print(f"[INFO] episode = {EPISODE_IDX}")
    print(f"[INFO] fps = {FPS}")
    print(f"[INFO] robot_ip = {REMOTE_IP}")

    # ===== 初始化机器人 =====
    robot_config = LeKiwiClientConfig(remote_ip=REMOTE_IP, id="lekiwi")
    robot = LeKiwiClient(robot_config)

    # ===== 加载数据集 =====
    dataset = LeRobotDataset(REPO_ID, episodes=[EPISODE_IDX])

    episode_frames = dataset.hf_dataset.filter(
        lambda x: x["episode_index"] == EPISODE_IDX
    )
    actions = episode_frames.select_columns(ACTION)

    # ===== 连接机器人 =====
    robot.connect()

    if not robot.is_connected:
        raise ValueError("Robot is not connected!")

    print("Starting replay loop...")
    log_say(f"Replaying episode {EPISODE_IDX}")

    # ===== 回放动作 =====
    for idx in range(len(episode_frames)):
        t0 = time.perf_counter()

        action = {
            name: float(actions[idx][ACTION][i])
            for i, name in enumerate(dataset.features[ACTION]["names"])
        }

        _ = robot.send_action(action)

        precise_sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))

    robot.disconnect()


if __name__ == "__main__":
    main()
