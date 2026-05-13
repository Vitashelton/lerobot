import time
import sys
import os
import cv2
import numpy as np

from lerobot.robots.lekiwi.config_lekiwi import LeKiwiClientConfig
from lerobot.robots.lekiwi.lekiwi_client import LeKiwiClient
from lerobot.utils.robot_utils import precise_sleep


# ===== 参数读取 =====
def get_arg(name, default):
    for arg in sys.argv:
        if arg.startswith(name + "="):
            return arg.split("=", 1)[1]
    return default


REPO_ID = get_arg("--repo_id", "auto_data")
FPS = int(get_arg("--fps", 10))
EPISODE_TIME = int(get_arg("--time", 120))
ROBOT_IP = get_arg("--ip", "192.168.3.215")


# ===== 创建保存目录 =====
SAVE_DIR = f"data/{REPO_ID}_{int(time.time())}"
os.makedirs(SAVE_DIR, exist_ok=True)


# ===== 简单避障 =====
def simple_avoidance(frame):
    h, w, _ = frame.shape

    left = frame[:, :w//3]
    center = frame[:, w//3:2*w//3]
    right = frame[:, 2*w//3:]

    left_score = np.mean(left)
    center_score = np.mean(center)
    right_score = np.mean(right)

    if center_score < 80:
        if left_score > right_score:
            return "left"
        else:
            return "right"
    else:
        return "forward"


def action_from_decision(decision):
    if decision == "forward":
        return {"linear": 0.2, "angular": 0.0}
    elif decision == "left":
        return {"linear": 0.0, "angular": 0.5}
    elif decision == "right":
        return {"linear": 0.0, "angular": -0.5}


# ===== 主程序 =====
def main():
    print(f"[INFO] Saving to: {SAVE_DIR}")

    robot = LeKiwiClient(LeKiwiClientConfig(remote_ip=ROBOT_IP))
    robot.connect()

    if not robot.is_connected:
        raise RuntimeError("Robot not connected")

    action_file = open(os.path.join(SAVE_DIR, "actions.txt"), "w")

    frame_id = 0
    start_time = time.time()

    while time.time() - start_time < EPISODE_TIME:
        t0 = time.perf_counter()

        obs = robot.get_observation()
        frame = obs["image"]

        decision = simple_avoidance(frame)
        action = action_from_decision(decision)

        robot.send_action(action)

        # ===== 保存图像 =====
        img_path = os.path.join(SAVE_DIR, f"{frame_id:06d}.jpg")
        cv2.imwrite(img_path, frame)

        # ===== 保存动作 =====
        action_file.write(f"{frame_id},{action['linear']},{action['angular']}\n")

        frame_id += 1

        precise_sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0))

    action_file.close()
    robot.disconnect()
    print("[INFO] Finished recording")


if __name__ == "__main__":
    main()