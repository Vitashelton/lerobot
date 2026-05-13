import zmq
import json
import base64
import cv2
import numpy as np
import time

# Pi Host IP
PI_HOST = "192.168.3.215"  # 改成你的 Pi IP

# ZMQ 初始化
context = zmq.Context()

# 发送动作
cmd_socket = context.socket(zmq.PUSH)
cmd_socket.connect(f"tcp://{PI_HOST}:5555")

# 接收观测
obs_socket = context.socket(zmq.PULL)
obs_socket.connect(f"tcp://{PI_HOST}:5556")

def send_target(target_x, target_y):
    """发送目标点给 Pi Host"""
    action = {"goal_x": target_x, "goal_y": target_y}
    cmd_socket.send_string(json.dumps(action))

def show_camera(obs_json):
    """解码 Pi 摄像头画面并显示"""
    obs = json.loads(obs_json)
    for cam_name, img_b64 in obs.items():
        img_bytes = base64.b64decode(img_b64)
        np_img = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        cv2.imshow(f"{cam_name} Camera", frame)
    cv2.waitKey(1)

if __name__ == "__main__":
    try:
        # 这里可以替换为 LoGoPlanner 输出的目标点序列
        target_path = [(1.0, 2.0), (2.0, 0.5), (1.5, 1.5)]
        for target in target_path:
            print(f"Sending target {target}")
            send_target(*target)
            time.sleep(0.1)

            # 等待 Pi 执行动作并显示摄像头画面
            start_time = time.time()
            while time.time() - start_time < 5:  # 每个目标停留 5 秒观测
                try:
                    obs_json = obs_socket.recv_string(flags=zmq.NOBLOCK)
                    show_camera(obs_json)
                except zmq.Again:
                    time.sleep(0.01)

    except KeyboardInterrupt:
        print("Exiting PC client")

    finally:
        cv2.destroyAllWindows()
        cmd_socket.close()
        obs_socket.close()
        context.term()