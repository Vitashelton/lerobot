#!/usr/bin/env python

import cv2
import time
import numpy as np
import zmq
import json
import threading
from collections import deque

from lerobot.robots.lekiwi.config_lekiwi import LeKiwiClientConfig
from lerobot.robots.lekiwi.lekiwi_client import LeKiwiClient

def smooth_depth(new_val, history_q, max_len=5):
    if new_val < 9000 and new_val > 0: history_q.append(new_val)
    if not history_q: return 9999
    return int(np.mean(history_q))

# 🔥 新增：专门接收 1D 雷达数据的后台监听类
class LidarReceiver:
    def __init__(self, ip):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.setsockopt(zmq.CONFLATE, 1)
        self.socket.connect(f"tcp://{ip}:7777")
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self.latest_lidar = None
        self.running = True
        self.thread = threading.Thread(target=self._recv_loop, daemon=True)
        self.thread.start()

    def _recv_loop(self):
        while self.running:
            try:
                msg = self.socket.recv_string(flags=zmq.NOBLOCK)
                self.latest_lidar = json.loads(msg)["lidar"]
            except zmq.Again:
                time.sleep(0.005)

def main():
    target_ip = "192.168.3.215"
    print(f"🚀 [INFO] 正在连接 Lekiwi 底盘 ({target_ip})...")
    
    # 1. 连接 Lekiwi 官方通道 (控制电机)
    robot_config = LeKiwiClientConfig(remote_ip=target_ip, id="lekiwi")
    robot = LeKiwiClient(robot_config)
    robot.connect()
    
    # 2. 连接我们的旁路接收通道 (获取雷达数据)
    lidar_receiver = LidarReceiver(target_ip)
    
    if not robot.is_connected:
        print("❌ [ERROR] 树莓派连接失败！")
        return

    print("✅ [INFO] 双通道连接成功！开始【1D 降维雷达】巡航...")
    
    SAFE_DIST_MM = 550
    CRITICAL_DIST_MM = 250
    hist_l, hist_c, hist_r = deque(maxlen=5), deque(maxlen=5), deque(maxlen=5)
    
    try:
        while True:
            start_time = time.time()
            
            # 从官方通道获取电机状态 (必须调用以保持通讯心跳)
            obs = robot.get_observation()
            
            # 🔥 从旁路通道秒读雷达数据
            pseudo_lidar = lidar_receiver.latest_lidar
            
            if not pseudo_lidar or len(pseudo_lidar) != 64:
                print(f"⚠️ [警告] 等待旁路通道 (5555) 发送雷达数据...")
                time.sleep(0.5)
                continue
                
            lidar_mm = np.array(pseudo_lidar) * 1000.0

            # --- 下方完全保留你的避障与渲染逻辑 ---
            zone_l_rays, zone_c_rays, zone_r_rays = lidar_mm[0:21], lidar_mm[21:43], lidar_mm[43:64]
            
            dist_l = smooth_depth(np.min(zone_l_rays) if len(zone_l_rays)>0 else 5000, hist_l)
            dist_c = smooth_depth(np.min(zone_c_rays) if len(zone_c_rays)>0 else 5000, hist_c)
            dist_r = smooth_depth(np.min(zone_r_rays) if len(zone_r_rays)>0 else 5000, hist_r)

            action_dict = {
                "arm_shoulder_pan.pos": obs.get("arm_shoulder_pan.pos", 0.0),
                "arm_shoulder_lift.pos": obs.get("arm_shoulder_lift.pos", 0.0),
                "arm_elbow_flex.pos": obs.get("arm_elbow_flex.pos", 0.0),
                "arm_wrist_flex.pos": obs.get("arm_wrist_flex.pos", 0.0),
                "arm_wrist_roll.pos": obs.get("arm_wrist_roll.pos", 0.0),
                "arm_gripper.pos": obs.get("arm_gripper.pos", 0.0),
                "x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0
            }
            
            state_text, color = "CRUISING", (0, 255, 0)
            
            if dist_c < CRITICAL_DIST_MM or dist_l < CRITICAL_DIST_MM or dist_r < CRITICAL_DIST_MM:
                state_text, color, action_dict["x.vel"] = "EMERGENCY: REVERSE!", (0, 0, 255), -0.15 
            elif dist_c < SAFE_DIST_MM:
                if dist_l > dist_r: state_text, color, action_dict["y.vel"] = "DODGE LEFT", (0, 165, 255), 0.3    
                else: state_text, color, action_dict["y.vel"] = "DODGE RIGHT", (0, 165, 255), -0.3   
            else:
                action_dict["x.vel"] = 0.2

            robot.send_action(action_dict)

            # --- HUD 渲染 ---
            hud_w, hud_h, bar_width, max_plot_dist = 640, 400, 10, 3000.0
            hud = np.zeros((hud_h, hud_w, 3), dtype=np.uint8)
            
            for i, dist in enumerate(lidar_mm):
                bar_h = int((min(dist, max_plot_dist) / max_plot_dist) * 250)
                bar_color = (0, 0, 255) if dist < CRITICAL_DIST_MM else ((0, 165, 255) if dist < SAFE_DIST_MM else (0, 255, 0))
                cv2.rectangle(hud, (i * bar_width, 300 - bar_h), ((i + 1) * bar_width, 300), bar_color, -1)
            
            cv2.line(hud, (21 * bar_width, 50), (21 * bar_width, 300), (255, 255, 255), 1)
            cv2.line(hud, (43 * bar_width, 50), (43 * bar_width, 300), (255, 255, 255), 1)
            cv2.putText(hud, f"L:{dist_l}mm", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(hud, f"C:{dist_c}mm", (21*bar_width + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(hud, f"R:{dist_r}mm", (43*bar_width + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            
            cv2.rectangle(hud, (0, hud_h-60), (hud_w, hud_h), (30,30,30), -1)
            if state_text != "CRUISING": cv2.rectangle(hud, (0,0), (hud_w-1, hud_h-1), color, 4) 
            fps = int(1.0 / (time.time() - start_time))
            cv2.putText(hud, f"ACT: {state_text}", (10, hud_h - 20), cv2.FONT_HERSHEY_DUPLEX, 0.8, color, 2)
            cv2.putText(hud, f"FPS: {fps}", (hud_w - 120, hud_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)

            cv2.imshow("1D Pseudo-Lidar HUD", hud)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

    except KeyboardInterrupt: pass
    finally:
        lidar_receiver.running = False
        robot.send_action({"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0})
        robot.disconnect()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()