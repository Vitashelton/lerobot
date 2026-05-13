import cv2
import numpy as np
import torch
import requests
import base64
import time
import threading

from lerobot.robots.lekiwi.config_lekiwi import LeKiwiClientConfig
from lerobot.robots.lekiwi.lekiwi_client import LeKiwiClient

# ========== 核心配置项 ==========
ROBOT_IP = "192.168.3.215"     
SERVER_URL = "http://10.33.5.238:8000/get_target" 
# ==============================

# 全局变量，用于线程间通信
global_target_x = None
global_target_y = None
global_target_time = 0.0  # 新增：记录坐标诞生的时间
global_latest_img = None

def to_float(value):
    if hasattr(value, "item"): return float(value.item())
    if isinstance(value, (list, np.ndarray, torch.Tensor)):
        return float(value[0]) if len(value) > 0 else 0.0
    return float(value)

def vlm_brain_thread():
    """这是异步的大脑线程：专门负责慢速思考，不卡顿底盘"""
    global global_target_x, global_target_y, global_latest_img
    
    print("[大脑] VLM 推理线程已启动...")
    while True:
        if global_latest_img is not None:
            # 拿到最新的一帧画面
            img_bgr = global_latest_img.copy()
            
            # 压缩图像
            small_img = cv2.resize(img_bgr, (320, 240))
            _, buffer = cv2.imencode('.jpg', small_img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            img_b64 = base64.b64encode(buffer).decode('utf-8')
            
            payload = {"image_b64": img_b64}
            
            try:
                start_t = time.time()
                # 🔥 放宽超时时间到 15 秒，给服务器显存加载和推理留足时间
                response = requests.post(SERVER_URL, json=payload, timeout=15.0, proxies={"http": None, "https": None})
                
                if response.status_code == 200:
                    result = response.json()
                    if result.get("status") == "success":
                        x = result.get("x")
                        y = result.get("y")
                        
                        if x != -1 and y != -1:
                            orig_h, orig_w = img_bgr.shape[:2]
                            global_target_x = int(x * (orig_w / 320.0))
                            global_target_y = int(y * (orig_h / 240.0))
                            global_target_time = time.time() # 🌟 盖上最新的时间戳
                            print(f"[大脑] 发现目标! 坐标: ({global_target_x}, {global_target_y})")
                        else:
                            global_target_x, global_target_y = None, None
                            print(f"[大脑] 视野中无纸箱。耗时: {time.time() - start_t:.2f}s")
            except Exception as e:
                print(f"[大脑] 请求失败或超时 (正常现象，等待下次重试): {e}")
                
        time.sleep(0.1) # 休息一下，防止把服务器 CPU 榨干

def main():
    global global_target_x, global_target_y, global_latest_img
    
    print("[小脑] 正在连接底盘...")
    robot = LeKiwiClient(LeKiwiClientConfig(remote_ip=ROBOT_IP, id="lekiwi"))
    robot.connect()
    print("[小脑] 底盘就绪！")

    # 启动异步大脑线程
    brain = threading.Thread(target=vlm_brain_thread, daemon=True)
    brain.start()

    try:
        while True:
            # 1. 高频获取底层状态与图像
            obs = robot.get_observation()
            img_front = obs.get('front')
            
            if img_front is None: 
                continue
                
            if hasattr(img_front, 'cpu'):
                img_front = img_front.permute(1, 2, 0).cpu().numpy()
            img_bgr = cv2.cvtColor((img_front * 255).astype(np.uint8) if img_front.max() <= 1.0 else img_front, cv2.COLOR_RGB2BGR)
            img_h, img_w = img_bgr.shape[:2]
            
            # 把最新画面交给大脑线程
            global_latest_img = img_bgr.copy()

            # 2. 构造全身 Action 数据包
            action = {k: to_float(obs.get(k, 0.0)) for k in robot._state_order}
            
            # 3. 小脑运动控制逻辑 (极速运行，防止过度转向)
            current_time = time.time()
            
            # 如果存在目标，并且这个目标的坐标是最近 0.8 秒内算出来的
            if global_target_x is not None and (current_time - global_target_time < 0.8):
                error_x = (img_w / 2 - global_target_x) / img_w
                
                # VLM 推理有延迟，降低 P 控制器的响应系数，避免车身抽搐
                v = 0.12                
                w = error_x * 0.4  # 转向力度调柔和
                
                cv2.circle(img_bgr, (global_target_x, global_target_y), 15, (0, 0, 255), -1)
                cv2.putText(img_bgr, "TRACKING", (20, 50), 2, 1.0, (0, 255, 0), 2)
                
            # 如果坐标已经“过期”（大模型卡住了），或者压根没找到
            else:
                if global_target_x is not None:
                    # 坐标过期：大模型还在算下一帧，此时保持非常微弱的直行，【绝对不要继续转向】！
                    v = 0.05
                    w = 0.0
                    cv2.putText(img_bgr, "WAITING VLM...", (20, 50), 2, 1.0, (0, 165, 255), 2)
                else:
                    # 压根没找到目标：原地缓慢扫视
                    v = 0.0
                    w = 0.3  
                    cv2.putText(img_bgr, "SEARCHING...", (20, 50), 2, 1.0, (0, 255, 255), 2)

            # 更新速度并下发
            action.update({"x.vel": 0.0, "y.vel": v, "theta.vel": w})
            robot.send_action({k: to_float(val) for k, val in action.items()})

            # 显示监控画面
            cv2.imshow("VLM End-to-End Navigation", img_bgr)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\n[系统] 收到中断信号，正在停车...")
    finally:
        obs = robot.get_observation()
        action = {k: to_float(obs.get(k, 0.0)) for k in robot._state_order}
        action.update({"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0})
        robot.send_action(action)
        robot.disconnect()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()