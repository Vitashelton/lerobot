# !/usr/bin/env python

import cv2
import time
import numpy as np
import math

from lerobot.robots.lekiwi.config_lekiwi import LeKiwiClientConfig
from lerobot.robots.lekiwi.lekiwi_client import LeKiwiClient

def main():
    print("🚀 [INFO] 正在连接 Lekiwi 底盘 (人工势场法 APF)...")
    robot_config = LeKiwiClientConfig(remote_ip="192.168.3.215", id="lekiwi")
    if "wrist" in robot_config.cameras:
        del robot_config.cameras["wrist"]
        
    robot = LeKiwiClient(robot_config)
    robot.connect()
    
    if not robot.is_connected:
        print("❌ [ERROR] 树莓派连接失败！")
        return

    print("✅ [INFO] 连接成功！APF 势场引擎已启动...")
    
    # ================== APF 核心调参区 ==================
    V_FORWARD = 0.25        # 基础前进引力 (向前的默认速度 m/s)
    K_REP = 8.0             # 斥力系数 (越大躲得越猛)
    INFLUENCE_RADIUS = 700  # 影响半径 (单位 mm，70厘米内的事物才产生斥力)
    MIN_SAFE_DIST = 200     # 绝对死区 (单位 mm，小于20厘米直接触发急停倒车)
    CAMERA_FOV_DEG = 87     # RealSense D435 水平视角大概 87 度
    # ====================================================

    try:
        while True:
            start_time = time.time()
            obs = robot.get_observation()
            
            # 1. 抓取深度图
            depth_key = next((k for k in obs.keys() if 'depth' in k.lower()), None)
            if not depth_key:
                time.sleep(0.1)
                continue
                
            depth_data = obs[depth_key]
            depth_np = depth_data.squeeze().cpu().numpy() if hasattr(depth_data, 'cpu') else np.squeeze(depth_data)
            
            if depth_np.dtype in [np.float32, np.float64]:
                depth_np = (depth_np * 1000).astype(np.uint16) if depth_np.max() < 100 else depth_np.astype(np.uint16)

            h, w = depth_np.shape
            
            # 2. 降维打击：从 3D 深度图提取 2D 激光雷达扫描线
            # 提取画面正中间偏下的一条宽带 (避开地面和太高的物体)
            scan_band = depth_np[h//2 - 30 : h//2 + 30, :]
            
            # 把这条宽带沿垂直方向取最小值（把障碍物压扁成一条线）
            # 为了防止噪点，先过滤掉 0，然后取 10% 分位数代替简单的 min
            scan_band[scan_band == 0] = 9999
            lidar_scan = np.percentile(scan_band, 10, axis=0)

            # 3. 人工势场计算 (Artificial Potential Field)
            F_x = V_FORWARD  # 引力：始终渴望向前
            F_y = 0.0        # 初始横向力为 0
            
            emergency_stop = False
            
            # 遍历这根扫描线上的每一个点 (相当于遍历雷达的每一根线)
            for col in range(w):
                dist = lidar_scan[col]
                
                if dist < MIN_SAFE_DIST:
                    emergency_stop = True
                    break
                    
                if dist < INFLUENCE_RADIUS:
                    # 将像素列转换为物理角度 (-43.5度 到 +43.5度)
                    angle_deg = (col / w - 0.5) * CAMERA_FOV_DEG
                    angle_rad = math.radians(angle_deg)
                    
                    # 计算斥力大小：距离越近，斥力呈指数级增大
                    # 公式：F = K * (1/d - 1/R) / d^2
                    rep_magnitude = K_REP * ((1.0 / dist) - (1.0 / INFLUENCE_RADIUS)) / (dist * dist / 100000.0) 
                    
                    # 将斥力分解为 X (前后) 和 Y (左右) 向量
                    # 注意：如果障碍物在正前方(angle=0)，向后的推力最大。
                    # 如果障碍物在左侧(angle<0)，产生向右的推力(Y为负)。
                    F_x -= rep_magnitude * math.cos(angle_rad)
                    F_y -= rep_magnitude * math.sin(angle_rad)

            # 4. 速度限制与指令下发
            action_dict = {
                "arm_shoulder_pan.pos": 0.0, "arm_shoulder_lift.pos": 0.0,
                "arm_elbow_flex.pos": 0.0, "arm_wrist_flex.pos": 0.0,
                "arm_wrist_roll.pos": 0.0, "arm_gripper.pos": 0.0,
                "x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0
            }

            if emergency_stop:
                action_dict["x.vel"] = -0.15  # 极其危险，倒车退让
                state_txt = "EMERGENCY REVERSE!"
                color = (0, 0, 255)
            else:
                # 速度截断，防止算出的合力太大导致车飞出去
                v_x_cmd = max(-0.2, min(0.3, F_x))
                v_y_cmd = max(-0.3, min(0.3, F_y))
                
                action_dict["x.vel"] = v_x_cmd
                action_dict["y.vel"] = v_y_cmd
                # 纯平移避障，不旋转车头，展示全向轮特长！
                action_dict["theta.vel"] = 0.0 
                
                state_txt = f"APF -> Vx:{v_x_cmd:.2f} Vy:{v_y_cmd:.2f}"
                color = (0, 255, 0) if v_x_cmd > 0.1 else (0, 165, 255)

            robot.send_action(action_dict)

            # ================== HUD 可视化 ==================
            vis_depth = np.clip(depth_np, 0, INFLUENCE_RADIUS + 300)
            vis_depth = (vis_depth / float(INFLUENCE_RADIUS + 300) * 255).astype(np.uint8)
            vis_depth = cv2.bitwise_not(vis_depth)
            depth_colormap = cv2.applyColorMap(vis_depth, cv2.COLORMAP_JET)
            
            # 画出扫描带
            cv2.rectangle(depth_colormap, (0, h//2 - 30), (w, h//2 + 30), (255,255,255), 2)
            
            # 绘制合力指针仪
            cx, cy = w // 2, h - 80
            cv2.circle(depth_colormap, (cx, cy), 50, (0,0,0), -1)
            cv2.circle(depth_colormap, (cx, cy), 50, (255,255,255), 2)
            # 根据算出的速度画一根线，直观展示车的移动趋势
            pointer_x = int(cx - action_dict["y.vel"] * 150)
            pointer_y = int(cy - action_dict["x.vel"] * 150)
            cv2.arrowedLine(depth_colormap, (cx, cy), (pointer_x, pointer_y), color, 4, tipLength=0.3)

            cv2.putText(depth_colormap, state_txt, (10, 30), cv2.FONT_HERSHEY_DUPLEX, 0.8, color, 2)
            
            cv2.imshow("APF Navigation", depth_colormap)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\n[INFO] 退出...")
    finally:
        robot.send_action({"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0})
        robot.disconnect()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()