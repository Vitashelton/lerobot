#!/usr/bin/env python
import cv2
import time
import numpy as np
from ultralytics import YOLO

# 导入你的 Lekiwi 专属库
from lerobot.robots.lekiwi.config_lekiwi import LeKiwiClientConfig
from lerobot.robots.lekiwi.lekiwi_client import LeKiwiClient

def main():
    # 1. 加载预训练的 YOLO 模型
    print("[INFO] 正在加载 YOLOv8n 模型...")
    model = YOLO('yolov8n.pt')

    # 2. 连接你的 Lekiwi 树莓派
    print("[INFO] 正在连接 Lekiwi 底盘 (192.168.3.215)...")
    robot_config = LeKiwiClientConfig(remote_ip="192.168.3.215", id="lekiwi")
    robot = LeKiwiClient(robot_config)
    robot.connect()
    
    if not robot.is_connected:
        print("[ERROR] 树莓派连接失败，请检查网络！")
        return

    print("[INFO] 连接成功！开始视觉巡航与避障...")
    
    try:
        while True:
            # ================== 【看】获取图像 ==================
            obs = robot.get_observation()
            
            # 【关键修复】：明确指定使用前置摄像头 'front'
            if 'front' not in obs:
                print(f"[警告] 没找到前置摄像头(front)数据！当前收到: {list(obs.keys())}")
                time.sleep(0.5)
                continue
                
            # 处理 LeRobot 传来的图像数据
            img_data = obs['front']
            if hasattr(img_data, 'cpu'):  # 如果是 PyTorch Tensor [C, H, W]
                img_np = img_data.permute(1, 2, 0).cpu().numpy()
                if img_np.dtype == np.float32 and img_np.max() <= 1.0:
                    img_np = (img_np * 255).astype(np.uint8)
            else:  # 如果已经是 NumPy array
                img_np = img_data
                
            # 转为 OpenCV 支持的 BGR 格式
            if len(img_np.shape) == 3 and img_np.shape[2] == 3:
                img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            else:
                img_bgr = img_np
            
            # ================== 【想】YOLO 推理 ==================
            # classes=[39] 代表只检测瓶子 (Bottle)
            results = model(img_bgr, classes=[39], verbose=False) 
            annotated_frame = results[0].plot()
            
            # ================== 【动】避障逻辑 ==================
            # 初始化速度字典 (全向轮：x为前后，y为左右，theta为旋转)
            action_dict = {
                "arm_shoulder_pan.pos": obs.get("arm_shoulder_pan.pos", 0.0),
                "arm_shoulder_lift.pos": obs.get("arm_shoulder_lift.pos", 0.0),
                "arm_elbow_flex.pos":  obs.get("arm_elbow_flex.pos", 0.0),
                "arm_wrist_flex.pos":  obs.get("arm_wrist_flex.pos", 0.0),
                "arm_wrist_roll.pos":  obs.get("arm_wrist_roll.pos", 0.0),
                "arm_gripper.pos":     obs.get("arm_gripper.pos", 0.0),
                "x.vel": 0.0,
                "y.vel": 0.0,
                "theta.vel": 0.0
            }
            
            boxes = results[0].boxes
            if len(boxes) == 0:
                # 没看到瓶子：安全巡航，慢慢前进
                cv2.putText(annotated_frame, "Status: Cruising", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                action_dict["x.vel"] = 0.2
            else:
                # 看到瓶子了！计算瓶子在画面里的位置和大小
                box = boxes[0].xyxy[0] 
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                
                box_area = (x2 - x1) * (y2 - y1)
                img_area = img_bgr.shape[0] * img_bgr.shape[1]
                box_center_x = (x1 + x2) / 2
                img_center_x = img_bgr.shape[1] / 2
                
                # 如果瓶子面积占屏幕比例超过 5% (说明离得很近了，触发避障)
                if box_area / img_area > 0.05:
                    cv2.putText(annotated_frame, "Status: EVASION!", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
                    # 判断瓶子在左边还是右边，执行反向平移
                    if box_center_x < img_center_x:
                        action_dict["y.vel"] = -0.3 # 瓶子在左，向右平移
                    else:
                        action_dict["y.vel"] = 0.3  # 瓶子在右，向左平移
                else:
                    # 瓶子还很远，继续慢慢走锁定目标
                    cv2.putText(annotated_frame, "Status: Target Locked (Far)", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    action_dict["x.vel"] = 0.1
            
            # 下发动作指令字典！
            robot.send_action(action_dict)

            # ================== 画面显示 ==================
            cv2.imshow("Lekiwi Deep Learning Vision", annotated_frame)
            
            # 按 'q' 键退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\n[INFO] 收到退出信号 (Ctrl+C)...")
    except Exception as e:
        print(f"\n[ERROR] 运行中发生错误: {e}")
    finally:
        # 安全退出，停止电机
        print("[INFO] 正在关闭连接并急停...")
        # 兼容 Client 中没有 stop_base 方法的情况
        if hasattr(robot, 'stop_base'):
            robot.stop_base()
        else:
            robot.send_action({"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0})
        
        robot.disconnect()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()