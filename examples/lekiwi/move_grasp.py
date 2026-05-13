#!/usr/bin/env python
import os
import time
import argparse
import numpy as np
import cv2
import torch
from ultralytics import YOLO

# LeRobot 核心导入
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
from lerobot.utils.utils import log_say

# 网络代理设置
os.environ["http_proxy"] = "http://127.0.0.1:7897" 
os.environ["https_proxy"] = "http://127.0.0.1:7897"

def parse_args():
    """解析官方标准参数与自定义视觉参数"""
    parser = argparse.ArgumentParser(description="Lekiwi Guided Teleoperation Record Script")
    
    # 官方标准参数
    parser.add_argument("--control.repo_id", dest="repo_id", type=str, required=True, help="HuggingFace dataset repo id")
    parser.add_argument("--control.num_episodes", dest="num_episodes", type=int, default=30)
    parser.add_argument("--control.episode_time_s", dest="episode_time_s", type=int, default=12)
    parser.add_argument("--control.reset_time_s", dest="reset_time_s", type=int, default=5)
    parser.add_argument("--control.single_task", dest="single_task", type=str, default="mobile grasp bottle")
    parser.add_argument("--control.fps", dest="fps", type=int, default=15)
    parser.add_argument("--control.tags", dest="tags", type=str, default='["grasping", "mobile"]')
    
    # 我们自定义的视觉参数 (完美适配瓶子)
# 我们自定义的视觉参数 (适配 YOLO-World 开放词汇与底边测距)
    parser.add_argument("--target_name", type=str, default="tissue box", help="你想抓什么？直接输入英文单词 (如 bottle, cup, tissue box)")
    parser.add_argument("--stop_y_ratio", type=float, default=0.85, help="停车警戒线比例 (0.85 表示屏幕最下方 15% 处停车)")
    parser.add_argument("--conf", type=float, default=0.10, help="YOLO 置信度 (World 模型对新物体的置信度偏低，建议设为 0.05-0.15 之间)")
    
    return parser.parse_args()

def to_float(value):
    if hasattr(value, "item"): return float(value.item())
    if isinstance(value, (list, np.ndarray, torch.Tensor)):
        return float(value[0]) if len(value) > 0 else 0.0
    return float(value)

def auto_align_phase(robot, model, args, events):
    """自动寻的阶段：螺旋搜寻 + 滤波视觉伺服 + 底边测距停车"""
    Kp_turn = 0.4      # 略微调大转向响应
    Kp_forward = 0.6   # 调大前进响应，因为用 y_max 算出来的误差比面积小
    last_seen_direction = 1.0  
    
    # 🌟 引入 EMA 滤波器状态变量
    smooth_ymax = 0
    smooth_center_x = 0
    alpha = 0.3 # 滤波系数：越小越平滑，越大越灵敏（0.3 是个好甜点）
    
    print(f"[SEARCH] 正在搜寻目标 (使用 YOLO-World)...")
    
    while not events.get("stop_recording", False):
        obs = robot.get_observation()
        img_front = obs.get('front')
        if img_front is None: continue
        
        if hasattr(img_front, 'cpu'):
            img_front = img_front.permute(1, 2, 0).cpu().numpy()
            
        img_bgr = cv2.cvtColor((img_front * 255).astype(np.uint8) if img_front.max() <= 1.0 else img_front, cv2.COLOR_RGB2BGR)
        img_h, img_w = img_bgr.shape[:2]
        
        # ⚠️ 注意：YOLO-World 只有我们设置的那一个类，不需要再传 classes 参数了
        results = model(img_bgr, conf=args.conf, verbose=False)
        boxes = results[0].boxes
        
        action = {k: to_float(obs.get(k, 0.0)) for k in robot._state_order}
        action.update({"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0})

        if len(boxes) == 0:
            # 盲回头机制
            action["y.vel"] = 0.0  
            action["theta.vel"] = 1.0 * last_seen_direction 
            status_text = "SEARCHING: Spinning..."
            color = (0, 165, 255)
        else:
            # 获取最大框
            box = boxes[0].xyxy[0].cpu().numpy()
            raw_center_x = (box[0] + box[2]) / 2
            raw_ymax = box[3] # 🌟 核心：只看物体的最下边缘 (接触地面的地方)
            
            # 🌟 滤波：消除 YOLO 闪烁导致的抽搐
            if smooth_ymax == 0: 
                smooth_ymax = raw_ymax
                smooth_center_x = raw_center_x
            else:
                smooth_ymax = alpha * raw_ymax + (1 - alpha) * smooth_ymax
                smooth_center_x = alpha * raw_center_x + (1 - alpha) * smooth_center_x
            
            # 计算误差
            error_x = (img_w / 2 - smooth_center_x) / img_w
            last_seen_direction = 1.0 if error_x > 0 else -1.0
            
            # 🌟 停车核心逻辑：让物体的底边，达到屏幕高度的 85% 处 (可根据实际抓取距离微调)
            target_ymax = img_h * args.stop_y_ratio 
            error_dist = (target_ymax - smooth_ymax) / img_h
            
            # 🌟 死区控制 (Deadband)：到达容忍范围内，强行让速度归零，防止原地哆嗦
            vel_y = error_dist * Kp_forward
            vel_theta = error_x * Kp_turn * 10
            
            if abs(error_x) < 0.08: vel_theta = 0.0 # 角度差不多就行了
            if abs(error_dist) < 0.05: vel_y = 0.0  # 距离差不多就行了
            
            # 判断是否彻底停稳满足抓取条件
            if vel_y == 0.0 and vel_theta == 0.0:
                action.update({"y.vel": 0.0, "theta.vel": 0.0})
                robot.send_action({k: to_float(v) for k, v in action.items()})
                print("[SUCCESS] 精准停靠！底边达标！准备录制...")
                return True
            
            action["theta.vel"] = vel_theta
            action["y.vel"] = vel_y
            
            status_text = f"ALIGN: dY {error_dist:.2f}, dX {error_x:.2f}"
            color = (0, 255, 0)
            # 渲染真实的框和滤波后的中心点
            cv2.rectangle(img_bgr, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
            cv2.circle(img_bgr, (int(smooth_center_x), int(smooth_ymax)), 5, (0, 0, 255), -1)
            # 画一条停车警戒线
            cv2.line(img_bgr, (0, int(target_ymax)), (img_w, int(target_ymax)), (255, 0, 0), 2)

        robot.send_action({k: to_float(v) for k, v in action.items()})
        cv2.putText(img_bgr, status_text, (20, 50), 2, 0.7, color, 2)
        cv2.imshow("Lekiwi Auto-Navigation View", img_bgr)
        
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            return False

def main():
    args = parse_args()
    
    print(f"\n[SYSTEM] 初始化录制流水线...")
    print(f"  - 仓库 ID: {args.repo_id}")
    print(f"  - 计划组数: {args.num_episodes} (时长: {args.episode_time_s}s/组)")

    model = YOLO('yolov8s-world.pt') 
    model.set_classes([args.target_name])
    robot = LeKiwiClient(LeKiwiClientConfig(remote_ip="192.168.3.215", id="lekiwi"))
    leader_arm = SO100Leader(SO100LeaderConfig(port="/dev/ttyACM0", id="leader"))
    keyboard = KeyboardTeleop(KeyboardTeleopConfig())
    
    tele_proc, rob_act_proc, rob_obs_proc = make_default_processors()
    
    robot.connect()
    action_features = hw_to_dataset_features(robot.action_features, ACTION)
    obs_features = hw_to_dataset_features(robot.observation_features, OBS_STR)
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id, fps=args.fps, features={**action_features, **obs_features}, 
        robot_type=robot.name, use_videos=True
    )

    leader_arm.connect()
    keyboard.connect()
    listener, events = init_keyboard_listener()

    print("\n[SYSTEM] 硬件就绪，按回车键开始全自动流程。")
    recorded_count = 0

    try:
        while recorded_count < args.num_episodes and not events.get("stop_recording", False):
            # --- STAGE 1: 自动寻的 ---
            success = auto_align_phase(robot, model, args, events)
            if not success: break
            
            # --- STAGE 2: 触发录制 ---
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            
            print("\n" + "="*40)
            log_say(f"Ready! Starting Episode {recorded_count + 1}. START GRASPING NOW!")
            print(f">>> 正在录制 ({args.episode_time_s}秒) ... 请抓取瓶子。")
            print("="*40 + "\n")
            
            record_loop(
                robot=robot, events=events, fps=args.fps, dataset=dataset,
                teleop=[leader_arm, keyboard], control_time_s=args.episode_time_s,
                single_task=args.single_task, teleop_action_processor=tele_proc,
                robot_action_processor=rob_act_proc, robot_observation_processor=rob_obs_proc,
            )
            
            dataset.save_episode()
            recorded_count += 1
            log_say(f"Episode {recorded_count} saved.")
            
            # --- STAGE 3: 丢件与倒车复位 (全关节平滑收回) ---
            if recorded_count < args.num_episodes:
                print(f"\n[INFO] 正在释放瓶子并平滑收回机械臂...")
                
                # 1. 获取刚刚抓完时的真实姿态，作为平滑过渡的起点
                current_obs = robot.get_observation()
                start_action = {k: to_float(current_obs.get(k, 0.0)) for k in robot._state_order}
                
                # 2. 定义 SOARM 六轴专属的“天鹅颈”安全待机姿态
                home_pose = start_action.copy()
                
                if "arm_shoulder_pan.pos" in home_pose: home_pose["arm_shoulder_pan.pos"] = -2.5    # 底座居中
                if "arm_shoulder_lift.pos" in home_pose: home_pose["arm_shoulder_lift.pos"] = -95.2  # 大臂绝对直立 (不再后仰)
                if "arm_elbow.pos" in home_pose: home_pose["arm_elbow.pos"] = 96.6         # 小臂向前水平弯曲
                if "arm_wrist_flex.pos" in home_pose: home_pose["arm_wrist_flex.pos"] = 71.8 # 手腕垂直指向地面
                if "arm_wrist_roll.pos" in home_pose: home_pose["arm_wrist_roll.pos"] = 1.8  # 夹爪保持水平不转
                if "arm_gripper.pos" in home_pose: home_pose["arm_gripper.pos"] = 100.0    # 彻底松开夹爪
                
                # 确保在收回手臂的这段时间里，底盘绝对静止
                home_pose.update({"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0})

                # 3. 平滑收回动作 (用 1.5 秒的时间，将机械臂从当前位置丝滑过渡到待机位置)
                transition_frames = int(args.fps * 1.5)
                for step in range(transition_frames):
                    alpha = (step + 1) / transition_frames # 进度条，从 0 慢慢涨到 1
                    interp_action = {}
                    
                    for k in start_action.keys():
                        if k in ["x.vel", "y.vel", "theta.vel"]:
                            interp_action[k] = 0.0
                        else:
                            # 核心公式：当前位置 = 起点 + 进度 * (终点 - 起点)
                            interp_action[k] = start_action[k] + alpha * (home_pose[k] - start_action[k])
                    
                    robot.send_action(interp_action)
                    time.sleep(1.0 / args.fps)

                print(f"[INFO] 手臂已收妥，开始倒车脱离...")

                # 4. 机械臂彻底收好后，底盘再开始倒车 (持续 1.5 秒)
                home_pose["y.vel"] = -0.15
                for _ in range(int(args.fps * 1.5)): 
                    robot.send_action(home_pose)
                    time.sleep(1.0 / args.fps)
                
                # 5. 彻底停稳
                home_pose["y.vel"] = 0.0
                robot.send_action(home_pose)
                
                print(f"[INFO] 已停稳。请在 {args.reset_time_s} 秒内把瓶子踢到新位置！")
                time.sleep(args.reset_time_s)

    except KeyboardInterrupt:
        print("\n[INFO] 收到键盘中断信号，停止录制。")
    finally:
        print("[INFO] 正在关闭连接并推送数据集...")
        robot.disconnect()
        leader_arm.disconnect()
        listener.stop()
        dataset.finalize()
        dataset.push_to_hub()
        cv2.destroyAllWindows()
        print("[SUCCESS] 任务彻底完成！")

if __name__ == "__main__":
    main()