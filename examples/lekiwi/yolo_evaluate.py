#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.

import os
import time
import argparse
import json
import torch
import numpy as np
import cv2
from ultralytics import YOLO

# LeRobot 核心导入
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.processor import make_default_processors
from lerobot.robots.lekiwi import LeKiwiClient, LeKiwiClientConfig
from lerobot.scripts.lerobot_record import record_loop
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun

# 代理设置（用于下载 HF 模型或 YOLO，按需保留）
os.environ["http_proxy"] = "http://127.0.0.1:7897" 
os.environ["https_proxy"] = "http://127.0.0.1:7897"

def str2bool(v):
    if isinstance(v, bool): return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'): return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'): return False
    else: raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
    parser = argparse.ArgumentParser(description="Auto Nav + ACT Grasp Evaluation")
    
    # 支持你要求的标准命令行参数
    parser.add_argument("--robot.type", dest="robot_type", type=str, default="lekiwi")
    parser.add_argument("--control.type", dest="control_type", type=str, default="record")
    parser.add_argument("--control.fps", dest="fps", type=int, default=15)
    parser.add_argument("--control.single_task", dest="single_task", type=str, required=True)
    parser.add_argument("--control.repo_id", dest="repo_id", type=str, required=True)
    parser.add_argument("--control.tags", dest="tags", type=str, default='["eval"]')
    
    parser.add_argument("--control.warmup_time_s", dest="warmup_time_s", type=int, default=5)
    parser.add_argument("--control.episode_time_s", dest="episode_time_s", type=int, default=15)
    parser.add_argument("--control.reset_time_s", dest="reset_time_s", type=int, default=5)
    parser.add_argument("--control.num_episodes", dest="num_episodes", type=int, default=10)
    parser.add_argument("--control.push_to_hub", dest="push_to_hub", type=str2bool, default=True)
    parser.add_argument("--control.policy.path", dest="policy_path", type=str, required=True)
    
    # 视觉导航参数
    parser.add_argument("--target_class", type=int, default=39, help="YOLO class id (39=bottle)")
    parser.add_argument("--area_ratio", type=float, default=0.25, help="Target area ratio to stop")
    parser.add_argument("--conf", type=float, default=0.40, help="YOLO confidence")
    
    return parser.parse_args()

def to_float(value):
    if hasattr(value, "item"): return float(value.item())
    if isinstance(value, (list, np.ndarray, torch.Tensor)):
        return float(value[0]) if len(value) > 0 else 0.0
    return float(value)

def auto_align_phase(robot, yolo_model, args, events):
    """阶段一：YOLO 自动寻的与贴脸"""
    Kp_turn = 0.15     
    Kp_forward = 0.25  
    last_seen_direction = 1.0  
    
    log_say(f"Phase 1: Auto Navigation to target {args.target_class}...")
    
    while not events.get("stop_recording", False):
        obs = robot.get_observation()
        img_front = obs.get('front')
        if img_front is None: continue
        
        if hasattr(img_front, 'cpu'):
            img_front = img_front.permute(1, 2, 0).cpu().numpy()
            
        img_bgr = cv2.cvtColor((img_front * 255).astype(np.uint8) if img_front.max() <= 1.0 else img_front, cv2.COLOR_RGB2BGR)
        img_h, img_w = img_bgr.shape[:2]
        
        results = yolo_model(img_bgr, classes=[args.target_class], conf=args.conf, verbose=False)
        boxes = results[0].boxes
        
        action = {k: to_float(obs.get(k, 0.0)) for k in robot._state_order}
        action.update({"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0})

        if len(boxes) == 0:
            action["y.vel"] = 0.0  
            action["theta.vel"] = 25.0 * last_seen_direction 
            status_text = "SEARCHING..."
        else:
            box = boxes[0].xyxy[0]
            box_center_x = (box[0] + box[2]) / 2
            current_ratio = ((box[2]-box[0])*(box[3]-box[1])) / (img_h * img_w)
            
            error_x = (img_w / 2 - box_center_x) / img_w
            error_dist = args.area_ratio - current_ratio
            last_seen_direction = 1.0 if error_x > 0 else -1.0
            
            condition_a = abs(error_x) < 0.10 and abs(error_dist) < 0.05
            condition_b = abs(error_x) < 0.15 and box[3] > img_h * 0.95 
            
            if condition_a or condition_b:
                action.update({"y.vel": 0.0, "theta.vel": 0.0})
                robot.send_action({k: to_float(v) for k, v in action.items()})
                log_say("Target locked! Ready for grasp.")
                return True
            
            action["theta.vel"] = error_x * Kp_turn * 100
            action["y.vel"] = error_dist * Kp_forward * 1.5
            status_text = "LOCK ON"

        robot.send_action({k: to_float(v) for k, v in action.items()})
        cv2.putText(img_bgr, status_text, (20, 50), 2, 0.7, (0, 255, 0), 2)
        cv2.imshow("Auto-Navigation", img_bgr)
        cv2.waitKey(1)
        
    return False

def auto_retreat_phase(robot, args):
    """阶段三：自动丢件与倒车 (使用你的完美姿势)"""
    log_say("Phase 3: Retreat and reset...")
    current_obs = robot.get_observation()
    start_action = {k: to_float(current_obs.get(k, 0.0)) for k in robot._state_order}
    
    # 你的完美复位姿势
    home_pose = start_action.copy()
    if "arm_shoulder_pan.pos" in home_pose: home_pose["arm_shoulder_pan.pos"] = -2.5
    if "arm_shoulder_lift.pos" in home_pose: home_pose["arm_shoulder_lift.pos"] = -95.2
    if "arm_elbow_flex.pos" in home_pose: home_pose["arm_elbow_flex.pos"] = 96.6 
    elif "arm_elbow.pos" in home_pose: home_pose["arm_elbow.pos"] = 96.6 
    if "arm_wrist_flex.pos" in home_pose: home_pose["arm_wrist_flex.pos"] = 71.8
    if "arm_wrist_roll.pos" in home_pose: home_pose["arm_wrist_roll.pos"] = 1.8
    if "arm_gripper.pos" in home_pose: home_pose["arm_gripper.pos"] = 2.4
    home_pose.update({"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0})

    # 平滑插值收回手臂 (1.5秒)
    transition_frames = int(args.fps * 1.5)
    for step in range(transition_frames):
        alpha = (step + 1) / transition_frames
        interp_action = {}
        for k in start_action.keys():
            if k in ["x.vel", "y.vel", "theta.vel"]:
                interp_action[k] = 0.0
            else:
                interp_action[k] = start_action[k] + alpha * (home_pose[k] - start_action[k])
        robot.send_action(interp_action)
        time.sleep(1.0 / args.fps)

    # 倒车 (1.5秒)
    home_pose["y.vel"] = -0.15
    for _ in range(int(args.fps * 1.5)): 
        robot.send_action(home_pose)
        time.sleep(1.0 / args.fps)
    
    # 彻底停稳
    home_pose["y.vel"] = 0.0
    robot.send_action(home_pose)
    print(f"[等待] 已退回，你有 {args.reset_time_s} 秒时间重新摆放瓶子...")
    time.sleep(args.reset_time_s)


def main():
    args = parse_args()

    # 1. 基础配置
    robot_config = LeKiwiClientConfig(remote_ip="192.168.3.215", id="lekiwi")
    robot = LeKiwiClient(robot_config)
    yolo_model = YOLO('yolov8n.pt')

    # 2. 加载策略与数据集
    print(f"Loading ACT Policy from: {args.policy_path}")
    policy = ACTPolicy.from_pretrained(args.policy_path)

    action_features = hw_to_dataset_features(robot.action_features, ACTION)
    obs_features = hw_to_dataset_features(robot.observation_features, OBS_STR)
    dataset_features = {**action_features, **obs_features}

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        features=dataset_features,
        robot_type=robot.name,
        use_videos=True,
        image_writer_threads=4,
    )

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy,
        pretrained_path=args.policy_path,
        dataset_stats=dataset.meta.stats,
        preprocessor_overrides={"device_processor": {"device": str(policy.config.device)}},
    )

    robot.connect()
    teleop_act_proc, robot_act_proc, robot_obs_proc = make_default_processors()

    listener, events = init_keyboard_listener()
    init_rerun(session_name="lekiwi_evaluate")

    try:
        if not robot.is_connected:
            raise ValueError("Robot is not connected!")

        print("Starting automated evaluate loop...")
        recorded_episodes = 0
        
        while recorded_episodes < args.num_episodes and not events["stop_recording"]:
            print(f"\n--- Episode {recorded_episodes + 1}/{args.num_episodes} ---")
            
            # --- Phase 1: YOLO 导航贴脸 ---
            success = auto_align_phase(robot, yolo_model, args, events)
            if not success: break
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            
            # --- Phase 2: ACT 模型接管抓取 (同时被官方框架录制) ---
            log_say(f"Phase 2: ACT Policy inference starting...")
            record_loop(
                robot=robot,
                events=events,
                fps=args.fps,
                policy=policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                dataset=dataset,
                control_time_s=args.episode_time_s,
                single_task=args.single_task,
                display_data=True,
                teleop_action_processor=teleop_act_proc,
                robot_action_processor=robot_act_proc,
                robot_observation_processor=robot_obs_proc,
            )

            # --- 异常处理 (人工中断重录) ---
            if events.get("rerecord_episode"):
                log_say("Re-record episode requested")
                events["rerecord_episode"] = False
                events["exit_early"] = False
                dataset.clear_episode_buffer()
                continue

            # 保存当前抓取数据
            dataset.save_episode()
            recorded_episodes += 1
            
            # --- Phase 3: 自动丢件与复位 ---
            if recorded_episodes < args.num_episodes and not events["stop_recording"]:
                auto_retreat_phase(robot, args)

    finally:
        log_say("Cleaning up...")
        robot.disconnect()
        listener.stop()
        cv2.destroyAllWindows()

        dataset.finalize()
        if args.push_to_hub:
            dataset.push_to_hub(tags=json.loads(args.tags) if isinstance(args.tags, str) else args.tags)
            print(f"Eval dataset pushed to {args.repo_id}")

if __name__ == "__main__":
    main()