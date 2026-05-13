# !/usr/bin/env python

import argparse
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# 🔥 核心修复一：将 ACTPolicy 改为 DiffusionPolicy
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

from lerobot.policies.factory import make_pre_post_processors
from lerobot.processor import make_default_processors
from lerobot.robots.lekiwi import LeKiwiClient, LeKiwiClientConfig
from lerobot.scripts.lerobot_record import record_loop
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import init_rerun

def parse_args():
    parser = argparse.ArgumentParser()
    # 兼容你命令行里的占位符参数 (不影响主逻辑，但保证你不报错)
    parser.add_argument("--robot.type", type=str, default="lekiwi")
    parser.add_argument("--control.type", type=str, default="record")
    parser.add_argument("--control.tags", type=str, default='["eval"]')
    parser.add_argument("--control.warmup_time_s", type=int, default=5)
    parser.add_argument("--control.push_to_hub", type=str, default="true")

    # 核心映射参数
    parser.add_argument("--control.fps", dest="fps", type=int, default=30)
    parser.add_argument("--control.single_task", dest="single_task", type=str, default="navigate")
    parser.add_argument("--control.repo_id", dest="repo_id", type=str, required=True, help="相当于原来的 HF_DATASET_ID")
    parser.add_argument("--control.episode_time_s", dest="episode_time_s", type=int, default=30)
    parser.add_argument("--control.reset_time_s", dest="reset_time_s", type=int, default=6)
    parser.add_argument("--control.num_episodes", dest="num_episodes", type=int, default=10)
    parser.add_argument("--control.policy.path", dest="policy_path", type=str, required=True, help="相当于原来的 HF_MODEL_ID")

    return parser.parse_args()

def main():
    args = parse_args()

    # 将命令行参数绑定到运行变量
    NUM_EPISODES = args.num_episodes
    FPS = args.fps
    EPISODE_TIME_SEC = args.episode_time_s
    RESET_TIME_SEC = args.reset_time_s
    TASK_DESCRIPTION = args.single_task
    HF_DATASET_ID = args.repo_id
    HF_MODEL_ID = args.policy_path

    print(f"\n🚀 [INFO] 启动模型评估部署模式...")
    print(f"👉 模型权重加载路径: {HF_MODEL_ID}")
    print(f"👉 评估数据集上传至: {HF_DATASET_ID}")
    print(f"👉 任务描述: {TASK_DESCRIPTION}")

    # 1. 初始化底盘配置
    robot_config = LeKiwiClientConfig(remote_ip="192.168.3.215", id="lekiwi")
    
    # 🔥 强杀腕部相机的护体神功
    if "wrist" in robot_config.cameras:
        del robot_config.cameras["wrist"]
        print("🛡️ [INFO] 已移除腕部相机 (wrist) 订阅。")

    robot = LeKiwiClient(robot_config)

    # 2. 加载训练好的策略模型
    print("\n🧠 [INFO] 正在将 Diffusion 模型加载至 GPU...")
    # 🔥 核心修复二：使用 DiffusionPolicy 的加载器
    policy = DiffusionPolicy.from_pretrained(HF_MODEL_ID)

    # 3. 配置数据集特征
    action_features = hw_to_dataset_features(robot.action_features, ACTION)
    obs_features = hw_to_dataset_features(robot.observation_features, OBS_STR)
    dataset_features = {**action_features, **obs_features}

    dataset = LeRobotDataset.create(
        repo_id=HF_DATASET_ID,
        fps=FPS,
        features=dataset_features,
        robot_type=robot.name,
        use_videos=True,
        image_writer_threads=4,
    )

    # 4. 构建数据预处理器与后处理器
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy,
        pretrained_path=HF_MODEL_ID,
        dataset_stats=dataset.meta.stats,
        preprocessor_overrides={"device_processor": {"device": str(policy.config.device)}},
    )

    # 5. 连接物理设备
    print("\n[INFO] 正在连接 Lekiwi 底盘...")
    robot.connect()

    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    listener, events = init_keyboard_listener()
    init_rerun(session_name="lekiwi_evaluate")

    try:
        if not robot.is_connected:
            raise ValueError("❌ [ERROR] 树莓派底盘未连接！")

        print("\n========================================================")
        print("✅ 模型推理就绪！双手离开键盘，看小车的表演吧！")
        print("========================================================\n")
        
        recorded_episodes = 0
        while recorded_episodes < NUM_EPISODES and not events["stop_recording"]:
            log_say(f"Running inference, recording eval episode {recorded_episodes} of {NUM_EPISODES}")

            # 实机闭环推理
            record_loop(
                robot=robot,
                events=events,
                fps=FPS,
                policy=policy,
                preprocessor=preprocessor,  
                postprocessor=postprocessor,
                dataset=dataset,
                control_time_s=EPISODE_TIME_SEC,
                single_task=TASK_DESCRIPTION,
                display_data=True,
                teleop_action_processor=teleop_action_processor,
                robot_action_processor=robot_action_processor,
                robot_observation_processor=robot_observation_processor,
            )

            if not events["stop_recording"] and (
                (recorded_episodes < NUM_EPISODES - 1) or events["rerecord_episode"]
            ):
                log_say("Reset the environment")
                record_loop(
                    robot=robot,
                    events=events,
                    fps=FPS,
                    control_time_s=RESET_TIME_SEC,
                    single_task=TASK_DESCRIPTION,
                    display_data=True,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                )

            if events["rerecord_episode"]:
                log_say("Re-record episode")
                events["rerecord_episode"] = False
                events["exit_early"] = False
                dataset.clear_episode_buffer()
                continue

            dataset.save_episode()
            recorded_episodes += 1

    finally:
        log_say("Stop inference & recording")
        robot.disconnect()
        listener.stop()

        if args.push_to_hub.lower() == "true":
            print("\n[INFO] 正在打包评估数据并上传至 Hugging Face...")
            dataset.finalize()
            dataset.push_to_hub()
            print("🎉 评估结束，数据已上传！")
        else:
            dataset.finalize()

if __name__ == "__main__":
    main()