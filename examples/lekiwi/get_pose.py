#!/usr/bin/env python
import os
import time
from lerobot.teleoperators.so_leader import SO100Leader, SO100LeaderConfig

# 防止 LeRobot 初始化时卡网络
os.environ["http_proxy"] = "http://127.0.0.1:7897" 
os.environ["https_proxy"] = "http://127.0.0.1:7897"

def main():
    print("正在连接 Leader 臂...")
    leader = SO100Leader(SO100LeaderConfig(port="/dev/ttyACM0", id="leader"))
    leader.connect()

    print("\n" + "="*50)
    print("✅ 姿势标定仪已启动！")
    print("现在，请用手把 Leader 臂掰成你心目中最完美的【收纳倒车姿势】。")
    print("屏幕会实时刷新它当前的各关节角度。")
    print("按 Ctrl+C 退出。")
    print("="*50 + "\n")

    try:
        while True:
            # 正确的 LeRobot API 读取方法
            action = leader.get_action()
            
            # 把字典里的数值保留一位小数，并精简一下键名方便查看
            formatted = {k.replace('arm_', '').replace('.pos', ''): round(float(v), 1) for k, v in action.items()}
            
            # 使用 \r 在同一行刷新数据，避免刷屏
            print(f"实时角度 -> {formatted}          ", end='\r') 
            time.sleep(0.2)
            
    except KeyboardInterrupt:
        print("\n\n[INFO] 标定结束。")
    finally:
        leader.disconnect()

if __name__ == "__main__":
    main()