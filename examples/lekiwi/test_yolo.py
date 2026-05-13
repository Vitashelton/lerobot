import cv2
import numpy as np
from ultralytics import YOLO
from lerobot.robots.lekiwi.config_lekiwi import LeKiwiClientConfig
from lerobot.robots.lekiwi.lekiwi_client import LeKiwiClient

def main():
    print("加载 YOLO 模型...")
    model = YOLO('yolov8n.pt') 
    
    print("连接树莓派...")
    robot = LeKiwiClient(LeKiwiClientConfig(remote_ip="192.168.3.215", id="lekiwi"))
    robot.connect()

    print("\n" + "="*40)
    print("照妖镜已启动！")
    print("把纸巾盒放到摄像头前，看看 AI 觉得它是什么。")
    print("按 'q' 退出。")
    print("="*40)

    try:
        while True:
            obs = robot.get_observation()
            img_front = obs.get('front')
            
            if img_front is None: continue
            if hasattr(img_front, 'cpu'): 
                img_front = img_front.permute(1, 2, 0).cpu().numpy()
                
            img_bgr = cv2.cvtColor((img_front * 255).astype(np.uint8) if img_front.max() <= 1.0 else img_front, cv2.COLOR_RGB2BGR)

            # 关键 1：不填 classes 参数，让它自由发挥
            # 关键 2：降低 conf（置信度），只要它觉得有 15% 的可能是个东西，就框出来
            results = model(img_bgr, conf=0.15, verbose=False)

            # 关键 3：直接用 YOLO 自带的超级画图功能，包含框、名字和概率
            annotated_frame = results[0].plot()

            cv2.imshow("YOLO Magic Mirror", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            
    finally:
        robot.disconnect()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()