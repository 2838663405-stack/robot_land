#!/usr/bin/env python3
import cv2
import numpy as np
import urllib.request
import time
import os

# 开发板的 web_video_server 地址
URL = "http://192.168.50.195:8080/stream?topic=/camera/camera/color/image_raw&type=mjpeg"

# 创建保存目录
name = input("请输入队员姓名（拼音）: ")
save_dir = f"face_data/{name}"
os.makedirs(save_dir, exist_ok=True)

print(f"保存目录: {save_dir}")
print("操作说明:")
print("  按 空格键 拍照")
print("  按 q 退出")
print(f"目标: 30 张，按空格键开始...")

# 打开视频流
cap = cv2.VideoCapture(URL)
if not cap.isOpened():
    print("无法连接，请确保 web_video_server 正在运行")
    print("在开发板上运行: ros2 run web_video_server web_video_server")
    exit(1)

count = 0
while count < 30:
    ret, frame = cap.read()
    if not ret:
        print("等待画面...")
        time.sleep(0.1)
        continue
    
    # 显示画面
    cv2.imshow("Capture - " + name, frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):  # 空格键
        filename = f"{save_dir}/{count}.jpg"
        cv2.imwrite(filename, frame)
        count += 1
        print(f"已保存 {count}/30 张 -> {filename}")
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"采集完成！共 {count} 张图片保存在 {save_dir}")
