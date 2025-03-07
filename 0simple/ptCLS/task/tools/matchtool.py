#!/usr/bin/env python3
""" ImageNet Training Script
"""
import os
import cv2
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

def capvide():
    video_path = r'F:\0match\AIdata\mask\path_to_your_video.mp4'
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        exit(-123)

    frameinx = 0
    while True:
        # 读取一帧
        ret, frame = cap.read()
        if not ret:
            print("End of video")
            break

        # 显示帧
        cv2.imshow('Video', frame)
        # cv2.imwrite('frame.jpg', frame)

        # 按 'q' 键退出
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    print("finish cap")


if __name__ == '__main__':
    capvide()
    print("finish work")
