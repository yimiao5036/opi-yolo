import cv2
import os

os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp'

# 改用正确的地址
rtsp_url = "rtsp://192.168.144.25:8554/main.264"

print(f"[测试] 正在尝试连接: {rtsp_url}")
cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

if not cap.isOpened():
    print("[结果] 无法打开视频流。")
else:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("[结果] 读取帧失败，视频流可能为空。")
    else:
        print(f"[结果] 成功读取第一帧！图像尺寸为: {frame.shape}")
    cap.release()