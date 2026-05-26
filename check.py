# ===================================
#           查看视频的帧率等属性
# ===================================
import cv2
import os

video_path = "saved_video/output_result.mp4"  # 替换成你上传的视频文件名

if not os.path.exists(video_path):
    print(f"错误：找不到文件 {video_path}")
    exit()

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("错误：无法打开视频文件！")
    exit()

# 获取视频自带的物理帧率 (FPS)
fps = cap.get(cv2.CAP_PROP_FPS)

# 获取视频的总帧数
total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

# 获取视频的分辨率宽度和高度
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 计算视频总时长（秒）
duration = total_frames / fps if fps > 0 else 0

print("==========================================")
print(f"视频文件: {os.path.basename(video_path)}")
print(f"视频分辨率: {width} x {height}")
print(f"视频原生帧率 (FPS): {fps:.2f} 帧/秒  <--- [核心原因看这里]")
print(f"视频总帧数: {int(total_frames)} 帧")
print(f"视频总时长: {duration:.2f} 秒")
print("==========================================")

# if fps <= 25.5:
#     print("💡 结论：不出所料，你的视频本身就被录制/压制成了 25 帧！")
#     print("OpenCV 的 cap.read() 会卡在 25fps 的播放速度上等下一帧，导致 NPU 闲置。")
#     print("你想测试 NPU 的真实上限，需要用前面提到的『纯随机数矩阵压测脚本』。")

cap.release()