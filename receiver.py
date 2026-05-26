import socket
import cv2
import numpy as np

PORT = 9999
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(('0.0.0.0', PORT))

print("📡 [笔记本地面站] 已启动，正在等待香橙派连线...")

# 初始化地面站的强悍 CPU 录像（对应发送端压缩后的分辨率）
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter("./saved_video/ground_station_record.mp4", fourcc, 24.0, (640, 480))

try:
    while True:
        packet, _ = sock.recvfrom(65535)
        data = np.frombuffer(packet, dtype=np.uint8)
        frame = cv2.imdecode(data, cv2.IMREAD_COLOR)

        if frame is not None:
            # 1. 地面站录像
            video_writer.write(frame)

            # 2. 实时显示监控
            cv2.imshow("🚀 UAV AI Ground Station Monitor", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
finally:
    sock.close()
    video_writer.release()
    cv2.destroyAllWindows()
    print("地面站已安全关闭，视频成功保存在本地。")