import socket
import cv2
import numpy as np
import threading
import queue
import time
import os

# ==================== 地面站核心配置 ====================
PORT = 9999
SAVE_DIR = "./asset/saved_video"
SAVE_PATH = os.path.join(SAVE_DIR, "ground_station_record.mp4")
RECORD_FPS = 24.0  # 录像固定的标准帧率
# =======================================================

# 确保保存视频的文件夹存在
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# 核心线程共享：最新的一帧画面与互斥锁（防止读写冲突）
latest_frame = None
frame_lock = threading.Lock()
is_running = True


def udp_receiver_thread():
    """ 📡 后台极速收包线程 """
    global latest_frame, is_running
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # 放大系统内核缓冲区到 4MB，全力应对图传电台的无线网络抖动
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)

    try:
        sock.bind(('0.0.0.0', PORT))
        print(f"📡 [地面站接收端] 已启动，正在监听 UDP 端口 {PORT}...")
    except Exception as e:
        print(f"❌ 端口绑定失败: {e}")
        is_running = False
        return

    while is_running:
        try:
            packet, addr = sock.recvfrom(65535)
            data = np.frombuffer(packet, dtype=np.uint8)
            frame = cv2.imdecode(data, cv2.IMREAD_COLOR)

            if frame is not None:
                # 安全地更新最新的一帧，瞬间覆盖老帧，杜绝延迟积压
                with frame_lock:
                    latest_frame = frame
        except Exception as e:
            if is_running:
                print(f"⚠️ 接收解码异常: {e}")
            time.sleep(0.01)

    sock.close()


def main():
    global latest_frame, is_running

    # 1. 启动后台无线电收包线程
    recv_t = threading.Thread(target=udp_receiver_thread, daemon=True)
    recv_t.start()

    # 2. 初始化地面站专用录像机 (对应香橙派最后一刻 resize 后的 640x480)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(SAVE_PATH, fourcc, RECORD_FPS, (640, 480))
    print(f"📹 视频录像机就绪，文件将安全保存在: {SAVE_PATH}")

    # 用于计算接收端实时刷新率
    fps_start_time = time.time()
    fps_counter = 0
    fps_display = 0

    # 严格按照发送端的帧率控制循环步长（1 / 24秒 ≈ 0.0416秒）
    frame_interval = 1.0 / RECORD_FPS
    last_frame_recorded = None

    print("🖥️ 监视窗口已建立，等待无人机连接推流...")

    try:
        while is_running:
            loop_start = time.time()

            # 从共享区取出最新帧
            current_frame = None
            with frame_lock:
                if latest_frame is not None:
                    current_frame = latest_frame.copy()

            # 无线图传时延容错核心逻辑：
            if current_frame is not None:
                # 拿到了图传新帧，更新最后记录的有效帧
                last_frame_recorded = current_frame
            elif last_frame_recorded is not None:
                # 【抗干扰核心】当前瞬间电台丢包没收到新图，复制上一帧顶替，防止视频时间轴缩水变快
                current_frame = last_frame_recorded.copy()

            # 如果已经有画面可渲染（无论是新帧还是补偿帧）
            if current_frame is not None:
                # 录像写入（PC 的强悍 CPU 在后台无感硬写磁盘，完全不拖累香橙派的 NPU）
                video_writer.write(current_frame)

                # 计算并渲染接收端的显示 FPS
                fps_counter += 1
                if (time.time() - fps_start_time) > 1.0:
                    fps_display = fps_counter
                    fps_counter = 0
                    fps_start_time = time.time()

                # 在右上角标注实时的图传显示帧率
                cv2.putText(current_frame, f"GCS FPS: {fps_display}", (current_frame.shape[1] - 140, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # 实时监控显示
                cv2.imshow("🚀 UAV AI Ground Station Monitor", current_frame)

            # 按键检测（按 'q' 安全退出）
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("👋 接收到关闭指令...")
                break

            # 严格控制写入频率，使其匹配 24.0 FPS 的时间流速
            time_used = time.time() - loop_start
            sleep_time = frame_interval - time_used
            if sleep_time > 0:
                time.sleep(sleep_time)

    finally:
        # 善后处理：确保内存日志和视频安全落盘
        is_running = False
        video_writer.release()
        cv2.destroyAllWindows()
        print("\n=======================================================")
        print(f"✅ 地面站已安全关闭。")
        print(f"🎬 带有精确检测框、对准线和控制误差的视频已成功保存在:\n   {SAVE_PATH}")
        print("=======================================================")


if __name__ == "__main__":
    main()