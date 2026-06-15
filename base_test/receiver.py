import socket
import cv2
import numpy as np
import threading
import queue
import time
import os
import sys
import logging

# =====================================================================
# 环境初始化：必须在 import av 之前将 imageio-ffmpeg 路径注入 PATH
# =====================================================================
try:
    import imageio_ffmpeg

    # 获取已经存在的 ffmpeg.exe 路径
    ffmpeg_exe_path = imageio_ffmpeg.get_ffmpeg_exe()
    ffmpeg_bin_dir = os.path.dirname(ffmpeg_exe_path)

    # 将该文本动态挂载到系统临时环境变量，确保 PyAV 能够识别并调用
    if ffmpeg_bin_dir not in os.environ["PATH"]:
        os.environ["PATH"] = ffmpeg_bin_dir + os.pathsep + os.environ["PATH"]
        print(f"✅ 成功挂载本地环境 FFmpeg: {ffmpeg_bin_dir}")
except ImportError:
    print("❌ 错误：未检测到 imageio_ffmpeg 库，请先执行 `pip install imageio-ffmpeg`")
    sys.exit(1)

import av

# 配置标准日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')


class JPEG2H264Streamer:
    """
    📡 地面端无人机图传转码代理类 (纯内存推流优化版)

    优化点：不再自己建立 UDP 接收端，而是暴露 push_frame 接口，
    直接接收主线程已经解码完成的 OpenCV BGR 矩阵，消除重复解码，极大降低 CPU 负担。
    """

    def __init__(self, qgc_ip="127.0.0.1", qgc_port=5600, width=640, height=480, fps=24):
        self.logger = logging.getLogger("ProxyStreamer")
        self.qgc_ip = qgc_ip
        self.qgc_port = qgc_port
        self.width = width
        self.height = height
        self.fps = fps

        # 帧缓冲区（只保留最新图像，防止网络抖动导致的画面累积延迟）
        self.frame_queue = queue.Queue(maxsize=3)
        self.is_running = False
        self.encoder_thread = None

    def start(self):
        """ 启动异步 H.264 编码推流线程 """
        self.is_running = True
        self.encoder_thread = threading.Thread(target=self._encoder_push_worker, daemon=True)
        self.encoder_thread.start()
        self.logger.info(f"🚀 H.264 转码推流引擎启动：目标 QGC -> {self.qgc_ip}:{self.qgc_port}")

    def push_frame(self, frame):
        """ 供外部（主线程）调用的纯内存图像注入接口 """
        if not self.is_running:
            return
        try:
            self.frame_queue.put_nowait(frame)
        except queue.Full:
            try:
                self.frame_queue.get_nowait()  # 弹出老帧
                self.frame_queue.put_nowait(frame)
            except queue.Empty:
                pass

    def _encoder_push_worker(self):
        """ 高性能 H.264 编码与推流核心进程 """
        output_url = f"udp://{self.qgc_ip}:{self.qgc_port}"
        output_container = None
        stream = None

        pts = 0
        frame_count = 0

        while self.is_running:
            try:
                # 初始化/重连机制
                if output_container is None:
                    output_container = av.open(output_url, mode='w', format='mpegts')
                    stream = output_container.add_stream('libx264', rate=self.fps)
                    stream.width = self.width
                    stream.height = self.height
                    stream.pix_fmt = 'yuv420p'

                    # 💡 极其关键的低延迟高级调优参数
                    stream.options = {
                        'preset': 'ultrafast',
                        'tune': 'zerolatency',
                        'g': str(self.fps)  # 每一秒强制发一个 I 帧
                    }
                    self.logger.info("PyAV H.264 极速低延迟编码器就绪")

                # 从队列获取主线程传过来的 OpenCV BGR 帧
                try:
                    frame = self.frame_queue.get(timeout=1.0)
                except queue.Empty:
                    continue

                # 确保分辨率对齐编码器
                if frame.shape[1] != self.width or frame.shape[0] != self.height:
                    frame = cv2.resize(frame, (self.width, self.height))

                # 转换色彩空间 (BGR -> RGB) 并封装为 PyAV 帧
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # 修改后
                av_frame = av.VideoFrame.from_ndarray(frame_rgb, format='rgb24')

                # 分配高精度时钟戳
                av_frame.pts = pts
                pts += int(90000 / self.fps)

                # 执行 H.264 实体压缩编码，并注入 UDP 发送端口
                for packet in stream.encode(av_frame):
                    output_container.mux(packet)

                frame_count += 1
                if frame_count % 100 == 0:
                    self.logger.info(f"📊 QGC 图传统计: 已成功向 QGC 转发 {frame_count} 帧 H.264 视频")

            except Exception as e:
                self.logger.error(f"转码或推流异常（准备重连）: {e}")
                output_container = None
                time.sleep(1.0)

        # 资源释放
        if output_container:
            try:
                for packet in stream.encode():
                    output_container.mux(packet)
                output_container.close()
            except Exception:
                pass

    def stop(self):
        """ 安全关闭代理服务 """
        self.is_running = False
        if self.encoder_thread:
            self.encoder_thread.join(timeout=2.0)
        self.logger.info("转码代理已安全关闭。")


# ==================== 地面站全局配置 ====================
PORT = 9999
SAVE_DIR = "../asset/saved_video"
SAVE_PATH = os.path.join(SAVE_DIR, "ground_station_record.mp4")
RECORD_FPS = 24.0  # 录像与转码统一固定的标准帧率

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# 核心线程共享区
latest_frame = None
frame_lock = threading.Lock()
is_running = True


def udp_receiver_thread():
    """ 📡 后台唯一极速收包线程 """
    global latest_frame, is_running
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
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
                with frame_lock:
                    latest_frame = frame
        except Exception as e:
            if is_running:
                print(f"⚠️ 接收解码异常: {e}")
            time.sleep(0.01)

    sock.close()


def main():
    global latest_frame, is_running

    # 1. 启动唯一的后台接收线程
    recv_t = threading.Thread(target=udp_receiver_thread, daemon=True)
    recv_t.start()

    # 2. 初始化本地录像机
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(SAVE_PATH, fourcc, RECORD_FPS, (640, 480))
    print(f"📹 视频录像机就绪，文件保存在: {SAVE_PATH}")

    # 3. 启动地面站高性能 H.264 代理组件 (对齐 24 FPS)
    proxy = JPEG2H264Streamer(
        qgc_ip="127.0.0.1",
        qgc_port=5600,
        width=640,
        height=480,
        fps=int(RECORD_FPS)
    )
    proxy.start()

    # FPS 统计变量
    fps_start_time = time.time()
    fps_counter = 0
    fps_display = 0

    frame_interval = 1.0 / RECORD_FPS
    last_frame_recorded = None

    print("🖥️ 监视窗口已建立，等待无人机连接...")

    try:
        while is_running:
            loop_start = time.time()

            # 从共享区取出最新帧
            current_frame = None
            with frame_lock:
                if latest_frame is not None:
                    current_frame = latest_frame.copy()
                    latest_frame = None  # 取出后置空，防止下一轮循环读到未更新的旧重复帧

            # 无线图传丢包容错核心
            if current_frame is not None:
                last_frame_recorded = current_frame
            elif last_frame_recorded is not None:
                current_frame = last_frame_recorded.copy()

            if current_frame is not None:
                # ─── 核心修改点：同时向 QGC 编码流中注入当前帧 ───
                proxy.push_frame(current_frame)

                # 本地硬盘录像
                video_writer.write(current_frame)

                # 计算并渲染本地显示的实时 FPS
                fps_counter += 1
                if (time.time() - fps_start_time) > 1.0:
                    fps_display = fps_counter
                    fps_counter = 0
                    fps_start_time = time.time()

                # 标注并进行本地窗口渲染显示
                cv2.putText(current_frame, f"GCS FPS: {fps_display}", (current_frame.shape[1] - 140, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow("🚀 UAV AI Ground Station Monitor", current_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("👋 接收到关闭指令...")
                break

            # 严格步长控制
            time_used = time.time() - loop_start
            sleep_time = frame_interval - time_used
            if sleep_time > 0:
                time.sleep(sleep_time)

    finally:
        is_running = False
        video_writer.release()
        cv2.destroyAllWindows()
        proxy.stop()  # 安全关闭转码流容器
        print(f"\n✅ 地面站双轨服务已安全关闭。录像保存在: {SAVE_PATH}")


if __name__ == "__main__":
    main()