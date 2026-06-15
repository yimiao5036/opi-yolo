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
# 1. 环境初始化：自动挂载本地虚拟环境中的 FFmpeg 7.1 核心
# =====================================================================
try:
    import imageio_ffmpeg

    ffmpeg_exe_path = imageio_ffmpeg.get_ffmpeg_exe()
    ffmpeg_bin_dir = os.path.dirname(ffmpeg_exe_path)
    if ffmpeg_bin_dir not in os.environ["PATH"]:
        os.environ["PATH"] = ffmpeg_bin_dir + os.pathsep + os.environ["PATH"]
except ImportError:
    sys.exit(1)

import av

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')


# =====================================================================
# 2. 极速转码推流引擎（移除所有不必要的队列等待与开销）
# =====================================================================
class JPEG2H264Streamer:
    def __init__(self, qgc_ip="127.0.0.1", qgc_port=5600, width=640, height=480, fps=24):
        self.logger = logging.getLogger("ProxyStreamer")
        self.qgc_ip = qgc_ip
        self.qgc_port = qgc_port
        self.width = width
        self.height = height
        self.fps = fps

        # 缓冲区缩减到 2，宁可丢帧，也绝不容忍 1 毫秒的累积排队延迟
        self.frame_queue = queue.Queue(maxsize=2)
        self.is_running = False
        self.encoder_thread = None

    def start(self):
        self.is_running = True
        self.encoder_thread = threading.Thread(target=self._encoder_push_worker, daemon=True)
        self.encoder_thread.start()

    def push_frame(self, frame):
        if not self.is_running:
            return
        try:
            self.frame_queue.put_nowait(frame)
        except queue.Full:
            try:
                self.frame_queue.get_nowait()  # 顶出老帧
                self.frame_queue.put_nowait(frame)
            except queue.Empty:
                pass

    def _encoder_push_worker(self):
        output_url = f"udp://{self.qgc_ip}:{self.qgc_port}"
        output_container = None
        stream = None
        pts = 0

        while self.is_running:
            try:
                if output_container is None:
                    output_container = av.open(output_url, mode='w', format='mpegts')
                    stream = output_container.add_stream('libx264', rate=self.fps)
                    stream.width = self.width
                    stream.height = self.height
                    stream.pix_fmt = 'yuv420p'

                    # 强悍的零延迟参数配置
                    stream.options = {
                        'preset': 'ultrafast',
                        'tune': 'zerolatency',
                        'g': str(self.fps)
                    }
                    self.logger.info("⚡ 极致低延迟转码推流内核就绪")

                try:
                    frame = self.frame_queue.get(timeout=1.0)
                except queue.Empty:
                    continue

                # 快速色彩空间转换 (BGR -> RGB)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                av_frame = av.VideoFrame.from_ndarray(frame_rgb, format='rgb24')

                av_frame.pts = pts
                pts += int(90000 / self.fps)

                for packet in stream.encode(av_frame):
                    output_container.mux(packet)

            except Exception as e:
                output_container = None
                time.sleep(1.0)

    def stop(self):
        self.is_running = False
        if self.encoder_thread:
            self.encoder_thread.join(timeout=1.0)


# =====================================================================
# 3. 核心全局配置与单套接字全速收包
# =====================================================================
PORT = 9999
RECORD_FPS = 24.0
frame_interval = 1.0 / RECORD_FPS

latest_frame = None
frame_lock = threading.Lock()
is_running = True


def udp_receiver_thread():
    global latest_frame, is_running
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # 扩大内核网络接收缓冲区，防止超高频数据包在高负载下被系统丢弃
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 * 1024 * 1024)
    sock.bind(('0.0.0.0', PORT))

    while is_running:
        try:
            packet, _ = sock.recvfrom(65535)
            data = np.frombuffer(packet, dtype=np.uint8)
            frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if frame is not None:
                with frame_lock:
                    latest_frame = frame
        except Exception:
            time.sleep(0.005)
    sock.close()


def main():
    global latest_frame, is_running

    # 启动全速收包
    recv_t = threading.Thread(target=udp_receiver_thread, daemon=True)
    recv_t.start()

    # 启动极速中继转码
    proxy = JPEG2H264Streamer(qgc_ip="127.0.0.1", qgc_port=5600, width=640, height=480, fps=int(RECORD_FPS))
    proxy.start()

    last_frame_recorded = None
    print("🚀 纯净版图传中继代理网关已在后台全速运行...（按 Ctrl+C 退出）")

    try:
        while is_running:
            loop_start = time.time()

            current_frame = None
            with frame_lock:
                if latest_frame is not None:
                    current_frame = latest_frame
                    latest_frame = None  # 取出即置空，拒绝重复读取

            # 无线丢包平滑补偿核心（保留此逻辑可以大幅减少 QGC 绿屏频率）
            if current_frame is not None:
                last_frame_recorded = current_frame
            elif last_frame_recorded is not None:
                current_frame = last_frame_recorded

            if current_frame is not None:
                # 内存直推，直接进入 FFmpeg 7.1 零延迟通道
                proxy.push_frame(current_frame)

            # 移除所有窗口渲染和 waitKey 阻塞，实行严格的高精时间步长控制
            time_used = time.time() - loop_start
            sleep_time = frame_interval - time_used
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n👋 正在安全关闭网关...")
    finally:
        is_running = False
        proxy.stop()
        print("✅ 网关服务已安全退出。")


if __name__ == "__main__":
    main()