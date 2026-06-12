import os
import cv2
import numpy as np
import time
import threading
import queue
import socket
import json
import sys
from ais_bench.infer.interface import InferSession

# ================== 导入控制模块 =====================
from drone_controller.base_control import DroneController
from drone_controller.pid_counter import PID
# ===================================================

class VideoStreaming:
    """ 📡 视频流采集与后台图传模块（支持 RTSP 自动重连） """

    def __init__(self, video_cfg, net_cfg):
        self.video_source = video_cfg.get("video_source", 0)
        self.jpeg_quality = video_cfg.get("jpeg_quality", 75)
        self.gs_ip = net_cfg.get("ground_station_ip", "127.0.0.1")
        self.udp_port = net_cfg.get("udp_port", 9999)

        # RTSP 特殊参数
        self.is_rtsp = isinstance(self.video_source, str) and self.video_source.startswith("rtsp://")
        self.cap = None
        self.frame_buffer = queue.Queue(maxsize=1)   # 只保留最新一帧
        self.is_running = True
        self.capture_thread = None

        # 启动后台采集线程
        self._start_capture_thread()

        # 图传发送队列与线程（原样保留）
        self.tx_queue = queue.Queue(maxsize=2)
        self.sender_thread = threading.Thread(target=self._udp_stream_sender_worker, daemon=True)
        self.sender_thread.start()

    def _start_capture_thread(self):
        """ 独立线程负责从摄像头/RTSP 读取帧，避免主循环阻塞 """
        self.capture_thread = threading.Thread(target=self._capture_worker, daemon=True)
        self.capture_thread.start()

    def _capture_worker(self):
        """ 后台不断读取帧，存入 frame_buffer """
        while self.is_running:
            # 如果摄像头未打开或已断开，尝试连接
            if self.cap is None or not self.cap.isOpened():
                self._open_camera()
                if self.cap is None or not self.cap.isOpened():
                    time.sleep(1)   # 等待重连
                    continue

            ret, frame = self.cap.read()
            if not ret:
                # 读取失败，关闭并标记需重连
                print("⚠️ RTSP 读取失败，准备重连...")
                self.cap.release()
                self.cap = None
                continue

            # 清空旧帧，只保留最新一帧
            while not self.frame_buffer.empty():
                try:
                    self.frame_buffer.get_nowait()
                except queue.Empty:
                    break
            self.frame_buffer.put(frame)

        if self.cap is not None:
            self.cap.release()

    def _open_camera(self):
        """ 打开摄像头或 RTSP 流，并设置低延迟参数 """
        try:
            print(f"📷 正在连接视频源: {self.video_source}")
            self.cap = cv2.VideoCapture(self.video_source)

            if self.is_rtsp:
                # 设置 RTSP 缓冲区大小（只保留 1 帧，降低延迟）
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                # 设置读取超时（OpenCV 没有直接超时参数，但可尝试设置后端属性）
                # 某些平台支持以下设置：
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('H', '2', '6', '4'))
                # 强制使用 tcp 协议（某些 rtsp 默认 udp 会丢包严重）
                # 如果 rtsp 地址支持，可以改成 "rtsp://...?tcp"
                if "?" not in self.video_source:
                    self.video_source += "?tcp"
                    # 注意：有些相机需要重新初始化，这里简单打印提示
                    print("💡 建议使用 TCP 传输: 在 RTSP URL 后添加 ?tcp")
            else:
                # USB 摄像头也可设置缓冲区
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            # 可选：强制指定分辨率（根据你的相机调整）
            # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            if self.cap.isOpened():
                print("✅ 视频源连接成功")
            else:
                print("❌ 视频源打开失败")
        except Exception as e:
            print(f"❌ 打开视频源异常: {e}")
            self.cap = None

    def read_frame(self):
        """ 主循环调用：从缓冲区获取最新一帧（非阻塞） """
        try:
            frame = self.frame_buffer.get_nowait()
            return True, frame
        except queue.Empty:
            # 没有新帧时返回 False，主循环可根据需要 sleep
            return False, None

    def push_to_stream(self, frame):
        """ 原图传推流逻辑保持不变 """
        try:
            send_frame = cv2.resize(frame, (640, 480))
            self.tx_queue.put_nowait(send_frame)
        except queue.Full:
            pass

    def _udp_stream_sender_worker(self):
        """ 原图传发送线程（未改动） """
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        server_address = (self.gs_ip, self.udp_port)
        print(f"📡 图传后台线程已启动，目标地面站 -> {self.gs_ip}:{self.udp_port}")

        while self.is_running:
            try:
                frame_to_send = self.tx_queue.get(timeout=1.0)
                if frame_to_send is None:
                    break
                result, img_encode = cv2.imencode('.jpg', frame_to_send,
                                                  [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
                if not result:
                    continue
                data = img_encode.tobytes()
                if len(data) > 65000:
                    continue
                sock.sendto(data, server_address)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ 图传线程异常: {e}")
                time.sleep(0.1)
        sock.close()

    def release(self):
        """ 释放资源 """
        self.is_running = False
        if self.capture_thread is not None:
            self.capture_thread.join(timeout=2.0)
        if self.cap is not None:
            self.cap.release()
        self.tx_queue.put(None)

class YOLO26UAVInfer:
    """ 🚀 昇腾 NPU 推理与图像后处理模块 """

    def __init__(self, model_cfg):
        model_path = model_cfg.get("model_path", "./om/yolo26n-balloon.om")
        device_id = model_cfg.get("device_id", 0)
        self.conf_threshold = model_cfg.get("conf_threshold", 0.25)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ 找不到模型文件: {model_path}")

        self.session = InferSession(device_id, model_path)
        self.input_width = 640
        self.input_height = 640

    def letterbox(self, img, new_shape=(640, 640), color=(114, 114, 114)):
        shape = img.shape[:2]  # [height, width]
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        dw /= 2
        dh /= 2
        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return img, r, (dw, dh)

    def preprocess(self, frame):
        """ 图像预处理，输出符合 NPU 输入的 Tensor """
        input_tensor, ratio, dwdh = self.letterbox(frame)
        input_tensor = input_tensor.astype(np.float32) / 255.0
        input_tensor = np.transpose(input_tensor, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0)
        input_tensor = np.ascontiguousarray(input_tensor)
        return input_tensor, ratio, dwdh

    def postprocess(self, outputs, ratio, dwdh, orig_shape):
        out = outputs[0]
        detections = []
        for i in range(out.shape[1]):
            box = out[0, i, :4]
            conf = out[0, i, 4]
            cls_id = int(out[0, i, 5])
            if conf > self.conf_threshold:
                x1, y1, x2, y2 = box
                # 基于原图比例还原坐标
                x1 = (x1 - dwdh[0]) / ratio
                y1 = (y1 - dwdh[1]) / ratio
                x2 = (x2 - dwdh[0]) / ratio
                y2 = (y2 - dwdh[1]) / ratio
                x1 = max(0, min(x1, orig_shape[1]))
                y1 = max(0, min(y1, orig_shape[0]))
                x2 = max(0, min(x2, orig_shape[1]))
                y2 = max(0, min(y2, orig_shape[0]))
                detections.append([x1, y1, x2, y2, conf, cls_id])
        return detections

    def draw_boxes(self, img, detections):
        """ 在图像上渲染精准目标框 """
        for det in detections:
            x1, y1, x2, y2, conf, cls_id = det
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            label = f"Target: {conf:.2f}"
            cv2.putText(img, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return img


class UAVControlLoop:
    """ 🛸 无人机飞控闭环控制核心 """

    def __init__(self, fc_cfg, pid_y_cfg, pid_z_cfg):
        connection_string = fc_cfg.get("connection_string", "udpout:127.0.0.1:14550")
        baud_rate = fc_cfg.get("baud_rate", 57600)

        # 初始化控制驱动端
        self.uav = DroneController(connection_string=connection_string, baud=baud_rate)

        # 从 JSON 参数动态初始化 PID 调节器
        self.pid_y = PID(
            kp=pid_y_cfg["kp"], ki=pid_y_cfg["ki"], kd=pid_y_cfg["kd"],
            max_out=pid_y_cfg["max_out"], min_out=pid_y_cfg["min_out"]
        )
        self.pid_z = PID(
            kp=pid_z_cfg["kp"], ki=pid_z_cfg["ki"], kd=pid_z_cfg["kd"],
            max_out=pid_z_cfg["max_out"], min_out=pid_z_cfg["min_out"]
        )

    def start_uav(self):
        """ 连接并解锁无人机 """
        self.uav.connect()
        if not self.uav.is_armed and self.uav.current_mode not in ['GUIDED', 'OFFBOARD']:
            self.uav.arm()
            self.uav.print_status()
            self.uav.set_mode('OFFBOARD')

    def process_control(self, detections, orig_shape, frame):
        """ 根据检测结果执行 MAVLink 闭环控制逻辑 """
        img_center_x = orig_shape[1] / 2
        img_center_y = orig_shape[0] / 2
        current_mode = getattr(self.uav, 'current_mode', 'UNKNOWN')

        if len(detections) > 0:
            # 寻找置信度最高的目标
            best_det = max(detections, key=lambda x: x[4])
            x1, y1, x2, y2, conf, cls_id = best_det
            target_x = (x1 + x2) / 2
            target_y = (y1 + y2) / 2

            # 计算归一化误差
            err_x = (target_x - img_center_x) / img_center_x
            err_y = (target_y - img_center_y) / img_center_y

            # PID 计算转化为物理速度需求
            vy_cmd = self.pid_y.update(err_x)  # 图像X正向 -> 飞机Vy右移
            vz_cmd = self.pid_z.update(err_y)  # 图像Y正向 -> 飞机Vz下移 (NED系)
            vx_cmd = 0.0  # 保持前后平移为0

            # 视效叠加
            cv2.line(frame, (int(img_center_x), int(img_center_y)), (int(target_x), int(target_y)), (255, 0, 0), 2)
            cv2.putText(frame, f"MAVLink Out -> Err_X: {err_x:+.3f} Err_Y: {err_y:+.3f}", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            # 执行机体速度控制
            if current_mode in ['GUIDED', 'OFFBOARD']:
                self.uav.send_body_velocity(vx=vx_cmd, vy=vy_cmd, vz=vz_cmd, yaw_rate=0.0)
            else:
                cv2.putText(frame, f"MANUAL MODE OVERRIDE ({current_mode})", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 0, 255), 2)
        else:
            # 目标丢失保护逻辑
            cv2.putText(frame, "MAVLink Out -> Target LOST (HOVER)", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 255), 2)

            # 清除 PID 积分，防止突跳
            if hasattr(self.pid_y, 'reset'): self.pid_y.reset()
            if hasattr(self.pid_z, 'reset'): self.pid_z.reset()

            if current_mode in ['GUIDED', 'OFFBOARD']:
                self.uav.send_body_velocity(vx=0.0, vy=0.0, vz=0.0, yaw_rate=0.0)


# ==================== 主程序入口 ======================
def main():
    # 默认加载同目录下的 config.json
    config_file = "config.json"

    # 允许在命令行指定 json 路径，例如: python3 infer_camera_modular.py config_outdoor.json
    if len(sys.argv) > 1:
        config_file = sys.argv[1]

    print(f"📖 正在从 {config_file} 加载系统配置...")
    try:
        with open(config_file, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception as e:
        print(f"❌ 读取配置文件失败: {e}")
        sys.exit(1)

    # 实例化各子系统并注入配置
    streamer = VideoStreaming(cfg["video"], cfg["network"])
    detector = YOLO26UAVInfer(cfg["model"])
    controller = UAVControlLoop(cfg["flight_control"], cfg["pid_y"], cfg["pid_z"])

    # 启动无人机连接
    controller.start_uav()


    print("🚀 NPU 精准推理 [JSON配置动态加载版] 主循环启动...")

    try:
        while True:
            ret, frame = streamer.read_frame()
            if not ret:
                time.sleep(0.005)
                continue

            orig_shape = frame.shape[:2]

            # 1. 图像预处理
            input_tensor, ratio, dwdh = detector.preprocess(frame)

            # 2. NPU 硬件推理
            t_start = time.time()
            outputs = detector.session.infer([input_tensor])
            t_end = time.time()

            fps_pure = 1.0 / (t_end - t_start)

            # 3. 坐标反求与控制计算
            detections = detector.postprocess(outputs, ratio, dwdh, orig_shape)
            controller.process_control(detections, orig_shape, frame)

            # 4. 界面渲染
            frame = detector.draw_boxes(frame, detections)
            cv2.putText(frame, f"NPU Pure FPS: {fps_pure:.1f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255),
                        2)

            # 5. 推送至后台异步图传队列
            streamer.push_to_stream(frame)

    except KeyboardInterrupt:
        print("\n👋 接收到终止信号，安全退出中...")
    finally:
        streamer.release()
        print("程序运行结束。")


if __name__ == "__main__":
    main()