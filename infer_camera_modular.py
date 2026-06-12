import os
import cv2
import numpy as np
import time
import threading
import queue
import socket
import json
import sys

# ================== 昇腾 NPU SDK（保护性导入） =====================
# 说明：本机可能无昇腾环境，try-except 确保 import 错误仅告警不崩溃。
# 部署到香橙派 AI Pro 时只需安装 ais_bench 即可无缝运行。
try:
    from ais_bench.infer.interface import InferSession
    _HAS_NPU = True
    print("✅ 昇腾 NPU SDK 导入成功")
except ImportError:
    _HAS_NPU = False
    print("⚠️ 昇腾 NPU SDK 未安装，YOLO26UAVInfer 将在实例化时报错（部署后正常）")
# ===============================================================

# ================== ZMQ Router 通信代理 =====================
from drone_controller.router_proxy import RouterProxy
from drone_controller.pid_counter import PID
# ===========================================================


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
    """ 🛸 无人机飞控闭环控制核心（基于 ZeroMQ Router 协议）

    使用 RouterProxy 取代底层 DroneController，所有 MAVLink 通信
    均通过 ZMQ REQ/SUB 转发给 Router 处理。

    【控制流】
      ┌─主循环─────────────┐
      │ update_detections() │──→ 写入共享检测结果
      └────────────────────┘
               │
      ┌─控制线程(20Hz)──────┐
      │ _compute_velocity() │──→ PID 计算机体速度
      │ proxy.send_setpoint │──→ ZMQ REQ → Router → PX4
      └────────────────────┘

    【保活设计】
      - 控制线程以 20Hz 持续发送 SETPOINT，满足协议 ≥20Hz 要求
      - 检测结果超过 500ms 未更新 → 强制零速刹车（防断流）
      - 无目标时 PID 自动复位 → 输出零速
    """

    def __init__(self, fc_cfg, pid_y_cfg, pid_z_cfg):
        # ---- 保存 config（供 start_uav 使用如 takeoff_alt 等） ----
        self.fc_cfg = fc_cfg
        # -----------------------------------------------------------

        # ---- 使用 RouterProxy 替换 DroneController ----
        self.proxy = RouterProxy(
            req_endpoint=fc_cfg.get("req_endpoint", "tcp://127.0.0.1:5555"),
            sub_endpoint=fc_cfg.get("sub_endpoint", "tcp://127.0.0.1:5556")
        )
        # ---------------------------------------------

        self.pid_y = PID(**pid_y_cfg)
        self.pid_z = PID(**pid_z_cfg)

        # 控制参数（按协议改为 20Hz，§4.1 要求 ≥20Hz）
        self.control_hz = 20.0
        self.control_interval = 1.0 / self.control_hz

        # 线程控制
        self.running = False
        self.control_thread = None
        self.current_target = (0.0, 0.0, 0.0)
        self.detections = []                   # 最新检测结果（线程间共享）
        self.frame_shape = (480, 640)          # 默认尺寸

        # 锁保护共享数据
        self.data_lock = threading.Lock()

        # ---- 防断流戳：主线程每次 update_detections 时更新 ----
        self._last_update_time = 0.0

    def start_uav(self):
        """连接 Router，执行标准起飞序列（协议 §3.3.1）

        流程：
          ① 启动 ZMQ 代理（开始收 STATE）
          ② 等待 Router 状态推送
          ③ 如需解锁 → ARM
          ④ 如需起飞 → WAYPOINT(TAKEOFF)
          ⑤ SETPOINT 预热（悬停）→ COMMAND(OFFBOARD)
          ⑥ 启动 20Hz 控制线程
        """
        # ---- ① 启动 ZMQ 代理 ----
        self.proxy.start()
        print("🔄 RouterProxy 已启动，等待 Router 状态推送...")

        # ---- ② 等待收到第一条 STATE ----
        state = None
        for i in range(50):  # 最多等 5 秒
            state = self.proxy.get_latest_state()
            if state is not None:
                print("✅ 收到 Router 状态推送")
                break
            time.sleep(0.1)

        if state is None:
            print("❌ 无法获取飞机状态，请检查 Router 是否运行，"
                  "且 SUB 端口 (tcp://127.0.0.1:5556) 可连通")
            return False

        armed = state["drone"]["armed"]
        mode = state["drone"]["mode"]
        alt_rel = state["drone"]["alt_rel"]
        age_us, _ = self.proxy.get_state_freshness(state)
        age_s = (age_us or 0) / 1_000_000

        print(f"📊 初始状态: mode={mode} | armed={armed} | "
              f"alt_rel={alt_rel:.1f}m | state_age={age_s:.2f}s")

        # ---- ③ 解锁（如未解锁） ----
        if not armed:
            if self._do_arm():
                armed = True
            else:
                return False

        # ---- ④ 起飞到目标高度（如高度不足） ----
        takeoff_alt = self.fc_cfg.get("takeoff_alt", 15.0)

        if alt_rel < takeoff_alt * 0.9:
            print(f"🛫 高度 {alt_rel:.1f}m < 目标 {takeoff_alt:.1f}m，执行起飞...")
            ok, ack = self.proxy.send_waypoint(
                action="TAKEOFF", alt=takeoff_alt,
                alt_frame="RELATIVE", speed=3.0
            )
            if not ok:
                print(f"⚠️ TAKEOFF 指令 ACK 失败: {ack}，尝试通过 SETPOINT 爬升...")
                # 降级方案：用 POSITION SETPOINT 爬升
                ok, ack = self.proxy.send_setpoint(
                    x=0, y=0, z=-takeoff_alt, yaw=0.0,
                    control_mode="POSITION"
                )
                if not ok:
                    print(f"❌ 爬升 SETPOINT 也失败: {ack}")
                    return False

            # 等待爬升
            for i in range(100):  # 最多等 10 秒
                state = self.proxy.get_latest_state()
                if state and state["drone"]["alt_rel"] >= takeoff_alt * 0.9:
                    alt_rel = state["drone"]["alt_rel"]
                    print(f"✅ 达到目标高度: {alt_rel:.1f}m")
                    break
                time.sleep(0.1)
            else:
                print(f"⚠️ 爬升等待超时，当前 alt_rel={state['drone']['alt_rel']:.1f}m，继续执行")

        # ---- ⑤ 预热 + 切换 OFFBOARD ----
        if mode != "OFFBOARD":
            # 获取最新高度用于预热悬停
            state = self.proxy.get_latest_state()
            current_alt = state["drone"]["alt_rel"] if state else takeoff_alt

            print(f"🔄 发送悬停 SETPOINT 预热 (alt={current_alt:.1f}m)...")
            ok, ack = self.proxy.send_setpoint(
                x=0, y=0, z=-current_alt, yaw=0.0,
                control_mode="POSITION"
            )
            if ok:
                time.sleep(0.2)  # 等流建立（协议 §3.3.1 ③→④）
            else:
                print(f"⚠️ 预热 SETPOINT ACK 异常: {ack}，尝试继续...")

            print("🔄 发送 OFFBOARD 指令...")
            ok, ack = self.proxy.send_command("OFFBOARD")
            if not ok:
                print(f"❌ OFFBOARD 指令失败: {ack}")
                return False

            # 等待模式切换
            for i in range(30):
                state = self.proxy.get_latest_state()
                if state and state["drone"]["mode"] == "OFFBOARD":
                    mode = "OFFBOARD"
                    print("✅ 已进入 OFFBOARD 模式")
                    break
                time.sleep(0.1)
            else:
                print(f"❌ OFFBOARD 模式切换超时，当前 mode={state['drone']['mode'] if state else 'N/A'}")
                return False

        # ---- ⑥ 启动独立控制线程 ----
        self.running = True
        self.control_thread = threading.Thread(
            target=self._control_loop, daemon=True, name="uav-control"
        )
        self.control_thread.start()
        print("✅ 控制线程已启动 (20Hz)")
        return True

    # ---- 辅助方法 ----

    def _do_arm(self):
        """执行解锁序列"""
        print("🔓 发送 ARM 指令...")
        ok, ack = self.proxy.send_command("ARM")
        if not ok:
            print(f"❌ ARM 指令发送失败: {ack}")
            return False
        # 等待飞控确认解锁
        for i in range(30):
            state = self.proxy.get_latest_state()
            if state and state["drone"]["armed"]:
                print("✅ 解锁成功")
                return True
            time.sleep(0.1)
        print("❌ 解锁超时（30 次轮询未确认 armed=true）")
        return False

    # ---- 主循环调用的接口 ----

    def update_detections(self, detections, frame_shape):
        """由主循环调用，更新最新检测结果和图像尺寸

        Args:
            detections:  postprocess 后的检测列表 [[x1,y1,x2,y2,conf,cls_id], ...]
            frame_shape: 原始帧形状 (height, width)
        """
        with self.data_lock:
            self.detections = detections
            self.frame_shape = frame_shape
            self._last_update_time = time.time()

    # ---- 内部控制线程 ----

    def _control_loop(self):
        """独立线程：以 20Hz 发送 SETPOINT，保证 OFFBOARD 保活

        无论是否有目标，每次迭代都发送 VELOCITY SETPOINT。
        无目标时速度全零（悬停刹车），满足协议 ≥20Hz 要求。
        """
        while self.running:
            start = time.time()

            # 获取最新的检测结果（加锁）
            with self.data_lock:
                dets = self.detections.copy()
                h, w = self.frame_shape[:2]
                last_update = self._last_update_time

            # ---- 防断流检测：超过 500ms 无更新则强制零速 ----
            # 覆盖主线程卡死、摄像头断流等场景
            if time.time() - last_update > 0.5:
                if len(dets) > 0:
                    # 只在首次过期时打印，避免刷屏
                    pass  # 保留静默刹车
                dets = []       # 清空检测 → PID 复位 → 零速
                self.pid_y.reset()
                self.pid_z.reset()
            # ------------------------------------------------

            # 计算速度指令
            vx, vy, vz = self._compute_velocity(dets, w, h)

            # 通过 ZMQ 发送 VELOCITY SETPOINT（协议 §3.1 速度控制）
            # 无论是否有目标，都持续发送以满足 20Hz 保活要求
            ok, ack = self.proxy.send_setpoint(
                vx=vx, vy=vy, vz=vz,
                yaw_rate=0.0,
                control_mode="VELOCITY"
            )
            if not ok and self.running:
                # ACK 失败记录日志，但不中断循环
                # recv 超时或格式错误由 _send_req 内部恢复 REQ socket
                pass

            # 精确控制循环频率（20Hz → 每 50ms）
            elapsed = time.time() - start
            sleep_time = self.control_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _compute_velocity(self, detections, frame_w, frame_h):
        """根据检测结果计算机体速度 (vx, vy, vz)

        与原始逻辑相同：
          - 无目标 → PID 复位 → 返回零速（悬停/刹车）
          - 有目标 → 取置信度最高的 → 偏差归一化 → PID 输出

        Returns:
            (vx, vy, vz): NED 坐标系速度 (m/s)
        """
        if not detections:
            # 无目标：悬停，并重置 PID 积分
            self.pid_y.reset()
            self.pid_z.reset()
            return 0.0, 0.0, 0.0

        # 取置信度最高的目标
        best = max(detections, key=lambda x: x[4])
        x1, y1, x2, y2, _conf, _cls_id = best
        target_x = (x1 + x2) / 2
        target_y = (y1 + y2) / 2
        img_center_x = frame_w / 2
        img_center_y = frame_h / 2

        err_x = (target_x - img_center_x) / img_center_x
        err_y = (target_y - img_center_y) / img_center_y

        vy = self.pid_y.update(err_x)   # 左右移动
        vz = self.pid_z.update(err_y)   # 上下移动（NED 中正为下）
        vx = 0.0                        # 前后移动

        return vx, vy, vz

    def stop(self):
        """停止控制线程，发送刹车指令，关闭 ZMQ 代理"""
        self.running = False
        if self.control_thread:
            self.control_thread.join(timeout=2.0)

        # 发送零速刹车指令
        print("🛑 发送刹车指令...")
        self.proxy.send_setpoint(
            vx=0.0, vy=0.0, vz=0.0,
            yaw_rate=0.0,
            control_mode="VELOCITY"
        )
        time.sleep(0.2)  # 确保指令发出

        self.proxy.close()
        print("控制已停止，ZMQ 代理已关闭")


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

    # 启动无人机连接（RouterProxy 启动 + 起飞序列）
    if not controller.start_uav():
        print("❌ 飞控启动失败，退出")
        sys.exit(1)

    print("🚀 NPU 精准推理 [ZMQ Router 协议版] 主循环启动...")
    loop_count = 0

    try:
        while True:
            ret, frame = streamer.read_frame()
            if not ret:
                time.sleep(0.005)
                continue

            loop_count += 1
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
            controller.update_detections(detections, orig_shape)

            # 4. 界面渲染
            frame = detector.draw_boxes(frame, detections)
            cv2.putText(frame, f"NPU Pure FPS: {fps_pure:.1f}", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # 5. 推送至后台异步图传队列
            streamer.push_to_stream(frame)

            # 6. 周期性打印 Router 状态（每 100 帧）
            if loop_count % 100 == 0:
                state = controller.proxy.get_latest_state()
                if state:
                    d = state["drone"]
                    age_us, _ = controller.proxy.get_state_freshness(state)
                    age_ms = (age_us or 0) / 1000
                    print(f"📡 [Router] mode={d['mode']} armed={d['armed']} "
                          f"alt={d['alt_rel']:.1f}m spd={d['ground_speed']:.1f}m/s "
                          f"batt={d['battery']:.0f}% state_age={age_ms:.0f}ms",
                          end="" if loop_count % 500 != 0 else "\n")

    except KeyboardInterrupt:
        print("\n👋 接收到终止信号，安全退出中...")
    finally:
        controller.stop()
        streamer.release()
        print("程序运行结束。")


if __name__ == "__main__":
    main()
