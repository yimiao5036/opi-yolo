import os
import cv2
import numpy as np
import time
import threading
import queue
import socket
import json
import sys
import logging

from mpmath.math2 import NAN

# ================== 昇腾 NPU SDK（保护性导入） =====================
# 开发机可能无昇腾环境，try-except 确保 import 错误仅告警不崩溃。
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
from drone_controller.target_tracker import TargetTracker
# ===========================================================


class VideoStreaming:
    """ 📡 视频流采集与后台图传模块（支持 RTSP 自动重连） """

    def __init__(self, video_cfg, net_cfg):
        self.logger = logging.getLogger("VideoStream")
        self.video_source = video_cfg.get("video_source", 0)
        self.jpeg_quality = video_cfg.get("jpeg_quality", 75)
        self.gs_ip = net_cfg.get("ground_station_ip", "127.0.0.1")
        self.udp_port = net_cfg.get("udp_port", 9999)

        # RTSP 特殊参数
        self.is_rtsp = isinstance(self.video_source, str) and self.video_source.startswith("rtsp://")
        # FFMPEG RTSP 传输协议统一设为 TCP（只设一次，避免每轮重试重复设置）
        os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp'
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
            # _open_camera() 内部自带带退避的无限重试，阻塞直到成功或被停止
            if self.cap is None or not self.cap.isOpened():
                self._open_camera()
                if self.cap is None or not self.cap.isOpened():
                    continue
            ret, frame = self.cap.read()
            if not ret:
                # 读取失败，关闭并标记需重连
                self.logger.warning("RTSP 读取失败，准备重连...")
                self.cap.release()
                self.cap = None
                continue

            # 清空旧帧，只保留最新一帧
            dropped = 0
            while not self.frame_buffer.empty():
                try:
                    self.frame_buffer.get_nowait()
                    dropped += 1
                except queue.Empty:
                    break
            if dropped > 0:
                self.logger.debug("丢弃旧帧 %d 帧（缓冲区刷新）", dropped)
            self.frame_buffer.put(frame)

        if self.cap is not None:
            self.cap.release()

    def _interruptible_sleep(self, duration, interval=1.0):
        """
        分段睡眠，每 interval 秒检查一次 self.is_running。
        替代 time.sleep(duration)，确保线程停止时秒级退出。

        Args:
            duration:  总睡眠时间（秒）
            interval:  每段睡眠时长（秒），默认 1 秒
        """
        end_time = time.time() + duration
        while time.time() < end_time and self.is_running:
            remaining = min(interval, end_time - time.time())
            if remaining <= 0:
                break
            time.sleep(remaining)

    def _open_camera(self):
        """
        打开摄像头或 RTSP 流，带自动退避重试机制

        【重试策略】
          1. 首轮正常尝试（INFO 日志，不打印 Warn）
          2. 每轮最多 5 次重试，每次间隔 1 秒，释放旧资源后再试
          3. 5 次全部失败 → 进入 30 秒冷却期（分段 sleep，可秒级响应停止信号）
          4. 无限外层 while self.is_running 循环
             → 硬件恢复后自动重连，手动停止时立即退出

        【日志规范】
          - 首次成功打开         → INFO
          - 每轮重试开始         → WARN（标明轮次）
          - 每次重试失败         → WARN（标明第几次/共5次）
          - 进入 30 秒冷却期     → ERROR
          - 重连成功后           → INFO（带 ✅ 标记）
        """
        RETRIES_PER_ROUND = 5
        RETRY_INTERVAL = 1.0      # 小循环重试间隔（秒）
        COOLDOWN = 30.0           # 大循环冷却期（秒）

        first_attempt = True       # 首轮特殊处理：不打印 Warn
        round_num = 0

        while self.is_running:
            round_num += 1

            if not first_attempt:
                self.logger.warning("摄像头重连第 %d 轮开始（共 %d 次尝试）...",
                                    round_num, RETRIES_PER_ROUND)

            for attempt in range(1, RETRIES_PER_ROUND + 1):
                if not self.is_running:
                    return

                # ---- 每次重试前释放旧资源，防止描述符泄漏 ----
                if self.cap is not None:
                    try:
                        self.cap.release()
                    except Exception:
                        pass
                    self.cap = None

                try:
                    self.logger.info("正在连接视频源: %s (尝试 %d/%d)",
                                     self.video_source, attempt, RETRIES_PER_ROUND)
                    self.cap = cv2.VideoCapture(self.video_source, cv2.CAP_FFMPEG)

                    if self.is_rtsp:
                        # 设置 RTSP 缓冲区大小（只保留 1 帧，降低延迟）
                        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        try:
                            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('H', '2', '6', '4'))
                        except Exception:
                            pass  # 编码设置失败不影响连接
                        # if "?" not in self.video_source:
                        #     self.video_source += "?tcp"
                    else:
                        # USB 摄像头也可设置缓冲区
                        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                    if self.cap.isOpened():
                        if first_attempt:
                            self.logger.info("视频源连接成功: %s", self.video_source)
                        else:
                            self.logger.info("✅ 摄像头重连成功: %s", self.video_source)
                        return   # 成功！退出 _open_camera
                    else:
                        self.cap.release()
                        self.cap = None
                        self.logger.warning("摄像头打开失败 (尝试 %d/%d)",
                                            attempt, RETRIES_PER_ROUND)

                except Exception as e:
                    if self.cap is not None:
                        try:
                            self.cap.release()
                        except Exception:
                            pass
                        self.cap = None
                    self.logger.warning("摄像头打开异常: %s (尝试 %d/%d)",
                                        e, attempt, RETRIES_PER_ROUND)

                # 小循环内重试间隔（分段 sleep，可响应停止信号）
                if attempt < RETRIES_PER_ROUND and self.is_running:
                    self._interruptible_sleep(RETRY_INTERVAL)

            first_attempt = False

            # ---- 5 次全部失败 → 进入冷却期 ----
            if self.is_running:
                self.logger.error("摄像头打开失败，进入 %.0f 秒冷却期...", COOLDOWN)
                self._interruptible_sleep(COOLDOWN)

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
        """ 原图传发送线程（异常降噪：首次报错后静默，恢复时输出日志） """
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        server_address = (self.gs_ip, self.udp_port)
        self.logger.info("图传后台线程已启动，目标地面站 -> %s:%s",
                         self.gs_ip, self.udp_port)
        send_count = 0
        tx_error_reported = False   # 标记当前是否处于异常状态，避免刷屏

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
                    self.logger.debug("图像数据过大 (%d bytes), 跳过发送", len(data))
                    continue
                sock.sendto(data, server_address)

                # ---- 发送成功：若之前处于异常状态，输出恢复日志 ----
                if tx_error_reported:
                    tx_error_reported = False
                    self.logger.info("✅ 图传已恢复: %s:%s", self.gs_ip, self.udp_port)

                send_count += 1
                if send_count % 100 == 0:
                    self.logger.info("图传统计: 已发送 %d 帧 UDP 数据包", send_count)

            except queue.Empty:
                continue
            except Exception as e:
                if not tx_error_reported:
                    tx_error_reported = True
                    self.logger.error("图传异常: %s (后续同类错误静默，恢复后输出日志)", e)
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
        self.logger = logging.getLogger("YOLO")
        model_path = model_cfg.get("model_path", "./om/yolo26n-balloon.om")
        device_id = model_cfg.get("device_id", 0)
        self.conf_threshold = model_cfg.get("conf_threshold", 0.25)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ 找不到模型文件: {model_path}")

        self.session = InferSession(device_id, model_path)
        self.input_width = 640
        self.input_height = 640
        self.logger.info("模型加载成功: path=%s input_size=%dx%d",
                         model_path, self.input_width, self.input_height)

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
        self.logger.debug("预处理: 输入形状 %s -> %s",
                          str(frame.shape), str(input_tensor.shape))
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
        if detections:
            best = max(detections, key=lambda d: d[4])
            self.logger.debug("检测到 %d 个目标, 最高置信度: %.3f 中心点: (%.0f, %.0f)",
                              len(detections), best[4],
                              (best[0] + best[2]) / 2, (best[1] + best[3]) / 2)
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

    def __init__(self, fc_cfg, pid_y_cfg, pid_z_cfg, proxy=None):
        self.logger = logging.getLogger("UAVControlLoop")

        # ---- 保存 config（供 start_uav 使用如 takeoff_alt 等） ----
        self.fc_cfg = fc_cfg
        # -----------------------------------------------------------

        # ---- 使用 RouterProxy（支持外部注入共享实例） ----
        if proxy is not None:
            self.proxy = proxy
            self._owns_proxy = False
            self.logger.info("使用外部注入的共享 RouterProxy")
        else:
            self.proxy = RouterProxy(
                req_endpoint=fc_cfg.get("req_endpoint", "tcp://127.0.0.1:5555"),
                sub_endpoint=fc_cfg.get("sub_endpoint", "tcp://127.0.0.1:5556")
            )
            self._owns_proxy = True
        # ---------------------------------------------

        self.pid_y = PID(**pid_y_cfg)
        self.pid_z = PID(**pid_z_cfg)

        # ---- 轻量级目标追踪过滤器 ----
        # 平滑 YOLO 检测、稳定 ID、coast 预测
        tracker_cfg = fc_cfg.get("target_tracker", {})
        self.tracker = TargetTracker(
            max_lost_frames=tracker_cfg.get("max_lost_frames", 8),
            max_association_dist=tracker_cfg.get("max_association_dist", 200.0),
            min_hits=tracker_cfg.get("min_hits", 3),
        )
        self._tracked_result = None   # 跟踪结果（锁保护）
        # 记录上一次跟踪状态，用于检测状态变化
        self._last_tracked = False
        # ------------------------------------------------

        # 控制参数（按协议改为 20Hz，§4.1 要求 ≥20Hz）
        self.control_hz = fc_cfg.get("control_hz", 20.0)
        self.control_interval = 1.0 / self.control_hz

        # 线程控制
        self.running = False
        self.control_thread = None
        self.current_target = (0.0, 0.0, 0.0)
        self.detections = []                   # 最新检测结果（仅用于渲染，线程间共享）
        self.frame_shape = (480, 640)          # 默认尺寸

        # 锁保护共享数据
        self.data_lock = threading.Lock()

        # ---- 防断流戳：主线程每次 update_detections 时更新 ----
        self._last_update_time = 0.0

        # ---- 安全刹车请求标志（主线程判丢 → 控制线程执行刹车） ----
        self._brake_requested = False

        # ---- 最近一次发送的速度指令（用于周期性日志打印） ----
        self._last_vx = 0.0
        self._last_vy = 0.0
        self._last_vz = 0.0

        # ---- 日志节流控制 ----
        self._log_throttle = {}  # key → 上次打印时间戳
        self._brake_last_logged = False  # 刹车是否已输出过日志

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
        self.logger.info("RouterProxy 已启动，等待 Router 状态推送...")

        # ---- ② 等待收到第一条 STATE ----
        state = None
        for i in range(50):  # 最多等 5 秒
            state = self.proxy.get_latest_state()
            if state is not None:
                self.logger.info("收到 Router 状态推送")
                break
            time.sleep(0.1)

        if state is None:
            self.logger.error("无法获取飞机状态，请检查 Router 是否运行，"
                              "且 SUB 端口 (tcp://127.0.0.1:5556) 可连通")
            return False

        armed = state["drone"]["armed"]
        mode = state["drone"]["mode"]
        alt_rel = state["drone"]["alt_rel"]
        age_us, _ = self.proxy.get_state_freshness(state)
        age_s = (age_us or 0) / 1_000_000

        self.logger.info("初始状态: mode=%s armed=%s alt_rel=%.1fm state_age=%.2fs",
                         mode, armed, alt_rel, age_s)

        # ---- ③ 解锁（如未解锁） ----
        if not armed:
            if self._do_arm():
                armed = True
            else:
                return False

        # ---- ④ 起飞到目标高度（如高度不足） ----
        takeoff_alt = self.fc_cfg.get("takeoff_alt", 5.0)

        if alt_rel < takeoff_alt * 0.9:
            self.logger.info("高度 %.1fm < 目标 %.1fm，执行起飞...", alt_rel, takeoff_alt)
            ok, ack = self.proxy.send_waypoint(
                lat = state["home"]["lat"],
                lon = state["home"]["lon"],
                alt = takeoff_alt,
                alt_frame="RELATIVE",
                speed=3.0,action="TAKEOFF",
                hover_time=5.0,yaw=state["drone"]["yaw"],
                acceptance_radius=0.5,
                is_last=False
            )
            if not ok:
                self.logger.warning("TAKEOFF 指令 ACK 失败: %s，尝试通过 SETPOINT 爬升...", ack)
                # 降级方案：用 POSITION SETPOINT 爬升
                ok, ack = self.proxy.send_setpoint(
                    x=0, y=0, z=-takeoff_alt, yaw=0.0,
                    control_mode="POSITION"
                )
                if not ok:
                    self.logger.error("爬升 SETPOINT 也失败: %s", ack)
                    return False

            # 等待爬升
            for i in range(100):  # 最多等 10 秒
                state = self.proxy.get_latest_state()
                if state and state["drone"]["alt_rel"] >= takeoff_alt * 0.9:
                    alt_rel = state["drone"]["alt_rel"]
                    self.logger.info("达到目标高度: %.1fm", alt_rel)
                    break
                time.sleep(0.1)
            else:
                alt_now = state["drone"]["alt_rel"] if state else 0
                self.logger.warning("爬升等待超时，当前 alt_rel=%.1fm，继续执行", alt_now)

        # ---- ⑤ 预热 + 切换 OFFBOARD ----
        if mode != "OFFBOARD":
            # 获取最新高度用于预热悬停
            state = self.proxy.get_latest_state()
            current_alt = state["drone"]["alt_rel"] if state else takeoff_alt

            self.logger.info("发送悬停 SETPOINT 预热 (alt=%.1fm)...", current_alt)
            ok, ack = self.proxy.send_setpoint(
                x=0, y=0, z=-current_alt, yaw=0.0,
                control_mode="POSITION"
            )
            if ok:
                time.sleep(0.2)  # 等流建立（协议 §3.3.1 ③→④）
            else:
                self.logger.warning("预热 SETPOINT ACK 异常: %s，尝试继续...", ack)

            self.logger.info("发送 OFFBOARD 指令...")
            ok, ack = self.proxy.send_command("OFFBOARD")
            if not ok:
                self.logger.error("OFFBOARD 指令失败: %s", ack)
                return False

            # 等待模式切换
            for i in range(30):
                state = self.proxy.get_latest_state()
                if state and state["drone"]["mode"] == "OFFBOARD":
                    mode = "OFFBOARD"
                    self.logger.info("已进入 OFFBOARD 模式")
                    break
                time.sleep(0.1)
            else:
                mode_str = state["drone"]["mode"] if state else "N/A"
                self.logger.error("OFFBOARD 模式切换超时，当前 mode=%s", mode_str)
                return False

        # ---- ⑥ 启动独立控制线程 ----
        self.running = True
        self.control_thread = threading.Thread(
            target=self._control_loop, daemon=True, name="uav-control"
        )
        self.control_thread.start()
        self.logger.info("控制线程已启动 (%.0fHz)", self.control_hz)
        return True

    # ---- 辅助方法 ----

    def _do_arm(self):
        """执行解锁序列"""
        self.logger.info("发送 ARM 指令...")
        ok, ack = self.proxy.send_command("ARM")
        if not ok:
            self.logger.error("ARM 指令发送失败: %s", ack)
            return False
        # 等待飞控确认解锁
        for i in range(30):
            state = self.proxy.get_latest_state()
            if state and state["drone"]["armed"]:
                self.logger.info("解锁成功")
                return True
            time.sleep(0.1)
        self.logger.error("解锁超时（30 次轮询未确认 armed=true）")
        return False

    # ---- 主循环调用的接口 ----

    def update_detections(self, detections, frame_shape):
        """由主循环调用，更新最新检测结果和图像尺寸

        处理流程：
          1. 用 YOLO 原始检测更新 TargetTracker（获取平滑/预测结果）
          2. Coast 滑行期间 → PID 积分清空（只靠 PD 稳住，不充积分）
          3. 判定彻底丢失 → 请求控制线程执行安全零速刹车
          4. 将结果写入线程安全共享区，供控制线程消费

        Args:
            detections:  postprocess 后的检测列表 [[x1,y1,x2,y2,conf,cls_id], ...]
            frame_shape: 原始帧形状 (height, width)

        Returns:
            (tracked: bool, result: dict)
            — tracked:  当前是否存在有效主目标
            — result:   跟踪器完整输出（含 center, box, is_predicted, lost_frames 等）
        """
        # ---- 跟踪过滤器：处理原始检测，输出平滑中心点 ----
        # 计算在锁外进行，避免阻塞控制线程（Kalman 计算 < 0.1ms 量级）
        result = self.tracker.update(detections, frame_shape)
        tracked = result.get("tracked", False)
        # ------------------------------------------------

        # ---- 检测跟踪状态变化 ----
        prev_tracked = self._last_tracked
        if tracked != prev_tracked:
            if tracked:
                self.logger.info("目标已锁定（丢失→锁定）")
            else:
                self.logger.info("目标已丢失（锁定→丢失）")
        self._last_tracked = tracked

        # ---- ① Coast 滑行期间：PID 积分清空 ----
        # 卡尔曼预测阶段目标可能漂移，积分项在此累积会导致
        # 目标重获瞬间积分暴冲（I-term windup）
        if tracked and result.get("is_predicted"):
            self.pid_y.reset_integral()
            self.pid_z.reset_integral()
            self.logger.debug("Coast 预测: 清空 PID 积分")

        # ---- ② 目标彻底丢失：请求控制线程执行刹车 ----
        # 注意：不在此处直接调 proxy.send_setpoint 以避免 REQ-REP
        # 并发冲突（控制线程也在使用同一 REQ socket），改用标志位
        # 由控制线程在下一 20Hz 周期拾取并发送刹车指令
        if not tracked:
            self.pid_y.reset_integral()
            self.pid_z.reset_integral()
            with self.data_lock:
                self._brake_requested = True
                # self.logger.info("目标丢失，请求刹车")

        with self.data_lock:
            self.detections = detections
            self.frame_shape = frame_shape
            self._last_update_time = time.time()
            self._tracked_result = result

        return tracked, result

    # ---- 内部控制线程 ----

    def _control_loop(self):
        """独立线程：以 20Hz 发送 SETPOINT，保证 OFFBOARD 保活

        控制流：
          1. 读取 TargetTracker 的平滑结果（免锁快照）
          2. 防断流检测（主线程 500ms 无更新 → 强制判丢）
          3. 跟踪器判丢 → PID 复位 → 零速刹车
          4. 跟踪器锁定 → 用平滑中心点计算 PID → VELOCITY SETPOINT

        无论是否有目标，每次迭代都发送 VELOCITY SETPOINT。
        无目标时速度全零（悬停刹车），满足协议 ≥20Hz 要求。
        """
        self.logger.info("控制循环线程已启动")
        while self.running:
            start = time.time()

            # ---- ① 读取跟踪结果和帧尺寸（加锁快照） ----
            with self.data_lock:
                tracked = self._tracked_result
                h, w = self.frame_shape[:2]
                last_update = self._last_update_time

            # ---- ② 防断流检测 ----
            # 主线程超过 500ms 未更新 → 强制判丢（覆盖摄像头断流/主线程卡死）
            if time.time() - last_update > 0.5:
                tracked = None
                # 节流：每 5 秒输出一次断流警告，不刷屏
                now = time.time()
                last_warn = self._log_throttle.get("antistale", 0)
                if now - last_warn > 5.0:
                    self._log_throttle["antistale"] = now
                    self.logger.warning("主线程超过 500ms 未更新检测结果，强制判丢")

            # ---- ③ 解析跟踪结果 → 速度指令（使用跟踪器输出的平滑中心点） ----
            vx, vy, vz = self._velocity_from_tracker(tracked, w, h)

            # ---- ③' 安全刹车请求：主线程判丢时紧急抱闸 ----
            # update_detections 检测到目标彻底丢失后设置此标志，
            # 控制线程在此拾取并执行一次全零 VELOCITY 刹车
            with self.data_lock:
                if self._brake_requested:
                    self._brake_requested = False
                    vx, vy, vz = 0.0, 0.0, 0.0
                    # 同时清空 PID（确保 PD 也归零）
                    self.pid_y.reset()
                    self.pid_z.reset()
                    # 仅第一次刹车输出日志，后续连续刹车静默
                    if not self._brake_last_logged:
                        self._brake_last_logged = True
                        self.logger.info("刹车指令已发送（控制线程执行）")
                else:
                    self._brake_last_logged = False

            # ---- 记录当前速度指令（供外部日志读取） ----
            self._last_vx, self._last_vy, self._last_vz = vx, vy, vz

            # ---- ④ 通过 ZMQ 发送 VELOCITY SETPOINT（协议 §3.1 速度控制） ----
            self.logger.debug("SETPOINT: control_mode=VELOCITY "
                              "vx=%.2f vy=%.2f vz=%.2f", vx, vy, vz)
            # ---- 防御性检查：飞控已切到 AUTO.LAND/AUTO.RTL 则不再发送 ----
            state = self.proxy.get_latest_state()
            if state:
                mode = state.get("drone", {}).get("mode", "")
                if mode in ("AUTO.LAND", "AUTO.RTL"):
                    # 静默跳过，不发送 SETPOINT
                    continue
            ok, ack = self.proxy.send_setpoint(
                vx=vx, vy=vy, vz=vz,
                yaw_rate=0.0,
                control_mode="VELOCITY"
            )
            if not ok and self.running:
                # ACK 失败记录日志，但不中断循环
                # recv 超时或格式错误由 _send_req 内部恢复 REQ socket
                self.logger.warning("SETPOINT ACK 失败: %s", ack)

            # ---- ⑤ 精确控制循环频率（20Hz → 每 50ms） ----
            elapsed = time.time() - start
            sleep_time = self.control_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _velocity_from_tracker(self, tracked, frame_w, frame_h):
        """
        根据跟踪器输出计算机体速度 (vx, vy, vz)

        这是控制线程使用的唯一速度计算路径。
        平滑后的中心点 → 归一化误差 → PID 输出。

        Args:
            tracked:  tracker.update() 返回的 result dict（或 None）
            frame_w:  画面宽度（像素）
            frame_h:  画面高度（像素）

        Returns:
            (vx, vy, vz): NED 坐标系速度 (m/s)
        """
        if tracked is None or not tracked.get("tracked"):
            self.pid_y.reset()
            self.pid_z.reset()
            return 0.0, 0.0, 0.0

        cx, cy = tracked["center"]
        img_cx = frame_w / 2.0
        img_cy = frame_h / 2.0

        err_x = (cx - img_cx) / img_cx
        err_y = (cy - img_cy) / img_cy

        vy = self.pid_y.update(err_x)   # 左右移动
        vz = self.pid_z.update(err_y)   # 上下移动（NED 中正为下）
        vx = 0.0                        # 前后维持不变

        # ---- 滑行预测期间：强制清空积分项 ----
        # 双重保险（update_detections 中也有一道）：
        # 确保 coast 帧只靠 PD 出力，积分项绝不充能
        if tracked.get("is_predicted"):
            self.pid_y.reset_integral()
            self.pid_z.reset_integral()

        return vx, vy, vz

    def stop(self):
        """停止控制线程，发送刹车指令，关闭 ZMQ 代理"""
        self.logger.info("正在停止控制线程...")
        self.running = False
        if self.control_thread:
            self.control_thread.join(timeout=2.0)

        # 发送零速刹车指令
        self.logger.info("发送刹车指令...")
        self.proxy.send_setpoint(
            vx=0.0, vy=0.0, vz=0.0,
            yaw_rate=0.0,
            control_mode="VELOCITY"
        )
        time.sleep(0.2)  # 确保指令发出

        self.tracker.reset()
        self._tracked_result = None

        if self._owns_proxy:
            self.proxy.close()
            self.logger.info("自有 RouterProxy 已关闭")
        else:
            self.logger.info("共享 RouterProxy 交由外部管理，跳过 close")
        self.logger.info("控制已停止，跟踪器已重置")


# ==================== 主程序入口 ======================
def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('./log/npu_infer.log'),  # 写入文件
            logging.StreamHandler()
        ]
    )
    main_logger = logging.getLogger("Main")

    # 默认加载同目录下的 config.json
    config_file = "config.json"

    # 允许在命令行指定 json 路径，例如: python3 infer_camera_modular.py config_outdoor.json
    if len(sys.argv) > 1:
        config_file = sys.argv[1]

    main_logger.info("正在从 %s 加载系统配置...", config_file)
    try:
        with open(config_file, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception as e:
        main_logger.error("读取配置文件失败: %s", e)
        sys.exit(1)

    # 实例化各子系统并注入配置
    streamer = VideoStreaming(cfg["video"], cfg["network"])
    detector = YOLO26UAVInfer(cfg["model"])
    controller = UAVControlLoop(cfg["flight_control"], cfg["pid_y"], cfg["pid_z"])

    # 启动无人机连接（RouterProxy 启动 + 起飞序列）
    if not controller.start_uav():
        main_logger.error("飞控启动失败")

    main_logger.info("NPU 精准推理 [ZMQ Router 协议版] 主循环启动...")
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
            tracked, track_res = controller.update_detections(detections, orig_shape)

            # ================================================================
            # 4. "所见即所得" 界面渲染
            #    控制与渲染的坐标绝对值统一：绘制追踪器的平滑框，不绘制原始 YOLO 框
            #    ┌──────────┬──────────┬──────────┐
            #    │ 状态     │ 框颜色   │ 标签     │
            #    ├──────────┼──────────┼──────────┤
            #    │ TRACK    │ 亮绿(0,255,0) │ ID:1 [MEAS] │
            #    │ COAST    │ 黄(0,255,255) │ ID:1 [COAST]│
            #    │ LOST     │ 不画框         │ 仅文字   │
            #    └──────────┴──────────┴──────────┘
            # ================================================================

            # 追踪器锁定 → 绘制平滑框
            if tracked and track_res.get("box") is not None:
                box = track_res.get("box")
                # 使用 np.clip 限制坐标，防止其画到屏幕外出现其他问题
                h_img, w_img = orig_shape
                bx1 = int(np.clip(box[0], 0, w_img-1))
                by1 = int(np.clip(box[1], 0, h_img-1))
                bx2 = int(np.clip(box[2], 0, w_img-1))
                by2 = int(np.clip(box[3], 0, h_img-1))
                is_coast = track_res.get("is_predicted", False)
                box_color = (0, 255, 255) if is_coast else (0, 255, 0)   # COAST=黄, MEAS=绿
                label = f"ID:{track_res.get('primary_id','?')}"
                label += " [COAST]" if is_coast else " [MEAS]"

                cv2.rectangle(frame, (int(bx1), int(by1)), (int(bx2), int(by2)),
                              box_color, 3)
                cv2.putText(frame, label, (int(bx1), int(by1) - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

            # 左上角状态面板
            cv2.putText(frame, f"NPU FPS: {fps_pure:.1f}", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            if track_res:
                status_str = ("✅" if tracked else "❌")
                lost = track_res.get("lost_frames", 0)
                n_act = track_res.get("n_active", 0)
                cv2.putText(frame, f"Tracker: {status_str} ID:{track_res.get('primary_id','-')} "
                            f"Lost:{lost}f Act:{n_act}",
                            (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            # 5. 推送至后台异步图传队列
            streamer.push_to_stream(frame)

            # 6. 周期性打印 Router + Tracker 状态（每 100 帧）
            if loop_count % 100 == 0:
                state = controller.proxy.get_latest_state()
                is_coast = track_res.get("is_predicted", False) if track_res else False
                lost_frames = track_res.get("lost_frames", 0) if track_res else 0
                n_active = track_res.get("n_active", 0) if track_res else 0
                primary_id = track_res.get("primary_id", "?") if track_res else "?"
                tracker_info = (f"TK:{'✅' if tracked else '❌'}"
                                f" ID:{primary_id}"
                                f" Lost:{lost_frames}f"
                                f" {'PRED' if is_coast else 'MEAS'}"
                                f" Act:{n_active}")
                if state:
                    d = state["drone"]
                    age_us, _ = controller.proxy.get_state_freshness(state)
                    age_ms = (age_us or 0) / 1000
                    main_logger.info(
                        "📡 Router: mode=%s armed=%s alt=%.1fm spd=%.1fm/s "
                        "batt=%.0f%% age=%.0fms v=(%.1f,%.1f,%.1f) %s",
                        d["mode"], d["armed"], d["alt_rel"],
                        d["ground_speed"], d["battery"], age_ms,
                        controller._last_vx, controller._last_vy,
                        controller._last_vz, tracker_info
                    )

    except KeyboardInterrupt:
        main_logger.info("接收到终止信号，安全退出中...")
        print()  # 换行，使输出整洁
    finally:
        controller.stop()
        streamer.release()
        main_logger.info("程序运行结束。")


if __name__ == "__main__":
    main()
