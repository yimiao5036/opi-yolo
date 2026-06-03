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

# ================== 导入基础控制与状态机模块 =====================
from drone_controller.base_control import DroneController
from drone_controller.pid_counter import PID
from drone_controller.mission_manager import MissionManager, MissionState, FailsafeTriggered
# ===================================================================

class VideoStreaming:
    """ 📡 视频流采集与后台图传模块 """

    def __init__(self, video_cfg, net_cfg):
        self.video_source = video_cfg.get("video_source", 0)
        self.jpeg_quality = video_cfg.get("jpeg_quality", 75)
        self.gs_ip = net_cfg.get("ground_station_ip", "127.0.0.1")
        self.udp_port = net_cfg.get("udp_port", 9999)

        self.cap = cv2.VideoCapture(self.video_source)
        self.tx_queue = queue.Queue(maxsize=2)
        self.is_running = True

        if not self.cap.isOpened():
            raise RuntimeError(f"❌ 无法打开视频源: {self.video_source}")

        self.sender_thread = threading.Thread(target=self._udp_stream_sender_worker, daemon=True)
        self.sender_thread.start()

    def read_frame(self):
        return self.cap.read()

    def push_frame(self, frame):
        if not self.tx_queue.full():
            self.tx_queue.put(frame.copy())

    def _udp_stream_sender_worker(self):
        udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        dest_addr = (self.gs_ip, self.udp_port)
        print(f"📡 [图传后台] UDP 推流服务已启动，目标地面站: {self.gs_ip}:{self.udp_port}")

        while self.is_running:
            try:
                frame = self.tx_queue.get(timeout=1.0)
                ret, jpeg_buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
                if not ret:
                    continue

                data = jpeg_buf.tobytes()
                packet_size = 60000
                total_size = len(data)

                for i in range(0, total_size, packet_size):
                    udp_socket.sendto(data[i:i + packet_size], dest_addr)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"⚠️ [图传异常] 发送失败: {e}")
        udp_socket.close()

    def release(self):
        self.is_running = False
        if self.cap.isOpened():
            self.cap.release()


class YOLO26UAVInfer:
    """ 🧠 昇腾 NPU YOLOv8 极致推理模块 """

    def __init__(self, model_cfg):
        self.model_path = model_cfg.get("model_path", "./om/yolo26n-balloon.om")
        self.conf_threshold = model_cfg.get("conf_threshold", 0.25)
        device_id = model_cfg.get("device_id", 0)

        print(f"🧠 [NPU 初始化] 正在载入昇腾 OM 模型: {self.model_path} ...")
        self.session = InferSession(device_id, self.model_path)

    def preprocess(self, img, img_size=640):
        h, w = img.shape[:2]
        r = min(img_size / h, img_size / w)
        unpad_h, unpad_w = int(round(h * r)), int(round(w * r))

        dw, dh = img_size - unpad_w, img_size - unpad_h
        dw /= 2
        dh /= 2

        if (w, h) != (unpad_w, unpad_h):
            img_resized = cv2.resize(img, (unpad_w, unpad_h), interpolation=cv2.INTER_LINEAR)
        else:
            img_resized = img

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                        value=(114, 114, 114))

        img_in = img_padded.transpose((2, 0, 1))[::-1]  # BGR to RGB
        img_in = np.ascontiguousarray(img_in).astype(np.float32) / 255.0
        return img_in, r, (dw, dh)

    def postprocess(self, outputs, ratio, dwdh, orig_shape):
        boxout = outputs[0][0]
        box_num = outputs[1][0]

        detections = []
        for i in range(box_num):
            box = boxout[i]
            score = box[4]
            cls_id = int(box[5])

            if score < self.conf_threshold:
                continue

            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
            x1 -= dwdh[0]
            x2 -= dwdh[0]
            y1 -= dwdh[1]
            y2 -= dwdh[1]

            x1, x2 = x1 / ratio, x2 / ratio
            y1, y2 = y1 / ratio, y2 / ratio

            x1 = max(0, min(x1, orig_shape[1]))
            y1 = max(0, min(y1, orig_shape[0]))
            x2 = max(0, min(x2, orig_shape[1]))
            y2 = max(0, min(y2, orig_shape[0]))

            detections.append({
                "box": [int(x1), int(y1), int(x2), int(y2)],
                "score": float(score),
                "class": cls_id
            })
        return detections

    def draw_boxes(self, frame, detections):
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Target: {det['score']:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return frame


class UAVControlLoop:
    """ ✈️ 融合型无人机控制闭环中心（深度适配状态机） """

    def __init__(self, flight_cfg, pid_y_cfg, pid_z_cfg):
        self.uav = DroneController(
            connection_string=flight_cfg.get("connection_string", "/dev/ttyUSB0"),
            baud=flight_cfg.get("baud_rate", 57600)
        )
        self.pid_y = PID(**pid_y_cfg)
        self.pid_z = PID(**pid_z_cfg)

    def start_uav(self):
        print("⚡ [底层通信] 正在建立与飞控的 MAVLink 链路...")
        if self.uav.connect():
            print("✅ [底层通信] 飞控监听守护线程就绪。")
            return True
        print("❌ [底层通信] 串口独占失败或未连上。")
        return False

    def process_control(self, detections, orig_shape, frame, mission_state):
        """
        根据当前状态机状态，决定是否发送闭环 PID 控制速度
        """
        h, w = orig_shape
        center_x_target, center_y_target = w / 2.0, h / 2.0

        # 绘制画面正中央的十字瞄准准心
        cv2.drawMarker(frame, (int(center_x_target), int(center_y_target)), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)

        # 判定是否包含有效检测目标
        if len(detections) > 0:
            best_det = max(detections, key=lambda x: x["score"])
            x1, y1, x2, y2 = best_det["box"]

            # 计算目标的像素中心点
            obj_x = (x1 + x2) / 2.0
            obj_y = (y1 + y2) / 2.0

            # 绘制目标质点追踪轨迹
            cv2.circle(frame, (int(obj_x), int(obj_y)), 5, (255, 0, 0), -1)
            cv2.line(frame, (int(center_x_target), int(center_y_target)), (int(obj_x), int(obj_y)), (255, 255, 0), 2)

            # 归一化百分比误差计算值
            err_x = (obj_x - center_x_target) / center_x_target
            err_y = (obj_y - center_y_target) / center_y_target

            # 核心控制逻辑决策：只有当状态机真正进入 VISUAL_TRACKING 状态，香橙派才拥有物理速度控制权！
            if mission_state == MissionState.VISUAL_TRACKING:
                vy_cmd = self.pid_y.update(err_x)
                vz_cmd = self.pid_z.update(err_y)

                # 下发机体系速度：前房保持0（悬停对准），vy左右平移，vz上下爬升
                self.uav.send_body_velocity(vx=0.0, vy=vy_cmd, vz=vz_cmd, yaw_rate=0.0)
        else:
            # 目标丢失兜底保护：如果在追踪状态下目标跟丢了，立即刹车原地悬停，并清除积分项
            if mission_state == MissionState.VISUAL_TRACKING:
                self.uav.send_body_velocity(0, 0, 0, 0)
                self.pid_y.reset()
                self.pid_z.reset()


def load_config(config_path="./config.json"):
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    # 模拟外部定义的自动化航点列表（NED坐标系：北, 东, 地。Z 轴向上为负，所以高度 1.5 米设为 -1.5）
    test_waypoints = [
        {'x': 1.0, 'y': 0.0, 'z': -1.5, 'yaw': 0},
        {'x': 2.0, 'y': 1.5, 'z': -1.5, 'yaw': 45},
        {'x': 0.0, 'y': 2.0, 'z': -1.5, 'yaw': 90}
    ]

    cfg = load_config()

    # 实例化子系统
    streamer = VideoStreaming(cfg["video"], cfg["network"])
    detector = YOLO26UAVInfer(cfg["model"])
    controller = UAVControlLoop(cfg["flight_control"], cfg["pid_y"], cfg["pid_z"])

    # 启动无人机串口连接
    if not controller.start_uav():
        sys.exit("❌ 飞控连接失败，无法继续业务")

    # 实例化复合状态机业务管理器，配置不发现目标时巡点悬停检索时间为 8 秒，飞完直接原地降落不返航
    mission = MissionManager(
        drone=controller.uav,
        waypoints=test_waypoints,
        target_altitude=1.5,
        arrival_radius=0.3,
        hold_duration=8.0,
        return_to_home=False
    )

    print("🚀 NPU 精准推理 [航点巡航与视觉追踪融合版] 主业务循环全面启动...")

    try:
        while True:
            ret, frame = streamer.read_frame()
            if not ret:
                print("❌ 未能读取到摄像头数据，退出中...")
                break

            orig_shape = frame.shape[:2]

            # 1. 图像预处理与硬件 NPU 高速推理
            input_tensor, ratio, dwdh = detector.preprocess(frame)
            outputs = detector.session.infer([input_tensor])

            # 2. 坐标反求与后处理
            detections = detector.postprocess(outputs, ratio, dwdh, orig_shape)

            # 判断当前帧是否捕捉到了目标
            target_detected = len(detections) > 0

            # 3. 驱动高层任务管理器状态机（传入视觉反馈）
            try:
                mission.update(target_detected=target_detected)
            except FailsafeTriggered:
                # 触发人肉拦截，跳出死循环，优雅关闭程序
                break

            # 4. 驱动底层视觉闭环速度计算（只有状态机进入 VISUAL_TRACKING 时才会真正发送速度指令）
            controller.process_control(detections, orig_shape, frame, mission.state)

            # 5. 地面站视频帧渲染绘制
            frame = detector.draw_boxes(frame, detections)

            # 在图传画面上实时叠加当前系统状态机状态
            status_text = f"State: {mission.state.name} | WP: {mission.current_wp_index + 1}/{len(test_waypoints)}"
            cv2.putText(frame, status_text, (20, 40), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 255), 2)

            # 推送视频到后台队列传输至 PC
            streamer.push_frame(frame)

            # 限制主循环高频速度（30Hz 左右），防止高频过度占用 CPU
            time.sleep(0.033)

    except KeyboardInterrupt:
        print("\n👋 接收到终端键盘中断指令，正在安全退出...")
    finally:
        # 优雅释放
        streamer.release()
        controller.uav.close()
        print("🏁 系统资源已安全回收，程序退出。")