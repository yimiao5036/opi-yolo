import os
import cv2
import numpy as np
import time
from ais_bench.infer.interface import InferSession

# 引入 MAVLink 核心库
from pymavlink import mavutil

# ==================== 进阶硬编码配置区域 ====================
MODEL_PATH = "./om/yolo26n-balloon.om"  # YOLO26 模型路径
CONF_THRESHOLD = 0.25  # 置信度过滤阈值
CAMERA_INDEX = 0  # 机载相机索引

# MAVLink 串口配置 (请根据香橙派实际使用的串口设备和飞控波特率修改)
MAVLINK_PORT = "/dev/ttyAMA1"  # 香橙派引脚对应的串口设备，例如 ttyAMA1 或 ttyS1
MAVLINK_BAUD = 921600  # 推荐 921600 保证高频数据不卡顿

# 地面站推流配置
GCS_IP = "114.114.114.114"  # 替换为您地面站电脑/平板的局域网静态 IP 地址
GCS_UDP_PORT = 5600  # QGroundControl 默认的视频监听端口是 5600


# ============================================================

class YOLO26UAVInfer:
    def __init__(self, model_path, device_id=0):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"找不到模型文件: {model_path}")
        self.session = InferSession(device_id, model_path)
        self.input_width = 640
        self.input_height = 640

    def letterbox(self, img, new_shape=(640, 640), color=(114, 114, 114)):
        shape = img.shape[:2]
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
        dw = new_shape[1] - new_unpad[0]
        dh = new_shape[0] - new_unpad[1]
        left, top = dw // 2, dh // 2
        right, bottom = dw - left, dh - top

        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return img, r, (left, top)

    def preprocess_frame(self, img_bgr):
        self.original_height, self.original_width = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized, self.ratio, (self.pad_left, self.pad_top) = self.letterbox(
            img_rgb, (self.input_width, self.input_height)
        )
        img_input = (img_resized.astype(np.float32) / 255.0).astype(np.float32)
        img_input = np.ascontiguousarray(np.transpose(img_input, (2, 0, 1))[np.newaxis, :, :, :])
        return img_input

    def postprocess(self, infer_result, conf_threshold):
        detections = []
        output = infer_result[0] if isinstance(infer_result, (list, tuple)) else infer_result
        if output.ndim == 3: output = output[0]

        for det in output:
            confidence = det[4]
            if confidence < conf_threshold: continue

            x1_scale, y1_scale, x2_scale, y2_scale = det[0], det[1], det[2], det[3]
            cls_id = int(det[5])

            x1 = max(0, min((x1_scale - self.pad_left) / self.ratio, self.original_width))
            y1 = max(0, min((y1_scale - self.pad_top) / self.ratio, self.original_height))
            x2 = max(0, min((x2_scale - self.pad_left) / self.ratio, self.original_width))
            y2 = max(0, min((y2_scale - self.pad_top) / self.ratio, self.original_height))

            detections.append([x1, y1, x2, y2, confidence, cls_id])
        return detections


def draw_boxes(image, detections):
    for det in detections:
        x1, y1, x2, y2, conf, cls_id = det
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"Target:{cls_id} {conf:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image


if __name__ == "__main__":
    print("====== 正在初始化机载 AI 计算机系统 ======")

    # 1. 初始化 MAVLink 串口连接到 PX4
    print(f"正在建立与 PX4 飞控的 MAVLink 串口连接 ({MAVLINK_PORT})...")
    try:
        master = mavutil.mavlink_connection(MAVLINK_PORT, baud=MAVLINK_BAUD)
        # 等待飞控心跳，确保连接成功
        master.wait_heartbeat(timeout=5)
        print("与 PX4 飞控握手成功（收到心跳包）!")
    except Exception as e:
        print(f"警告：无法连接到飞控串口 ({e})。数据发送将被跳过。")
        master = None

    # 2. 初始化 YOLOv26 NPU 会话
    yolo_infer = YOLO26UAVInfer(MODEL_PATH)

    # 3. 初始化机载相机
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"致命错误：无法开启机载相机 index {CAMERA_INDEX}")
        exit()

    # 设定适合推流的相机分辨率
    frame_w, frame_h = 640, 480
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_h)

    # 4. 初始化地面站 UDP GStreamer 推流管道
    # 这一行命令会通过 H.264 编码，将视频转化为连续帧以 UDP 格式推往地面站 IP 端口
    gst_pipeline = f"appsrc ! videoconvert ! x264enc tune=zerolatency bitrate=800 speed-preset=ultrafast ! rtph264pay config-interval=1 pt=96 ! udpsink host={GCS_IP} port={GCS_UDP_PORT}"
    out_stream = cv2.VideoWriter(gst_pipeline, cv2.CAP_GSTREAMER, 0, 30.0, (frame_w, frame_h), True)
    if not out_stream.isOpened():
        print("警告：GStreamer 推流管道初始化失败，请检查香橙派是否安装了 gstreamer 组件。将退回到纯后台模式。")
    else:
        print(f"地面站推流已开启：目标 IP -> {GCS_IP}:{GCS_UDP_PORT}")

    print("系统启动成功，正在进入任务循环...")
    fps_start_time = 0

    while True:
        ret, frame = cap.read()
        if not ret: continue

        # 计算图像物理中心点
        img_center_x = frame_w / 2
        img_center_y = frame_h / 2

        # NPU 执行推理
        input_tensor = yolo_infer.preprocess_frame(frame)
        raw_result = yolo_infer.infer(input_tensor)
        detections = yolo_infer.postprocess(raw_result, conf_threshold=CONF_THRESHOLD)

        # 核心逻辑：如果检测到了目标，计算目标相对画面中心的物理/像素偏差
        if len(detections) > 0:
            # 假设默认追踪置信度最高的那一个目标
            best_det = max(detections, key=lambda x: x[4])
            x1, y1, x2, y2, conf, cls_id = best_det

            # 计算目标的中心像素坐标
            target_center_x = (x1 + x2) / 2
            target_center_y = (y1 + y2) / 2

            # 计算目标相对于画面正中心的视角绝对角偏差 (或粗暴计算像素比偏差)
            # 弧度视角偏差计算公式例：(target_center - img_center) / 焦距
            # 这里简化为标准化像素误差 (范围 -1.0 到 1.0) 方便 PX4 接收
            err_x = (target_center_x - img_center_x) / img_center_x
            err_y = (target_center_y - img_center_y) / img_center_y

            # 通过 MAVLink 串口发送给 PX4 飞控
            if master is not None:
                # 使用 LANDING_TARGET 消息（常用于自主追踪降落或悬停对准）
                master.mav.landing_target_send(
                    time_usec=int(time.time() * 1e6),  # 时间戳
                    target_num=0,  # 目标编号
                    frame=mavutil.mavlink.MAV_FRAME_BODY_NED,  # 相对机体坐标系
                    angle_x=err_x,  # X轴视角弧度/误差
                    angle_y=err_y,  # Y轴视角弧度/误差
                    distance=0.0,  # 距离 (如果不准可以填0，飞控会结合气压计计算)
                    size_x=0.0, size_y=0.0  # 目标物理大小
                )

            # 在图像上额外标出追踪器连线
            cv2.line(frame, (int(img_center_x), int(img_center_y)),
                     (int(target_center_x), int(target_center_y)), (255, 0, 0), 2)

        # 绘制检测框与渲染实时机载 FPS
        frame = draw_boxes(frame, detections)
        fps_end_time = time.time()
        time_diff = fps_end_time - fps_start_time
        fps = 1 / time_diff if time_diff > 0 else 0
        fps_start_time = fps_end_time
        cv2.putText(frame, f"Air-FPS: {fps:.1f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # 通过数传网络将带有框和 FPS 的画面推向地面站
        if out_stream.isOpened():
            out_stream.write(frame)

    cap.release()
    if out_stream.isOpened(): out_stream.release()