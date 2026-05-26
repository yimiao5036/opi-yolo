import os
import cv2
import numpy as np
import time
import threading
import queue
import socket
from ais_bench.infer.interface import InferSession

# ==================== 配置区域 ====================
MODEL_PATH = "./om/yolo26n-balloon.om"  # YOLO26 模型路径
INPUT_VIDEO = "./test_video.mp4"  # 输入的测试视频（首飞时可改成 0 代表机载 USB 相机）
CONF_THRESHOLD = 0.25  # 置信度阈值

# -------------- 无线图传网络配置 --------------
GROUND_STATION_IP = "192.168.31.239"  # 地面站的局域网 IP
UDP_PORT = 9999
# ==================================================

# 初始化一个图传专用的轻量级队列，最大长度为2。如果后台发得慢导致队列满了，主推理环会触发丢帧，绝对不卡死。
tx_queue = queue.Queue(maxsize=2)


def udp_stream_sender_thread():
    """ 📡 后台图传专用线程 """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_address = (GROUND_STATION_IP, UDP_PORT)
    print(f"📡 图传后台线程已启动，正在持续向地面站 [{GROUND_STATION_IP}:{UDP_PORT}] 推流...")

    while True:
        try:
            # 阻塞等待主线程丢进来的实时画框图像
            frame = tx_queue.get()

            # 1. 预处理：将图片缩放到 640x480（极大降低空中无线带宽压力和延迟）
            small_frame = cv2.resize(frame, (640, 480))

            # 2. 图像进行轻量级 JPEG 压缩（质量设为 60，画面清晰且计算速度极快）
            ret, jpeg_img = cv2.imencode('.jpg', small_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])

            if ret:
                data = np.array(jpeg_img).tobytes()
                # 3. 确保单个 UDP 数据包不超过缓冲区上限（~65KB）
                if len(data) < 65000:
                    sock.sendto(data, server_address)

            tx_queue.task_done()
        except Exception:
            # 即使无线网络抖动或地面站未开启，后台线程也静默承受，绝不连累无人机主系统
            pass


class YOLO26UAVInfer:
    def __init__(self, model_path, device_id=0):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"找不到模型文件: {model_path}")
        self.session = InferSession(device_id, model_path)
        self.input_width = 640
        self.input_height = 640

    def letterbox(self, img, new_shape=(640, 640), color=(114, 114, 114)) :
        shape = img.shape[:2]
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


def preprocess(self, img):
    img_canvas, ratio, (dw, dh) = self.letterbox(img, (self.input_width, self.input_height))
    img_in = img_canvas[:, :, ::-1].transpose(2, 0, 1)
    img_in = np.ascontiguousarray(img_in).astype(np.float32) / 255.0
    img_in = np.expand_dims(img_in, axis=0)
    return img_in, ratio, (dw, dh)


def infer(self, input_tensor):
    outputs = self.session.infer([input_tensor])
    return outputs


def postprocess(self, outputs, ratio, pad, orig_shape, conf_thres=0.25):
    det_out = outputs[0][0]
    valid_detections = []
    for i in range(det_out.shape[0]):
        row = det_out[i]
        conf = row[4]
        if conf < conf_thres:
            continue
        cls_id = int(row[5])
        cx, cy, w, h = row[0], row[1], row[2], row[3]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        dw, dh = pad
        x1 = (x1 - dw) / ratio
        y1 = (y1 - dh) / ratio
        x2 = (x2 - dw) / ratio
        y2 = (y2 - dh) / ratio
        x1 = max(0, min(orig_shape[1], x1))
        y1 = max(0, min(orig_shape[0], y1))
        x2 = max(0, min(orig_shape[1], x2))
        y2 = max(0, min(orig_shape[0], y2))
        valid_detections.append([x1, y1, x2, y2, conf, cls_id])
    return valid_detections


def draw_boxes(img, detections):
    img_copy = img.copy()
    for det in detections:
        x1, y1, x2, y2, conf, cls_id = det
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"Target: {conf:.2f}"
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img_copy, (x1, y1 - label_h - 5), (x1 + label_w, y1), (0, 255, 0), -1)
        cv2.putText(img_copy, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return img_copy


if __name__ == "__main__":
    # 0. 提前拉起后台异步无线图传线程
    stream_thread = threading.Thread(target=udp_stream_sender_thread, daemon=True)
    stream_thread.start()

    # 初始化推理会话
    yolo_infer = YOLO26UAVInfer(MODEL_PATH)

    # 视频输入源
    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        print(f"❌ 无法打开输入视频: {INPUT_VIDEO}")
        exit()

    print("🚀 NPU 推理主循环已启动！")

    while True:
        t_start = time.time()
        ret, frame = cap.read()
        if not ret:
            print("视频播放结束或未捕获到帧。")
            break

        orig_shape = frame.shape[:2]
        img_center_x = orig_shape[1] / 2
        img_center_y = orig_shape[0] / 2

        # 1. 预处理
        input_tensor, ratio, pad = yolo_infer.preprocess(frame)

        # 2. 执行昇腾 NPU 硬件推理
        outputs = yolo_infer.infer(input_tensor)

        # 3. 后处理
        detections = yolo_infer.postprocess(outputs, ratio, pad, orig_shape, conf_thres=CONF_THRESHOLD)

        # 4. 执行你的核心控制逻辑（寻找最显著的目标并计算偏差）
        if len(detections) > 0:
            best_det = max(detections, key=lambda x: x[4])
            x1, y1, x2, y2, conf, cls_id = best_det
            target_x = (x1 + x2) / 2
            target_y = (y1 + y2) / 2

            err_x = (target_x - img_center_x) / img_center_x
            err_y = (target_y - img_center_y) / img_center_y

            # 画一条从画面中心指向目标中心的蓝色对准线，直观展示控制走向
            cv2.line(frame, (int(img_center_x), int(img_center_y)),
                     (int(target_x), int(target_y)), (255, 0, 0), 2)

            # 把计算好的、原本要发给飞控的误差值实时写在视频上
            cv2.putText(frame, f"MAVLink Out -> Err_X: {err_x:+.3f} Err_Y: {err_y:+.3f}",
                        (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        else:
            cv2.putText(frame, "MAVLink Out -> Target LOST (HOVER)",
                        (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # 5. 渲染基础边界框
        frame = draw_boxes(frame, detections)

        # 6. 计算单帧总耗时和纯计算 FPS
        t_end = time.time()
        fps = 1.0 / (t_end - t_start)
        cv2.putText(frame, f"NPU Process-FPS: {fps:.1f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # ==================== 异步丢帧推流 ====================
        try:
            # block=False 意味着如果无线网络太慢队列满了，程序不会原地死等，而是抛出异常并直接走下一次循环
            tx_queue.put(frame, block=False)
        except queue.Full:
            pass  # 主动丢帧
        # ===================================================================

        # 打印调试信息到终端
        print(f"当前帧推理完成 | 纯计算主环帧率: {fps:.1f} FPS")

    cap.release()
    print("香橙派端任务运行完毕。")