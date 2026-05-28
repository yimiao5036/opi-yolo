import os
import cv2
import numpy as np
import time
import threading
import queue
import socket
from ais_bench.infer.interface import InferSession

# ==================== 配置区域 ======================
MODEL_PATH = "./om/yolo26n-balloon.om"          # YOLO26 模型路径
INPUT_VIDEO = "./test_video.mp4"                # 输入的测试视频（或 0 代表摄像头）
CONF_THRESHOLD = 0.25                           # 置信度阈值

# -------------- 无线图传网络配置 --------------
GROUND_STATION_IP = "192.168.31.239"            # 地面站的局域网 IP
UDP_PORT = 9999
JPEG_QUALITY = 75                               # JPEG 压缩质量 (0-100)
# ==================================================

# 限制队列大小，发不完就直接丢帧，保证飞控算法实时性
tx_queue = queue.Queue(maxsize=2)


def udp_stream_sender_thread():
    """ 📡 后台图传专用线程 """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_address = (GROUND_STATION_IP, UDP_PORT)
    print(f"📡 图传后台线程已启动...")

    while True:
        try:
            frame_to_send = tx_queue.get()
            if frame_to_send is None:
                break

            # 压缩为 JPEG 字节流
            result, img_encode = cv2.imencode('.jpg', frame_to_send, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if not result:
                continue

            data = img_encode.tobytes()
            if len(data) > 65000:  # 严格防丢包/报错限制
                continue

            sock.sendto(data, server_address)
        except Exception as e:
            print(f"❌ 图传线程异常: {e}")
            time.sleep(0.1)

    sock.close()


class YOLO26UAVInfer:
    def __init__(self, model_path, device_id=0):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"找不到模型文件: {model_path}")
        self.session = InferSession(device_id, model_path)
        self.input_width = 640
        self.input_height = 640

    def letterbox(self, img, new_shape=(640, 640), color=(114, 114, 114)):
        shape = img.shape[:2]  # 原始图的高宽 [height, width]
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

    def postprocess(self, outputs, ratio, dwdh, orig_shape):
        out = outputs[0]
        detections = []
        for i in range(out.shape[1]):
            box = out[0, i, :4]
            conf = out[0, i, 4]
            cls_id = int(out[0, i, 5])
            if conf > CONF_THRESHOLD:
                x1, y1, x2, y2 = box
                # 严格基于原图比例还原坐标
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


def draw_boxes(img, detections):
    for det in detections:
        x1, y1, x2, y2, conf, cls_id = det
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        label = f"Target: {conf:.2f}"
        cv2.putText(img, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return img


def main():
    detector = YOLO26UAVInfer(MODEL_PATH)
    cap = cv2.VideoCapture(INPUT_VIDEO)
    from control import UAVController

    uav = UAVController()

    if not cap.isOpened():
        print("❌ 无法打开视频源")
        return

    # 启动后台无线电图传线程
    sender_thread = threading.Thread(target=udp_stream_sender_thread, daemon=True)
    sender_thread.start()

    print("🚀 NPU 精准推理主循环启动...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 完全保留读取到的原图大小
        orig_shape = frame.shape[:2]
        img_center_x = orig_shape[1] / 2
        img_center_y = orig_shape[0] / 2

        # 1. 严格使用原图进行标准无畸变 letterbox 图像预处理
        input_tensor, ratio, dwdh = detector.letterbox(frame)
        input_tensor = input_tensor.astype(np.float32) / 255.0
        input_tensor = np.transpose(input_tensor, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0)
        input_tensor = np.ascontiguousarray(input_tensor)

        t_start = time.time()
        # 2. NPU 硬件推理
        outputs = detector.session.infer([input_tensor])
        # 3. 坐标反求（恢复到大图的分辨率，保证百分之百精准）
        detections = detector.postprocess(outputs, ratio, dwdh, orig_shape)
        t_end = time.time()

        fps_pure = 1.0 / (t_end - t_start)

        # 4. 寻找置信度最高的目标并计算 MAVLink 飞控误差
        if len(detections) > 0:
            best_det = max(detections, key=lambda x: x[4])
            x1, y1, x2, y2, conf, cls_id = best_det
            target_x = (x1 + x2) / 2
            target_y = (y1 + y2) / 2

            err_x = (target_x - img_center_x) / img_center_x
            err_y = (target_y - img_center_y) / img_center_y

            # 画中心对准线
            cv2.line(frame, (int(img_center_x), int(img_center_y)), (int(target_x), int(target_y)), (255, 0, 0), 2)
            cv2.putText(frame, f"MAVLink Out -> Err_X: {err_x:+.3f} Err_Y: {err_y:+.3f}", (20, 70),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            # ================ MAVLink 动作调用 =================
            uav.update_control(has_target=True, err_x=err_x, err_y=err_y)
            # =================================================
        else:
            cv2.putText(frame, "MAVLink Out -> Target LOST (HOVER)", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 255), 2)
            # ================ MAVLink 丢失保护 =================
            uav.update_control(has_target=False)
            # =================================================

        # 5. 在原始分辨率大图上，渲染精准目标框
        frame = draw_boxes(frame, detections)
        cv2.putText(frame, f"NPU Pure FPS: {fps_pure:.1f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # 只有在进入发送队列的前一秒，才将画好完美框的图 resize 为轻量化分辨率
        send_frame = cv2.resize(frame, (640, 480))

        # 6. 非阻塞式尝试推入图传队列
        try:
            tx_queue.put_nowait(send_frame)
        except queue.Full:
            pass  # 队列满则丢弃该帧，绝不积压延迟

    cap.release()
    tx_queue.put(None)
    print("程序运行结束。")


if __name__ == "__main__":
    main()