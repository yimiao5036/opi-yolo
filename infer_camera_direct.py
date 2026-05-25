import os
import cv2
import numpy as np
import time
from ais_bench.infer.interface import InferSession

# ==================== 硬编码配置区域 ====================
MODEL_PATH = "./yolo26_8t.om"  # 你的 YOLO26 模型路径
CONF_THRESHOLD = 0.25  # 置信度过滤阈值 (YOLO26 推荐 0.25)
CAMERA_INDEX = 0  # 摄像头索引 (默认0，如果打不开可以尝试1或2)


# ========================================================

class YOLO26CameraInfer:
    def __init__(self, model_path, device_id=0):
        self.device_id = device_id
        self.model_path = model_path

        # 自动检查模型文件是否存在，防止硬编码路径写错导致程序崩溃
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"找不到硬编码的模型文件: {model_path}，请检查路径是否正确。")

        # 创建 ais_bench 推理会话
        self.session = InferSession(device_id, model_path)
        self.input_width = 640
        self.input_height = 640

    def letterbox(self, img, new_shape=(640, 640), color=(114, 114, 114)):
        shape = img.shape[:2]
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
        dw = new_shape[1] - new_unpad[0]
        dh = new_shape[0] - new_unpad[1]
        left = dw // 2
        right = dw - left
        top = dh // 2
        bottom = dh - top

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

    def infer(self, img_input):
        return self.session.infer([img_input])

    def postprocess(self, infer_result, conf_threshold):
        detections = []
        if isinstance(infer_result, (list, tuple)):
            output = infer_result[0]
        else:
            output = infer_result

        if output.ndim == 3:
            output = output[0]

        for det in output:
            confidence = det[4]
            if confidence < conf_threshold:
                continue

            x1_scale, y1_scale, x2_scale, y2_scale = det[0], det[1], det[2], det[3]
            cls_id = int(det[5])

            x1 = (x1_scale - self.pad_left) / self.ratio
            y1 = (y1_scale - self.pad_top) / self.ratio
            x2 = (x2_scale - self.pad_left) / self.ratio
            y2 = (y2_scale - self.pad_top) / self.ratio

            x1 = max(0, min(x1, self.original_width))
            y1 = max(0, min(y1, self.original_height))
            x2 = max(0, min(x2, self.original_width))
            y2 = max(0, min(y2, self.original_height))

            detections.append([x1, y1, x2, y2, confidence, cls_id])

        return detections


def draw_boxes(image, detections):
    for det in detections:
        x1, y1, x2, y2, conf, cls_id = det
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"Class: {cls_id} {conf:.2f}"

        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (x1, y1 - label_h - 5), (x1 + label_w, y1), (0, 255, 0), -1)
        cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return image


if __name__ == "__main__":
    print("正在初始化 YOLOv26 离线推理引擎...")
    # 传入硬编码的模型路径
    yolo_infer = YOLO26CameraInfer(MODEL_PATH)

    # 打开硬编码指定的摄像头
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"错误：无法打开摄像头索引 {CAMERA_INDEX}，请检查物理连接。")
        exit()

    # 尝试设置合适的分辨率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print(f"初始化成功！当前模型：{MODEL_PATH} | 过滤阈值：{CONF_THRESHOLD}")
    print("开始实时检测，在画面窗口按下 'q' 键可退出...")

    fps_start_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # 预处理与 NPU 推理
        input_tensor = yolo_infer.preprocess_frame(frame)
        raw_result = yolo_infer.infer(input_tensor)

        # 后处理（使用硬编码的置信度）
        detections = yolo_infer.postprocess(raw_result, conf_threshold=CONF_THRESHOLD)

        # 绘制结果
        frame = draw_boxes(frame, detections)

        # FPS 实时计算与显示
        fps_end_time = time.time()
        time_diff = fps_end_time - fps_start_time
        fps = 1 / time_diff if time_diff > 0 else 0
        fps_start_time = fps_end_time

        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        cv2.imshow("YOLOv26 Real-time Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("检测已结束。")