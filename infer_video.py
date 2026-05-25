import os
import cv2
import numpy as np
import time
from ais_bench.infer.interface import InferSession

# ==================== 模拟测试硬编码配置 ====================
MODEL_PATH = "./yolo26_8t.om"  # 你的 YOLO26 模型路径
VIDEO_PATH = "./drone_test.mp4"  # 用于模拟相机的测试视频文件
CONF_THRESHOLD = 0.25  # 置信度阈值


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
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
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

    def infer(self, img_input):
        return self.session.infer([img_input])

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


if __name__ == "__main__":
    print("====== 启动香橙派无人机检测算法模拟测试 ======")
    yolo_infer = YOLO26UAVInfer(MODEL_PATH)

    # 用视频文件模拟机载相机输入
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"错误：无法打开模拟视频文件 {VIDEO_PATH}")
        exit()

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    img_center_x, img_center_y = frame_w / 2, frame_h / 2

    print(f"成功加载视频，分辨率: {frame_w}x{frame_h} | 正在进行 NPU 循环推理...")
    fps_start_time = time.time()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("视频播放结束或无法读取。")
            break

        frame_count += 1

        # 1. 预处理与 NPU 推理
        input_tensor = yolo_infer.preprocess_frame(frame)
        raw_result = yolo_infer.infer(input_tensor)
        detections = yolo_infer.postprocess(raw_result, conf_threshold=CONF_THRESHOLD)

        # 2. 模拟数据下发接口（核心优化点）
        if len(detections) > 0:
            best_det = max(detections, key=lambda x: x[4])  # 追踪置信度最高的目标
            x1, y1, x2, y2, conf, cls_id = best_det
            target_x, target_y = (x1 + x2) / 2, (y1 + y2) / 2

            # 计算标准化像素偏差（-1.0 到 1.0）
            err_x = (target_x - img_center_x) / img_center_x
            err_y = (target_y - img_center_y) / img_center_y

            # 模拟发送给飞控
            print(
                f"[Frame {frame_count:04d}] 检测到目标(类别:{cls_id}/置信度:{conf:.2f}) -> 模拟下发飞控控制量 err_x: {err_x:+.4f}, err_y: {err_y:+.4f}")
        else:
            print(f"[Frame {frame_count:04d}] 未检测到目标 -> 告知飞控保持悬停对准...")

        # 3. 计算端到端真实 FPS（包含预处理、推理和后处理）
        if frame_count % 30 == 0:
            end_time = time.time()
            avg_fps = 30 / (end_time - fps_start_time)
            print(f"----------------> 当前香橙派端到端平均处理速度: {avg_fps:.1f} FPS")
            fps_start_time = time.time()

    cap.release()