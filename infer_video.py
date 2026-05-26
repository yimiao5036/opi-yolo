import os
import cv2
import numpy as np
import time
from ais_bench.infer.interface import InferSession

# ==================== 配置区域 ====================
MODEL_PATH = "./om/yolo26n-balloon.om"  # 你的 YOLO26 模型路径
INPUT_VIDEO = "./test_video.mp4"  # 输入的 24 帧测试视频
OUTPUT_VIDEO = "./output_result.mp4"  # 检测完成后保存的新视频路径
CONF_THRESHOLD = 0.25  # 置信度阈值


# ==================================================

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


def draw_boxes(image, detections):
    for det in detections:
        x1, y1, x2, y2, conf, cls_id = det
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        # 画绿色的目标检测框
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"Target:{cls_id} {conf:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image


if __name__ == "__main__":
    yolo_infer = YOLO26UAVInfer(MODEL_PATH)

    # 1. 打开输入视频
    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        print(f"错误：无法打开输入视频 {INPUT_VIDEO}")
        exit()

    # 读取输入视频的各种属性，用于配置保存视频的写入器
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    img_center_x, img_center_y = frame_w / 2, frame_h / 2

    # 2. 【核心点】初始化视频保存器 (VideoWriter)
    # mp4v 是最通用、在 Linux 下不需要额外配置库的编码格式
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, video_fps, (frame_w, frame_h))

    print(f"开始处理视频并渲染可视化结果...")
    print(f"原视频属性: {frame_w}x{frame_h} @ {video_fps:.2f} FPS")
    print(f"处理后的视频将实时保存到: {OUTPUT_VIDEO}")

    fps_start_time = time.time()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # NPU 执行推理
        input_tensor = yolo_infer.preprocess_frame(frame)
        raw_result = yolo_infer.infer(input_tensor)
        detections = yolo_infer.postprocess(raw_result, conf_threshold=CONF_THRESHOLD)

        # 如果检测到目标，计算偏差并画可视化指示线
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

        # 渲染框和实时运行 FPS
        frame = draw_boxes(frame, detections)

        end_time = time.time()
        curr_fps = 1 / (end_time - fps_start_time) if (end_time - fps_start_time) > 0 else video_fps
        fps_start_time = end_time
        cv2.putText(frame, f"NPU Process-FPS: {curr_fps:.1f}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 3. 【核心点】把这一帧画好的画面写入新视频文件
        video_writer.write(frame)

        if frame_count % 24 == 0:
            print(f"已处理并写入 {frame_count} 帧...")

    # 4. 释放资源，这一步至关重要！不 release 视频文件会损坏打不开
    cap.release()
    video_writer.release()
    print(f"🎉 处理完毕！完整的检测结果视频已成功保存至: {OUTPUT_VIDEO}")