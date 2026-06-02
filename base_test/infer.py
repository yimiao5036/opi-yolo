import os
import cv2
import numpy as np
import argparse
from ais_bench.infer.interface import InferSession

class YOLO26Infer:
    def __init__(self, model_path, device_id=0):
        """
        初始化推理器
        :param model_path: .om 模型的本地路径
        :param device_id: NPU 设备ID
        """
        self.device_id = device_id
        self.model_path = model_path
        # 创建 ais_bench 推理会话
        self.session = InferSession(device_id, model_path)
        # 获取模型输入尺寸，用于后续预处理
        self.input_width = 640
        self.input_height = 640

    def letterbox(self, img, new_shape=(640, 640), color=(114, 114, 114)):
        shape = img.shape[:2]
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        # 计算缩放后的新尺寸（四舍五入）
        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
        # 计算需要填充的总宽度和高度
        dw = new_shape[1] - new_unpad[0]
        dh = new_shape[0] - new_unpad[1]
        # 分别计算左右/上下填充量（允许奇数不对称）
        left = dw // 2
        right = dw - left
        top = dh // 2
        bottom = dh - top

        # 缩放图片
        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

        # 添加填充（可能左右或上下不对称，但总尺寸正好是 new_shape）
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return img, r, (left, top)  # 返回填充的左上角偏移量

    def preprocess(self, image_path):
        """
        预处理图片，使其符合模型输入要求
        """
        # 使用OpenCV读取图片 (BGR格式)
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise FileNotFoundError(f"图片未找到: {image_path}")

        # 获取原始图片尺寸，用于后处理时的坐标转换
        self.original_height, self.original_width = img_bgr.shape[:2]

        # 将图片从BGR转换为RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # 调整图片尺寸并保持宽高比，进行Letterbox填充
        img_resized, self.ratio, (self.pad_left, self.pad_top) = self.letterbox(
            img_rgb, (self.input_width, self.input_height)
        )

        # 归一化：将像素值从 [0, 255] 转为 [0, 1]
        img_input = (img_resized.astype(np.float32) / 255.0).astype(np.float32)

        # 调整维度：HWC -> NCHW，并转换为连续数组
        img_input = np.ascontiguousarray(np.transpose(img_input, (2, 0, 1))[np.newaxis, :, :, :])

        return img_input

    def infer(self, img_input):
        """
        执行推理
        """
        # 使用会话的 infer 方法进行推理
        # infer 方法接收一个列表，包含输入张量，返回一个列表，包含输出张量
        result = self.session.infer([img_input])
        print(f"Input tensor shape: {img_input.shape}")
        print(f"Input tensor dtype: {img_input.dtype}")
        print(f"Input tensor size: {img_input.nbytes}")
        return result

    def postprocess(self, infer_result, conf_threshold=0.5):
        """
        对模型输出进行后处理
        :param infer_result: 模型原始输出，一个列表，通常第一个元素是输出张量
        :param conf_threshold: 置信度阈值
        :return: 包含检测结果的列表，每个元素为 [x1, y1, x2, y2, confidence, class_id]
        """
        detections = []
        # 获取输出张量，假设输出列表的第一个元素是输出数组
        if isinstance(infer_result, (list, tuple)):
            output = infer_result[0]
        else:
            output = infer_result

        # ========== 调试信息 ==========
        # print(f"[DEBUG] output shape: {output.shape}")
        # print(f"[DEBUG] output dtype: {output.dtype}")
        # print(f"[DEBUG] output min: {output.min()}, max: {output.max()}")
        # # 打印前5个检测框（如果形状是 [N,6] 或 [1,N,6]）
        # if output.ndim == 3:
        #     flat_output = output[0]
        # else:
        #     flat_output = output
        # print(f"[DEBUG] first 5 rows:\n{flat_output[:5]}")

        # 假设输出维度为 [batch, num_detections, 6]
        # 移除batch维度，并只取第一个batch的结果
        if output.ndim == 3:
            output = output[0]

        # 遍历所有检测框
        for det in output:
            confidence = det[4]
            if confidence < conf_threshold:
                continue

            # 获取原始输出坐标 (x1, y1, x2, y2) 和 类别ID
            x1_scale, y1_scale, x2_scale, y2_scale = det[0], det[1], det[2], det[3]
            cls_id = int(det[5])

            # 将坐标从模型输入尺寸映射回原始图片尺寸
            x1 = (x1_scale - self.pad_left) / self.ratio
            y1 = (y1_scale - self.pad_top) / self.ratio
            x2 = (x2_scale - self.pad_left) / self.ratio
            y2 = (y2_scale - self.pad_top) / self.ratio

            # 边界限制，防止超出原始图片范围
            x1 = max(0, min(x1, self.original_width))
            y1 = max(0, min(y1, self.original_height))
            x2 = max(0, min(x2, self.original_width))
            y2 = max(0, min(y2, self.original_height))

            detections.append([x1, y1, x2, y2, confidence, cls_id])

        return detections

def draw_boxes(image, detections, class_names=None):
    """
    在图片上绘制检测框和标签
    :param image: 原始图片数组 (BGR格式)
    :param detections: 后处理得到的结果列表
    :param class_names: 类别名称列表，默认使用类别ID
    """
    img_copy = image.copy()
    for det in detections:
        x1, y1, x2, y2, conf, cls_id = det
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # 绘制矩形框
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 构建标签文本
        label = f"Class: {cls_id}" if class_names is None else class_names[cls_id]
        label += f" {conf:.2f}"

        # 设置标签背景
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img_copy, (x1, y1 - label_h - 5), (x1 + label_w, y1), (0, 255, 0), -1)
        cv2.putText(img_copy, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return img_copy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv26 目标检测推理")
    parser.add_argument("--model", type=str, required=True, help="YOLOv26 OM模型文件路径")
    parser.add_argument("--image", type=str, required=True, help="输入图片路径")
    parser.add_argument("--conf", type=float, default=0.5, help="置信度阈值")
    args = parser.parse_args()

    # 初始化推理器
    yolo_infer = YOLO26Infer(args.model)

    # 图片预处理
    print("预处理图片...")
    input_tensor = yolo_infer.preprocess(args.image)

    # 执行推理
    print("执行推理...")
    raw_result = yolo_infer.infer(input_tensor)

    # 后处理
    print("后处理...")
    detections = yolo_infer.postprocess(raw_result, conf_threshold=args.conf)

    # 打印结果
    print(f"检测到 {len(detections)} 个目标:")
    for det in detections:
        print(f"类别ID: {int(det[5])}, 置信度: {det[4]:.4f}, 边界框: ({det[0]:.2f}, {det[1]:.2f}, {det[2]:.2f}, {det[3]:.2f})")

    # 可选：绘制结果并保存
    original_img = cv2.imread(args.image)
    result_img = draw_boxes(original_img, detections)
    output_path = "output_" + os.path.basename(args.image)
    cv2.imwrite(output_path, result_img)
    print(f"结果图片已保存至: {output_path}")