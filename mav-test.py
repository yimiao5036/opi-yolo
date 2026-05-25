import cv2
import math
import time
import serial
import numpy as np
from ais_bench.infer.interface import InferSession
from pymavlink import mavutil


class DroneDetector:
    def __init__(self, model_path, serial_port='/dev/ttyAMA0', baudrate=57600):
        self.session = InferSession(0, model_path)
        self.input_size = 640
        # 串口/MAVLink
        self.serial_port = serial_port
        self.baudrate = baudrate
        self.mav = None
        self.init_serial()

    def init_serial(self):
        """初始化串口（如果用于自定义协议）或 MAVLink 连接"""
        try:
            self.mav = mavutil.mavlink_connection(self.serial_port, baud=self.baudrate)
            print(f"MAVLink 已连接到 {self.serial_port} 波特率 {self.baudrate}")
        except Exception as e:
            print(f"串口打开失败: {e}")
            self.mav = None

    def letterbox(self, img, color=(114, 114, 114)):
        shape = img.shape[:2]
        r = min(self.input_size / shape[0], self.input_size / shape[1])
        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
        dw = self.input_size - new_unpad[0]
        dh = self.input_size - new_unpad[1]
        left = dw // 2
        right = dw - left
        top = dh // 2
        bottom = dh - top
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return img, r, (left, top)

    def preprocess(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized, self.ratio, (self.pad_left, self.pad_top) = self.letterbox(img_rgb)
        img_input = img_resized.astype(np.float32) / 255.0
        img_input = np.ascontiguousarray(np.transpose(img_input, (2, 0, 1))[np.newaxis, ...])
        return img_input

    def postprocess(self, output, conf_thresh=0.5):
        detections = []
        if output.ndim == 3:
            output = output[0]
        for det in output:
            raw_conf = det[4]
            conf = 1.0 / (1.0 + math.exp(-raw_conf))
            if conf < conf_thresh:
                continue
            x1, y1, x2, y2 = det[0], det[1], det[2], det[3]
            # 映射回原图坐标
            x1 = (x1 - self.pad_left) / self.ratio
            y1 = (y1 - self.pad_top) / self.ratio
            x2 = (x2 - self.pad_left) / self.ratio
            y2 = (y2 - self.pad_top) / self.ratio
            cls_id = int(det[5])
            detections.append({
                'bbox': (x1, y1, x2, y2),
                'confidence': conf,
                'class': cls_id,
                'center': ((x1 + x2) / 2, (y1 + y2) / 2)
            })
        return detections

    def send_mavlink_obstacle(self, angle_deg, distance_m):
        """发送 DISTANCE_SENSOR 消息，表示目标方向与距离"""
        if not self.mav:
            return
        self.mav.mav.distance_sensor_send(
            0,  # time_boot_ms
            0,  # min_distance cm
            10000,  # max_distance cm
            int(distance_m * 100),  # current_distance cm
            0,  # type (0=laser)
            0,  # id
            int(angle_deg / 360.0 * 65535),  # orientation (0~360° -> 0~65535)
            0  # covariance
        )

    def send_custom_serial(self, detections):
        """通过串口发送自定义格式的数据（例如 JSON）"""
        if not self.serial or not self.serial.is_open:
            return
        import json
        data = {
            'timestamp': time.time(),
            'detections': detections
        }
        self.serial.write((json.dumps(data) + '\n').encode())

    def detect_and_send(self, frame, send_method='mavlink', conf_thresh=0.5):
        """对单帧图像进行检测，并发送结果"""
        input_tensor = self.preprocess(frame)
        outputs = self.session.infer([input_tensor])
        detections = self.postprocess(outputs[0], conf_thresh)
        if not detections:
            return detections
        # 假设只发送置信度最高的那个目标
        best = detections[0]
        if send_method == 'mavlink':
            # 计算角度和距离（需要根据相机参数和实际尺寸估算）
            # 这里简单示例：根据目标在图像中的偏移量估算角度
            cx = best['center'][0]
            img_w = frame.shape[1]
            angle = (cx - img_w / 2) / (img_w / 2) * 45  # 水平视场角90°，偏移量映射为±45°
            # 距离估算：根据目标在图像中的高度（假设目标实际高度1m）
            # 需要相机焦距等，简化：使用固定值10米
            distance = 10.0
            self.send_mavlink_obstacle(angle, distance)
        elif send_method == 'custom':
            self.send_custom_serial(detections)
        return detections

    def run_on_camera(self, camera_id=0, conf_thresh=0.5, send_method='mavlink'):
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print("摄像头打开失败")
            return
        print("开始检测，按 q 退出")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            detections = self.detect_and_send(frame, send_method, conf_thresh)
            # 绘制检测框（调试）
            for det in detections:
                x1, y1, x2, y2 = map(int, det['bbox'])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{det['class']}:{det['confidence']:.2f}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.imshow("Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    def run_on_image(self, image_path, conf_thresh=0.5, send_method='mavlink'):
        frame = cv2.imread(image_path)
        if frame is None:
            print("图片读取失败")
            return
        detections = self.detect_and_send(frame, send_method, conf_thresh)
        print(f"检测到 {len(detections)} 个目标")
        for d in detections:
            print(d)
        # 显示结果
        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{det['class']}:{det['confidence']:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imshow("Detection", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    MODEL_PATH = "./om/yolo26n-drone.om"
    SERIAL_PORT = "/dev/ttyAMA0"  # 根据实际修改
    BAUD = 57600

    detector = DroneDetector(MODEL_PATH, SERIAL_PORT, BAUD)

    # 选择模式：摄像头或单张图片
    # detector.run_on_camera(camera_id=0, conf_thresh=0.5, send_method='mavlink')
    detector.run_on_image("test.jpg", conf_thresh=0.5, send_method='mavlink')