import cv2
import numpy as np

# 导入昇腾核心推理库（板子自带）
from acllite_model import AclLiteModel
from acllite_resource import AclLiteResource

def preprocess(img_path, model_w=640, model_h=640):
    """图片预处理：缩放、通道转换、归一化"""
    orig_img = cv2.imread(img_path)
    h, w, _ = orig_img.shape
    
    # 1. 缩放到模型要求的 640x640
    img = cv2.resize(orig_img, (model_w, model_h))
    # 2. BGR 转 RGB (YOLO 训练时通常是 RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 3. 归一化并调整维度为 [1, 3, 640, 640]
    img = img.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1) # HWC -> CHW
    img = np.expand_dims(img, axis=0) # CHW -> NCHW
    
    # 必须将其转换为连续的内存，方便 NPU 读取
    return np.ascontiguousarray(img), orig_img

def main():
    # 1. 初始化昇腾 NPU 资源
    acl_resource = AclLiteResource()
    acl_resource.init()

    # 2. 加载你的 YOLO26 .om 模型
    model_path = "./yolo26_8t.om"
    model = AclLiteModel(model_path)

    # 3. 图像预处理
    input_data, orig_img = preprocess("./test.jpg")

    # 4. 送入 NPU 进行硬件加速推理
    # execute 接收一个列表，里面是模型的输入 Tensor
    result_list = model.execute([input_data])

    # 5. 后处理（针对 YOLO26 的端到端特性）
    # YOLO26 输出的直接是最终预测框，result_list[0] 形状通常是 [1, 300, 6] 或类似
    # 6个维度的含义一般为：[x_min, y_min, x_max, y_max, score, class_id]
    predictions = result_list[0][0] 

    CONF_THRESHOLD = 0.25 # 设定置信度阈值
    orig_h, orig_w = orig_img.shape[:2]

    box_count = 0
    for pred in predictions:
        # 取出置信度和类别
        score = pred[4]
        class_id = int(pred[5])
        
        # 过滤掉低置信度的框
        if score > CONF_THRESHOLD:
            box_count += 1
            # 取出缩放后的坐标 [0~640]
            x1, y1, x2, y2 = pred[0:4]
            
            # 将坐标还原到原图尺寸
            x1 = int(x1 * orig_w / 640)
            y1 = int(y1 * orig_h / 640)
            x2 = int(x2 * orig_w / 640)
            y2 = int(y2 * orig_h / 640)
            
            # 在原图上画框和标签
            cv2.rectangle(orig_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(orig_img, f"Class {class_id}: {score:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    print(f"推理完成，共检测到 {box_count} 个目标。")
    
    # 6. 保存检测结果图
    cv2.imwrite("./result.jpg", orig_img)

if __name__ == "__main__":
    main()