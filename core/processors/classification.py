import cv2
import os
from flask import current_app
import numpy as np


def process_classification(image_path, model):
    """图像分类处理"""
    img = cv2.imread(image_path)
    if img is None:
        return None, None, None

    # 预处理图像
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 使用模型进行预测
    results = model(img_rgb)

    # 获取预测结果
    if hasattr(results, 'probs'):  # 如果是分类模型
        probs = results.probs
        class_index = int(probs.top1)
        confidence = float(probs.top1conf)
        class_name = model.names[class_index] if hasattr(model, 'names') else f"Class {class_index}"
    else:
        # 备用方案：直接使用模型预测
        class_index = int(results[0].probs.top1) if hasattr(results[0].probs, 'top1') else 0
        confidence = float(results[0].probs.top1conf) if hasattr(results[0].probs, 'top1conf') else 0.0
        class_name = model.names[class_index] if hasattr(model, 'names') else f"Class {class_index}"

    # 在图像上添加标签
    label = f"{class_name}: {confidence:.2f}"
    cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 保存结果图片
    result_filename = 'result_' + os.path.basename(image_path)
    result_path = os.path.join(current_app.config['RESULT_FOLDER'], result_filename)
    cv2.imwrite(result_path, img)

    return class_name, confidence, result_filename
