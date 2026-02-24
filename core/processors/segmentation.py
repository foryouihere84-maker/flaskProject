import cv2
import os
from flask import current_app
import numpy as np


def process_segmentation(image_path, model):
    """图像分割处理"""
    img = cv2.imread(image_path)
    if img is None:
        return None, None

    # 使用模型进行分割
    results = model(img)
    result = results[0]

    # 创建分割掩码可视化
    if hasattr(result, 'masks') and result.masks is not None:
        masks = result.masks.data.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy() if result.boxes is not None else []

        # 为每个检测到的对象创建彩色掩码
        mask_overlay = np.zeros_like(img)
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]

        for i, mask in enumerate(masks):
            color = colors[i % len(colors)]
            mask_binary = (mask > 0.5).astype(np.uint8) * 255
            mask_colored = np.zeros_like(img)
            mask_colored[mask_binary > 0] = color

            # 将掩码叠加到原图上
            mask_overlay = cv2.addWeighted(mask_overlay, 1.0, mask_colored, 0.5, 0)

            # 添加标签
            if len(classes) > i and hasattr(model, 'names'):
                class_name = model.names[int(classes[i])]
                cv2.putText(img, class_name, (10, 30 + i * 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # 将掩码叠加到原图
        img = cv2.addWeighted(img, 1.0, mask_overlay, 0.7, 0)

    # 保存结果图片
    result_filename = 'result_' + os.path.basename(image_path)
    result_path = os.path.join(current_app.config['RESULT_FOLDER'], result_filename)
    cv2.imwrite(result_path, img)

    return result_filename, len(masks) if 'masks' in locals() else 0
