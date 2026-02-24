import cv2
import os
from flask import current_app
import traceback


def process_detection(image_path, model):
    """YOLO 目标检测处理"""
    try:
        print(f"Processing image: {image_path}")

        # 读取图片
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图片文件: {image_path}")

        print(f"Image shape: {img.shape}")

        # 执行检测
        print("Running model inference...")
        results = model(img)
        result = results[0]
        print("Model inference completed")

        detections = []

        # 处理检测结果
        if hasattr(result, 'boxes') and result.boxes is not None:
            print("Processing bounding boxes...")
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()

            print(f"Found {len(boxes)} detections")

            for i, (box, conf, cls) in enumerate(zip(boxes, confs, classes)):
                x1, y1, x2, y2 = map(int, box)
                class_name = model.names[int(cls)] if hasattr(model, 'names') else f"Class {int(cls)}"

                label = f"{class_name}: {conf:.2f}"
                print(f"Detection {i}: {label} at [{x1}, {y1}, {x2}, {y2}]")

                # 绘制边界框
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                detections.append({
                    'class': class_name,
                    'confidence': float(conf),
                    'bbox': [x1, y1, x2, y2]
                })
        else:
            print("No bounding boxes detected")

        # 保存结果图片
        result_filename = 'result_' + os.path.basename(image_path)
        result_path = os.path.join(current_app.config['RESULT_FOLDER'], result_filename)

        print(f"Saving result to: {result_path}")
        success = cv2.imwrite(result_path, img)
        if not success:
            raise ValueError(f"无法保存结果图片到: {result_path}")

        print(f"Detection processing completed. Found {len(detections)} objects.")
        return detections, result_filename

    except Exception as e:
        print(f"Error in process_detection: {str(e)}")
        print(traceback.format_exc())
        raise
