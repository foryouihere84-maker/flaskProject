# utils/model_downloader.py
import os
import requests
from tqdm import tqdm
import torch
from ultralytics import YOLO


def download_yolo_models(base_dir):
    """下载YOLO预训练模型"""
    models_dir = os.path.join(base_dir, 'models', 'detection')
    os.makedirs(models_dir, exist_ok=True)

    # YOLO模型下载列表
    yolo_models = {
        'yolov8n': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt',
        'yolov8s': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt',
        'yolov8m': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt'
    }

    downloaded_models = []

    for model_name, url in yolo_models.items():
        model_path = os.path.join(models_dir, f'{model_name}.pt')

        if os.path.exists(model_path):
            print(f"✓ 模型 {model_name} 已存在")
            downloaded_models.append(model_name)
            continue

        print(f"正在下载 {model_name}...")
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))

            with open(model_path, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=model_name) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))

            # 验证模型完整性
            try:
                model = YOLO(model_path)
                print(f"✓ {model_name} 下载并验证成功")
                downloaded_models.append(model_name)
            except Exception as e:
                print(f"✗ {model_name} 验证失败: {str(e)}")
                os.remove(model_path)

        except Exception as e:
            print(f"✗ {model_name} 下载失败: {str(e)}")

    return downloaded_models


def verify_detection_models(base_dir):
    """验证检测模型的完整性和可用性"""
    models_dir = os.path.join(base_dir, 'models', 'detection')
    if not os.path.exists(models_dir):
        return []

    available_models = []

    for model_file in os.listdir(models_dir):
        if model_file.endswith('.pt'):
            model_path = os.path.join(models_dir, model_file)
            model_name = os.path.splitext(model_file)[0]

            try:
                model = YOLO(model_path)
                dummy_input = torch.zeros(1, 3, 640, 640)
                with torch.no_grad():
                    _ = model(dummy_input, verbose=False)

                print(f"✓ 模型 {model_name} 验证通过")
                available_models.append(model_name)

            except Exception as e:
                print(f"✗ 模型 {model_name} 验证失败: {str(e)}")

    return available_models
