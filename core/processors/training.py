import os
import zipfile
import shutil
import time
from flask import current_app
from ultralytics import YOLO
import cv2
from pathlib import Path
import traceback


def process_model_training(model_name, dataset_path, training_params=None):
    """处理模型训练"""
    try:
        print(f"Starting training for model: {model_name}")
        print(f"Dataset path: {dataset_path}")
        
        # 1. 解压数据集
        dataset_dir = extract_dataset(dataset_path)
        print(f"Dataset extracted to: {dataset_dir}")
        
        # 2. 准备训练数据
        train_data_dir = prepare_training_data(dataset_dir)
        print(f"Training data prepared at: {train_data_dir}")
        
        # 3. 加载基础模型
        base_model_path = os.path.join(current_app.config['BASE_DIR'], 'models', 'detection', f'{model_name}.pt')
        if not os.path.exists(base_model_path):
            raise FileNotFoundError(f"Base model not found: {base_model_path}")
        
        model = YOLO(base_model_path)
        print(f"Base model loaded: {model_name}")
        
        # 4. 创建新的模型名称（原模型名+时间戳）
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        new_model_name = f"{model_name}_trained_{timestamp}"
        new_model_path = os.path.join(current_app.config['BASE_DIR'], 'models', 'detection', f"{new_model_name}.pt")
        
        # 5. 配置训练参数（使用用户自定义参数或默认值）
        if training_params is None:
            training_params = {}
            
        training_config = {
            'data': train_data_dir,  # 训练数据配置文件路径
            'epochs': int(training_params.get('epochs', 10)),           # 训练轮数
            'batch': int(training_params.get('batch', 16)),            # 批次大小
            'imgsz': int(training_params.get('imgsz', 640)),           # 图像尺寸
            'patience': int(training_params.get('patience', 3)),          # 早停耐心值
            'device': training_params.get('device', 'cpu'),        # 训练设备
            'exist_ok': True,       # 允许覆盖已有结果
            'project': os.path.join(current_app.config['BASE_DIR'], 'runs'),  # 训练结果保存路径
            'name': new_model_name, # 实验名称
            'save_period': -1,      # 不定期保存检查点
            'cache': training_params.get('cache', False),         # 是否缓存数据
            'optimizer': training_params.get('optimizer', 'SGD'),     # 优化器
            'lr0': float(training_params.get('lr0', 0.001)),           # 初始学习率
            'lrf': float(training_params.get('lrf', 0.01)),            # 最终学习率
            'weight_decay': float(training_params.get('weight_decay', 0.0005)), # 权重衰减
            'warmup_epochs': int(training_params.get('warmup_epochs', 1)),     # 预热轮数
            'freeze': [int(x) for x in training_params.get('freeze', '0,1,2,3,4,5,6,7,8,9').split(',')]  # 冻结层数
        }
        
        print("Starting model training...")
        print(f"Training config: {training_config}")
        
        # 6. 开始训练
        results = model.train(**training_config)
        
        # 7. 保存训练好的模型
        trained_model_path = os.path.join(
            current_app.config['BASE_DIR'], 
            'runs', 
            new_model_name, 
            'weights', 
            'best.pt'
        )
        
        if os.path.exists(trained_model_path):
            shutil.copy2(trained_model_path, new_model_path)
            print(f"Trained model saved to: {new_model_path}")
        else:
            # 如果没有best.pt，使用last.pt
            last_model_path = os.path.join(
                current_app.config['BASE_DIR'], 
                'runs', 
                new_model_name, 
                'weights', 
                'last.pt'
            )
            if os.path.exists(last_model_path):
                shutil.copy2(last_model_path, new_model_path)
                print(f"Last model saved to: {new_model_path}")
            else:
                raise FileNotFoundError("No trained model weights found")
        
        # 8. 清理临时文件
        cleanup_temp_files(dataset_dir, train_data_dir)
        
        # 9. 返回结果
        training_info = {
            'base_model': model_name,
            'new_model': new_model_name,
            'epochs_completed': training_config['epochs'],
            'training_time': '约几分钟',  # 实际时间取决于数据量和硬件
            'training_params': training_params  # 返回使用的训练参数
        }
        
        print(f"Training completed successfully!")
        print(f"New model: {new_model_name}")
        
        return {
            'new_model_path': new_model_path,
            'new_model_name': new_model_name,
            'training_info': training_info
        }
        
    except Exception as e:
        print(f"Model training process error: {str(e)}")
        print(traceback.format_exc())
        # 清理可能的临时文件
        try:
            if 'dataset_dir' in locals():
                cleanup_temp_files(dataset_dir, train_data_dir if 'train_data_dir' in locals() else None)
        except:
            pass
        raise


def extract_dataset(dataset_path):
    """解压数据集"""
    try:
        # 创建解压目录
        timestamp = int(time.time())
        extract_dir = os.path.join(current_app.config['UPLOAD_FOLDER'], f'dataset_extract_{timestamp}')
        os.makedirs(extract_dir, exist_ok=True)
        
        # 解压ZIP文件
        with zipfile.ZipFile(dataset_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        
        # 删除上传的压缩包
        os.remove(dataset_path)
        
        return extract_dir
        
    except Exception as e:
        print(f"Dataset extraction error: {str(e)}")
        raise


def prepare_training_data(dataset_dir):
    """准备训练数据"""
    try:
        # 查找图片文件
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        images = []
        
        for root, dirs, files in os.walk(dataset_dir):
            for file in files:
                if os.path.splitext(file)[1].lower() in image_extensions:
                    images.append(os.path.join(root, file))
        
        if not images:
            raise ValueError("数据集中未找到任何图片文件")
        
        print(f"Found {len(images)} images in dataset")
        
        # 创建YOLO格式的数据目录结构
        timestamp = int(time.time())
        train_data_dir = os.path.join(current_app.config['UPLOAD_FOLDER'], f'yolo_data_{timestamp}')
        images_dir = os.path.join(train_data_dir, 'images')
        labels_dir = os.path.join(train_data_dir, 'labels')
        
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)
        
        # 复制图片文件
        for i, img_path in enumerate(images):
            # 生成新文件名
            ext = os.path.splitext(img_path)[1]
            new_name = f"image_{i:06d}{ext}"
            new_path = os.path.join(images_dir, new_name)
            
            # 复制图片
            shutil.copy2(img_path, new_path)
            
            # 创建对应的空标签文件（如果没有的话）
            label_name = f"image_{i:06d}.txt"
            label_path = os.path.join(labels_dir, label_name)
            if not os.path.exists(label_path):
                with open(label_path, 'w') as f:
                    pass  # 创建空标签文件
        
        # 创建YOLO数据配置文件
        data_yaml_path = os.path.join(train_data_dir, 'data.yaml')
        with open(data_yaml_path, 'w', encoding='utf-8') as f:
            f.write(f"""path: {train_data_dir}
train: images
val: images

nc: 80
names: ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush']
""")
        
        return data_yaml_path
        
    except Exception as e:
        print(f"Training data preparation error: {str(e)}")
        raise


def cleanup_temp_files(dataset_dir, train_data_dir):
    """清理临时文件"""
    try:
        # 删除解压的数据集目录
        if dataset_dir and os.path.exists(dataset_dir):
            shutil.rmtree(dataset_dir)
            print(f"Cleaned up dataset directory: {dataset_dir}")
        
        # 删除训练数据目录
        if train_data_dir and os.path.exists(os.path.dirname(train_data_dir)):
            shutil.rmtree(os.path.dirname(train_data_dir))
            print(f"Cleaned up training data directory: {train_data_dir}")
            
    except Exception as e:
        print(f"Cleanup error: {str(e)}")
        # 不抛出异常，因为这是清理步骤