from ultralytics import YOLO
import torch
import torchvision.models as tv_models
from flask import current_app
import os


class ModelManager:
    def __init__(self):
        self._models = {}  # 缓存已加载的模型实例

    def load_model(self, model_key, model_info):
        """根据配置加载模型（只加载一次）"""
        if model_key in self._models:
            return self._models[model_key]

        model_type = model_info['type']
        model_path = model_info['path']

        # 确保路径是绝对路径
        if not os.path.isabs(model_path):
            base_dir = current_app.config['BASE_DIR']
            model_path = os.path.join(base_dir, model_path)

        print(f"Loading model from: {model_path}")  # 调试信息

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        if model_type == 'yolo':
            model = YOLO(model_path)
        elif model_type == 'torchvision':
            # 示例：加载 torchvision 预训练模型（需根据实际调整）
            model = tv_models.resnet50(pretrained=False)
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
            model.eval()
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        self._models[model_key] = model
        print(f"Model loaded successfully: {model_key}")
        return model

    def get_model(self, category, model_name):
        """根据功能类别和模型名称获取模型实例"""
        model_key = f"{category}:{model_name}"

        # 获取模型配置
        models_config = current_app.config['MODELS']
        if category not in models_config or model_name not in models_config[category]:
            available_models = list(models_config.get(category, {}).keys())
            raise ValueError(
                f"Model {model_name} not found in category {category}. Available models: {available_models}")

        model_info = models_config[category][model_name]
        return self.load_model(model_key, model_info)


# 创建全局管理器实例
model_manager = ModelManager()
