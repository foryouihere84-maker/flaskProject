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
        model_name = model_info.get('name', '')
        model_path = model_info.get('path', '')

        print(f"Loading model: {model_key}, type: {model_type}, name: {model_name}")  # 调试信息

        if model_type == 'yolo':
            # 确保路径是绝对路径
            if not os.path.isabs(model_path):
                base_dir = current_app.config['BASE_DIR']
                model_path = os.path.join(base_dir, model_path)
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            model = YOLO(model_path)
        elif model_type == 'torchvision':
            # 如果提供了模型路径，则从文件加载；否则使用预训练模型
            if model_path and os.path.exists(model_path):
                print(f"Loading model from file: {model_path}")
                # 从.pth文件加载模型
                if 'resnet18' in model_name.lower():
                    model = tv_models.resnet18()
                    model.load_state_dict(torch.load(model_path, map_location='cpu'))
                elif 'resnet50' in model_name.lower():
                    model = tv_models.resnet50()
                    model.load_state_dict(torch.load(model_path, map_location='cpu'))
                elif 'vgg16' in model_name.lower():
                    model = tv_models.vgg16()
                    model.load_state_dict(torch.load(model_path, map_location='cpu'))
                elif 'mobilenet' in model_name.lower():
                    model = tv_models.mobilenet_v2()
                    model.load_state_dict(torch.load(model_path, map_location='cpu'))
                else:
                    # 默认使用resnet50
                    model = tv_models.resnet50()
                    model.load_state_dict(torch.load(model_path, map_location='cpu'))
            else:
                # 使用torchvision预训练模型
                print(f"Using pretrained model: {model_name}")
                if model_name == 'resnet18':
                    model = tv_models.resnet18(pretrained=True)
                elif model_name == 'resnet34':
                    model = tv_models.resnet34(pretrained=True)
                elif model_name == 'resnet50':
                    model = tv_models.resnet50(pretrained=True)
                elif model_name == 'resnet101':
                    model = tv_models.resnet101(pretrained=True)
                elif model_name == 'resnet152':
                    model = tv_models.resnet152(pretrained=True)
                elif model_name == 'vgg16':
                    model = tv_models.vgg16(pretrained=True)
                elif model_name == 'vgg19':
                    model = tv_models.vgg19(pretrained=True)
                elif model_name == 'mobilenet_v2':
                    model = tv_models.mobilenet_v2(pretrained=True)
                elif model_name == 'efficientnet_b0':
                    model = tv_models.efficientnet_b0(pretrained=True)
                else:
                    # 默认使用resnet50
                    model = tv_models.resnet50(pretrained=True)
            
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
            
        # 如果模型不在配置中，尝试动态扫描并更新配置
        if category not in models_config or model_name not in models_config[category]:
            print(f"Model {model_name} not found in config, scanning for new models...")
            self._refresh_models_config()
                
            # 重新获取配置
            models_config = current_app.config['MODELS']
                
            # 再次检查
            if category not in models_config or model_name not in models_config[category]:
                available_models = list(models_config.get(category, {}).keys())
                raise ValueError(
                    f"Model {model_name} not found in category {category}. Available models: {available_models}")
            
        model_info = models_config[category][model_name]
        return self.load_model(model_key, model_info)
        
    def _refresh_models_config(self):
        """动态刷新模型配置"""
        from utils.file_utils import get_detection_models, get_adversarial_models, get_classification_models
            
        # 重新扫描模型文件
        detection_models = get_detection_models()
        adversarial_models = get_adversarial_models()
        classification_models = get_classification_models()
            
        # 重新构建检测模型配置
        detection_config = {}
        for model_name in detection_models:
            detection_config[model_name] = {
                'path': os.path.join('models', 'detection', f'{model_name}.pt'),
                'type': 'yolo'
            }
            
        # 重新构建对抗攻击模型配置
        adversarial_config = {}
        for model_name in adversarial_models:
            model_path = os.path.join('models', 'adversarial', f'{model_name}.pth')
            adversarial_config[model_name] = {
                'name': model_name,
                'path': model_path,
                'type': 'torchvision'
            }
            
        # 如果没有扫描到模型文件，使用预训练模型
        for model_name in classification_models:
            if model_name not in adversarial_config:
                adversarial_config[model_name] = {
                    'name': model_name,
                    'type': 'torchvision'
                }
            
        # 更新应用配置
        current_app.config['MODELS'] = {
            'detection': detection_config,
            'adversarial': adversarial_config
        }
            
        print(f"Models config refreshed. Detection models: {list(detection_config.keys())}")


# 创建全局管理器实例
model_manager = ModelManager()
