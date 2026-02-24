import os
from flask import Flask, render_template
from blueprints.detection import detection_bp
from blueprints.classification import classification_bp
from blueprints.segmentation import segmentation_bp
import config
from utils.file_utils import get_detection_models, get_classification_models, get_segmentation_models, \
    get_adversarial_models
from blueprints.adversarial import adversarial_bp

def init_models_config(app):
    """在应用上下文中初始化模型配置"""
    with app.app_context():
        # 动态获取模型列表
        detection_models = get_detection_models()
        classification_models = get_classification_models()
        segmentation_models = get_segmentation_models()
        adversarial_models = get_adversarial_models()  # 添加对抗攻击模型

        # 构建检测模型配置
        detection_config = {}
        for model_name in detection_models:
            detection_config[model_name] = {
                'path': os.path.join('models', 'detection', f'{model_name}.pt'),
                'type': 'yolo'
            }

        # 构建分类模型配置
        classification_config = {}
        for model_name in classification_models:
            classification_config[model_name] = {
                'path': os.path.join('models', 'classification', f'{model_name}.pth'),
                'type': 'torchvision'
            }

        # 构建分割模型配置
        segmentation_config = {}
        for model_name in segmentation_models:
            segmentation_config[model_name] = {
                'path': os.path.join('models', 'segmentation', f'{model_name}.pt'),
                'type': 'yolo'
            }

         # 构建对抗攻击模型配置
        adversarial_config = {}
        for model_name in adversarial_models:
            adversarial_config[model_name] = {
                'path': os.path.join('models', 'adversarial', f'{model_name}.pth'),
                'type': 'torchvision'  # 或其他合适的类型
            }

        # 更新应用配置
        app.config['MODELS'] = {
            'detection': detection_config,
            'classification': classification_config,
            'segmentation': segmentation_config,
            'adversarial': adversarial_config
        }

        # 设置默认模型
        app.config['DEFAULT_MODEL'] = detection_models[0] if detection_models else 'yolov8n'


def create_app(config_class='config.Config'):
    app = Flask(__name__)

    # 初始化配置
    app.config.from_object(config_class)

    # 在应用上下文中初始化模型配置
    init_models_config(app)

    # 确保上传和结果目录存在
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

    # 注册蓝图
    app.register_blueprint(detection_bp)
    app.register_blueprint(classification_bp)
    app.register_blueprint(segmentation_bp)
    app.register_blueprint(adversarial_bp)

    @app.route('/')
    def home():
        # 首页：展示功能入口
        return render_template('index.html')

    return app


if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5000)
