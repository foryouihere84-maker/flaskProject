from flask import Blueprint, request, render_template, current_app, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename
from core.processors.training import process_model_training
from utils.file_utils import allowed_file, get_detection_models
import os
import traceback
import time


training_bp = Blueprint('training', __name__, url_prefix='/training')


@training_bp.route('/', methods=['GET'])
def index():
    """模型训练页面"""
    try:
        # 获取当前可用的检测模型列表
        models_available = get_detection_models()
        
        return render_template('training.html',
                             models=models_available)
    except Exception as e:
        print(f"Training page error: {str(e)}")
        print(traceback.format_exc())
        flash(f'系统错误: {str(e)}', 'error')
        return redirect(url_for('detection.index'))


@training_bp.route('/upload_dataset', methods=['POST'])
def upload_dataset():
    """上传训练数据集"""
    try:
        # 检查文件
        if 'dataset' not in request.files:
            return jsonify({'success': False, 'error': '没有选择文件'})
        
        file = request.files['dataset']
        if file.filename == '':
            return jsonify({'success': False, 'error': '没有选择文件'})
        
        # 检查文件类型
        if not allowed_file(file.filename, {'zip'}):
            return jsonify({'success': False, 'error': '只支持ZIP格式的压缩包'})
        
        # 保存上传的压缩包
        timestamp = int(time.time())
        dataset_filename = f"dataset_{timestamp}.zip"
        dataset_path = os.path.join(current_app.config['UPLOAD_FOLDER'], dataset_filename)
        file.save(dataset_path)
        
        print(f"Dataset uploaded to: {dataset_path}")
        
        return jsonify({
            'success': True,
            'dataset_path': dataset_path,
            'dataset_filename': dataset_filename
        })
        
    except Exception as e:
        print(f"Upload dataset error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'success': False, 'error': f'上传失败: {str(e)}'})


@training_bp.route('/check_cuda', methods=['GET'])
def check_cuda():
    """检查CUDA环境可用性"""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        device_count = torch.cuda.device_count() if cuda_available else 0
        
        devices = []
        if cuda_available:
            for i in range(device_count):
                devices.append({
                    'id': i,
                    'name': torch.cuda.get_device_name(i)
                })
        
        return jsonify({
            'cuda_available': cuda_available,
            'device_count': device_count,
            'devices': devices
        })
    except Exception as e:
        print(f"CUDA check error: {str(e)}")
        return jsonify({
            'cuda_available': False,
            'device_count': 0,
            'devices': [],
            'error': str(e)
        })


@training_bp.route('/train', methods=['POST'])
def train_model():
    """开始模型训练"""
    try:
        # 获取参数
        model_name = request.form.get('model_name')
        dataset_path = request.form.get('dataset_path')
        
        # 获取训练参数
        training_params = {
            'epochs': request.form.get('epochs'),
            'batch': request.form.get('batch'),
            'imgsz': request.form.get('imgsz'),
            'patience': request.form.get('patience'),
            'device': request.form.get('device'),
            'cache': request.form.get('cache') == 'true',
            'optimizer': request.form.get('optimizer'),
            'lr0': request.form.get('lr0'),
            'lrf': request.form.get('lrf'),
            'weight_decay': request.form.get('weight_decay'),
            'warmup_epochs': request.form.get('warmup_epochs'),
            'freeze': request.form.get('freeze')
        }
        # 过滤掉None值和空字符串
        training_params = {k: v for k, v in training_params.items() if v is not None and v != ''}
        
        # 参数验证
        if not model_name or not dataset_path:
            return jsonify({'success': False, 'error': '缺少必要参数'})
        
        if not os.path.exists(dataset_path):
            return jsonify({'success': False, 'error': '数据集文件不存在'})
        
        # 获取模型列表验证
        models_available = get_detection_models()
        if model_name not in models_available:
            return jsonify({'success': False, 'error': f'模型 {model_name} 不存在'})
        
        # 开始训练
        result = process_model_training(model_name, dataset_path, training_params)
        
        return jsonify({
            'success': True,
            'new_model_path': result['new_model_path'],
            'new_model_name': result['new_model_name'],
            'training_info': result['training_info']
        })
        
    except Exception as e:
        print(f"Model training error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'success': False, 'error': f'训练失败: {str(e)}'})