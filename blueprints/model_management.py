from flask import Blueprint, render_template, request, jsonify, flash, redirect, url_for, send_file
import os
import glob
from werkzeug.utils import secure_filename
from utils.file_utils import scan_model_files

model_management_bp = Blueprint('model_management', __name__, url_prefix='/model-management')

def get_trained_models():
    """
    获取所有包含 'trained' 关键词的模型文件
    """
    from flask import current_app
    
    # 获取 detection 目录路径
    detection_dir = os.path.join(current_app.config['BASE_DIR'], 'models', 'detection')
    
    # 确保目录存在
    if not os.path.exists(detection_dir):
        os.makedirs(detection_dir, exist_ok=True)
        return []
    
    # 查找包含 'trained' 的 .pt 文件
    pattern = os.path.join(detection_dir, '*trained*.pt')
    trained_files = glob.glob(pattern)
    
    # 提取模型信息
    models = []
    for file_path in trained_files:
        filename = os.path.basename(file_path)
        model_name = os.path.splitext(filename)[0]  # 移除扩展名
        file_size = os.path.getsize(file_path)
        
        # 格式化文件大小
        if file_size < 1024:
            size_str = f"{file_size} B"
        elif file_size < 1024 * 1024:
            size_str = f"{file_size / 1024:.1f} KB"
        elif file_size < 1024 * 1024 * 1024:
            size_str = f"{file_size / (1024 * 1024):.1f} MB"
        else:
            size_str = f"{file_size / (1024 * 1024 * 1024):.1f} GB"
        
        models.append({
            'name': model_name,
            'filename': filename,
            'path': file_path,
            'size': size_str,
            'raw_size': file_size
        })
    
    # 按名称排序
    return sorted(models, key=lambda x: x['name'])

@model_management_bp.route('/')
def model_list():
    """
    显示模型管理页面
    """
    models = get_trained_models()
    return render_template('model_management.html', models=models)

@model_management_bp.route('/delete/<model_name>', methods=['POST'])
def delete_model(model_name):
    """
    删除指定的模型文件
    """
    from flask import current_app
    
    try:
        # 验证模型名称安全性
        safe_model_name = secure_filename(model_name)
        if safe_model_name != model_name:
            return jsonify({'success': False, 'message': '无效的模型名称'})
        
        # 构建文件路径
        detection_dir = os.path.join(current_app.config['BASE_DIR'], 'models', 'detection')
        model_path = os.path.join(detection_dir, f"{model_name}.pt")
        
        # 检查文件是否存在且确实包含 'trained'
        if not os.path.exists(model_path):
            return jsonify({'success': False, 'message': '模型文件不存在'})
        
        if 'trained' not in model_name.lower():
            return jsonify({'success': False, 'message': '只能删除包含 "trained" 关键词的模型文件'})
        
        # 删除文件
        os.remove(model_path)
        
        # 重新初始化模型配置
        from app import init_models_config
        init_models_config(current_app)
        
        return jsonify({'success': True, 'message': f'模型 {model_name} 已成功删除'})
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'删除失败: {str(e)}'})

@model_management_bp.route('/refresh')
def refresh_models():
    """
    刷新模型列表
    """
    models = get_trained_models()
    return jsonify({
        'success': True,
        'models': models,
        'count': len(models)
    })

@model_management_bp.route('/export/<model_name>')
def export_model(model_name):
    """
    导出指定的模型文件供用户下载
    """
    from flask import current_app
    
    try:
        # 验证模型名称安全性
        safe_model_name = secure_filename(model_name)
        if safe_model_name != model_name:
            return jsonify({'success': False, 'message': '无效的模型名称'}), 400
        
        # 构建文件路径
        detection_dir = os.path.join(current_app.config['BASE_DIR'], 'models', 'detection')
        model_path = os.path.join(detection_dir, f"{model_name}.pt")
        
        # 检查文件是否存在
        if not os.path.exists(model_path):
            return jsonify({'success': False, 'message': '模型文件不存在'}), 404
        
        # 发送文件供下载
        return send_file(
            model_path,
            as_attachment=True,
            download_name=f"{model_name}.pt",
            mimetype='application/octet-stream'
        )
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'导出失败: {str(e)}'}), 500