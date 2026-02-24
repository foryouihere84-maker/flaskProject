from flask import Blueprint, request, render_template, current_app, redirect, url_for, flash
from werkzeug.utils import secure_filename
from core.model_manager import model_manager
from core.processors.detection import process_detection
from utils.file_utils import allowed_file, get_detection_models
import os
import traceback


detection_bp = Blueprint('detection', __name__, url_prefix='/detection')


@detection_bp.route('/', methods=['GET', 'POST'])
def index():
    try:
        # 获取当前可用的检测模型列表（GET和POST都需要）
        models_available = get_detection_models()
        default_model = current_app.config['DEFAULT_MODEL'] if current_app.config[
                                                                   'DEFAULT_MODEL'] in models_available else \
            models_available[0] if models_available else ''

        if request.method == 'POST':
            # 检查文件
            if 'file' not in request.files:
                flash('没有选择文件', 'error')
                return redirect(request.url)

            file = request.files['file']
            if file.filename == '':
                flash('没有选择文件', 'error')
                return redirect(request.url)

            if file and allowed_file(file.filename, current_app.config['ALLOWED_EXTENSIONS']):
                try:
                    # 保存文件
                    filename = secure_filename(file.filename)
                    upload_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
                    file.save(upload_path)
                    print(f"File saved to: {upload_path}")

                    # 获取模型选择
                    model_name = request.form.get('model', default_model)
                    category = 'detection'

                    print(f"Selected model: {model_name}")
                    print(f"Available models: {list(current_app.config['MODELS']['detection'].keys())}")

                    # 获取模型
                    model = model_manager.get_model(category, model_name)

                    # 处理检测
                    detections, result_filename = process_detection(upload_path, model)
                    print(f"Detection completed. Found {len(detections)} objects.")

                    # 检测完成后仍然传递模型列表数据
                    return render_template('detection.html',
                                           uploaded_image=filename,
                                           result_image=result_filename,
                                           detections=detections,
                                           model_name=model_name,
                                           models=models_available)  # 关键：传递models变量

                except Exception as e:
                    print(f"Processing error: {str(e)}")
                    print(traceback.format_exc())
                    flash(f'处理图片时出错: {str(e)}', 'error')
                    # 删除可能已保存的文件
                    if 'upload_path' in locals() and os.path.exists(upload_path):
                        os.remove(upload_path)
                    return redirect(request.url)

        # GET 请求
        return render_template('detection.html',
                               models=models_available,
                               model_name=default_model)

    except Exception as e:
        print(f"General error: {str(e)}")
        print(traceback.format_exc())
        flash(f'系统错误: {str(e)}', 'error')
        # 即使出错也要传递模型列表
        models_available = get_detection_models()
        return render_template('detection.html',
                               models=models_available,
                               model_name=models_available[0] if models_available else '')
