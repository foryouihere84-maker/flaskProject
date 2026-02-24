import os
from flask import Blueprint, request, render_template, current_app, redirect, url_for
from werkzeug.utils import secure_filename
from core.model_manager import model_manager
from core.processors.classification import process_classification
from utils.file_utils import allowed_file

# 创建蓝图对象，注意变量名必须为 classification_bp
classification_bp = Blueprint('classification', __name__, url_prefix='/classification')

@classification_bp.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # 检查文件上传
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file(file.filename, current_app.config['ALLOWED_EXTENSIONS']):
            # 保存上传文件
            filename = secure_filename(file.filename)
            upload_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
            file.save(upload_path)

            # 获取用户选择的模型（从表单中获取，默认为 'resnet50'）
            model_name = request.form.get('model', 'resnet50')
            category = 'classification'
            model = model_manager.get_model(category, model_name)

            # 调用分类处理函数（需自行实现）
            class_name, confidence, result_filename = process_classification(upload_path, model)

            return render_template('classification.html',
                                   uploaded_image=filename,
                                   result_image=result_filename,
                                   class_name=class_name,
                                   confidence=confidence,
                                   model_name=model_name)

    # GET 请求：显示上传页面，并传递可用的模型列表
    models_available = current_app.config['MODELS']['classification'].keys()
    return render_template('classification.html', models=models_available)