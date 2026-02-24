# blueprints/adversarial.py (修改后)
from flask import Blueprint, request, render_template, current_app, redirect, url_for, flash
from werkzeug.utils import secure_filename
from core.processors.adversarial import physical_processor
from core.processors.official_adversarial import (
    process_official_shadow_attack,
    process_official_advcam_attack,
    process_combined_official_attack
)
from utils.file_utils import allowed_file, get_classification_models
import os
import traceback

adversarial_bp = Blueprint('adversarial', __name__, url_prefix='/adversarial')

@adversarial_bp.route('/', methods=['GET', 'POST'])
def index():
    try:
        # 获取分类模型列表（用于展示）
        models_available = get_classification_models()
        default_model = models_available[0] if models_available else 'resnet50'

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
                    # 保存原始文件
                    filename = secure_filename(file.filename)
                    upload_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
                    file.save(upload_path)
                    print(f"Original file saved to: {upload_path}")

                    # 获取攻击参数
                    attack_type = request.form.get('attack_type', 'cam')
                    intensity = float(request.form.get('intensity', 0.1))
                    style_type = request.form.get('style_type', 'natural')  # 用于AdvCam风格选择

                    print(f"Attack type: {attack_type}")
                    print(f"Intensity: {intensity}")
                    print(f"Style type: {style_type}")

                    # 应用对抗攻击 - 区分官方和自研算法
                    if attack_type == 'cam':
                        # 自研 AdvCam
                        adv_path, adv_image = physical_processor.adv_cam_attack(upload_path, intensity)
                    elif attack_type == 'shadow':
                        # 自研 AdvShadow
                        adv_path, adv_image = physical_processor.adv_shadow_attack(upload_path, intensity)
                    elif attack_type == 'combined':
                        # 自研组合攻击
                        adv_path, adv_image = physical_processor.combined_attack(upload_path, intensity, intensity*1.5)
                    elif attack_type == 'official_shadow':
                        # 官方 ShadowAttack
                        fast_mode = request.form.get('fast_mode') == 'true'
                        adv_path, adv_image = process_official_shadow_attack(upload_path, intensity)
                    elif attack_type == 'official_cam':
                        # 官方 AdvCam
                        fast_mode = request.form.get('fast_mode') == 'true'
                        adv_path, adv_image = process_official_advcam_attack(upload_path, intensity, style_type)
                    elif attack_type == 'official_combined':
                        # 官方组合攻击
                        fast_mode = request.form.get('fast_mode') == 'true'
                        shadow_intensity = float(request.form.get('shadow_intensity', 0.3))
                        cam_intensity = float(request.form.get('cam_intensity', 0.1))
                        adv_path, adv_image = process_combined_official_attack(
                            upload_path, shadow_intensity, cam_intensity
                        )
                    else:
                        raise ValueError(f"Unsupported attack type: {attack_type}")

                    print(f"Adversarial sample generated: {os.path.basename(adv_path)}")

                    # 返回结果页面
                    return render_template('adversarial.html',
                                         original_image=filename,
                                         adversarial_image=os.path.basename(adv_path),
                                         attack_type=attack_type,
                                         intensity=intensity,
                                         style_type=style_type if 'style_type' in locals() else None,
                                         models=models_available)

                except Exception as e:
                    print(f"Processing error: {str(e)}")
                    print(traceback.format_exc())
                    flash(f'生成对抗样本时出错: {str(e)}', 'error')
                    # 清理已保存的文件
                    if 'upload_path' in locals() and os.path.exists(upload_path):
                        os.remove(upload_path)
                    if 'adv_path' in locals() and os.path.exists(adv_path):
                        os.remove(adv_path)
                    return redirect(request.url)

        # GET 请求
        return render_template('adversarial.html',
                               models=models_available,
                               attack_type='cam',
                               intensity=0.1,
                               style_type='natural')

    except Exception as e:
        print(f"General error: {str(e)}")
        print(traceback.format_exc())
        flash(f'系统错误: {str(e)}', 'error')
        models_available = get_classification_models()
        return render_template('adversarial.html',
                               models=models_available,
                               attack_type='cam',
                               intensity=0.1,
                               style_type='natural')
