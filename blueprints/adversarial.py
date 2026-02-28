# blueprints/adversarial.py (修改后)
import os
import time
import traceback

from flask import Blueprint, request, render_template, current_app, redirect, flash
from flask import jsonify
from werkzeug.utils import secure_filename

from core.model_manager import model_manager
from core.processors.adversarial import physical_processor, traditional_processor
from core.processors.detection import process_detection
from core.processors.official_adversarial import (
    process_official_shadow_attack,
    process_official_advcam_attack,
    process_combined_official_attack
)
from core.processors.patch_attack import process_hybrid_adversarial
from utils.file_utils import allowed_file, get_classification_models
from utils.history_manager import history_manager

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
                    from utils.file_utils import safe_filename
                    filename = safe_filename(file.filename)

                    # 添加额外的安全检查
                    if not filename or len(filename) < 5:  # 文件名太短可能有问题
                        filename = f"upload_{int(time.time())}{os.path.splitext(file.filename)[1].lower()}"
                    upload_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
                    file.save(upload_path)
                    print(f"Original file saved to: {upload_path}")

                    # 获取攻击参数
                    attack_mode = request.form.get('attack_mode', 'patch_only')
                    attack_type = request.form.get('attack_type', '')
                    intensity = float(request.form.get('intensity', 0.1))
                    style_type = request.form.get('style_type', 'natural')  # 用于AdvCam风格选择
                    
                    # 补丁攻击参数
                    use_patch = request.form.get('use_patch') == 'true'
                    patch_type = request.form.get('patch_type', 'eot_optimized')
                    patch_size = int(request.form.get('patch_size', 40))
                    patch_position = request.form.get('patch_position', 'random')
                    patch_shape = request.form.get('patch_shape', 'rectangle')
                    patch_alpha = float(request.form.get('patch_alpha', 0.8))
                    patch_color = request.form.get('patch_color', 'high_contrast')

                    print(f"Attack mode: {attack_mode}")
                    print(f"Attack type: {attack_type}")
                    print(f"Intensity: {intensity}")
                    print(f"Style type: {style_type}")
                    print(f"Use patch: {use_patch}")

                    # 构建攻击配置
                    attack_config = {
                        'attack_mode': attack_mode,
                        'use_patch': use_patch,
                        'patch_params': {
                            'patch_type': patch_type,
                            'patch_size': patch_size,
                            'patch_position': patch_position,
                            'patch_shape': patch_shape,
                            'alpha': patch_alpha,
                            'color_preset': patch_color
                        },
                        'base_attack': attack_type,
                        'base_params': {'epsilon': intensity},
                        'model_name': request.form.get('model_name', 'resnet50')
                    }
                    
                    # 根据攻击模式处理参数
                    if attack_mode == 'patch_only':
                        # 仅补丁攻击模式
                        attack_config['use_patch'] = True
                        attack_config['base_attack'] = None
                    elif attack_mode == 'traditional_only':
                        # 仅传统攻击模式
                        attack_config['use_patch'] = False
                    elif attack_mode == 'physical_only':
                        # 仅物理攻击模式
                        attack_config['use_patch'] = False
                    elif attack_mode == 'hybrid':
                        # 混合攻击模式
                        attack_config['use_patch'] = use_patch
                        
                    # 如果是官方组合攻击，需要特殊处理参数
                    if attack_type == 'official_combined':
                        attack_config['base_params'] = {
                            'shadow_intensity': float(request.form.get('shadow_intensity', 0.3)),
                            'cam_intensity': float(request.form.get('cam_intensity', 0.1)),
                            'fast_mode': request.form.get('fast_mode') == 'true'
                        }
                    elif attack_type == 'combined':
                        attack_config['base_params'] = {
                            'cam_intensity': intensity,
                            'shadow_intensity': intensity * 1.5
                        }
                    elif attack_type == 'official_cam':
                        attack_config['base_params']['style_type'] = style_type
                        attack_config['base_params']['fast_mode'] = request.form.get('fast_mode') == 'true'
                    elif attack_type == 'official_shadow':
                        attack_config['base_params']['fast_mode'] = request.form.get('fast_mode') == 'true'
                    
                    print(f"Attack config: {attack_config}")
                    
                    # 应用对抗攻击 - 使用混合处理器
                    try:
                        adv_path, adv_image = process_hybrid_adversarial(upload_path, attack_config)
                    except Exception as e:
                        print(f"Hybrid processor error: {str(e)}")
                        # 回退到传统处理方式
                        if attack_mode == 'patch_only':
                            # 单独的补丁攻击
                            from core.processors.patch_attack import AdvancedPatchAttackProcessor
                            patch_processor = AdvancedPatchAttackProcessor()
                            adv_path, adv_image = patch_processor.apply_patch_attack(upload_path, attack_config['patch_params'])
                        elif attack_mode in ['traditional_only', 'physical_only']:
                            # 传统或物理攻击
                            if attack_type in ['fgsm', 'fgsm_plus_plus', 'targeted_fgsm', 'universal_fgsm', 
                                             'pgd', 'momentum_pgd', 'pgd_linf', 'pgd_l2', 'cw']:
                                # 数字域攻击
                                model_name = request.form.get('model_name', 'resnet50')
                                try:
                                    # 获取分类模型
                                    model = model_manager.get_model('adversarial', model_name)
                                    adv_path, adv_image = traditional_processor.generate_adversarial_example(
                                        upload_path, model, attack_type, epsilon=intensity
                                    )
                                except Exception as e:
                                    # 如果模型不可用，使用默认模型
                                    print(f"Model {model_name} not available, using default model: {str(e)}")
                                    model = model_manager.get_model('adversarial', 'resnet50')
                                    adv_path, adv_image = traditional_processor.generate_adversarial_example(
                                        upload_path, model, attack_type, epsilon=intensity
                                    )
                            elif attack_type in ['cam', 'shadow', 'combined', 'official_shadow', 'official_cam', 'official_combined']:
                                # 物理域攻击
                                if attack_type == 'cam':
                                    adv_path, adv_image = physical_processor.adv_cam_attack(upload_path, intensity)
                                elif attack_type == 'shadow':
                                    adv_path, adv_image = physical_processor.adv_shadow_attack(upload_path, intensity)
                                elif attack_type == 'combined':
                                    adv_path, adv_image = physical_processor.combined_attack(upload_path, intensity, intensity*1.5)
                                elif attack_type == 'official_shadow':
                                    fast_mode = request.form.get('fast_mode') == 'true'
                                    adv_path, adv_image = process_official_shadow_attack(upload_path, intensity)
                                elif attack_type == 'official_cam':
                                    fast_mode = request.form.get('fast_mode') == 'true'
                                    adv_path, adv_image = process_official_advcam_attack(upload_path, intensity, style_type)
                                elif attack_type == 'official_combined':
                                    fast_mode = request.form.get('fast_mode') == 'true'
                                    shadow_intensity = float(request.form.get('shadow_intensity', 0.3))
                                    cam_intensity = float(request.form.get('cam_intensity', 0.1))
                                    adv_path, adv_image = process_combined_official_attack(
                                        upload_path, shadow_intensity, cam_intensity
                                    )
                            else:
                                raise ValueError(f"Unsupported attack type: {attack_type}")
                        else:
                            raise e

                    print(f"Adversarial sample generated: {os.path.basename(adv_path)}")

                    # 保存历史记录
                    history_manager.add_record(
                        operation_type='adversarial',
                        original_image=filename,
                        result_image=os.path.basename(adv_path),
                        attack_type=attack_type,
                        intensity=intensity,
                        style_type=style_type if 'style_type' in locals() else None,
                        use_patch=use_patch,
                        patch_type=patch_type if use_patch else None,
                        model_name=request.form.get('model_name', 'resnet50')
                    )
                    
                    # 返回结果页面
                    return render_template('adversarial.html',
                                         original_image=filename,
                                         adversarial_image=os.path.basename(adv_path),
                                         attack_type=attack_type,
                                         intensity=intensity,
                                         style_type=style_type if 'style_type' in locals() else None,
                                         use_patch=use_patch,
                                         patch_type=patch_type,
                                         patch_size=patch_size,
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
                               attack_mode='patch_only',
                               attack_type='',
                               intensity=0.1,
                               style_type='natural',
                               use_patch=False)

    except Exception as e:
        print(f"General error: {str(e)}")
        print(traceback.format_exc())
        flash(f'系统错误: {str(e)}', 'error')
        models_available = get_classification_models()
        return render_template('adversarial.html',
                               models=models_available,
                               attack_mode='patch_only',
                               attack_type='',
                               intensity=0.1,
                               style_type='natural',
                               use_patch=False)


# 添加新的路由用于对抗检测对比
@adversarial_bp.route('/detect_compare', methods=['POST'])
def detect_compare():
    """对抗样本检测对比功能 - 增强版"""
    try:
        # 获取POST数据
        original_image = request.form.get('original_image')
        adversarial_image = request.form.get('adversarial_image')
        model_name = request.form.get('model_name')

        # 参数验证
        if not all([original_image, adversarial_image, model_name]):
            return jsonify({'error': '缺少必要参数'}), 400

        # 构建完整路径
        original_path = os.path.join(current_app.config['UPLOAD_FOLDER'], original_image)
        adversarial_path = os.path.join(current_app.config['RESULT_FOLDER'], adversarial_image)

        # 检查文件是否存在
        if not os.path.exists(original_path):
            return jsonify({'error': f'原始图像不存在: {original_image}'}), 404
        if not os.path.exists(adversarial_path):
            return jsonify({'error': f'对抗样本不存在: {adversarial_image}'}), 404

        # 获取检测模型
        try:
            model = model_manager.get_model('detection', model_name)
        except Exception as e:
            return jsonify({'error': f'模型加载失败: {str(e)}'}), 500

        # 分别对原始图像和对抗样本进行检测
        try:
            original_detections, original_result = process_detection(original_path, model)
            adversarial_detections, adversarial_result = process_detection(adversarial_path, model)
        except Exception as e:
            return jsonify({'error': f'检测过程出错: {str(e)}'}), 500

        # 分析检测结果差异
        comparison_result = analyze_detection_difference(original_detections, adversarial_detections)

        # 返回JSON结果
        return jsonify({
            'success': True,
            'original_result': original_result,
            'adversarial_result': adversarial_result,
            'original_detections': original_detections or [],
            'adversarial_detections': adversarial_detections or [],
            'comparison': comparison_result
        })

    except Exception as e:
        print(f"Detection comparison error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': f'系统错误: {str(e)}'}), 500


def analyze_detection_difference(orig_dets, adv_dets):
    """分析检测结果差异 - 改进版（基于IoU的真实匹配）"""
    result = {
        'total_objects_orig': len(orig_dets),
        'total_objects_adv': len(adv_dets),
        'objects_lost': 0,
        'objects_added': 0,
        'objects_matched': 0,
        'confidence_changes': [],
        'class_changes': [],
        'position_changes': [],
        'detailed_matches': []
    }

    # 边界情况处理
    if not orig_dets and not adv_dets:
        return result
    if not orig_dets:
        result['objects_added'] = len(adv_dets)
        return result
    if not adv_dets:
        result['objects_lost'] = len(orig_dets)
        return result

    # 基于IoU的真实匹配算法
    matched_pairs = []
    used_adv_indices = set()
    used_orig_indices = set()
    
    # 为每个原始检测寻找最佳匹配
    for i, orig_det in enumerate(orig_dets):
        best_match_idx = -1
        best_iou = 0
        
        for j, adv_det in enumerate(adv_dets):
            if j in used_adv_indices:
                continue
                
            iou = calculate_iou(orig_det['bbox'], adv_det['bbox'])
            # IOU阈值设为0.3，可根据需要调整
            if iou > 0.3 and iou > best_iou:
                best_iou = iou
                best_match_idx = j
        
        if best_match_idx != -1:
            matched_pairs.append((i, best_match_idx, best_iou))
            used_adv_indices.add(best_match_idx)
            used_orig_indices.add(i)
    
    # 统计匹配结果
    result['objects_matched'] = len(matched_pairs)
    result['objects_lost'] = len(orig_dets) - len(matched_pairs)
    result['objects_added'] = len(adv_dets) - len(matched_pairs)
    
    # 处理匹配的目标
    for orig_idx, adv_idx, iou in matched_pairs:
        orig_det = orig_dets[orig_idx]
        adv_det = adv_dets[adv_idx]
        
        # 记录置信度变化
        conf_diff = adv_det['confidence'] - orig_det['confidence']
        result['confidence_changes'].append({
            'class': orig_det['class'],
            'confidence_change': conf_diff,
            'iou': iou,
            'orig_confidence': orig_det['confidence'],
            'adv_confidence': adv_det['confidence']
        })
        
        # 记录类别变化
        if orig_det['class'] != adv_det['class']:
            result['class_changes'].append({
                'from_class': orig_det['class'],
                'to_class': adv_det['class'],
                'confidence_orig': orig_det['confidence'],
                'confidence_adv': adv_det['confidence'],
                'iou': iou
            })
        
        # 记录位置变化（中心点距离）
        orig_center = [(orig_det['bbox'][0] + orig_det['bbox'][2]) / 2, 
                      (orig_det['bbox'][1] + orig_det['bbox'][3]) / 2]
        adv_center = [(adv_det['bbox'][0] + adv_det['bbox'][2]) / 2, 
                     (adv_det['bbox'][1] + adv_det['bbox'][3]) / 2]
        position_change = ((orig_center[0] - adv_center[0]) ** 2 + 
                          (orig_center[1] - adv_center[1]) ** 2) ** 0.5
        
        result['position_changes'].append({
            'class': orig_det['class'],
            'position_change': position_change,
            'iou': iou
        })
        
        # 详细匹配信息
        result['detailed_matches'].append({
            'orig_index': orig_idx,
            'adv_index': adv_idx,
            'class': orig_det['class'],
            'iou': iou,
            'confidence_orig': orig_det['confidence'],
            'confidence_adv': adv_det['confidence'],
            'position_change': position_change
        })
    
    # 计算平均置信度变化
    if result['confidence_changes']:
        total_conf_change = sum(item['confidence_change'] for item in result['confidence_changes'])
        result['avg_confidence_change'] = total_conf_change / len(result['confidence_changes'])
    
    print(f"Debug - Orig: {len(orig_dets)}, Adv: {len(adv_dets)}")
    print(f"Debug - Matched: {result['objects_matched']}, Lost: {result['objects_lost']}, Added: {result['objects_added']}")
    
    return result


def calculate_iou(box1, box2):
    """计算两个边界框的IOU"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    # 计算交集
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0

    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)

    # 计算并集
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = area1 + area2 - inter_area

    return inter_area / union_area if union_area > 0 else 0.0
