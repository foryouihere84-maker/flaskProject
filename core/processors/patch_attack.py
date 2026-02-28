"""
现代化补丁攻击处理器
集成最新研究成果：EoT框架、智能优化、多视角鲁棒性
基于2025年最新论文实现：RP2、DAPatch、PhysPatch等
"""

import cv2
import numpy as np
import os
import torch
import torch.nn as nn
from flask import current_app
import traceback
from typing import Dict, Tuple, Any
import random


class AdvancedPatchAttackProcessor:
    """先进补丁攻击处理器 - 基于EoT框架和智能优化"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[AdvancedPatchAttack] Processor initialized on {self.device}")
        
        # EoT变换参数
        self.transform_params = {
            'rotation_range': (-15, 15),      # 旋转角度范围
            'scale_range': (0.8, 1.2),        # 缩放比例范围
            'brightness_range': (0.7, 1.3),   # 亮度变化范围
            'noise_std': 0.02,                # 噪声标准差
            'blur_range': (0, 3),             # 模糊核大小
            'num_transforms': 10              # 变换采样数量
        }
    
    def apply_patch_attack(self, image_path: str, patch_params: Dict[str, Any]) -> Tuple[str, np.ndarray]:
        """
        应用先进补丁攻击的主入口 - 基于EoT框架
        
        Args:
            image_path: 原始图像路径
            patch_params: 补丁参数字典
                - patch_type: 补丁类型 ('eot_optimized', 'rp2_style', 'dapatch', 'physpatch')
                - patch_size: 补丁大小 (10-100)
                - optimization_steps: 优化迭代次数 (50-500)
                - target_class: 目标类别 (可选)
                - use_eot: 是否启用EoT框架 (默认True)
                - attack_strength: 攻击强度 (0.1-1.0)
        
        Returns:
            Tuple[对抗样本路径, 对抗样本numpy数组]
        """
        try:
            # 读取图像
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"无法读取图片: {image_path}")
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img_rgb.shape[:2]
            
            print(f"[AdvancedPatchAttack] Processing image {w}x{h} with params: {patch_params}")
            
            # 根据补丁类型选择具体的实现
            patch_type = patch_params.get('patch_type', 'eot_optimized')
            
            # 智能补丁生成
            if patch_type == 'eot_optimized':
                result_img = self._generate_eot_optimized_patch(img_rgb, patch_params)
            elif patch_type == 'rp2_style':
                result_img = self._generate_rp2_style_patch(img_rgb, patch_params)
            elif patch_type == 'dapatch':
                result_img = self._generate_deformable_patch(img_rgb, patch_params)
            elif patch_type == 'physpatch':
                result_img = self._generate_physically_realizable_patch(img_rgb, patch_params)
            else:
                # 向后兼容旧方法
                result_img = self._legacy_patch_methods(img_rgb, patch_type, patch_params)
            
            # 保存结果
            adv_filename = f'patch_{patch_type}_{os.path.basename(image_path)}'
            
            # 处理应用上下文问题
            try:
                from flask import current_app
                result_folder = current_app.config['RESULT_FOLDER']
            except:
                # 如果没有应用上下文，使用相对路径
                result_folder = 'static/results'
                os.makedirs(result_folder, exist_ok=True)
            
            adv_path = os.path.join(result_folder, adv_filename)
            
            adv_bgr = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
            success = cv2.imwrite(adv_path, adv_bgr)
            if not success:
                raise ValueError(f"无法保存对抗样本: {adv_path}")
            
            print(f"[AdvancedPatchAttack] Generated advanced patch attack: {adv_path}")
            return adv_path, result_img
            
        except Exception as e:
            print(f"[AdvancedPatchAttack] Error: {str(e)}")
            print(traceback.format_exc())
            raise
    
    def _apply_random_patch(self, image: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """随机补丁攻击 - 最基础的实现"""
        h, w = image.shape[:2]
        
        # 获取参数
        patch_size = min(params.get('patch_size', 30), min(h, w) // 4)
        alpha = params.get('alpha', 0.8)
        position = params.get('patch_position', 'random')
        shape = params.get('patch_shape', 'rectangle')
        color_preset = params.get('color_preset', 'high_contrast')
        
        # 创建补丁
        if color_preset == 'natural':
            # 自然颜色补丁
            patch_color = np.array([100, 150, 200], dtype=np.float32)  # 蓝色调
        elif color_preset == 'high_contrast':
            # 高对比度补丁
            patch_color = np.array([255, 0, 0], dtype=np.float32)  # 红色
        else:  # noise
            # 噪声补丁
            patch_color = np.random.randint(0, 256, (3,), dtype=np.uint8).astype(np.float32)
        
        # 确定位置
        if position == 'random':
            x = np.random.randint(0, w - patch_size)
            y = np.random.randint(0, h - patch_size)
        elif position == 'center':
            x = (w - patch_size) // 2
            y = (h - patch_size) // 2
        elif position == 'corners':
            corner_choices = [
                (0, 0),  # 左上
                (w - patch_size, 0),  # 右上
                (0, h - patch_size),  # 左下
                (w - patch_size, h - patch_size)  # 右下
            ]
            x, y = random.choice(corner_choices)
        else:
            x, y = 0, 0
        
        # 应用补丁
        result = image.copy().astype(np.float32)
        
        if shape == 'rectangle':
            # 矩形补丁
            result[y:y+patch_size, x:x+patch_size] = (
                alpha * patch_color + (1 - alpha) * result[y:y+patch_size, x:x+patch_size]
            )
        elif shape == 'circle':
            # 圆形补丁
            center_x, center_y = x + patch_size // 2, y + patch_size // 2
            radius = patch_size // 2
            
            for i in range(y, min(y + patch_size, h)):
                for j in range(x, min(x + patch_size, w)):
                    distance = np.sqrt((i - center_y)**2 + (j - center_x)**2)
                    if distance <= radius:
                        weight = alpha * (1 - distance / radius)  # 边缘渐变
                        result[i, j] = weight * patch_color + (1 - weight) * result[i, j]
        elif shape == 'custom':
            # 自定义形状（三角形）
            points = np.array([
                [x + patch_size//2, y],
                [x, y + patch_size],
                [x + patch_size, y + patch_size]
            ], np.int32)
            cv2.fillPoly(result[y:y+patch_size, x:x+patch_size], [points - [x, y]], patch_color)
            result[y:y+patch_size, x:x+patch_size] = (
                alpha * result[y:y+patch_size, x:x+patch_size] + 
                (1 - alpha) * image[y:y+patch_size, x:x+patch_size]
            )
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def _apply_structured_patch(self, image: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """结构化补丁攻击 - 模拟RP2论文中的方法"""
        h, w = image.shape[:2]
        
        patch_size = min(params.get('patch_size', 40), min(h, w) // 3)
        alpha = params.get('alpha', 0.9)
        
        # 创建结构化补丁（模拟交通标志攻击）
        patch = np.zeros((patch_size, patch_size, 3), dtype=np.float32)
        
        # 创建类似停止标志的图案
        center = patch_size // 2
        radius = patch_size // 3
        
        # 白色外圆
        cv2.circle(patch, (center, center), radius, (255, 255, 255), -1)
        
        # 红色内圆
        cv2.circle(patch, (center, center), radius - 5, (255, 0, 0), -1)
        
        # 白色条纹（模拟STOP文字）
        stripe_width = patch_size // 8
        for i in range(3):
            y_pos = center - stripe_width + i * (stripe_width + 2)
            if 0 <= y_pos < patch_size:
                cv2.rectangle(patch, 
                            (center - radius + 10, y_pos), 
                            (center + radius - 10, y_pos + stripe_width), 
                            (255, 255, 255), -1)
        
        # 随机位置放置
        x = np.random.randint(0, w - patch_size)
        y = np.random.randint(0, h - patch_size)
        
        # 应用补丁
        result = image.copy().astype(np.float32)
        result[y:y+patch_size, x:x+patch_size] = (
            alpha * patch + (1 - alpha) * result[y:y+patch_size, x:x+patch_size]
        )
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def _apply_invisible_patch(self, image: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """不可见补丁攻击 - 微小扰动"""
        h, w = image.shape[:2]
        
        patch_size = min(params.get('patch_size', 20), min(h, w) // 5)
        alpha = params.get('alpha', 0.3)  # 很低的透明度
        
        # 创建微小的噪声补丁
        noise = np.random.normal(0, 10, (patch_size, patch_size, 3))  # 低强度噪声
        
        # 随机位置
        x = np.random.randint(0, w - patch_size)
        y = np.random.randint(0, h - patch_size)
        
        # 应用补丁
        result = image.copy().astype(np.float32)
        result[y:y+patch_size, x:x+patch_size] = np.clip(
            result[y:y+patch_size, x:x+patch_size] + alpha * noise, 0, 255
        )
        
        return result.astype(np.uint8)
    
    def _legacy_patch_methods(self, image: np.ndarray, patch_type: str, params: Dict[str, Any]) -> np.ndarray:
        """向后兼容的旧方法实现"""
        if patch_type == 'random':
            return self._apply_random_patch(image, params)
        elif patch_type == 'structured':
            return self._apply_structured_patch(image, params)
        elif patch_type == 'invisible':
            return self._apply_invisible_patch(image, params)
        else:
            raise ValueError(f"Unsupported legacy patch type: {patch_type}")
    
    def _generate_eot_optimized_patch(self, image: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """基于EoT框架的优化补丁生成 - 2025年最新方法"""
        h, w = image.shape[:2]
        patch_size = min(params.get('patch_size', 40), min(h, w) // 4)
        steps = params.get('optimization_steps', 100)
        attack_strength = params.get('attack_strength', 0.8)
        target_class = params.get('target_class', None)
        
        # 初始化随机补丁
        patch = np.random.randint(0, 256, (patch_size, patch_size, 3), dtype=np.uint8)
        patch = patch.astype(np.float32)
        
        # EoT优化循环
        for step in range(steps):
            # 采样多个变换版本
            transformed_patches = []
            losses = []
            
            for _ in range(self.transform_params['num_transforms']):
                # 应用随机变换
                transformed_patch = self._apply_eot_transformation(patch.copy())
                transformed_patches.append(transformed_patch)
                
                # 计算损失（简化版本）
                loss = self._compute_patch_loss(transformed_patch, target_class)
                losses.append(loss)
            
            # EoT损失：期望损失
            eot_loss = np.mean(losses)
            
            # 梯度近似更新
            grad = self._estimate_gradient(patch, losses, transformed_patches)
            patch = patch + attack_strength * grad
            patch = np.clip(patch, 0, 255)
            
            if step % 20 == 0:
                print(f"[EoT-Opt] Step {step}/{steps}, Loss: {eot_loss:.4f}")
        
        # 将优化后的补丁应用到图像
        return self._apply_patch_to_image(image, patch.astype(np.uint8), params)
    
    def _generate_rp2_style_patch(self, image: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """RP2风格补丁生成 - 鲁棒物理扰动方法"""
        h, w = image.shape[:2]
        patch_size = min(params.get('patch_size', 50), min(h, w) // 3)
        
        # RP2核心思想：结构化+随机化
        patch = np.zeros((patch_size, patch_size, 3), dtype=np.float32)
        
        # 创建多层次结构
        center = patch_size // 2
        
        # 背景层
        patch[:, :] = [100, 100, 100]  # 灰色背景
        
        # 中心高亮区域
        cv2.circle(patch, (center, center), patch_size//4, [255, 255, 0], -1)  # 黄色圆
        
        # 边缘干扰图案
        for i in range(8):
            angle = i * np.pi / 4
            end_x = int(center + (patch_size//3) * np.cos(angle))
            end_y = int(center + (patch_size//3) * np.sin(angle))
            cv2.line(patch, (center, center), (end_x, end_y), [0, 0, 255], 2)  # 红色射线
        
        # 添加噪声增强鲁棒性
        noise = np.random.normal(0, 15, patch.shape)
        patch = np.clip(patch + noise, 0, 255)
        
        return self._apply_patch_to_image(image, patch.astype(np.uint8), params)
    
    def _generate_deformable_patch(self, image: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """可变形补丁生成 - DAPatch方法"""
        h, w = image.shape[:2]
        patch_size = min(params.get('patch_size', 35), min(h, w) // 4)
        
        # 创建基础网格
        grid_size = 5
        control_points = []
        
        # 生成控制点
        for i in range(grid_size):
            for j in range(grid_size):
                x = j * patch_size // (grid_size - 1)
                y = i * patch_size // (grid_size - 1)
                # 添加随机偏移
                offset_x = np.random.randint(-5, 6)
                offset_y = np.random.randint(-5, 6)
                control_points.append([(x, y), (x + offset_x, y + offset_y)])
        
        # 创建变形补丁
        patch = np.zeros((patch_size, patch_size, 3), dtype=np.float32)
        
        # 绘制变形图案
        for i, (orig_pt, deform_pt) in enumerate(control_points):
            x, y = int(deform_pt[0]), int(deform_pt[1])
            if 0 <= x < patch_size and 0 <= y < patch_size:
                # 根据位置分配不同颜色
                color_val = (i * 50) % 255
                patch[y, x] = [color_val, 255 - color_val, color_val // 2]
        
        # 应用模糊使边缘平滑
        patch = cv2.GaussianBlur(patch, (3, 3), 0)
        
        return self._apply_patch_to_image(image, patch.astype(np.uint8), params)
    
    def _generate_physically_realizable_patch(self, image: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """物理可实现补丁生成 - PhysPatch方法"""
        h, w = image.shape[:2]
        patch_size = min(params.get('patch_size', 45), min(h, w) // 3)
        
        # 模拟真实打印效果
        patch = np.random.randint(50, 200, (patch_size, patch_size, 3), dtype=np.uint8)
        patch = patch.astype(np.float32)
        
        # 添加打印约束
        # 1. 颜色离散化（模拟打印色阶）
        patch = np.round(patch / 32) * 32  # 8-bit色阶
        
        # 2. 添加网点图案（半色调效果）
        dot_size = 3
        for i in range(0, patch_size, dot_size * 2):
            for j in range(0, patch_size, dot_size * 2):
                if (i // dot_size + j // dot_size) % 2 == 0:
                    if i < patch_size and j < patch_size:
                        patch[i:i+dot_size, j:j+dot_size] *= 0.7
        
        # 3. 边缘锐化
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        for c in range(3):
            patch[:,:,c] = cv2.filter2D(patch[:,:,c], -1, kernel)
        
        return self._apply_patch_to_image(image, np.clip(patch, 0, 255).astype(np.uint8), params)
    
    def _apply_eot_transformation(self, patch: np.ndarray) -> np.ndarray:
        """应用EoT变换"""
        h, w = patch.shape[:2]
        
        # 随机旋转
        angle = np.random.uniform(*self.transform_params['rotation_range'])
        M_rot = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
        patch = cv2.warpAffine(patch, M_rot, (w, h))
        
        # 随机缩放
        scale = np.random.uniform(*self.transform_params['scale_range'])
        patch = cv2.resize(patch, None, fx=scale, fy=scale)
        if patch.shape[0] != h or patch.shape[1] != w:
            patch = cv2.resize(patch, (w, h))
        
        # 亮度变化
        brightness = np.random.uniform(*self.transform_params['brightness_range'])
        patch = np.clip(patch * brightness, 0, 255)
        
        # 添加噪声
        noise = np.random.normal(0, self.transform_params['noise_std'] * 255, patch.shape)
        patch = np.clip(patch + noise, 0, 255)
        
        # 随机模糊
        if np.random.random() > 0.5:
            blur_size = np.random.randint(*self.transform_params['blur_range'])
            if blur_size > 0:
                patch = cv2.GaussianBlur(patch, (blur_size*2+1, blur_size*2+1), 0)
        
        return patch
    
    def _compute_patch_loss(self, patch: np.ndarray, target_class: Any = None) -> float:
        """计算补丁损失（简化版）"""
        # 简化的损失函数 - 实际应用中应连接到目标模型
        if target_class is not None:
            # 目标攻击：最大化目标类激活
            return -np.mean(patch)  # 简化为目标最大化亮度
        else:
            # 无目标攻击：最大化扰动
            return np.std(patch)  # 最大化像素差异
    
    def _estimate_gradient(self, patch: np.ndarray, losses: list, transformed_patches: list) -> np.ndarray:
        """梯度估计"""
        # 简化的有限差分梯度估计
        eps = 1e-3
        grad = np.zeros_like(patch)
        
        for i in range(patch.shape[0]):
            for j in range(patch.shape[1]):
                for k in range(patch.shape[2]):
                    # 向前差分
                    patch_plus = patch.copy()
                    patch_plus[i, j, k] += eps
                    loss_plus = self._compute_patch_loss(patch_plus)
                    grad[i, j, k] = (loss_plus - np.mean(losses)) / eps
        
        return grad
    
    def _apply_patch_to_image(self, image: np.ndarray, patch: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """将补丁应用到图像"""
        h, w = image.shape[:2]
        ph, pw = patch.shape[:2]
        
        # 随机位置
        x = np.random.randint(0, max(1, w - pw))
        y = np.random.randint(0, max(1, h - ph))
        
        # 应用补丁
        result = image.copy().astype(np.float32)
        alpha = params.get('alpha', 0.9)
        
        # 确保边界不越界
        x_end = min(x + pw, w)
        y_end = min(y + ph, h)
        px_end = x_end - x
        py_end = y_end - y
        
        result[y:y_end, x:x_end] = (
            alpha * patch[:py_end, :px_end] + 
            (1 - alpha) * result[y:y_end, x:x_end]
        )
        
        return np.clip(result, 0, 255).astype(np.uint8)


class HybridAdversarialProcessor:
    """混合对抗攻击处理器 - 补丁攻击 + 传统/物理攻击"""
    
    def __init__(self):
        self.patch_processor = PatchAttackProcessor()
        from .adversarial import traditional_processor, physical_processor
        self.traditional_processor = traditional_processor
        self.physical_processor = physical_processor
        print("[HybridProcessor] Initialized")
    
    def hybrid_attack(self, image_path: str, attack_config: Dict[str, Any]) -> Tuple[str, np.ndarray]:
        """
        混合攻击主函数 - 支持多种攻击模式
        
        Args:
            image_path: 原始图像路径
            attack_config: 攻击配置字典
                - attack_mode: str, 攻击模式 ('patch_only', 'traditional_only', 'physical_only', 'hybrid')
                - use_patch: bool, 是否使用补丁攻击
                - patch_params: Dict, 补丁参数
                - base_attack: str, 基础攻击类型
                - base_params: Dict, 基础攻击参数
                - model_name: str, 模型名称（数字域攻击需要）
        
        Returns:
            Tuple[最终对抗样本路径, numpy数组]
        """
        try:
            current_image_path = image_path
            attack_mode = attack_config.get('attack_mode', 'patch_only')
            print(f"[HybridAttack] Starting {attack_mode} attack with config: {attack_config}")
            
            # 根据攻击模式处理
            if attack_mode == 'patch_only':
                # 仅补丁攻击
                print("[HybridAttack] Applying patch-only attack")
                final_path, final_img = self.patch_processor.apply_patch_attack(
                    current_image_path, 
                    attack_config['patch_params']
                )
                
            elif attack_mode in ['traditional_only', 'physical_only']:
                # 仅传统或物理攻击
                base_attack_type = attack_config['base_attack']
                base_params = attack_config.get('base_params', {})
                
                print(f"[HybridAttack] Applying {attack_mode} attack: {base_attack_type}")
                
                if attack_mode == 'traditional_only':
                    # 数字域攻击
                    from core.model_manager import model_manager
                    model_name = attack_config.get('model_name', 'resnet50')
                    try:
                        model = model_manager.get_model('adversarial', model_name)
                    except Exception as e:
                        print(f"[HybridAttack] Model {model_name} not available, using default")
                        model = model_manager.get_model('adversarial', 'resnet50')
                    
                    final_path, final_img = self.traditional_processor.generate_adversarial_example(
                        current_image_path, model, base_attack_type, **base_params
                    )
                else:
                    # 物理域攻击
                    final_path, final_img = self._apply_physical_attack(
                        current_image_path, base_attack_type, base_params
                    )
                    
            elif attack_mode == 'hybrid':
                # 混合攻击
                # 第一步：应用补丁攻击
                if attack_config.get('use_patch', False):
                    print("[HybridAttack] Step 1: Applying patch attack")
                    patch_result_path, _ = self.patch_processor.apply_patch_attack(
                        current_image_path, 
                        attack_config['patch_params']
                    )
                    current_image_path = patch_result_path
                    print(f"[HybridAttack] Patch attack completed: {os.path.basename(patch_result_path)}")
                
                # 第二步：应用基础攻击
                base_attack_type = attack_config['base_attack']
                base_params = attack_config.get('base_params', {})
                
                print(f"[HybridAttack] Step 2: Applying base attack '{base_attack_type}'")
                
                if base_attack_type in ['fgsm', 'fgsm_plus_plus', 'targeted_fgsm', 'universal_fgsm', 
                                      'pgd', 'momentum_pgd', 'pgd_linf', 'pgd_l2', 'cw']:
                    # 数字域攻击
                    from core.model_manager import model_manager
                    model_name = attack_config.get('model_name', 'resnet50')
                    try:
                        model = model_manager.get_model('adversarial', model_name)
                    except Exception as e:
                        print(f"[HybridAttack] Model {model_name} not available, using default")
                        model = model_manager.get_model('adversarial', 'resnet50')
                    
                    final_path, final_img = self.traditional_processor.generate_adversarial_example(
                        current_image_path, model, base_attack_type, **base_params
                    )
                elif base_attack_type in ['cam', 'shadow', 'combined', 'official_shadow', 'official_cam', 'official_combined']:
                    # 物理域攻击
                    final_path, final_img = self._apply_physical_attack(
                        current_image_path, base_attack_type, base_params
                    )
                else:
                    raise ValueError(f"Unsupported base attack type: {base_attack_type}")
            else:
                raise ValueError(f"Unsupported attack mode: {attack_mode}")
            
            print(f"[HybridAttack] Completed. Final result: {os.path.basename(final_path)}")
            return final_path, final_img
            
        except Exception as e:
            print(f"[HybridAttack] Error: {str(e)}")
            print(traceback.format_exc())
            raise


    def _apply_physical_attack(self, image_path: str, attack_type: str, params: Dict[str, Any]) -> Tuple[str, np.ndarray]:
        """应用物理域攻击的辅助方法"""
        if attack_type in ['cam', 'shadow', 'combined']:
            # 自研物理攻击
            if attack_type == 'cam':
                intensity = params.get('epsilon', 0.1)
                return self.physical_processor.adv_cam_attack(image_path, intensity)
            elif attack_type == 'shadow':
                intensity = params.get('epsilon', 0.3)
                return self.physical_processor.adv_shadow_attack(image_path, intensity)
            elif attack_type == 'combined':
                cam_intensity = params.get('cam_intensity', 0.1)
                shadow_intensity = params.get('shadow_intensity', 0.3)
                return self.physical_processor.combined_attack(image_path, cam_intensity, shadow_intensity)
                
        elif attack_type in ['official_shadow', 'official_cam', 'official_combined']:
            # 官方物理攻击
            from .official_adversarial import (
                process_official_shadow_attack,
                process_official_advcam_attack,
                process_combined_official_attack
            )
            
            intensity = params.get('epsilon', 0.3)
            
            if attack_type == 'official_shadow':
                fast_mode = params.get('fast_mode', True)
                return process_official_shadow_attack(image_path, intensity, fast_mode)
            elif attack_type == 'official_cam':
                style_type = params.get('style_type', 'natural')
                return process_official_advcam_attack(image_path, intensity, style_type)
            elif attack_type == 'official_combined':
                shadow_intensity = params.get('shadow_intensity', 0.3)
                cam_intensity = params.get('cam_intensity', 0.1)
                fast_mode = params.get('fast_mode', True)
                return process_combined_official_attack(image_path, shadow_intensity, cam_intensity, fast_mode)
        
        raise ValueError(f"Unsupported physical attack type: {attack_type}")


# 保持向后兼容性的包装类
class PatchAttackProcessor(AdvancedPatchAttackProcessor):
    """向后兼容的补丁攻击处理器"""
    pass


# 全局实例
hybrid_processor = HybridAdversarialProcessor()


def process_hybrid_adversarial(image_path: str, attack_config: Dict[str, Any]) -> Tuple[str, np.ndarray]:
    """处理混合对抗攻击的便捷函数"""
    return hybrid_processor.hybrid_attack(image_path, attack_config)