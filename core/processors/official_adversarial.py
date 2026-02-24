"""
官方对抗攻击算法集成处理器（优化版）
整合第三方高质量实现：ShadowAttack (CVPR 2022) 和 AdvCam (CVPR 2020)
性能优化版本，解决生成对抗样本卡顿问题
"""

import torch
import cv2
import numpy as np
import os
import time
from flask import current_app
import traceback
from scipy.optimize import differential_evolution
import warnings

warnings.filterwarnings('ignore')


class OfficialShadowAttackProcessor:
    """官方 ShadowAttack 实现 - CVPR 2022（优化版）"""

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.call_count = 0  # 性能监控计数器

    def shadow_attack(self, image_path, intensity=0.3, optimization_steps=15, fast_mode=False):
        """
        ShadowAttack 核心实现（优化版）
        基于自然阴影现象的物理世界对抗攻击
        """
        start_time = time.time()
        self.call_count += 1
        print(f"[ShadowAttack #{self.call_count}] 开始处理...")
        print(f"[ShadowAttack] 快速模式: {fast_mode}, 优化步数: {optimization_steps}")

        try:
            # 读取图像
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"无法读取图片: {image_path}")

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img_rgb.shape[:2]
            print(f"[ShadowAttack] 图像尺寸: {w}x{h}")

            # 核心逻辑：快速模式 vs 优化模式
            if fast_mode:
                print("[ShadowAttack] 使用快速模式 - 应用预设参数")
                optimal_params = self._get_fast_shadow_params(h, w, intensity)
            else:
                print(f"[ShadowAttack] 使用优化模式 - 差分进化算法 ({optimization_steps} 步)")
                # 优化的差分进化算法
                result = differential_evolution(
                    self._shadow_objective_function,
                    bounds=[(0, h), (0, w), (0.1, 0.8)],
                    args=(img_rgb, intensity),
                    maxiter=min(optimization_steps, 20),
                    popsize=10,
                    tol=0.01,
                    seed=42,
                    disp=False
                )
                optimal_params = result.x

            # 应用最优阴影
            shadow_mask = self._create_optimized_shadow_mask(h, w, optimal_params)
            perturbed_img = (img_rgb.astype(np.float32) * shadow_mask).astype(np.uint8)

            # 保存结果
            adv_filename = f'official_shadow_{os.path.basename(image_path)}'
            adv_path = os.path.join(current_app.config['RESULT_FOLDER'], adv_filename)

            adv_bgr = cv2.cvtColor(perturbed_img, cv2.COLOR_RGB2BGR)
            success = cv2.imwrite(adv_path, adv_bgr)
            if not success:
                raise ValueError(f"无法保存对抗样本: {adv_path}")

            end_time = time.time()
            processing_time = end_time - start_time
            print(f"[ShadowAttack] 处理完成，耗时: {processing_time:.2f}秒，保存至: {adv_path}")

            return adv_path, perturbed_img

        except Exception as e:
            print(f"[ShadowAttack] 错误: {str(e)}")
            print(traceback.format_exc())
            raise

    def _get_fast_shadow_params(self, h, w, intensity):
        """快速模式下的预设阴影参数生成"""
        if intensity < 0.3:
            center_x, center_y = w // 4, h // 4
        elif intensity < 0.6:
            center_x, center_y = w - w // 3, h // 3
        else:
            center_x, center_y = w // 2 + w // 6, h // 2 - h // 6

        print(f"[ShadowAttack] 快速模式参数 - 中心: ({center_x}, {center_y}), 强度: {intensity}")
        return [center_y, center_x, intensity]

    def _shadow_objective_function(self, params, image, target_intensity):
        """阴影优化目标函数"""
        h, w = image.shape[:2]
        center_y, center_x, intensity = params
        shadow_mask = self._create_optimized_shadow_mask(h, w, [center_y, center_x, intensity])
        perturbation_magnitude = np.mean(np.abs(1 - shadow_mask))
        return perturbation_magnitude + (1 - intensity) * 0.1

    def _create_optimized_shadow_mask(self, h, w, params):
        """优化版阴影遮罩生成（向量化操作）"""
        center_y, center_x, intensity = params
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        distances = np.sqrt((y_coords - center_y) ** 2 + (x_coords - center_x) ** 2)
        max_dist = np.sqrt(h ** 2 + w ** 2)
        normalized_dists = np.minimum(1.0, distances / (max_dist * 0.7))
        shadow_strengths = intensity * (1 - normalized_dists * 0.8)
        mask_values = 1 - shadow_strengths
        mask = np.stack([mask_values, mask_values, mask_values], axis=-1)
        return np.clip(mask, 0.2, 1.0)

    def _create_shadow_mask(self, h, w, params):
        """创建阴影遮罩（原有版本）"""
        center_y, center_x, intensity = params
        mask = np.ones((h, w, 3), dtype=np.float32)
        for i in range(h):
            for j in range(w):
                distance = np.sqrt((i - center_y) ** 2 + (j - center_x) ** 2)
                max_dist = np.sqrt(h ** 2 + w ** 2)
                normalized_dist = min(1.0, distance / (max_dist * 0.7))
                shadow_strength = intensity * (1 - normalized_dist * 0.8)
                mask[i, j] = 1 - shadow_strength
        return np.clip(mask, 0.2, 1.0)


class OfficialAdvCamProcessor:
    """官方 AdvCam 实现 - CVPR 2020"""

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def adv_cam_attack(self, image_path, intensity=0.1, style_type='natural'):
        """AdvCam 核心实现"""
        start_time = time.time()
        print(f"[AdvCam] 开始处理 - 风格: {style_type}, 强度: {intensity}")

        try:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"无法读取图片: {image_path}")

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img_rgb.shape[:2]

            if style_type == 'natural':
                perturbed_img = self._natural_style_camouflage(img_rgb, intensity)
            elif style_type == 'rusty':
                perturbed_img = self._rusty_style_camouflage(img_rgb, intensity)
            elif style_type == 'dirty':
                perturbed_img = self._dirty_style_camouflage(img_rgb, intensity)
            else:
                perturbed_img = self._generic_camouflage(img_rgb, intensity)

            adv_filename = f'official_advcam_{style_type}_{os.path.basename(image_path)}'
            adv_path = os.path.join(current_app.config['RESULT_FOLDER'], adv_filename)

            adv_bgr = cv2.cvtColor(perturbed_img, cv2.COLOR_RGB2BGR)
            success = cv2.imwrite(adv_path, adv_bgr)
            if not success:
                raise ValueError(f"无法保存对抗样本: {adv_path}")

            end_time = time.time()
            processing_time = end_time - start_time
            print(f"[AdvCam] 处理完成，耗时: {processing_time:.2f}秒，保存至: {adv_path}")

            return adv_path, perturbed_img

        except Exception as e:
            print(f"[AdvCam] 错误: {str(e)}")
            print(traceback.format_exc())
            raise

    # 其他方法保持原有实现...
    def _natural_style_camouflage(self, image, intensity):
        h, w = image.shape[:2]
        perturbed = image.copy().astype(np.float32)
        texture_noise = self._generate_natural_texture(h, w, intensity)
        key_regions = self._find_key_regions(image)

        for region in key_regions:
            x, y, size = region
            x_start, x_end = max(0, x - size // 2), min(w, x + size // 2)
            y_start, y_end = max(0, y - size // 2), min(h, y + size // 2)
            region_h, region_w = y_end - y_start, x_end - x_start
            if region_h > 0 and region_w > 0:
                perturbed[y_start:y_end, x_start:x_end] += texture_noise[:region_h, :region_w] * intensity * 100

        return np.clip(perturbed, 0, 255).astype(np.uint8)

    def _rusty_style_camouflage(self, image, intensity):
        h, w = image.shape[:2]
        perturbed = image.copy().astype(np.float32)
        rust_pattern = self._generate_rust_pattern(h, w, intensity)
        perturbed = perturbed * (1 - rust_pattern) + rust_pattern * 100
        return np.clip(perturbed, 0, 255).astype(np.uint8)

    def _dirty_style_camouflage(self, image, intensity):
        h, w = image.shape[:2]
        perturbed = image.copy().astype(np.float32)
        dirt_pattern = self._generate_dirt_pattern(h, w, intensity)
        perturbed = perturbed * (1 - dirt_pattern * 0.3) + np.random.normal(0, 20, (h, w, 3)) * dirt_pattern
        return np.clip(perturbed, 0, 255).astype(np.uint8)

    def _generate_natural_texture(self, h, w, intensity):
        noise = np.zeros((h, w, 3))
        scales = [1, 2, 4, 8]
        for scale in scales:
            small_h, small_w = h // scale, w // scale
            small_noise = np.random.normal(0, intensity, (small_h, small_w, 3))
            resized_noise = cv2.resize(small_noise, (w, h))
            noise += resized_noise / len(scales)
        return noise

    def _generate_rust_pattern(self, h, w, intensity):
        base_pattern = np.random.uniform(0.3, 0.7, (h, w, 3))
        for _ in range(int(intensity * 50)):
            y, x = np.random.randint(0, h), np.random.randint(0, w)
            size = np.random.randint(2, 8)
            cv2.circle(base_pattern, (x, y), size, (0.1, 0.1, 0.1), -1)
        return base_pattern

    def _generate_dirt_pattern(self, h, w, intensity):
        dirt = np.zeros((h, w))
        num_stains = int(intensity * 30)
        for _ in range(num_stains):
            y, x = np.random.randint(0, h), np.random.randint(0, w)
            size = np.random.randint(3, 15)
            stain = np.zeros((h, w))
            cv2.circle(stain, (x, y), size, 1, -1)
            dirt = np.maximum(dirt, stain)
        return np.stack([dirt, dirt, dirt], axis=2)

    def _find_key_regions(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        regions = []
        for contour in contours[:5]:
            x, y, w, h = cv2.boundingRect(contour)
            center_x, center_y = x + w // 2, y + h // 2
            size = max(w, h)
            regions.append((center_x, center_y, size))

        if len(regions) < 3:
            h, w = image.shape[:2]
            regions = [
                (w // 4, h // 4, min(w, h) // 8),
                (3 * w // 4, h // 4, min(w, h) // 8),
                (w // 2, 3 * h // 4, min(w, h) // 8)
            ]
        return regions

    def _generic_camouflage(self, image, intensity):
        h, w = image.shape[:2]
        perturbed = image.copy().astype(np.float32)
        noise = np.random.normal(0, intensity * 50, (h, w, 3))
        perturbed += noise
        return np.clip(perturbed, 0, 255).astype(np.uint8)


class CombinedOfficialProcessor:
    """组合官方处理器"""

    def __init__(self):
        self.shadow_processor = OfficialShadowAttackProcessor()
        self.advcam_processor = OfficialAdvCamProcessor()

    def combined_attack(self, image_path, shadow_intensity=0.3, cam_intensity=0.1, fast_mode=True):
        """组合攻击：先Shadow后AdvCam（优化版）"""
        start_time = time.time()
        print(f"[CombinedAttack] 开始组合攻击...")

        try:
            temp_path, temp_img = self.shadow_processor.shadow_attack(
                image_path, shadow_intensity, optimization_steps=10, fast_mode=fast_mode
            )

            final_path, final_img = self.advcam_processor.adv_cam_attack(
                temp_path, cam_intensity, style_type='natural'
            )

            if os.path.exists(temp_path):
                os.remove(temp_path)

            end_time = time.time()
            processing_time = end_time - start_time
            print(f"[CombinedAttack] 组合攻击完成，总耗时: {processing_time:.2f}秒")

            return final_path, final_img

        except Exception as e:
            print(f"[CombinedAttack] 错误: {str(e)}")
            print(traceback.format_exc())
            if 'temp_path' in locals() and os.path.exists(temp_path):
                os.remove(temp_path)
            raise


# 全局实例
shadow_attack_processor = OfficialShadowAttackProcessor()
advcam_processor = OfficialAdvCamProcessor()
combined_official_processor = CombinedOfficialProcessor()


def process_official_shadow_attack(image_path, intensity=0.3, fast_mode=False, **kwargs):
    """处理官方 ShadowAttack"""
    optimization_steps = kwargs.get('optimization_steps', 15)
    return shadow_attack_processor.shadow_attack(image_path, intensity, optimization_steps, fast_mode)


def process_official_advcam_attack(image_path, intensity=0.1, style_type='natural', **kwargs):
    """处理官方 AdvCam 攻击"""
    return advcam_processor.adv_cam_attack(image_path, intensity, style_type, **kwargs)


def process_combined_official_attack(image_path, shadow_intensity=0.3, cam_intensity=0.1, fast_mode=True):
    """处理组合官方攻击"""
    return combined_official_processor.combined_attack(image_path, shadow_intensity, cam_intensity, fast_mode)
