import torch
import torch.nn as nn
import cv2
import numpy as np
import os
from flask import current_app
import traceback
from torchvision import transforms
from PIL import Image


class AdversarialAttackProcessor:
    """传统的对抗攻击处理器（FGSM、PGD、CW）"""

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def fgsm_attack(self, image_tensor, model, target_label=None, epsilon=0.03):
        """FGSM (Fast Gradient Sign Method) 攻击"""
        try:
            image_tensor.requires_grad = True

            # 前向传播
            outputs = model(image_tensor)

            # 如果指定了目标标签，则进行目标攻击
            if target_label is not None:
                loss = -nn.CrossEntropyLoss()(outputs, torch.tensor([target_label]).to(self.device))
            else:
                # 非目标攻击：最大化损失
                _, predicted = torch.max(outputs.data, 1)
                loss = nn.CrossEntropyLoss()(outputs, predicted)

            # 计算梯度
            model.zero_grad()
            loss.backward()

            # 生成对抗扰动
            data_grad = image_tensor.grad.data
            sign_data_grad = data_grad.sign()

            # 应用扰动
            perturbed_image = image_tensor + epsilon * sign_data_grad
            perturbed_image = torch.clamp(perturbed_image, 0, 1)

            return perturbed_image.detach()

        except Exception as e:
            print(f"FGSM attack error: {str(e)}")
            print(traceback.format_exc())
            raise

    def pgd_attack(self, image_tensor, model, target_label=None, epsilon=0.03, alpha=0.01, iterations=10):
        """PGD (Projected Gradient Descent) 攻击"""
        try:
            original_image = image_tensor.clone()

            # 初始化扰动图像
            perturbed_image = image_tensor.clone() + torch.randn_like(image_tensor) * 0.01
            perturbed_image = torch.clamp(perturbed_image, 0, 1)

            for i in range(iterations):
                perturbed_image.requires_grad = True

                # 前向传播
                outputs = model(perturbed_image)

                # 计算损失
                if target_label is not None:
                    loss = -nn.CrossEntropyLoss()(outputs, torch.tensor([target_label]).to(self.device))
                else:
                    _, predicted = torch.max(outputs.data, 1)
                    loss = nn.CrossEntropyLoss()(outputs, predicted)

                # 计算梯度
                model.zero_grad()
                loss.backward()

                # 更新扰动
                data_grad = perturbed_image.grad.data
                perturbed_image = perturbed_image + alpha * data_grad.sign()

                # 投影到epsilon范围内
                eta = torch.clamp(perturbed_image - original_image, -epsilon, epsilon)
                perturbed_image = original_image + eta
                perturbed_image = torch.clamp(perturbed_image, 0, 1).detach_()

            return perturbed_image

        except Exception as e:
            print(f"PGD attack error: {str(e)}")
            print(traceback.format_exc())
            raise

    def cw_attack(self, image_tensor, model, target_label=None, c=0.1, kappa=0, max_iter=1000):
        """CW (Carlini & Wagner) L2 攻击"""
        try:
            original_image = image_tensor.clone()
            w = torch.zeros_like(image_tensor, requires_grad=True)

            optimizer = torch.optim.Adam([w], lr=0.01)

            for i in range(max_iter):
                # 将w转换回图像空间
                perturbed_image = 0.5 * (torch.tanh(w) + 1)

                # 计算损失
                outputs = model(perturbed_image)

                if target_label is not None:
                    # 目标攻击
                    target_loss = nn.CrossEntropyLoss()(outputs, torch.tensor([target_label]).to(self.device))
                    confidence_loss = torch.max(outputs) - outputs[0, target_label]
                    loss = c * torch.norm(perturbed_image - original_image, p=2) + confidence_loss
                else:
                    # 非目标攻击
                    _, predicted = torch.max(outputs.data, 1)
                    loss = nn.CrossEntropyLoss()(outputs, predicted)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # 检查是否成功攻击
                if i % 100 == 0:
                    pred_label = torch.argmax(outputs).item()
                    if target_label is not None and pred_label == target_label:
                        break
                    elif target_label is None and pred_label != torch.argmax(model(original_image)).item():
                        break

            final_image = 0.5 * (torch.tanh(w) + 1)
            return torch.clamp(final_image, 0, 1).detach()

        except Exception as e:
            print(f"CW attack error: {str(e)}")
            print(traceback.format_exc())
            raise

    def generate_adversarial_example(self, image_path, model, attack_method='fgsm', **kwargs):
        """统一的数字域对抗样本生成接口"""
        try:
            # 1. 加载和预处理图像
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            
            # 2. 根据攻击方法调用具体实现
            if attack_method == 'fgsm':
                perturbed_tensor = self.fgsm_attack(image_tensor, model, **kwargs)
            elif attack_method == 'fgsm_plus_plus':
                perturbed_tensor = self.fgsm_plus_plus(image_tensor, model, **kwargs)
            elif attack_method == 'targeted_fgsm':
                perturbed_tensor = self.targeted_fgsm(image_tensor, model, **kwargs)
            elif attack_method == 'universal_fgsm':
                perturbed_tensor = self.universal_fgsm(image_tensor, model, **kwargs)
            elif attack_method == 'pgd':
                perturbed_tensor = self.pgd_attack(image_tensor, model, **kwargs)
            elif attack_method == 'momentum_pgd':
                perturbed_tensor = self.momentum_pgd(image_tensor, model, **kwargs)
            elif attack_method == 'pgd_linf':
                perturbed_tensor = self.pgd_linf(image_tensor, model, **kwargs)
            elif attack_method == 'pgd_l2':
                perturbed_tensor = self.pgd_l2(image_tensor, model, **kwargs)
            elif attack_method == 'cw':
                perturbed_tensor = self.cw_attack(image_tensor, model, **kwargs)
            else:
                raise ValueError(f"Unsupported attack method: {attack_method}")
            
            # 3. 保存结果图像
            return self._save_adversarial_image(perturbed_tensor, image_path, attack_method)
            
        except Exception as e:
            print(f"Generate adversarial example error: {str(e)}")
            raise

    def _save_adversarial_image(self, tensor, original_path, attack_method):
        """保存对抗样本图像"""
        try:
            # 转换tensor到numpy图像
            image_np = tensor.squeeze().cpu().numpy()
            image_np = np.transpose(image_np, (1, 2, 0))
            # 反归一化
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image_np = (image_np * std + mean) * 255
            image_np = np.clip(image_np, 0, 255).astype(np.uint8)
            
            # 保存图像
            adv_filename = f'{attack_method}_{os.path.basename(original_path)}'
            adv_path = os.path.join(current_app.config['RESULT_FOLDER'], adv_filename)
            
            cv2.imwrite(adv_path, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
            print(f"Adversarial image saved: {adv_path}")
            return adv_path, image_np
            
        except Exception as e:
            print(f"Save adversarial image error: {str(e)}")
            raise

    def fgsm_plus_plus(self, image_tensor, model, target_label=None, epsilon=0.03, steps=5):
        """FGSM++ 多步FGSM攻击"""
        try:
            perturbed = image_tensor.clone()
            step_size = epsilon / steps
            
            for i in range(steps):
                perturbed.requires_grad = True
                outputs = model(perturbed)
                
                if target_label is not None:
                    loss = -nn.CrossEntropyLoss()(outputs, torch.tensor([target_label]).to(self.device))
                else:
                    _, predicted = torch.max(outputs.data, 1)
                    loss = nn.CrossEntropyLoss()(outputs, predicted)
                    
                model.zero_grad()
                loss.backward()
                
                data_grad = perturbed.grad.data.sign()
                perturbed = perturbed + step_size * data_grad
                perturbed = torch.clamp(perturbed, 0, 1).detach()
            
            return perturbed
        except Exception as e:
            print(f"FGSM++ attack error: {str(e)}")
            raise

    def targeted_fgsm(self, image_tensor, model, target_label, epsilon=0.03):
        """目标导向FGSM攻击"""
        try:
            image_tensor.requires_grad = True
            outputs = model(image_tensor)
            
            # 目标攻击：最小化目标类别的损失
            loss = -nn.CrossEntropyLoss()(outputs, torch.tensor([target_label]).to(self.device))
            
            model.zero_grad()
            loss.backward()
            
            data_grad = image_tensor.grad.data
            sign_data_grad = data_grad.sign()
            
            perturbed_image = image_tensor - epsilon * sign_data_grad  # 注意这里是减号
            perturbed_image = torch.clamp(perturbed_image, 0, 1)
            
            return perturbed_image.detach()
        except Exception as e:
            print(f"Targeted FGSM attack error: {str(e)}")
            raise

    def universal_fgsm(self, image_tensor, model, universal_perturbation=None, epsilon=0.03):
        """通用扰动FGSM攻击"""
        try:
            if universal_perturbation is None:
                # 生成通用扰动
                universal_perturbation = torch.randn_like(image_tensor) * 0.01
                universal_perturbation = torch.clamp(universal_perturbation, -epsilon, epsilon)
            
            image_tensor.requires_grad = True
            perturbed_image = image_tensor + universal_perturbation
            perturbed_image = torch.clamp(perturbed_image, 0, 1)
            
            outputs = model(perturbed_image)
            _, predicted = torch.max(outputs.data, 1)
            loss = nn.CrossEntropyLoss()(outputs, predicted)
            
            model.zero_grad()
            loss.backward()
            
            # 更新通用扰动
            grad_sign = image_tensor.grad.data.sign()
            new_perturbation = universal_perturbation + epsilon * grad_sign
            new_perturbation = torch.clamp(new_perturbation, -epsilon, epsilon)
            
            final_perturbed = image_tensor + new_perturbation
            final_perturbed = torch.clamp(final_perturbed, 0, 1)
            
            return final_perturbed.detach()
        except Exception as e:
            print(f"Universal FGSM attack error: {str(e)}")
            raise

    def momentum_pgd(self, image_tensor, model, target_label=None, epsilon=0.03, alpha=0.01, iterations=10, momentum=0.9):
        """带动量的PGD攻击"""
        try:
            original_image = image_tensor.clone()
            perturbed_image = image_tensor.clone()
            grad_accum = torch.zeros_like(image_tensor)
            
            for i in range(iterations):
                perturbed_image.requires_grad = True
                outputs = model(perturbed_image)
                
                if target_label is not None:
                    loss = -nn.CrossEntropyLoss()(outputs, torch.tensor([target_label]).to(self.device))
                else:
                    _, predicted = torch.max(outputs.data, 1)
                    loss = nn.CrossEntropyLoss()(outputs, predicted)
                    
                model.zero_grad()
                loss.backward()
                
                # 动量更新
                grad_current = perturbed_image.grad.data
                grad_accum = momentum * grad_accum + grad_current / torch.norm(grad_current, p=1)
                
                perturbed_image = perturbed_image + alpha * grad_accum.sign()
                eta = torch.clamp(perturbed_image - original_image, -epsilon, epsilon)
                perturbed_image = original_image + eta
                perturbed_image = torch.clamp(perturbed_image, 0, 1).detach_()
            
            return perturbed_image
        except Exception as e:
            print(f"Momentum PGD attack error: {str(e)}")
            raise

    def pgd_linf(self, image_tensor, model, target_label=None, epsilon=0.03, alpha=0.01, iterations=10):
        """Linf范数约束的PGD攻击"""
        try:
            original_image = image_tensor.clone()
            perturbed_image = image_tensor.clone() + torch.randn_like(image_tensor) * 0.01
            perturbed_image = torch.clamp(perturbed_image, 0, 1)
            
            for i in range(iterations):
                perturbed_image.requires_grad = True
                outputs = model(perturbed_image)
                
                if target_label is not None:
                    loss = -nn.CrossEntropyLoss()(outputs, torch.tensor([target_label]).to(self.device))
                else:
                    _, predicted = torch.max(outputs.data, 1)
                    loss = nn.CrossEntropyLoss()(outputs, predicted)
                
                model.zero_grad()
                loss.backward()
                
                # Linf投影
                data_grad = perturbed_image.grad.data.sign()
                perturbed_image = perturbed_image + alpha * data_grad
                eta = torch.clamp(perturbed_image - original_image, -epsilon, epsilon)
                perturbed_image = original_image + eta
                perturbed_image = torch.clamp(perturbed_image, 0, 1).detach_()
            
            return perturbed_image
        except Exception as e:
            print(f"PGD Linf attack error: {str(e)}")
            raise

    def pgd_l2(self, image_tensor, model, target_label=None, epsilon=0.03, alpha=0.01, iterations=10):
        """L2范数约束的PGD攻击"""
        try:
            original_image = image_tensor.clone()
            perturbed_image = image_tensor.clone() + torch.randn_like(image_tensor) * 0.01
            perturbed_image = torch.clamp(perturbed_image, 0, 1)
            
            for i in range(iterations):
                perturbed_image.requires_grad = True
                outputs = model(perturbed_image)
                
                if target_label is not None:
                    loss = -nn.CrossEntropyLoss()(outputs, torch.tensor([target_label]).to(self.device))
                else:
                    _, predicted = torch.max(outputs.data, 1)
                    loss = nn.CrossEntropyLoss()(outputs, predicted)
                
                model.zero_grad()
                loss.backward()
                
                # L2投影
                data_grad = perturbed_image.grad.data
                grad_norm = torch.norm(data_grad, p=2)
                normalized_grad = data_grad / (grad_norm + 1e-12)
                
                perturbed_image = perturbed_image + alpha * normalized_grad
                
                # L2球投影
                delta = perturbed_image - original_image
                delta_norm = torch.norm(delta, p=2)
                factor = torch.min(torch.tensor(1.0), epsilon / (delta_norm + 1e-12))
                delta = delta * factor
                
                perturbed_image = original_image + delta
                perturbed_image = torch.clamp(perturbed_image, 0, 1).detach_()
            
            return perturbed_image
        except Exception as e:
            print(f"PGD L2 attack error: {str(e)}")
            raise


class AdvCamShadowProcessor:
    """AdvCam和AdvShadow物理世界对抗攻击处理器"""

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def adv_cam_attack(self, image_path, intensity=0.1):
        """AdvCam攻击 - 相机镜头扰动模拟"""
        try:
            # 读取图像
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"无法读取图片: {image_path}")

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img_rgb.shape[:2]

            # 创建扰动贴纸位置
            patch_size = min(50, min(h, w) // 8)
            patches = [
                (0, 0),  # 左上角
                (w - patch_size, 0),  # 右上角
                (0, h - patch_size),  # 左下角
                (w // 2 - patch_size // 2, 0)  # 顶部中央
            ]

            # 复制图像进行扰动
            perturbed_img = img_rgb.copy().astype(np.float32)

            # 在指定位置添加噪声贴纸
            for x, y in patches:
                # 生成随机噪声
                noise = np.random.normal(0, intensity * 255, (patch_size, patch_size, 3))
                # 确保在图像边界内
                end_x = min(x + patch_size, w)
                end_y = min(y + patch_size, h)
                start_x, start_y = x, y

                # 应用噪声扰动
                patch_h, patch_w = end_y - start_y, end_x - start_x
                if patch_h > 0 and patch_w > 0:
                    perturbed_img[start_y:end_y, start_x:end_x] = np.clip(
                        perturbed_img[start_y:end_y, start_x:end_x] +
                        noise[:patch_h, :patch_w], 0, 255
                    )

            # 转换为uint8
            perturbed_img = perturbed_img.astype(np.uint8)

            # 保存对抗样本
            adv_filename = f'adv_cam_{os.path.basename(image_path)}'
            adv_path = os.path.join(current_app.config['RESULT_FOLDER'], adv_filename)

            # 转换回BGR保存
            adv_bgr = cv2.cvtColor(perturbed_img, cv2.COLOR_RGB2BGR)
            success = cv2.imwrite(adv_path, adv_bgr)
            if not success:
                raise ValueError(f"无法保存对抗样本: {adv_path}")

            print(f"AdvCam attack completed. Saved to: {adv_path}")
            return adv_path, perturbed_img

        except Exception as e:
            print(f"AdvCam attack error: {str(e)}")
            print(traceback.format_exc())
            raise

    def adv_shadow_attack(self, image_path, intensity=0.3):
        """AdvShadow攻击 - 阴影投射攻击"""
        try:
            # 读取图像
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"无法读取图片: {image_path}")

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img_rgb.shape[:2]

            # 创建渐变阴影遮罩
            shadow_mask = np.ones((h, w, 3), dtype=np.float32)

            # 创建径向渐变阴影（从左上角向外扩散）
            center_x, center_y = 0, 0  # 阴影源点
            max_distance = np.sqrt(h ** 2 + w ** 2)

            for i in range(h):
                for j in range(w):
                    # 计算到阴影源点的距离
                    distance = np.sqrt((i - center_y) ** 2 + (j - center_x) ** 2)
                    # 归一化距离
                    normalized_dist = distance / max_distance
                    # 计算阴影强度
                    shadow_strength = intensity * (1 - normalized_dist * 0.7)
                    shadow_mask[i, j] = 1 - shadow_strength

            # 应用阴影
            perturbed_img = (img_rgb.astype(np.float32) * shadow_mask).astype(np.uint8)

            # 保存对抗样本
            adv_filename = f'adv_shadow_{os.path.basename(image_path)}'
            adv_path = os.path.join(current_app.config['RESULT_FOLDER'], adv_filename)

            # 转换回BGR保存
            adv_bgr = cv2.cvtColor(perturbed_img, cv2.COLOR_RGB2BGR)
            success = cv2.imwrite(adv_path, adv_bgr)
            if not success:
                raise ValueError(f"无法保存对抗样本: {adv_path}")

            print(f"AdvShadow attack completed. Saved to: {adv_path}")
            return adv_path, perturbed_img

        except Exception as e:
            print(f"AdvShadow attack error: {str(e)}")
            print(traceback.format_exc())
            raise

    def combined_attack(self, image_path, cam_intensity=0.1, shadow_intensity=0.3):
        """组合攻击 - 先CAM后Shadow"""
        try:
            # 先应用CAM攻击
            temp_path, temp_img = self.adv_cam_attack(image_path, cam_intensity)

            # 再应用Shadow攻击
            final_path, final_img = self.adv_shadow_attack(temp_path, shadow_intensity)

            # 清理临时文件
            if os.path.exists(temp_path):
                os.remove(temp_path)

            print(f"Combined attack completed. Final result: {os.path.basename(final_path)}")
            return final_path, final_img

        except Exception as e:
            print(f"Combined attack error: {str(e)}")
            print(traceback.format_exc())
            # 清理可能的临时文件
            if 'temp_path' in locals() and os.path.exists(temp_path):
                os.remove(temp_path)
            raise


# 全局处理器实例
traditional_processor = AdversarialAttackProcessor()
physical_processor = AdvCamShadowProcessor()


def process_traditional_adversarial(image_path, model, attack_method='fgsm', **kwargs):
    """处理传统数字域对抗攻击"""
    return traditional_processor.generate_adversarial_example(image_path, model, attack_method, **kwargs)


def process_physical_adversarial(image_path, attack_type='cam', **kwargs):
    """处理物理世界对抗攻击"""
    if attack_type == 'cam':
        return physical_processor.adv_cam_attack(image_path, **kwargs)
    elif attack_type == 'shadow':
        return physical_processor.adv_shadow_attack(image_path, **kwargs)
    elif attack_type == 'combined':
        return physical_processor.combined_attack(image_path, **kwargs)
    else:
        raise ValueError(f"Unsupported attack type: {attack_type}")


# 为了向后兼容，保留原来的接口
def process_adversarial(image_path, model, attack_method='fgsm', **kwargs):
    """向后兼容的主处理接口"""
    if attack_method in ['fgsm', 'pgd', 'cw']:
        return process_traditional_adversarial(image_path, model, attack_method, **kwargs)
    elif attack_method in ['cam', 'shadow', 'combined']:
        return process_physical_adversarial(image_path, attack_method, **kwargs)
    else:
        raise ValueError(f"Unsupported attack method: {attack_method}")
