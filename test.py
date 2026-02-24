import torch
import requests
import os
from tqdm import tqdm


def download_robustbench_model(model_name, save_path):
    """从RobustBench下载模型"""

    # RobustBench模型下载链接映射
    model_urls = {
        'Standard_RN50': 'https://github.com/RobustBench/robustbench/releases/download/v1.0/ImageNet.pt',
        'TRADES': 'https://github.com/RobustBench/robustbench/releases/download/v1.0/ImageNet_TRADES.pt',
        'FreeAT': 'https://github.com/RobustBench/robustbench/releases/download/v1.0/ImageNet_FreeAT.pt',
        'FastAT': 'https://github.com/RobustBench/robustbench/releases/download/v1.0/ImageNet_FastAT.pt'
    }

    if model_name not in model_urls:
        print(f"不支持的模型: {model_name}")
        return False

    url = model_urls[model_name]

    # 创建保存目录
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print(f"正在下载 {model_name}...")

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))

        with open(save_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=model_name) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))

        print(f"✓ {model_name} 下载完成: {save_path}")
        return True

    except Exception as e:
        print(f"✗ 下载失败: {str(e)}")
        return False


# 使用示例
if __name__ == "__main__":
    # 确保models/adversarial目录存在
    save_dir = "models/adversarial"

    # 下载推荐模型
    models_to_download = [
        'Standard_RN50',
        'TRADES',
        'FreeAT',
        'FastAT'
    ]

    for model_name in models_to_download:
        save_path = os.path.join(save_dir, f"{model_name}.pth")
        download_robustbench_model(model_name, save_path)
