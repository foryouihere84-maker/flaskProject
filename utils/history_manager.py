import os
import json
import time
from datetime import datetime
from flask import current_app
import threading
import glob

class HistoryManager:
    """历史记录管理器"""
    
    def __init__(self):
        self.history_file = None
        self.lock = threading.Lock()
        
    def _get_history_file_path(self):
        """延迟获取历史文件路径（在应用上下文中）"""
        if self.history_file is None:
            self.history_file = os.path.join(current_app.config['BASE_DIR'], 'history.json')
            self._ensure_history_file()
        return self.history_file
    
    def _ensure_history_file(self):
        """确保历史记录文件存在"""
        if not os.path.exists(self.history_file):
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump([], f, ensure_ascii=False, indent=2)
    
    def _load_history(self):
        """加载历史记录"""
        self._get_history_file_path()  # 确保文件路径已初始化
        try:
            with open(self.history_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []
    
    def _save_history(self, history):
        """保存历史记录"""
        self._get_history_file_path()  # 确保文件路径已初始化
        with open(self.history_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    
    def add_record(self, operation_type, original_image, result_image=None, additional_images=None, **kwargs):
        """
        添加历史记录
        
        Args:
            operation_type: 操作类型 ('detection' 或 'adversarial')
            original_image: 原始图片文件名
            result_image: 结果图片文件名（可选）
            additional_images: 额外的图片文件名列表（如对抗攻击中的中间步骤图片）
            **kwargs: 其他参数（如模型名称、攻击参数等）
        """
        with self.lock:
            history = self._load_history()
            
            record = {
                'id': str(int(time.time() * 1000000)),  # 使用微秒时间戳作为唯一ID
                'operation_type': operation_type,
                'original_image': original_image,
                'result_image': result_image,
                'additional_images': additional_images or [],
                'timestamp': datetime.now().isoformat(),
                'params': kwargs
            }
            
            history.append(record)
            self._save_history(history)
            return record['id']
    
    def get_records(self, limit=None, offset=0):
        """
        获取历史记录
        
        Args:
            limit: 限制返回记录数
            offset: 偏移量
            
        Returns:
            历史记录列表（按时间倒序排列）
        """
        history = self._load_history()
        # 按时间倒序排列
        history.sort(key=lambda x: x['timestamp'], reverse=True)
        
        if limit:
            return history[offset:offset + limit]
        return history
    
    def get_record(self, record_id):
        """根据ID获取单条记录"""
        history = self._load_history()
        for record in history:
            if record['id'] == record_id:
                return record
        return None
    
    def delete_record(self, record_id):
        """
        删除历史记录及对应的图片文件
        
        Args:
            record_id: 记录ID
            
        Returns:
            bool: 删除是否成功
        """
        with self.lock:
            history = self._load_history()
            record_to_delete = None
            
            # 查找要删除的记录
            for i, record in enumerate(history):
                if record['id'] == record_id:
                    record_to_delete = record
                    del history[i]
                    break
            
            if not record_to_delete:
                return False
            
            # 删除对应的图片文件
            self._delete_associated_files(record_to_delete)
            
            # 保存更新后的历史记录
            self._save_history(history)
            return True
    
    def _delete_associated_files(self, record):
        """删除与记录关联的图片文件（包括智能匹配的results目录文件）"""
        upload_folder = current_app.config['UPLOAD_FOLDER']
        result_folder = current_app.config['RESULT_FOLDER']
        
        # 删除原始图片
        if record['original_image']:
            original_path = os.path.join(upload_folder, record['original_image'])
            if os.path.exists(original_path):
                try:
                    # 获取原始图片的创建时间
                    original_ctime = os.path.getctime(original_path)
                    os.remove(original_path)
                    print(f"Deleted original image: {original_path}")
                except Exception as e:
                    print(f"Failed to delete original image {original_path}: {e}")
                    original_ctime = None
            else:
                original_ctime = None
        else:
            original_ctime = None
        
        # 删除结果图片
        if record['result_image']:
            result_path = os.path.join(result_folder, record['result_image'])
            if os.path.exists(result_path):
                try:
                    os.remove(result_path)
                    print(f"Deleted result image: {result_path}")
                except Exception as e:
                    print(f"Failed to delete result image {result_path}: {e}")
        
        # 删除额外的图片
        for image_name in record.get('additional_images', []):
            # 额外图片可能在uploads或results目录中
            for folder in [upload_folder, result_folder]:
                image_path = os.path.join(folder, image_name)
                if os.path.exists(image_path):
                    try:
                        os.remove(image_path)
                        print(f"Deleted additional image: {image_path}")
                        break
                    except Exception as e:
                        print(f"Failed to delete additional image {image_path}: {e}")
        
        # 智能删除results目录下相关的图片文件
        if record['original_image'] and original_ctime:
            self._delete_related_results_files(record['original_image'], original_ctime, result_folder)
    
    def _delete_related_results_files(self, original_filename, original_ctime, result_folder):
        """删除results目录下与原图片相关的文件（名称包含原图片名且创建时间在5分钟内）"""
        try:
            # 获取原图片的基本名称（不含扩展名）
            original_basename = os.path.splitext(original_filename)[0]
            
            # 在results目录中查找所有图片文件
            image_patterns = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif']
            related_files = []
            
            for pattern in image_patterns:
                pattern_path = os.path.join(result_folder, pattern)
                files = glob.glob(pattern_path)
                related_files.extend(files)
            
            # 时间窗口：5分钟（300秒）
            time_window = 300
            deleted_count = 0
            
            print(f"Searching for related files containing '{original_basename}' in results folder...")
            
            for file_path in related_files:
                try:
                    # 检查文件名是否包含原图片的基本名称
                    filename = os.path.basename(file_path)
                    filename_without_ext = os.path.splitext(filename)[0]
                    
                    # 检查是否包含原图片名（支持部分匹配）
                    if original_basename in filename_without_ext or filename_without_ext in original_basename:
                        # 检查创建时间是否在5分钟内
                        file_ctime = os.path.getctime(file_path)
                        time_diff = abs(file_ctime - original_ctime)
                        
                        if time_diff <= time_window:
                            # 删除文件
                            os.remove(file_path)
                            deleted_count += 1
                            print(f"Deleted related result file: {filename} (time diff: {time_diff:.1f}s)")
                        else:
                            print(f"Skipped file {filename} - time difference too large: {time_diff:.1f}s")
                    else:
                        # 文件名不匹配，跳过
                        pass
                        
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")
                    continue
            
            print(f"Deleted {deleted_count} related result files for original image: {original_filename}")
            
        except Exception as e:
            print(f"Error in _delete_related_results_files: {e}")
    
    def get_statistics(self):
        """获取历史记录统计信息"""
        history = self._load_history()
        
        stats = {
            'total_records': len(history),
            'detection_count': 0,
            'adversarial_count': 0,
            'today_count': 0
        }
        
        today = datetime.now().date()
        
        for record in history:
            # 统计操作类型
            if record['operation_type'] == 'detection':
                stats['detection_count'] += 1
            elif record['operation_type'] == 'adversarial':
                stats['adversarial_count'] += 1
            
            # 统计今天的记录
            record_date = datetime.fromisoformat(record['timestamp']).date()
            if record_date == today:
                stats['today_count'] += 1
        
        return stats

# 全局历史管理器实例
history_manager = HistoryManager()