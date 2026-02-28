from flask import Blueprint, render_template, request, jsonify, current_app
from utils.history_manager import history_manager
import traceback

history_bp = Blueprint('history', __name__, url_prefix='/history')

@history_bp.route('/', methods=['GET'])
def history_index():
    """历史记录主页"""
    try:
        # 获取所有历史记录
        records = history_manager.get_records()
        
        # 获取统计信息
        stats = history_manager.get_statistics()
        
        return render_template('history.html', records=records, stats=stats)
        
    except Exception as e:
        print(f"History page error: {str(e)}")
        print(traceback.format_exc())
        # 返回空记录页面
        return render_template('history.html', records=[], stats={
            'total_records': 0,
            'detection_count': 0,
            'adversarial_count': 0,
            'today_count': 0
        })

@history_bp.route('/delete/<record_id>', methods=['POST'])
def delete_record(record_id):
    """删除历史记录"""
    try:
        success = history_manager.delete_record(record_id)
        
        if success:
            return jsonify({
                'success': True,
                'message': '记录删除成功'
            })
        else:
            return jsonify({
                'success': False,
                'message': '记录不存在或删除失败'
            }), 404
            
    except Exception as e:
        print(f"Delete record error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            'success': False,
            'message': f'删除过程中发生错误: {str(e)}'
        }), 500

@history_bp.route('/api/stats', methods=['GET'])
def api_stats():
    """API获取统计信息"""
    try:
        stats = history_manager.get_statistics()
        return jsonify(stats)
    except Exception as e:
        print(f"API stats error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@history_bp.route('/api/records', methods=['GET'])
def api_records():
    """API获取记录列表"""
    try:
        limit = request.args.get('limit', type=int)
        offset = request.args.get('offset', 0, type=int)
        operation_type = request.args.get('type')
        
        records = history_manager.get_records(limit=limit, offset=offset)
        
        # 如果指定了操作类型，进行过滤
        if operation_type:
            records = [r for r in records if r['operation_type'] == operation_type]
        
        return jsonify(records)
    except Exception as e:
        print(f"API records error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@history_bp.route('/clear', methods=['POST'])
def clear_history():
    """清空所有历史记录"""
    try:
        # 获取所有记录
        records = history_manager.get_records()
        
        # 逐个删除
        deleted_count = 0
        for record in records:
            if history_manager.delete_record(record['id']):
                deleted_count += 1
        
        return jsonify({
            'success': True,
            'message': f'成功删除 {deleted_count} 条记录'
        })
        
    except Exception as e:
        print(f"Clear history error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            'success': False,
            'message': f'清空过程中发生错误: {str(e)}'
        }), 500