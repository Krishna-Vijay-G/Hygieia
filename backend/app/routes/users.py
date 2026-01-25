"""
User Management Routes (Admin only)
"""

from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from app import db
from app.models import User, Analysis, AuditLog

users_bp = Blueprint('users', __name__)


def admin_required(fn):
    """Decorator to require admin access"""
    from functools import wraps
    
    @wraps(fn)
    @jwt_required()
    def wrapper(*args, **kwargs):
        current_user_id = get_jwt_identity()
        user = User.query.get(current_user_id)
        if not user or not user.is_admin:
            return jsonify({'error': 'Admin access required'}), 403
        return fn(*args, **kwargs)
    return wrapper


@users_bp.route('', methods=['GET'])
@admin_required
def get_all_users():
    """Get all users (admin only)"""
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 50, type=int)
    search = request.args.get('search', '')
    
    query = User.query
    
    if search:
        search_term = f"%{search}%"
        query = query.filter(
            (User.username.ilike(search_term)) |
            (User.email.ilike(search_term)) |
            (User.first_name.ilike(search_term)) |
            (User.last_name.ilike(search_term))
        )
    
    pagination = query.order_by(User.created_at.desc()).paginate(
        page=page, per_page=per_page, error_out=False
    )
    
    # Add analysis count to each user
    users_with_counts = []
    for user in pagination.items:
        user_dict = user.to_dict(include_private=True)
        analysis_count = Analysis.query.filter_by(user_id=user.id).count()
        user_dict['analysis_count'] = analysis_count
        users_with_counts.append(user_dict)
    
    return jsonify({
        'users': users_with_counts,
        'total': pagination.total,
        'pages': pagination.pages,
        'current_page': page
    })


@users_bp.route('/<user_id>', methods=['GET'])
@admin_required
def get_user(user_id):
    """Get specific user profile (admin only)"""
    user = User.query.get(user_id)
    
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    return jsonify({
        'user': user.to_dict(include_private=True)
    })


@users_bp.route('/<user_id>/analyses', methods=['GET'])
@admin_required
def get_user_analyses(user_id):
    """Get specific user's analyses (admin only)"""
    user = User.query.get(user_id)
    
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 20, type=int)
    
    pagination = Analysis.query.filter_by(user_id=user_id).order_by(
        Analysis.created_at.desc()
    ).paginate(page=page, per_page=per_page, error_out=False)
    
    # Get counts by analysis type
    from sqlalchemy import func
    count_query = db.session.query(
        Analysis.analysis_type,
        func.count(Analysis.id).label('count')
    ).filter_by(user_id=user_id).group_by(Analysis.analysis_type)
    
    counts = {row.analysis_type: row.count for row in count_query.all()}
    
    return jsonify({
        'user': user.to_dict(),
        'analyses': [a.to_dict() for a in pagination.items],
        'total': pagination.total,
        'pages': pagination.pages,
        'current_page': page,
        'counts': counts
    })


@users_bp.route('/<user_id>/toggle-admin', methods=['POST'])
@admin_required
def toggle_admin(user_id):
    """Toggle user admin status"""
    current_user_id = get_jwt_identity()
    
    if user_id == current_user_id:
        return jsonify({'error': 'Cannot modify your own admin status'}), 400
    
    user = User.query.get(user_id)
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    # Prevent modification of owner users
    if user.is_owner:
        return jsonify({'error': 'Cannot modify owner user permissions. Owner accounts are protected.'}), 403
    
    user.is_admin = not user.is_admin
    
    # Log action
    audit = AuditLog(
        user_id=current_user_id,
        action='toggle_admin',
        resource_type='user',
        resource_id=user_id,
        details={'new_status': user.is_admin},
        ip_address=request.remote_addr
    )
    db.session.add(audit)
    db.session.commit()
    
    return jsonify({
        'message': f"Admin status {'granted' if user.is_admin else 'revoked'}",
        'user': user.to_dict(include_private=True)
    })


@users_bp.route('/<user_id>/toggle-active', methods=['POST'])
@admin_required
def toggle_active(user_id):
    """Toggle user active status"""
    current_user_id = get_jwt_identity()
    
    if user_id == current_user_id:
        return jsonify({'error': 'Cannot deactivate your own account'}), 400
    
    user = User.query.get(user_id)
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    # Prevent modification of owner users
    if user.is_owner:
        return jsonify({'error': 'Cannot modify owner user status. Owner accounts are protected.'}), 403
    
    user.is_active = not user.is_active
    
    # Log action
    audit = AuditLog(
        user_id=current_user_id,
        action='toggle_active',
        resource_type='user',
        resource_id=user_id,
        details={'new_status': user.is_active},
        ip_address=request.remote_addr
    )
    db.session.add(audit)
    db.session.commit()
    
    return jsonify({
        'message': f"User {'activated' if user.is_active else 'deactivated'}",
        'user': user.to_dict(include_private=True)
    })


@users_bp.route('/<user_id>', methods=['DELETE'])
@admin_required
def delete_user(user_id):
    """Delete a user (admin only)"""
    current_user_id = get_jwt_identity()
    
    if user_id == current_user_id:
        return jsonify({'error': 'Cannot delete your own account'}), 400
    
    user = User.query.get(user_id)
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    # Prevent deletion of owner users
    if user.is_owner:
        return jsonify({'error': 'Cannot delete owner users. Owner accounts are protected for system integrity.'}), 403
    
    username = user.username
    
    # Log action before deletion
    audit = AuditLog(
        user_id=current_user_id,
        action='delete_user',
        resource_type='user',
        resource_id=user_id,
        details={'deleted_username': username},
        ip_address=request.remote_addr
    )
    db.session.add(audit)
    
    db.session.delete(user)
    db.session.commit()
    
    return jsonify({'message': f"User '{username}' deleted successfully"})


@users_bp.route('/stats', methods=['GET'])
@admin_required
def get_user_stats():
    """Get user statistics"""
    from sqlalchemy import func
    from datetime import datetime, timedelta
    
    total_users = User.query.count()
    active_users = User.query.filter_by(is_active=True).count()
    admin_users = User.query.filter_by(is_admin=True).count()
    
    # New users this month
    month_ago = datetime.utcnow() - timedelta(days=30)
    new_users_month = User.query.filter(User.created_at >= month_ago).count()
    
    # Users by login activity
    week_ago = datetime.utcnow() - timedelta(days=7)
    active_this_week = User.query.filter(User.last_login >= week_ago).count()
    
    return jsonify({
        'total_users': total_users,
        'active_users': active_users,
        'admin_users': admin_users,
        'new_users_month': new_users_month,
        'active_this_week': active_this_week
    })
