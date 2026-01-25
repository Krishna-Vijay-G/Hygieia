"""
Authentication Routes
"""

from flask import Blueprint, request, jsonify, send_from_directory
from flask_jwt_extended import (
    create_access_token, create_refresh_token, 
    jwt_required, get_jwt_identity, get_jwt
)
from datetime import datetime
from app import db
from app.models import User, AuditLog
import os
import base64
import uuid
import shutil
from datetime import datetime as _dt

auth_bp = Blueprint('auth', __name__)


@auth_bp.route('/register', methods=['POST'])
def register():
    """Register a new user"""
    data = request.get_json()
    
    # Validate required fields
    required_fields = ['username', 'email', 'password', 'first_name', 'last_name']
    for field in required_fields:
        if not data.get(field):
            return jsonify({'error': f'{field} is required'}), 400
    
    # Check if username already exists
    if User.query.filter_by(username=data['username']).first():
        return jsonify({'error': 'Username already taken'}), 409
    
    # Check if email already exists
    if User.query.filter_by(email=data['email']).first():
        return jsonify({'error': 'Email already registered'}), 409
    
    # Validate password strength
    if len(data['password']) < 8:
        return jsonify({'error': 'Password must be at least 8 characters'}), 400
    
    # Create new user
    user = User(
        username=data['username'],
        email=data['email'].lower(),
        first_name=data['first_name'],
        last_name=data['last_name'],
        phone=data.get('phone'),
        is_admin=False
    )
    user.set_password(data['password'])
    
    try:
        db.session.add(user)
        db.session.commit()
        
        # Log registration
        audit = AuditLog(
            user_id=user.id,
            action='user_registered',
            resource_type='user',
            resource_id=user.id,
            ip_address=request.remote_addr,
            user_agent=request.user_agent.string[:500] if request.user_agent else None
        )
        db.session.add(audit)
        db.session.commit()
        
        # Create tokens
        access_token = create_access_token(identity=user.id)
        refresh_token = create_refresh_token(identity=user.id)
        
        return jsonify({
            'message': 'Registration successful',
            'user': user.to_dict(include_private=True),
            'access_token': access_token,
            'refresh_token': refresh_token
        }), 201
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': 'Registration failed', 'details': str(e)}), 500


@auth_bp.route('/login', methods=['POST'])
def login():
    """User login"""
    data = request.get_json()
    
    if not data.get('login') or not data.get('password'):
        return jsonify({'error': 'Login credentials required'}), 400
    
    # Find user by username or email
    login_value = data['login'].lower()
    user = User.query.filter(
        (User.username == login_value) | (User.email == login_value)
    ).first()
    
    if not user or not user.check_password(data['password']):
        return jsonify({'error': 'Invalid credentials'}), 401
    
    if not user.is_active:
        return jsonify({'error': 'Account is deactivated'}), 403
    
    # Update last login
    user.last_login = datetime.utcnow()
    
    # Log login
    audit = AuditLog(
        user_id=user.id,
        action='user_login',
        resource_type='user',
        resource_id=user.id,
        ip_address=request.remote_addr,
        user_agent=request.user_agent.string[:500] if request.user_agent else None
    )
    db.session.add(audit)
    db.session.commit()
    
    # Create tokens
    access_token = create_access_token(identity=user.id)
    refresh_token = create_refresh_token(identity=user.id)
    
    return jsonify({
        'message': 'Login successful',
        'user': user.to_dict(include_private=True),
        'access_token': access_token,
        'refresh_token': refresh_token
    })


@auth_bp.route('/refresh', methods=['POST'])
@jwt_required(refresh=True)
def refresh():
    """Refresh access token"""
    current_user_id = get_jwt_identity()
    user = User.query.get(current_user_id)
    
    if not user or not user.is_active:
        return jsonify({'error': 'Invalid user'}), 401
    
    access_token = create_access_token(identity=current_user_id)
    return jsonify({'access_token': access_token})


@auth_bp.route('/me', methods=['GET'])
@jwt_required()
def get_current_user():
    """Get current user profile"""
    current_user_id = get_jwt_identity()
    user = User.query.get(current_user_id)
    
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    return jsonify({'user': user.to_dict(include_private=True)})


@auth_bp.route('/me', methods=['PUT'])
@jwt_required()
def update_profile():
    """Update current user profile"""
    current_user_id = get_jwt_identity()
    user = User.query.get(current_user_id)
    
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    data = request.get_json()
    
    # Update allowed fields
    if 'first_name' in data:
        user.first_name = data['first_name']
    if 'last_name' in data:
        user.last_name = data['last_name']
    if 'phone' in data:
        user.phone = data['phone']
    if 'avatar_url' in data:
        # Accept either a URL/path or a data URL (base64). If it's a data URL, decode and save.
        avatar_val = data['avatar_url']
        try:
            if isinstance(avatar_val, str) and avatar_val.startswith('data:'):
                # parse data URL
                header, b64 = avatar_val.split(',', 1)
                mime = header.split(';')[0].split(':')[1] if ';' in header else 'image/png'
                ext = 'png'
                if 'jpeg' in mime or 'jpg' in mime:
                    ext = 'jpg'
                elif 'png' in mime:
                    ext = 'png'
                elif 'gif' in mime:
                    ext = 'gif'

                uploads_root = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'uploads'))
                avatars_dir = os.path.join(uploads_root, 'avatars')
                os.makedirs(avatars_dir, exist_ok=True)

                # If the user has an existing local avatar, archive it instead of deleting
                try:
                    old_avatar = user.avatar_url
                    if old_avatar and isinstance(old_avatar, str) and old_avatar.startswith('/uploads/'):
                        # derive relative path under uploads
                        rel_path = old_avatar[len('/uploads/'):]  # e.g. 'avatars/old.png'
                        old_full_path = os.path.join(uploads_root, rel_path)
                        if os.path.exists(old_full_path):
                            # create archive folder with username and timestamp
                            ts = _dt.utcnow().strftime('%Y%m%dT%H%M%SZ')
                            safe_username = (user.username or f'user_{user.id}').replace(' ', '_')
                            archive_dir = os.path.join(uploads_root, 'avatars_archive', f"{safe_username}_{ts}")
                            os.makedirs(archive_dir, exist_ok=True)
                            try:
                                shutil.move(old_full_path, os.path.join(archive_dir, os.path.basename(old_full_path)))
                            except Exception:
                                # non-fatal: continue even if archive fails
                                pass
                except Exception:
                    # ignore archive errors and continue
                    pass

                filename = f"avatar_{user.id}_{uuid.uuid4().hex}.{ext}"
                file_path = os.path.join(avatars_dir, filename)

                # Decode and write file
                file_bytes = base64.b64decode(b64)
                with open(file_path, 'wb') as f:
                    f.write(file_bytes)

                # Set URL path for client to load (served from /uploads/... configured in app)
                user.avatar_url = f"/uploads/avatars/{filename}"
            else:
                # plain URL or path
                user.avatar_url = avatar_val
        except Exception as e:
            return jsonify({'error': 'Failed to process avatar image', 'details': str(e)}), 400
    
    # Update password if provided
    if data.get('new_password'):
        if not data.get('current_password'):
            return jsonify({'error': 'Current password required'}), 400
        if not user.check_password(data['current_password']):
            return jsonify({'error': 'Current password is incorrect'}), 401
        if len(data['new_password']) < 8:
            return jsonify({'error': 'New password must be at least 8 characters'}), 400
        user.set_password(data['new_password'])
    
    try:
        db.session.commit()
        return jsonify({
            'message': 'Profile updated',
            'user': user.to_dict(include_private=True)
        })
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': 'Update failed', 'details': str(e)}), 500


@auth_bp.route('/change-password', methods=['PUT'])
@jwt_required()
def change_password():
    """Change current user's password"""
    current_user_id = get_jwt_identity()
    user = User.query.get(current_user_id)
    
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    data = request.get_json()
    
    if not data.get('current_password') or not data.get('new_password'):
        return jsonify({'error': 'Current password and new password are required'}), 400
    
    if not user.check_password(data['current_password']):
        return jsonify({'error': 'Current password is incorrect'}), 401
    
    if len(data['new_password']) < 8:
        return jsonify({'error': 'New password must be at least 8 characters'}), 400
    
    user.set_password(data['new_password'])
    
    try:
        db.session.commit()
        
        # Log password change
        audit = AuditLog(
            user_id=user.id,
            action='password_changed',
            resource_type='user',
            resource_id=user.id,
            ip_address=request.remote_addr
        )
        db.session.add(audit)
        db.session.commit()
        
        return jsonify({'message': 'Password changed successfully'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': 'Password change failed', 'details': str(e)}), 500


@auth_bp.route('/deactivate', methods=['POST'])
@jwt_required()
def deactivate_account():
    """Deactivate current user's account"""
    current_user_id = get_jwt_identity()
    user = User.query.get(current_user_id)
    
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    if user.is_admin:
        # Check if this is the only admin
        admin_count = User.query.filter_by(is_admin=True, is_active=True).count()
        if admin_count <= 1:
            return jsonify({'error': 'Cannot deactivate the last active admin account'}), 400
    
    user.is_active = False
    
    try:
        db.session.commit()
        
        # Log deactivation
        audit = AuditLog(
            user_id=user.id,
            action='account_deactivated',
            resource_type='user',
            resource_id=user.id,
            ip_address=request.remote_addr
        )
        db.session.add(audit)
        db.session.commit()
        
        return jsonify({'message': 'Account deactivated successfully'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': 'Account deactivation failed', 'details': str(e)}), 500


@auth_bp.route('/logout', methods=['POST'])
@jwt_required()
def logout():
    """User logout (client should discard tokens)"""
    current_user_id = get_jwt_identity()
    
    # Log logout
    audit = AuditLog(
        user_id=current_user_id,
        action='user_logout',
        resource_type='user',
        resource_id=current_user_id,
        ip_address=request.remote_addr
    )
    db.session.add(audit)
    db.session.commit()
    
    return jsonify({'message': 'Logout successful'})


# Note: uploads are served by the main app at /uploads/<path:filename>
