"""
Database Models for Hygieia
"""

from datetime import datetime, timezone
from werkzeug.security import generate_password_hash, check_password_hash
from app import db
import uuid


def utcnow():
    """Return timezone-aware UTC datetime"""
    return datetime.now(timezone.utc)


class User(db.Model):
    """User account model"""
    __tablename__ = 'users'
    
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(256), nullable=False)
    first_name = db.Column(db.String(80), nullable=False)
    last_name = db.Column(db.String(80), nullable=False)
    phone = db.Column(db.String(20), nullable=True)
    avatar_url = db.Column(db.String(500), nullable=True)
    is_admin = db.Column(db.Boolean, default=False, nullable=False)
    is_owner = db.Column(db.Boolean, default=False, nullable=False)
    is_active = db.Column(db.Boolean, default=True, nullable=False)
    is_verified = db.Column(db.Boolean, default=False, nullable=False)
    created_at = db.Column(db.DateTime(timezone=True), default=utcnow, nullable=False)
    updated_at = db.Column(db.DateTime(timezone=True), default=utcnow, onupdate=utcnow)
    last_login = db.Column(db.DateTime(timezone=True), nullable=True)
    
    # Relationships
    analyses = db.relationship('Analysis', backref='user', lazy='dynamic', cascade='all, delete-orphan')
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
    
    def to_dict(self, include_private=False):
        data = {
            'id': self.id,
            'username': self.username,
            'first_name': self.first_name,
            'last_name': self.last_name,
            'full_name': f"{self.first_name} {self.last_name}",
            'avatar_url': self.avatar_url,
            'is_admin': self.is_admin,
            'is_owner': self.is_owner,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }
        if include_private:
            data.update({
                'email': self.email,
                'phone': self.phone,
                'is_active': self.is_active,
                'is_verified': self.is_verified,
                'last_login': self.last_login.isoformat() if self.last_login else None,
                'updated_at': self.updated_at.isoformat() if self.updated_at else None
            })
        return data


class Analysis(db.Model):
    """Analysis results model"""
    __tablename__ = 'analyses'
    
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = db.Column(db.String(36), db.ForeignKey('users.id', ondelete='CASCADE'), nullable=False, index=True)
    analysis_type = db.Column(db.String(50), nullable=False, index=True)  # breast_diagnosis, breast_risk, skin_diagnosis, heart_risk, diabetes_risk
    model_name = db.Column(db.String(100), nullable=True)
    input_data = db.Column(db.JSON, nullable=True)
    result = db.Column(db.JSON, nullable=False)
    risk_level = db.Column(db.String(50), nullable=True)
    confidence = db.Column(db.Float, nullable=True)
    risk_score = db.Column(db.Float, nullable=True)
    image_path = db.Column(db.String(500), nullable=True)
    blockchain_hash = db.Column(db.String(256), nullable=True, index=True)
    ai_summary = db.Column(db.Text, nullable=True)  # Auto-generated AI summary, stored permanently
    created_at = db.Column(db.DateTime(timezone=True), default=utcnow, nullable=False, index=True)
    
    def to_dict(self, include_user=False):
        data = {
            'id': self.id,
            'user_id': self.user_id,
            'analysis_type': self.analysis_type,
            'model_name': self.model_name,
            'input_data': self.input_data,
            'result': self.result,
            'result_data': self.result,  # Alias for frontend compatibility
            'risk_level': self.risk_level,
            'confidence': self.confidence,
            'risk_score': self.risk_score,
            'image_path': self.image_path,
            'blockchain_hash': self.blockchain_hash,
            'ai_summary': self.ai_summary,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }
        if include_user and self.user:
            data['user'] = {
                'id': self.user.id,
                'username': self.user.username,
                'full_name': f"{self.user.first_name} {self.user.last_name}"
            }
        return data


class BlockchainRecord(db.Model):
    """Blockchain verification records"""
    __tablename__ = 'blockchain_records'
    
    id = db.Column(db.Integer, primary_key=True)
    block_index = db.Column(db.Integer, nullable=False, unique=True, index=True)
    timestamp = db.Column(db.DateTime(timezone=True), default=utcnow, nullable=False)
    data_hash = db.Column(db.String(256), nullable=False, index=True)
    previous_hash = db.Column(db.String(256), nullable=False)
    current_hash = db.Column(db.String(256), nullable=False, unique=True, index=True)
    nonce = db.Column(db.Integer, default=0)
    analysis_id = db.Column(db.String(36), db.ForeignKey('analyses.id', ondelete='SET NULL'), nullable=True, index=True)
    
    # Relationship
    analysis = db.relationship('Analysis', backref='blockchain_record', uselist=False)
    
    def to_dict(self, include_analysis=False):
        data = {
            'id': self.id,
            'block_index': self.block_index,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'data_hash': self.data_hash,
            'previous_hash': self.previous_hash,
            'current_hash': self.current_hash,
            'nonce': self.nonce,
            'analysis_id': self.analysis_id
        }
        if include_analysis and self.analysis:
            data['analysis'] = self.analysis.to_dict(include_user=True)
        return data


class ModelBenchmark(db.Model):
    """Model benchmark results"""
    __tablename__ = 'model_benchmarks'
    
    id = db.Column(db.Integer, primary_key=True)
    model_type = db.Column(db.String(50), nullable=False, index=True)  # breast_diagnosis, breast_risk, skin_diagnosis, heart_risk, diabetes_risk
    model_name = db.Column(db.String(100), nullable=False)
    accuracy = db.Column(db.Float, nullable=True)
    precision_score = db.Column(db.Float, nullable=True)
    recall = db.Column(db.Float, nullable=True)
    f1_score = db.Column(db.Float, nullable=True)
    auc_roc = db.Column(db.Float, nullable=True)
    test_samples = db.Column(db.Integer, nullable=True)
    training_samples = db.Column(db.Integer, nullable=True)
    benchmark_date = db.Column(db.DateTime(timezone=True), default=utcnow, nullable=False)
    additional_metrics = db.Column(db.JSON, nullable=True)
    
    def to_dict(self):
        return {
            'id': self.id,
            'model_type': self.model_type,
            'model_name': self.model_name,
            'accuracy': self.accuracy,
            'precision_score': self.precision_score,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'auc_roc': self.auc_roc,
            'test_samples': self.test_samples,
            'training_samples': self.training_samples,
            'benchmark_date': self.benchmark_date.isoformat() if self.benchmark_date else None,
            'additional_metrics': self.additional_metrics
        }


class AuditLog(db.Model):
    """Audit log for tracking system activities"""
    __tablename__ = 'audit_logs'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(36), db.ForeignKey('users.id', ondelete='SET NULL'), nullable=True, index=True)
    action = db.Column(db.String(100), nullable=False, index=True)
    resource_type = db.Column(db.String(50), nullable=True)
    resource_id = db.Column(db.String(36), nullable=True)
    details = db.Column(db.JSON, nullable=True)
    ip_address = db.Column(db.String(45), nullable=True)
    user_agent = db.Column(db.String(500), nullable=True)
    created_at = db.Column(db.DateTime(timezone=True), default=utcnow, nullable=False, index=True)
    
    # Relationship
    user = db.relationship('User', backref='audit_logs')
    
    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'action': self.action,
            'resource_type': self.resource_type,
            'resource_id': self.resource_id,
            'details': self.details,
            'ip_address': self.ip_address,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


class ChatSession(db.Model):
    """Chat session model for Dr. Hygieia AI conversations with messages stored as JSON"""
    __tablename__ = 'chat_sessions'
    
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = db.Column(db.String(36), db.ForeignKey('users.id', ondelete='CASCADE'), nullable=False, index=True)
    title = db.Column(db.String(255), default='New Conversation')
    analysis_id = db.Column(db.String(36), db.ForeignKey('analyses.id', ondelete='SET NULL'), nullable=True, index=True)
    context_type = db.Column(db.String(50), nullable=True)  # 'general', 'analysis', 'follow-up'
    context_data = db.Column(db.JSON, nullable=True)
    messages = db.Column(db.JSON, nullable=False, default=list)  # Store messages as JSON array
    is_active = db.Column(db.Boolean, default=True, nullable=False)
    created_at = db.Column(db.DateTime(timezone=True), default=utcnow, nullable=False, index=True)
    updated_at = db.Column(db.DateTime(timezone=True), default=utcnow, onupdate=utcnow)
    
    # Relationships
    user = db.relationship('User', backref=db.backref('chat_sessions', lazy='dynamic', cascade='all, delete-orphan'))
    analysis = db.relationship('Analysis', backref=db.backref('chat_sessions', lazy='dynamic'))
    
    def add_message(self, role, content, metadata=None):
        """Add a message to the session"""
        message = {
            'id': str(uuid.uuid4()),
            'role': role,
            'content': content,
            'metadata': metadata,
            'created_at': datetime.now(timezone.utc).isoformat()
        }
        if not isinstance(self.messages, list):
            self.messages = []
        self.messages = self.messages + [message]  # Create new list to trigger SQLAlchemy update
        self.updated_at = datetime.now(timezone.utc)
        return message
    
    def get_messages(self, exclude_system=False):
        """Get messages, optionally excluding system messages"""
        messages = self.messages if isinstance(self.messages, list) else []
        if exclude_system:
            return [m for m in messages if m.get('role') != 'system']
        return messages
    
    def to_dict(self, include_messages=False, include_analysis=False, include_user=False):
        data = {
            'id': self.id,
            'user_id': self.user_id,
            'title': self.title,
            'analysis_id': self.analysis_id,
            'context_type': self.context_type,
            'context_data': self.context_data,
            'is_active': self.is_active,
            'message_count': len(self.messages) if isinstance(self.messages, list) else 0,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }
        if include_messages:
            data['messages'] = self.get_messages()
        if include_analysis and self.analysis:
            data['analysis'] = self.analysis.to_dict()
        if include_user and self.user:
            data['user'] = {
                'id': self.user.id,
                'username': self.user.username,
                'first_name': self.user.first_name,
                'last_name': self.user.last_name,
                'full_name': f"{self.user.first_name} {self.user.last_name}"
            }
        return data
