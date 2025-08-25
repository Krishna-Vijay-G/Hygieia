"""
Data models for storing diagnostic results, chat conversations, and patient information.
"""
from datetime import datetime
from typing import Dict, List, Optional
import uuid
from app import db
from sqlalchemy import Text, DateTime, JSON
import json

class DiagnosticResult(db.Model):
    """Model for storing diagnostic results"""
    __tablename__ = 'diagnostic_results'
    
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    result_type = db.Column(db.String(50), nullable=False)  # 'dermatology', 'heart_disease', 'breast_cancer', 'diabetes'
    input_data = db.Column(JSON, nullable=False)
    prediction = db.Column(JSON, nullable=False)
    educational_content = db.Column(Text, default="")
    image_path = db.Column(db.String(255), default="")
    created_at = db.Column(DateTime, default=datetime.utcnow)
    disclaimer_shown = db.Column(db.Boolean, default=True)
    
    # Relationship to chat conversations
    chat_conversations = db.relationship('ChatConversation', backref='diagnostic_result', lazy=True)

class ChatConversation(db.Model):
    """Model for storing chat conversations"""
    __tablename__ = 'chat_conversations'
    
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = db.Column(db.String(100), nullable=False)
    title = db.Column(db.String(200), default="Medical Consultation")
    created_at = db.Column(DateTime, default=datetime.utcnow)
    updated_at = db.Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Foreign key to diagnostic result (optional)
    diagnostic_result_id = db.Column(db.String(36), db.ForeignKey('diagnostic_results.id'), nullable=True)
    
    # Relationship to messages
    messages = db.relationship('ChatMessage', backref='conversation', lazy=True, order_by='ChatMessage.created_at')

class ChatMessage(db.Model):
    """Model for storing individual chat messages"""
    __tablename__ = 'chat_messages'
    
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    conversation_id = db.Column(db.String(36), db.ForeignKey('chat_conversations.id'), nullable=False)
    
    role = db.Column(db.String(20), nullable=False)  # 'user' or 'assistant'
    content = db.Column(Text, nullable=False)
    message_type = db.Column(db.String(50), default='text')  # 'text', 'image_analysis', 'prescription'
    
    # Optional message metadata for AI analysis
    message_metadata = db.Column(JSON, nullable=True)
    
    created_at = db.Column(DateTime, default=datetime.utcnow)

# Legacy functions for backward compatibility
def save_result(result_data: dict) -> str:
    """Save diagnostic result and return ID"""
    result = DiagnosticResult(
        result_type=result_data['result_type'],
        input_data=result_data['input_data'],
        prediction=result_data['prediction'],
        educational_content=result_data.get('educational_content', ''),
        image_path=result_data.get('image_path', '')
    )
    db.session.add(result)
    db.session.commit()
    return result.id

def get_result(result_id: str) -> Optional[DiagnosticResult]:
    """Get diagnostic result by ID"""
    return DiagnosticResult.query.get(result_id)

def get_all_results() -> List[DiagnosticResult]:
    """Get all diagnostic results"""
    return DiagnosticResult.query.all()

# Chat functions
def create_chat_conversation(session_id: str, title: str = "Medical Consultation", diagnostic_result_id: str = None) -> ChatConversation:
    """Create a new chat conversation"""
    conversation = ChatConversation(
        session_id=session_id,
        title=title,
        diagnostic_result_id=diagnostic_result_id
    )
    db.session.add(conversation)
    db.session.commit()
    return conversation

def get_chat_conversation(conversation_id: str) -> Optional[ChatConversation]:
    """Get chat conversation by ID"""
    return ChatConversation.query.get(conversation_id)

def get_conversations_by_session(session_id: str) -> List[ChatConversation]:
    """Get all conversations for a session"""
    return ChatConversation.query.filter_by(session_id=session_id).order_by(ChatConversation.updated_at.desc()).all()

def add_chat_message(conversation_id: str, role: str, content: str, message_type: str = 'text', message_metadata: dict = None) -> ChatMessage:
    """Add a message to a chat conversation"""
    message = ChatMessage(
        conversation_id=conversation_id,
        role=role,
        content=content,
        message_type=message_type,
        message_metadata=message_metadata
    )
    db.session.add(message)
    
    # Update conversation timestamp
    conversation = ChatConversation.query.get(conversation_id)
    if conversation:
        conversation.updated_at = datetime.utcnow()
    
    db.session.commit()
    return message
