"""
Chat Routes for Dr. Hygieia AI Assistant
========================================
Provides API endpoints for AI-powered chat functionality.
"""

from flask import Blueprint, request, jsonify, Response, stream_with_context
from flask_jwt_extended import jwt_required, get_jwt_identity
from app import db
from app.models import User, Analysis, ChatSession
from app.services.chat_service import dr_hygieia
from uuid import UUID
import json


def serialize_for_json(obj):
    """Convert UUID and other non-JSON-serializable objects to strings"""
    if isinstance(obj, UUID):
        return str(obj)
    elif isinstance(obj, dict):
        return {k: serialize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [serialize_for_json(item) for item in obj]
    return obj


chat_bp = Blueprint('chat', __name__)


@chat_bp.route('/status', methods=['GET'])
@jwt_required()
def get_chat_status():
    """Get the status of the AI chat service"""
    return jsonify({
        'available': dr_hygieia.is_available,
        'model': dr_hygieia.model if dr_hygieia.is_available else None,
        'assistant_name': 'Dr. Hygieia'
    })


@chat_bp.route('/sessions', methods=['GET'])
@jwt_required()
def get_sessions():
    """Get all chat sessions for the current user (or all sessions for admins)"""
    user_id = get_jwt_identity()
    
    user = User.query.get(user_id)
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 50, type=int)  # Increased for admins
    
    if user.is_admin:
        # Admins see all sessions
        sessions = ChatSession.query.filter_by(
            is_active=True
        ).order_by(
            ChatSession.updated_at.desc()
        ).paginate(page=page, per_page=per_page, error_out=False)
        session_dicts = [s.to_dict(include_user=True) for s in sessions.items]
    else:
        # Regular users see only their sessions
        sessions = ChatSession.query.filter_by(
            user_id=user_id,
            is_active=True
        ).order_by(
            ChatSession.updated_at.desc()
        ).paginate(page=page, per_page=per_page, error_out=False)
        session_dicts = [s.to_dict() for s in sessions.items]
    
    return jsonify({
        'sessions': session_dicts,
        'total': sessions.total,
        'page': sessions.page,
        'pages': sessions.pages,
        'per_page': sessions.per_page
    })


@chat_bp.route('/sessions', methods=['POST'])
@jwt_required()
def create_session():
    """Create a new chat session"""
    user_id = get_jwt_identity()
    data = request.get_json() or {}
    
    user = User.query.get(user_id)
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    # Check for analysis context
    analysis_id = data.get('analysis_id')
    analysis = None
    context_type = data.get('context_type', 'general')
    context_data = data.get('context_data')
    
    if analysis_id:
        analysis = Analysis.query.get(analysis_id)
        if analysis and str(analysis.user_id) != str(user_id):
            return jsonify({'error': 'Unauthorized access to analysis'}), 403
        if analysis:
            context_type = 'analysis'
            # Rich context data including analysis date, input parameters, and results
            # Convert all UUID objects to strings for JSON serialization
            context_data = {
                'analysis_id': str(analysis.id),
                'analysis_type': analysis.analysis_type,
                'model_name': analysis.model_name,
                'risk_level': analysis.risk_level,
                'confidence': analysis.confidence,
                'risk_score': analysis.risk_score,
                'input_data': analysis.input_data,
                'result': analysis.result,
                'created_at': analysis.created_at.isoformat() if analysis.created_at else None,
                'ai_summary': analysis.ai_summary
            }
    
    # Create session
    session = ChatSession(
        user_id=user_id,
        title=data.get('title', 'New Conversation'),
        analysis_id=analysis_id if analysis else None,
        context_type=context_type,
        context_data=context_data,
        messages=[]  # Initialize with empty messages array
    )
    
    db.session.add(session)
    db.session.flush()  # Flush to get the session ID without committing
    
    # If there's analysis context, use the stored AI summary
    if analysis:
        # Use stored summary if available, otherwise generate one
        summary = analysis.ai_summary
        if not summary:
            summary = dr_hygieia.generate_analysis_summary(
                analysis.to_dict(),
                user.to_dict()
            )
            # Save the generated summary for future use
            analysis.ai_summary = summary
            db.session.add(analysis)
        
        # Add system message with rich context
        system_context = f"""This conversation is about the user's {analysis.analysis_type} analysis.
Analysis Date: {analysis.created_at.strftime('%B %d, %Y at %I:%M %p') if analysis.created_at else 'Unknown'}
Risk Level: {analysis.risk_level or 'N/A'}
Confidence: {f'{analysis.confidence:.1%}' if analysis.confidence else 'N/A'}
Input Parameters: {json.dumps(serialize_for_json(analysis.input_data), indent=2) if analysis.input_data else 'N/A'}
Result Summary: {json.dumps(serialize_for_json(analysis.result), indent=2) if analysis.result else 'N/A'}"""
        
        session.add_message(
            role='system',
            content=system_context,
            metadata={'analysis_id': analysis_id}
        )
        
        # Add AI greeting with stored summary
        analysis_type_name = analysis.analysis_type.replace('-', ' ').title()
        greeting = f"Hello {user.first_name}! I'm Dr. Hygieia, your AI health assistant. I've reviewed your recent {analysis_type_name} analysis from {analysis.created_at.strftime('%B %d, %Y') if analysis.created_at else 'your records'}.\n\n{summary}\n\nFeel free to ask me any questions about your results or health in general!"
        
        session.add_message(
            role='assistant',
            content=greeting
        )
    else:
        # Add a general greeting for non-analysis sessions
        greeting = f"Hello {user.first_name}! I'm Dr. Hygieia, your AI health assistant. I'm here to help you understand your health analysis results and answer any health-related questions you may have.\n\nHow can I assist you today?"
        
        session.add_message(
            role='assistant',
            content=greeting
        )
    
    db.session.commit()
    
    return jsonify({
        'session': session.to_dict(include_messages=True)
    }), 201


@chat_bp.route('/sessions/<session_id>', methods=['GET'])
@jwt_required()
def get_session(session_id):
    """Get a specific chat session with messages"""
    user_id = get_jwt_identity()
    
    user = User.query.get(user_id)
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    session = ChatSession.query.get(session_id)
    if not session:
        return jsonify({'error': 'Session not found'}), 404
    
    print(f"GET session - Session user_id: {session.user_id}, Request user_id: {user_id}, is_admin: {user.is_admin}")
    
    # Allow access if user owns the session or is admin
    if str(session.user_id) != str(user_id) and not user.is_admin:
        return jsonify({'error': 'Unauthorized'}), 403
    
    return jsonify({
        'session': session.to_dict(include_messages=True, include_analysis=True),
        'is_owner': str(session.user_id) == str(user_id),
        'session_owner': session.user.to_dict() if session.user else None
    })


@chat_bp.route('/sessions/<session_id>', methods=['PUT'])
@jwt_required()
def update_session(session_id):
    """Update a chat session (title, etc.)"""
    user_id = get_jwt_identity()
    data = request.get_json() or {}
    
    session = ChatSession.query.get(session_id)
    if not session:
        return jsonify({'error': 'Session not found'}), 404
    
    if str(session.user_id) != str(user_id):
        return jsonify({'error': 'Unauthorized'}), 403
    
    if 'title' in data:
        session.title = data['title'][:255]
    
    db.session.commit()
    
    return jsonify({
        'session': session.to_dict()
    })


@chat_bp.route('/sessions/<session_id>', methods=['DELETE'])
@jwt_required()
def delete_session(session_id):
    """Delete a chat session (soft delete)"""
    user_id = get_jwt_identity()
    
    session = ChatSession.query.get(session_id)
    if not session:
        return jsonify({'error': 'Session not found'}), 404
    
    if str(session.user_id) != str(user_id):
        return jsonify({'error': 'Unauthorized'}), 403
    
    # Soft delete
    session.is_active = False
    db.session.commit()
    
    return jsonify({'message': 'Session deleted'})


@chat_bp.route('/sessions/<session_id>/messages', methods=['POST'])
@jwt_required()
def send_message(session_id):
    """Send a message and get an AI response"""
    user_id = get_jwt_identity()
    
    session = ChatSession.query.get(session_id)
    if not session:
        return jsonify({'error': 'Session not found'}), 404
    
    # Only the session owner can send messages
    if str(session.user_id) != str(user_id):
        return jsonify({'error': 'Unauthorized - Only session owner can send messages'}), 403
    
    data = request.get_json() or {}
    
    if not data.get('content'):
        return jsonify({'error': 'Message content is required'}), 400
    
    user = User.query.get(user_id)
    
    # Add user message to session
    user_msg = session.add_message(
        role='user',
        content=data['content']
    )
    db.session.commit()
    
    # Get conversation history (exclude system messages)
    messages = [
        {'role': msg['role'], 'content': msg['content']}
        for msg in session.get_messages(exclude_system=True)
    ]
    
    # Get analysis context if available
    analysis_dict = session.analysis.to_dict() if session.analysis else None
    user_dict = user.to_dict() if user else None
    
    # Generate AI response
    ai_response = dr_hygieia.chat(
        messages=messages,
        analysis=analysis_dict,
        user=user_dict
    )
    
    # Add AI response to session
    ai_msg = session.add_message(
        role='assistant',
        content=ai_response
    )
    
    # Update session title if it's the first real exchange
    if session.title == 'New Conversation' and len(messages) <= 2:
        new_title = dr_hygieia.generate_title(messages)
        session.title = new_title
    
    db.session.commit()
    
    return jsonify({
        'user_message': user_msg,
        'assistant_message': ai_msg
    })


@chat_bp.route('/sessions/<session_id>/messages/stream', methods=['POST'])
@jwt_required()
def send_message_stream(session_id):
    """Send a message and stream the AI response"""
    user_id = get_jwt_identity()
    data = request.get_json() or {}
    
    if not data.get('content'):
        return jsonify({'error': 'Message content is required'}), 400
    
    session = ChatSession.query.get(session_id)
    if not session:
        return jsonify({'error': 'Session not found'}), 404
    
    print(f"Stream - Session user_id: {session.user_id}, Request user_id: {user_id}")
    
    if str(session.user_id) != str(user_id):
        return jsonify({'error': 'Unauthorized'}), 403
    
    user = User.query.get(user_id)
    
    # Add user message to session
    user_msg = session.add_message(
        role='user',
        content=data['content']
    )
    db.session.commit()
    
    # Get conversation history (exclude system messages)
    messages = [
        {'role': msg['role'], 'content': msg['content']}
        for msg in session.get_messages(exclude_system=True)
    ]
    
    # Get analysis context if available
    analysis_dict = session.analysis.to_dict() if session.analysis else None
    user_dict = user.to_dict() if user else None
    
    def generate():
        full_response = ""
        
        # Send user message first
        yield f"data: {json.dumps({'type': 'user_message', 'message': user_msg})}\n\n"
        
        # Stream AI response
        for chunk in dr_hygieia.chat(
            messages=messages,
            analysis=analysis_dict,
            user=user_dict,
            stream=True
        ):
            full_response += chunk
            yield f"data: {json.dumps({'type': 'chunk', 'content': chunk})}\n\n"
        
        # Save complete response to session
        ai_msg = session.add_message(
            role='assistant',
            content=full_response
        )
        
        # Update title if needed
        if session.title == 'New Conversation' and len(messages) <= 2:
            session.title = dr_hygieia.generate_title(messages)
        
        db.session.commit()
        
        # Send completion
        yield f"data: {json.dumps({'type': 'complete', 'message': ai_msg})}\n\n"
    
    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no'
        }
    )


@chat_bp.route('/analysis/<analysis_id>/summary', methods=['GET'])
@jwt_required()
def get_analysis_summary(analysis_id):
    """Get the stored AI summary for an analysis (uses pre-generated summary from DB)"""
    user_id = get_jwt_identity()
    
    analysis = Analysis.query.get(analysis_id)
    if not analysis:
        return jsonify({'error': 'Analysis not found'}), 404
    
    # Check access - users can only view their own analyses, admins can view all
    user = User.query.get(user_id)
    is_owner = str(analysis.user_id) == str(user_id)
    is_admin = user and user.is_admin
    
    if not is_owner and not is_admin:
        return jsonify({'error': 'Unauthorized'}), 403
    
    # Use stored summary if available
    summary = analysis.ai_summary
    
    # Generate and save if not already present (for legacy analyses)
    if not summary:
        user = User.query.get(user_id)
        summary = dr_hygieia.generate_analysis_summary(
            analysis.to_dict(),
            user.to_dict() if user else None
        )
        # Save for future requests
        analysis.ai_summary = summary
        db.session.commit()
    
    return jsonify({
        'summary': summary,
        'analysis_id': analysis_id,
        'analysis_type': analysis.analysis_type,
        'created_at': analysis.created_at.isoformat() if analysis.created_at else None
    })


@chat_bp.route('/quick-chat', methods=['POST'])
@jwt_required()
def quick_chat():
    """Send a one-off message without creating a session (for simple queries)"""
    user_id = get_jwt_identity()
    data = request.get_json() or {}
    
    if not data.get('message'):
        return jsonify({'error': 'Message is required'}), 400
    
    user = User.query.get(user_id)
    
    # Check for optional analysis context
    analysis_dict = None
    if data.get('analysis_id'):
        analysis = Analysis.query.get(data['analysis_id'])
        if analysis and analysis.user_id == user_id:
            analysis_dict = analysis.to_dict()
    
    messages = [{'role': 'user', 'content': data['message']}]
    
    response = dr_hygieia.chat(
        messages=messages,
        analysis=analysis_dict,
        user=user.to_dict() if user else None
    )
    
    return jsonify({
        'response': response
    })
