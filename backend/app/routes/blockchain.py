"""
Blockchain Routes - Verification and audit trail
"""

from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from app import db
from app.models import User, BlockchainRecord, Analysis

blockchain_bp = Blueprint('blockchain', __name__)


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


@blockchain_bp.route('/records', methods=['GET'])
@admin_required
def get_blockchain_records():
    """Get all blockchain records (admin only)"""
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 50, type=int)
    
    pagination = BlockchainRecord.query.order_by(
        BlockchainRecord.block_index.desc()
    ).paginate(page=page, per_page=per_page, error_out=False)
    
    records = []
    for record in pagination.items:
        record_data = record.to_dict(include_analysis=True)
        records.append(record_data)
    
    return jsonify({
        'records': records,
        'total': pagination.total,
        'pages': pagination.pages,
        'current_page': page
    })


@blockchain_bp.route('/record/<int:block_index>', methods=['GET'])
@admin_required
def get_blockchain_record(block_index):
    """Get specific blockchain record by index"""
    record = BlockchainRecord.query.filter_by(block_index=block_index).first()
    
    if not record:
        return jsonify({'error': 'Block not found'}), 404
    
    return jsonify({'record': record.to_dict(include_analysis=True)})


@blockchain_bp.route('/verify/<hash>', methods=['GET'])
@admin_required
def verify_hash(hash):
    """Verify a blockchain hash"""
    record = BlockchainRecord.query.filter_by(current_hash=hash).first()
    
    if not record:
        return jsonify({
            'verified': False,
            'message': 'Hash not found in blockchain'
        })
    
    return jsonify({
        'verified': True,
        'record': record.to_dict(include_analysis=True)
    })


@blockchain_bp.route('/validate', methods=['GET'])
@admin_required
def validate_chain():
    """Validate the entire blockchain"""
    records = BlockchainRecord.query.order_by(BlockchainRecord.block_index.asc()).all()
    
    if not records:
        return jsonify({
            'valid': True,
            'message': 'Blockchain is empty',
            'blocks': 0
        })
    
    valid = True
    invalid_blocks = []
    
    for i, record in enumerate(records):
        if i == 0:
            # Genesis block
            if record.previous_hash != '0' * 64:
                valid = False
                invalid_blocks.append({
                    'block_index': record.block_index,
                    'issue': 'Invalid genesis block previous hash'
                })
        else:
            # Check chain continuity
            if record.previous_hash != records[i-1].current_hash:
                valid = False
                invalid_blocks.append({
                    'block_index': record.block_index,
                    'issue': 'Chain broken - previous hash mismatch'
                })
    
    return jsonify({
        'valid': valid,
        'blocks': len(records),
        'invalid_blocks': invalid_blocks,
        'message': 'Blockchain integrity verified' if valid else 'Blockchain integrity compromised'
    })


@blockchain_bp.route('/stats', methods=['GET'])
@admin_required
def get_blockchain_stats():
    """Get blockchain statistics"""
    total_blocks = BlockchainRecord.query.count()
    total_analyses = Analysis.query.count()
    
    # Get analysis type breakdown
    from sqlalchemy import func
    type_stats = db.session.query(
        Analysis.analysis_type,
        func.count(Analysis.id)
    ).group_by(Analysis.analysis_type).all()
    
    type_breakdown = {t: c for t, c in type_stats}
    
    # Recent activity
    recent_blocks = BlockchainRecord.query.order_by(
        BlockchainRecord.timestamp.desc()
    ).limit(5).all()
    
    return jsonify({
        'total_blocks': total_blocks,
        'total_analyses': total_analyses,
        'analysis_breakdown': type_breakdown,
        'recent_blocks': [b.to_dict() for b in recent_blocks]
    })
