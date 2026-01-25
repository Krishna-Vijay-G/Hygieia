"""
Analysis Routes - Medical diagnostic endpoints
"""

from flask import Blueprint, request, jsonify, current_app
from flask_jwt_extended import jwt_required, get_jwt_identity
from werkzeug.utils import secure_filename
import os
import uuid
import hashlib
import json
from datetime import datetime

from app import db
from app.models import User, Analysis, BlockchainRecord, AuditLog
from app.services.chat_service import dr_hygieia

analysis_bp = Blueprint('analysis', __name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def create_blockchain_hash(analysis_data):
    """Create a hash for blockchain verification"""
    data_string = json.dumps(analysis_data, sort_keys=True, default=str)
    return hashlib.sha256(data_string.encode()).hexdigest()


def add_to_blockchain(analysis):
    """Add analysis to blockchain"""
    try:
        # Get the last block
        last_block = BlockchainRecord.query.order_by(BlockchainRecord.block_index.desc()).first()
        
        if last_block:
            block_index = last_block.block_index + 1
            previous_hash = last_block.current_hash
        else:
            block_index = 0
            previous_hash = '0' * 64
        
        # Create data hash
        data_to_hash = {
            'analysis_id': analysis.id,
            'user_id': analysis.user_id,
            'analysis_type': analysis.analysis_type,
            'result': analysis.result,
            'timestamp': analysis.created_at.isoformat()
        }
        data_hash = create_blockchain_hash(data_to_hash)
        
        # Create block hash
        block_data = f"{block_index}{data_hash}{previous_hash}{datetime.utcnow().isoformat()}"
        current_hash = hashlib.sha256(block_data.encode()).hexdigest()
        
        # Create blockchain record
        block = BlockchainRecord(
            block_index=block_index,
            data_hash=data_hash,
            previous_hash=previous_hash,
            current_hash=current_hash,
            analysis_id=analysis.id
        )
        
        db.session.add(block)
        db.session.commit()
        
        # Update analysis with blockchain hash
        analysis.blockchain_hash = current_hash
        db.session.commit()
        
        return current_hash
    except Exception as e:
        db.session.rollback()
        print(f"Blockchain error: {e}")
        return None


def generate_and_save_ai_summary(analysis, user):
    """Auto-generate AI summary and save to analysis record"""
    try:
        user_dict = user.to_dict() if user else None
        summary = dr_hygieia.generate_analysis_summary(analysis.to_dict(), user_dict)
        analysis.ai_summary = summary
        db.session.commit()
        return summary
    except Exception as e:
        print(f"AI Summary generation error: {e}")
        return None


@analysis_bp.route('/types', methods=['GET'])
def get_analysis_types():
    """Get available analysis types"""
    return jsonify({
        'types': [
            {
                'id': 'heart-prediction',
                'name': 'Heart Risk Prediction',
                'description': 'Heart risk evaluation based on symptoms and risk factors',
                'requires_image': False,
                'accuracy': '99.4%'
            },
            {
                'id': 'diabetes-prediction',
                'name': 'Diabetes Risk Prediction',
                'description': 'Diabetes risk assessment based on symptoms and lifestyle',
                'requires_image': False,
                'accuracy': '98.1%'
            },
            {
                'id': 'skin-diagnosis',
                'name': 'Skin Lesion Diagnosis',
                'description': 'AI-powered skin lesion and condition analysis',
                'requires_image': True,
                'accuracy': '96.8%'
            },
            {
                'id': 'breast-prediction',
                'name': 'Breast Cancer Prediction',
                'description': 'Clinical risk assessment using biomarkers and risk factors',
                'requires_image': False,
                'accuracy': '81.3%'
            },
            {
                'id': 'breast-diagnosis',
                'name': 'Breast Tissue Diagnosis',
                'description': 'Tissue-level tumor diagnosis using FNA measurements',
                'requires_image': False,
                'accuracy': '97.2%'
            }
        ]
    })


@analysis_bp.route('/heart-prediction', methods=['POST'])
@jwt_required()
def analyze_heart_prediction():
    """Analyze heart disease risk"""
    current_user_id = get_jwt_identity()
    data = request.get_json()
    
    try:
        # Prepare input data - convert snake_case to PascalCase for heart risk model
        snake_to_pascal = {
            'gender': 'Gender',
            'age': 'Age',
            'chest_pain': 'Chest_Pain',
            'shortness_of_breath': 'Shortness_of_Breath',
            'fatigue': 'Fatigue',
            'palpitations': 'Palpitations',
            'dizziness': 'Dizziness',
            'swelling': 'Swelling',
            'pain_arms_jaw_back': 'Pain_Arms_Jaw_Back',
            'cold_sweats_nausea': 'Cold_Sweats_Nausea',
            'high_bp': 'High_BP',
            'high_cholesterol': 'High_Cholesterol',
            'diabetes': 'Diabetes',
            'smoking': 'Smoking',
            'obesity': 'Obesity',
            'sedentary_lifestyle': 'Sedentary_Lifestyle',
            'family_history': 'Family_History',
            'chronic_stress': 'Chronic_Stress'
        }
        
        input_data = {}
        for snake_key, pascal_key in snake_to_pascal.items():
            input_data[pascal_key] = int(data.get(snake_key, 0))
        
        # Import and run model
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        from model_bridge import predict_heart_risk
        
        result = predict_heart_risk(input_data)
        
        # Create analysis record
        analysis = Analysis(
            user_id=current_user_id,
            analysis_type='heart-prediction',
            model_name='Heart Risk Predictive Model',
            input_data=input_data,
            result=result,
            risk_level=result.get('risk_level'),
            confidence=result.get('confidence'),
            risk_score=result.get('risk_score')
        )
        
        db.session.add(analysis)
        db.session.commit()
        
        # Add to blockchain
        blockchain_hash = add_to_blockchain(analysis)
        
        # Auto-generate AI summary
        user = User.query.get(current_user_id)
        generate_and_save_ai_summary(analysis, user)
        
        # Log analysis
        audit = AuditLog(
            user_id=current_user_id,
            action='heart-prediction analysis',
            resource_type='analysis',
            resource_id=analysis.id,
            ip_address=request.remote_addr
        )
        db.session.add(audit)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'analysis_id': analysis.id,
            'analysis': analysis.to_dict(),
            'blockchain_hash': blockchain_hash
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e), 'message': 'Analysis failed. Please try again.'}), 500


@analysis_bp.route('/diabetes-prediction', methods=['POST'])
@jwt_required()
def analyze_diabetes_prediction():
    """Analyze diabetes risk"""
    current_user_id = get_jwt_identity()
    data = request.get_json()
    
    try:
        # Map frontend data to diabetes model expected format
        input_data = {
            'Age': int(data.get('age', 0)),
            'Gender': data.get('gender', 'Male'),
            'Polyuria': data.get('polyuria', 'No'),
            'Polydipsia': data.get('polydipsia', 'No'),
            'sudden weight loss': data.get('sudden_weight_loss', 'No'),
            'weakness': data.get('weakness', 'No'),
            'Polyphagia': data.get('polyphagia', 'No'),
            'Genital thrush': data.get('genital_thrush', 'No'),
            'visual blurring': data.get('visual_blurring', 'No'),
            'Itching': data.get('itching', 'No'),
            'Irritability': data.get('irritability', 'No'),
            'delayed healing': data.get('delayed_healing', 'No'),
            'partial paresis': data.get('partial_paresis', 'No'),
            'muscle stiffness': data.get('muscle_stiffness', 'No'),
            'Alopecia': data.get('alopecia', 'No'),
            'Obesity': data.get('obesity', 'No')
        }
        
        # Import and run model
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        from model_bridge import predict_diabetes_risk
        
        result = predict_diabetes_risk(input_data)
        
        # Create analysis record
        analysis = Analysis(
            user_id=current_user_id,
            analysis_type='diabetes-prediction',
            model_name='Diabetes Risk Predictive Model',
            input_data=data,
            result=result,
            risk_level=result.get('risk_level'),
            confidence=result.get('confidence'),
            risk_score=result.get('risk_score')
        )
        
        db.session.add(analysis)
        db.session.commit()
        
        # Add to blockchain
        blockchain_hash = add_to_blockchain(analysis)
        
        # Auto-generate AI summary
        user = User.query.get(current_user_id)
        generate_and_save_ai_summary(analysis, user)
        
        # Log analysis
        audit = AuditLog(
            user_id=current_user_id,
            action='diabetes-prediction analysis',
            resource_type='analysis',
            resource_id=analysis.id,
            ip_address=request.remote_addr
        )
        db.session.add(audit)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'analysis_id': analysis.id,
            'analysis': analysis.to_dict(),
            'blockchain_hash': blockchain_hash
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e), 'message': 'Analysis failed. Please try again.'}), 500


@analysis_bp.route('/skin-diagnosis', methods=['POST'])
@jwt_required()
def analyze_skin_diagnosis():
    """Analyze skin image for dermatological conditions"""
    current_user_id = get_jwt_identity()
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Allowed: PNG, JPG, JPEG, GIF'}), 400
    
    try:
        # Save file
        filename = f"{uuid.uuid4()}_{secure_filename(file.filename)}"
        filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Import and run model
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        from model_bridge import diagnose_skin_lesion
        
        result = diagnose_skin_lesion(filepath)
        
        # Create analysis record
        analysis = Analysis(
            user_id=current_user_id,
            analysis_type='skin-diagnosis',
            model_name='Skin Lesion Diagnostic Model',
            result=result,
            risk_level=result.get('risk_level'),
            confidence=result.get('confidence'),
            image_path=filename
        )
        
        db.session.add(analysis)
        db.session.commit()
        
        # Add to blockchain
        blockchain_hash = add_to_blockchain(analysis)
        
        # Auto-generate AI summary
        user = User.query.get(current_user_id)
        generate_and_save_ai_summary(analysis, user)
        
        # Log analysis
        audit = AuditLog(
            user_id=current_user_id,
            action='skin-diagnosis analysis',
            resource_type='analysis',
            resource_id=analysis.id,
            ip_address=request.remote_addr
        )
        db.session.add(audit)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'analysis_id': analysis.id,
            'analysis': analysis.to_dict(),
            'blockchain_hash': blockchain_hash
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500


@analysis_bp.route('/breast-prediction', methods=['POST'])
@jwt_required()
def analyze_breast_prediction():
    """Analyze breast cancer clinical risk (biomarkers and anthropometric data)"""
    current_user_id = get_jwt_identity()
    data = request.get_json()
    
    try:
        # Store original user input BEFORE any processing
        original_input = data.copy()
        
        # Import and run model for clinical risk prediction
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        from model_bridge import predict_breast_cancer
        
        # Create a copy for model processing to avoid modifying original
        model_input = data.copy()
        result = predict_breast_cancer(model_input)
        
        # Prepare input data for storage - preserve frontend's user-friendly actual_* values
        input_data = data.copy()  # This already contains the actual_* fields from frontend
        
        # Create analysis record
        analysis = Analysis(
            user_id=current_user_id,
            analysis_type='breast-prediction',
            model_name='BC Predictive Model',
            input_data=input_data,
            result=result,
            risk_level=result.get('risk_level'),
            confidence=result.get('confidence')
        )
        
        db.session.add(analysis)
        db.session.commit()
        
        # Add to blockchain
        blockchain_hash = add_to_blockchain(analysis)
        
        # Auto-generate AI summary
        user = User.query.get(current_user_id)
        generate_and_save_ai_summary(analysis, user)
        
        # Log analysis
        audit = AuditLog(
            user_id=current_user_id,
            action='breast-prediction analysis',
            resource_type='analysis',
            resource_id=analysis.id,
            ip_address=request.remote_addr
        )
        db.session.add(audit)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'analysis_id': analysis.id,
            'analysis': analysis.to_dict(),
            'blockchain_hash': blockchain_hash
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e), 'message': 'Analysis failed. Please try again.'}), 500


@analysis_bp.route('/breast-diagnosis', methods=['POST'])
@jwt_required()
def analyze_breast_diagnosis():
    """Analyze breast tissue diagnosis using Wisconsin dataset features"""
    current_user_id = get_jwt_identity()
    data = request.get_json()
    
    try:
        # Import and run model for tissue diagnosis
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        from model_bridge import diagnose_breast_cancer
        
        result = diagnose_breast_cancer(data)
        
        # Create analysis record
        analysis = Analysis(
            user_id=current_user_id,
            analysis_type='breast-diagnosis',
            model_name='BC Diagnostic Model',
            input_data=data,
            result=result,
            risk_level=result.get('risk_level'),
            confidence=result.get('confidence')
        )
        
        db.session.add(analysis)
        db.session.commit()
        
        # Add to blockchain
        blockchain_hash = add_to_blockchain(analysis)
        
        # Auto-generate AI summary
        user = User.query.get(current_user_id)
        generate_and_save_ai_summary(analysis, user)
        
        # Log analysis
        audit = AuditLog(
            user_id=current_user_id,
            action='breast-diagnosis analysis',
            resource_type='analysis',
            resource_id=analysis.id,
            ip_address=request.remote_addr
        )
        db.session.add(audit)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'analysis_id': analysis.id,
            'analysis': analysis.to_dict(),
            'blockchain_hash': blockchain_hash
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e), 'message': 'Analysis failed. Please try again.'}), 500


@analysis_bp.route('/history', methods=['GET'])
@jwt_required()
def get_analysis_history():
    """Get current user's analysis history"""
    current_user_id = get_jwt_identity()
    
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 20, type=int)
    analysis_type = request.args.get('type')
    
    query = Analysis.query.filter_by(user_id=current_user_id)
    
    if analysis_type:
        query = query.filter_by(analysis_type=analysis_type)
    
    query = query.order_by(Analysis.created_at.desc())
    pagination = query.paginate(page=page, per_page=per_page, error_out=False)
    
    # Get counts by analysis type
    from sqlalchemy import func
    count_query = db.session.query(
        Analysis.analysis_type,
        func.count(Analysis.id).label('count')
    ).filter_by(user_id=current_user_id).group_by(Analysis.analysis_type)
    
    counts = {row.analysis_type: row.count for row in count_query.all()}
    
    # Get all blockchain hashes for these analyses at once to avoid N+1 queries
    analysis_ids = [a.id for a in pagination.items]
    valid_hashes = set()
    if analysis_ids:
        records = BlockchainRecord.query.filter(BlockchainRecord.analysis_id.in_(analysis_ids)).all()
        valid_hashes = {r.current_hash for r in records}
    
    analyses_data = []
    for a in pagination.items:
        a_dict = a.to_dict()
        if a.blockchain_hash:
            if a.blockchain_hash in valid_hashes:
                a_dict['blockchain_status'] = 'verified'
            else:
                a_dict['blockchain_status'] = 'failed'
        else:
            a_dict['blockchain_status'] = 'not_secured'
        analyses_data.append(a_dict)
    
    return jsonify({
        'analyses': analyses_data,
        'total': pagination.total,
        'pages': pagination.pages,
        'current_page': page,
        'counts': counts
    })


@analysis_bp.route('/<analysis_id>', methods=['GET'])
@jwt_required()
def get_analysis(analysis_id):
    """Get specific analysis result"""
    current_user_id = get_jwt_identity()
    user = User.query.get(current_user_id)
    # Validate analysis_id is a UUID-like string to avoid DB errors
    import uuid as _uuid
    try:
        # This will raise a ValueError if not a valid UUID
        _uuid.UUID(str(analysis_id))
    except Exception:
        return jsonify({'error': 'Invalid analysis id'}), 400

    analysis = Analysis.query.get(analysis_id)
    if not analysis:
        return jsonify({'error': 'Analysis not found'}), 404
    
    # Check access - users can only view their own analyses, admins can view all
    is_owner = str(analysis.user_id) == str(current_user_id)
    is_admin = user and user.is_admin
    
    if not is_owner and not is_admin:
        return jsonify({'error': 'Access denied - You can only view your own analyses'}), 403
    
    analysis_data = analysis.to_dict(include_user=True)
    
    # Verify blockchain hash existence to ensure it's still "verified"
    analysis_data['blockchain_status'] = 'verified' if analysis.blockchain_hash else 'not_secured'
    
    if analysis.blockchain_hash:
        record = BlockchainRecord.query.filter_by(current_hash=analysis.blockchain_hash).first()
        if not record:
            # Hash exists in analysis record but no corresponding block found
            # This happens if a block was deleted - analysis is no longer verified
            analysis_data['blockchain_status'] = 'failed'
            
    return jsonify({'analysis': analysis_data})


@analysis_bp.route('/stats', methods=['GET'])
@jwt_required()
def get_analysis_stats():
    """Get analysis statistics (admin only)"""
    current_user_id = get_jwt_identity()
    user = User.query.get(current_user_id)
    
    if not user or not user.is_admin:
        return jsonify({'error': 'Admin access required'}), 403
    
    from sqlalchemy import func
    
    # Get total analyses
    total_analyses = Analysis.query.count()
    
    # Get counts by analysis type
    type_counts = db.session.query(
        Analysis.analysis_type,
        func.count(Analysis.id).label('count')
    ).group_by(Analysis.analysis_type).all()
    
    # Convert to dict
    analysis_counts = {row.analysis_type: row.count for row in type_counts}
    
    return jsonify({
        'total_analyses': total_analyses,
        'by_type': analysis_counts
    })
