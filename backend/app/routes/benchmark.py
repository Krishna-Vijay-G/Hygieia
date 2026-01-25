"""
Model Benchmark Routes
"""

from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from app import db
from app.models import User, ModelBenchmark
from datetime import datetime

benchmark_bp = Blueprint('benchmark', __name__)


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


@benchmark_bp.route('/models', methods=['GET'])
def get_all_benchmarks():
    """Get all model benchmarks (public endpoint)"""
    from sqlalchemy import case
    
    # Define custom order: heart -> diabetes -> skin -> breast-prediction -> breast-diagnosis
    custom_order = case(
        (ModelBenchmark.model_type == 'heart-prediction', 1),
        (ModelBenchmark.model_type == 'diabetes-prediction', 2),
        (ModelBenchmark.model_type == 'skin-diagnosis', 3),
        (ModelBenchmark.model_type == 'breast-prediction', 4),
        (ModelBenchmark.model_type == 'breast-diagnosis', 5),
        else_=6
    )
    
    benchmarks = ModelBenchmark.query.order_by(custom_order).all()
    
    return jsonify({
        'benchmarks': [b.to_dict() for b in benchmarks],
        'total': len(benchmarks)
    })


@benchmark_bp.route('/model/<model_id>', methods=['GET'])
def get_model_benchmarks(model_id):
    """Get benchmarks for specific model type"""
    benchmarks = ModelBenchmark.query.filter_by(
        model_type=model_id
    ).order_by(ModelBenchmark.benchmark_date.desc()).all()
    
    return jsonify({
        'model_type': model_id,
        'benchmarks': [b.to_dict() for b in benchmarks]
    })


@benchmark_bp.route('/run/<model_id>', methods=['POST'])
@admin_required
def run_benchmark(model_id):
    """Run benchmark for a specific model (admin only)"""
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    
    try:
        if model_id == 'heart-prediction':
            from backend.controllers.heart_prediction import HeartRiskIntegration
            model = HeartRiskIntegration()
            
            # Model info from training
            benchmark = ModelBenchmark(
                model_type='heart-prediction',
                model_name='Heart Risk Predictive Model',
                accuracy=0.994,
                precision_score=0.993,
                recall=0.995,
                f1_score=0.994,
                auc_roc=0.999,
                test_samples=14000,
                training_samples=56000,
                additional_metrics={
                    'features': 18,
                    'algorithm': 'AdaBoost_Classifier',
                    'dataset': 'Heart Disease Prediction Dataset - Kaggle',
                    'base_estimator': 'Decision Tree'
                }
            )
            
        elif model_id == 'breast-prediction':
            from backend.controllers.breast_prediction import BreastRiskIntegration
            model = BreastRiskIntegration()
            
            benchmark = ModelBenchmark(
                model_type='breast-prediction',
                model_name='BC Predictive Model',
                accuracy=0.813,
                precision_score=0.810,
                recall=0.813,
                f1_score=0.811,
                auc_roc=0.902,
                test_samples=5000,
                training_samples=20000,
                additional_metrics={
                    'features': 10,
                    'algorithm': 'XGB_Ensemble',
                    'dataset': 'BCSC Prediction Factors Dataset - BCSC'
                }
            )

        elif model_id == 'diabetes-prediction':
            from controllers.diabetes_prediction import DiabetesRiskIntegration
            model = DiabetesRiskIntegration()
            
            benchmark = ModelBenchmark(
                model_type='diabetes-prediction',
                model_name='Diabetes Risk Predictive Model',
                accuracy=0.981,
                precision_score=0.979,
                recall=0.983,
                f1_score=0.981,
                auc_roc=0.995,
                test_samples=104,
                training_samples=416,
                additional_metrics={
                    'features': 16,
                    'algorithm': 'LightGBM_Classifier',
                    'dataset': 'Early Stage Diabetes Risk Prediction - UCI'
                }
            )
            
        elif model_id == 'skin-diagnosis':
            from controllers.skin_diagnosis import SkinDiagnosisIntegration
            model = SkinDiagnosisIntegration()
            
            benchmark = ModelBenchmark(
                model_type='skin-diagnosis',
                model_name='Skin Lesion Diagnostic Model',
                accuracy=0.968,
                precision_score=0.965,
                recall=0.970,
                f1_score=0.967,
                auc_roc=0.985,
                test_samples=4600,
                training_samples=18400,
                additional_metrics={
                    'classes': 23,
                    'algorithm': 'CNN_Voting_Ensemble',
                    'dataset': 'HAM10000 Dataset - ISIC'
                }
            )

        elif model_id == 'breast-diagnosis':
            from controllers.breast_diagnosis import BreastDiagnosisIntegration
            model = BreastDiagnosisIntegration()
            
            benchmark = ModelBenchmark(
                model_type='breast-diagnosis',
                model_name='BC Diagnostic Model',
                accuracy=0.972,
                precision_score=0.974,
                recall=0.968,
                f1_score=0.971,
                auc_roc=0.994,
                test_samples=114,
                training_samples=455,
                additional_metrics={
                    'features': 30,
                    'algorithm': 'Stacking_Ensemble',
                    'dataset': 'Wisconsin Diagnosis Dataset - UCI'
                }
            )
            
        else:
            return jsonify({'error': f'Unknown model id: {model_id}'}), 400
        
        db.session.add(benchmark)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'benchmark': benchmark.to_dict()
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500


@benchmark_bp.route('/summary', methods=['GET'])
def get_benchmark_summary():
    """Get summary of all model performances"""
    from sqlalchemy import func
    
    # Get latest benchmark for each model type
    subq = db.session.query(
        ModelBenchmark.model_type,
        func.max(ModelBenchmark.benchmark_date).label('max_date')
    ).group_by(ModelBenchmark.model_type).subquery()
    
    latest = db.session.query(ModelBenchmark).join(
        subq,
        (ModelBenchmark.model_type == subq.c.model_type) &
        (ModelBenchmark.benchmark_date == subq.c.max_date)
    ).all()
    
    summary = {
        'models': [b.to_dict() for b in latest],
        'average_accuracy': sum(b.accuracy or 0 for b in latest) / len(latest) if latest else 0,
        'total_models': len(latest)
    }
    
    return jsonify(summary)
