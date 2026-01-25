"""
Model Bridge - Routes model predictions to the appropriate controllers
This module provides a unified interface for model predictions across the application.
"""

import os
import sys
from typing import Dict, Any

# Add controllers to path
sys.path.insert(0, os.path.dirname(__file__))

from controllers.heart_prediction import HeartRiskIntegration
from controllers.diabetes_prediction import DiabetesRiskIntegration
from controllers.breast_diagnosis import BreastDiagnosisIntegration
from controllers.breast_prediction import BreastRiskIntegration
from controllers.skin_diagnosis import SkinDiagnosisIntegration


def predict_heart_risk(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Predict heart risk

    Args:
        input_data: Dictionary with heart risk factors (PascalCase format)

    Returns:
        Dictionary with prediction results
    """
    try:
        model = HeartRiskIntegration()
        if not model.is_loaded:
            raise Exception("Model failed to load")

        # Get prediction from model
        result = model.predict(input_data)

        if not result.get('success'):
            raise Exception(result.get('error', 'Prediction failed'))

        return {
            'success': True,
            'prediction': result.get('prediction'),
            'risk_level': result.get('risk_level'),
            'confidence': result.get('confidence'),
            'confidence_percent': result.get('confidence_percent'),
            'risk_score': result.get('risk_score'),
            'severity': result.get('severity'),
            'method': result.get('method'),
        }
    except Exception as e:
        print(f"Heart Risk prediction error: {str(e)}")
        raise Exception(f"Heart Risk prediction error: {str(e)}")

def predict_diabetes_risk(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Predict diabetes risk
    
    Args:
        input_data: Dictionary with diabetes risk factors
        
    Returns:
        Dictionary with prediction results
    """
    try:
        model = DiabetesRiskIntegration()

        # Convert boolean values to 'Yes'/'No' which the diabetes model expects
        converted_data = input_data.copy()
        for k, v in list(converted_data.items()):
            if isinstance(v, bool):
                converted_data[k] = 'Yes' if v else 'No'

        # Get prediction from model
        result = model.predict(converted_data)

        if not result.get('success'):
            raise Exception(result.get('error', 'Prediction failed'))

        return {
            'success': True,
            'prediction': result.get('prediction'),
            'probability': result.get('probability'),
            'risk_level': result.get('risk_level'),
            'confidence': result.get('confidence'),
            'confidence_percent': result.get('confidence_percent'),
            'risk_score': result.get('risk_score'),
            'interpretation': result.get('interpretation'),
            'risk_factors': result.get('risk_factors'),
            'method': result.get('method'),
        }
    except Exception as e:
        print(f"Diabetes prediction error: {str(e)}")
        raise Exception(f"Diabetes prediction error: {str(e)}")

def diagnose_breast_cancer(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Predict breast cancer diagnosis using tissue analysis
    
    Args:
        input_data: Dictionary with tissue FNA measurements
        
    Returns:
        Dictionary with prediction results
    """
    try:
        model = BreastDiagnosisIntegration()
        if not model.is_loaded:
            raise Exception("Breast diagnosis model failed to load")

        # Get prediction from model
        result = model.predict(input_data)

        if not result.get('success'):
            raise Exception(result.get('error', 'Prediction failed'))

        return {
            'success': True,
            'prediction': result.get('prediction'),
            'probability': result.get('probability'),
            'risk_level': result.get('risk_level'),
            'confidence': result.get('confidence'),
            'interpretation': result.get('interpretation'),
            'condition_name': result.get('condition_name'),
            'risk_factors': result.get('risk_factors'),
            'threshold_used': result.get('threshold_used'),
            'method': result.get('method'),
        }
    except Exception as e:
        print(f"Breast diagnosis prediction error: {str(e)}")
        raise Exception(f"Breast diagnosis prediction error: {str(e)}")

def predict_breast_cancer(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Predict breast cancer risk using clinical factors
    
    Args:
        input_data: Dictionary with clinical risk factors
        
    Returns:
        Dictionary with prediction results
    """
    try:
        model = BreastRiskIntegration()
        if not model.is_loaded:
            raise Exception("Breast risk prediction model failed to load")

        # Get prediction from model
        result = model.predict(input_data)

        if not result.get('success'):
            raise Exception(result.get('error', 'Prediction failed'))

        return {
            'success': True,
            'prediction': result.get('prediction'),
            'probability': result.get('probability'),
            'risk_level': result.get('risk_level'),
            'confidence': result.get('confidence'),
            'interpretation': result.get('interpretation'),
            'condition_name': result.get('condition_name'),
            'risk_factors': result.get('risk_factors'),
            'threshold_used': result.get('threshold_used'),
            'method': result.get('method'),
        }
    except Exception as e:
        print(f"Breast risk prediction error: {str(e)}")
        raise Exception(f"Breast risk prediction error: {str(e)}")


def diagnose_skin_lesion(image_input) -> Dict[str, Any]:
    """
    Predict skin condition from image

    Args:
        image_input: Either image file path (str) or numpy array

    Returns:
        Dictionary with prediction results
    """
    try:
        model = SkinDiagnosisIntegration()
        if not model.is_loaded:
            raise Exception("Skin diagnosis model failed to load")

        # Get prediction from model
        result = model.predict(image_input)

        # Check for error in skin diagnosis result (different from other models)
        if result.get('error') or result.get('prediction') == 'error':
            raise Exception(result.get('error', 'Prediction failed'))

        return {
            'success': True,
            'prediction': result.get('prediction'),
            'probability': result.get('confidence'),  # Skin model uses 'confidence' not 'probability'
            'risk_level': result.get('risk_level'),
            'confidence': result.get('confidence'),
            'interpretation': result.get('condition_name'),
            'condition_name': result.get('condition_name'),
            'risk_factors': result.get('probabilities'),  # Skin model provides probabilities
            'method': result.get('method'),
        }
    except Exception as e:
        print(f"Skin diagnosis error: {str(e)}")
        raise Exception(f"Skin diagnosis error: {str(e)}")