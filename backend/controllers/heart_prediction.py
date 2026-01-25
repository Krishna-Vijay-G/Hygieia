#!/usr/bin/env python3
"""
Heart Risk Predictive Model Integration for Hygieia
"""

import os
import joblib
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from typing import Dict

# logging
import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# constants
INPUT_FEATURES = [
    'Chest_Pain', 'Shortness_of_Breath', 'Fatigue', 'Palpitations',
    'Dizziness', 'Swelling', 'Pain_Arms_Jaw_Back', 'Cold_Sweats_Nausea',
    'High_BP', 'High_Cholesterol', 'Diabetes', 'Smoking',
    'Obesity', 'Sedentary_Lifestyle', 'Family_History', 'Chronic_Stress',
    'Gender', 'Age'
]

FEATURE_DESCRIPTIONS = {
    'Chest_Pain': 'Chest pain or discomfort',
    'Shortness_of_Breath': 'Difficulty breathing',
    'Fatigue': 'Persistent unexplained tiredness',
    'Palpitations': 'Irregular or rapid heartbeat',
    'Dizziness': 'Lightheadedness or fainting',
    'Swelling': 'Swelling in legs or ankles',
    'Pain_Arms_Jaw_Back': 'Pain radiating to arms, jaw, or back',
    'Cold_Sweats_Nausea': 'Cold sweats and nausea',
    'High_BP': 'History of high blood pressure',
    'High_Cholesterol': 'Elevated cholesterol levels',
    'Diabetes': 'Diabetes diagnosis',
    'Smoking': 'Current or past smoking history',
    'Obesity': 'Obesity (BMI > 30)',
    'Sedentary_Lifestyle': 'Low physical activity',
    'Family_History': 'Family history of heart disease',
    'Chronic_Stress': 'High chronic stress levels',
    'Gender': 'Gender (0=Female, 1=Male)',
    'Age': 'Age in years'
}

SYMPTOM_FEATURES = [
    'Chest_Pain', 'Shortness_of_Breath', 'Fatigue', 'Palpitations',
    'Dizziness', 'Swelling', 'Pain_Arms_Jaw_Back', 'Cold_Sweats_Nausea'
]

RISK_FACTOR_FEATURES = [
    'High_BP', 'High_Cholesterol', 'Diabetes', 'Smoking',
    'Obesity', 'Sedentary_Lifestyle', 'Family_History', 'Chronic_Stress'
]

DEMOGRAPHIC_FEATURES = ['Gender', 'Age']

MODEL_INFO = {
    'name': 'Heart Risk Prediction',
    'id': 'heart-risk',
    'model_name': 'Heart Risk Predictive Model',
    'method': 'AdaBoost_Classifier',
    'description': 'Heart disease risk prediction using AdaBoost with multiple weak learners',
    'version': '1.0',
    'dataset': 'Heart Disease Prediction Dataset - Kaggle',
    'modified_date': '2026-01-04',
    'author': 'Krishna Vijay G',
    'auth_url': 'https://Krishna-Vijay-G.github.io',
    'training_date': '2025-12-31',
    'performance': {
        'test_accuracy': 0.9936,
        'validation_accuracy': 0.9933,
        'roc_auc': 0.9997,
        'f1_score': 0.9936,
        'precision': 0.9936,
        'recall': 0.9936
    },
    'training_details': {
        'training_samples': 42000,
        'validation_samples': 14000,
        'test_samples': 14000,
        'total_samples': 70000,
        'features': 18,
        'classes': 2
    }
}

class HeartRiskIntegration:
    """Integration class for heart disease risk prediction model."""

    def __init__(self, model_path: str = None):
        """Initialize the heart disease model integration."""
        if model_path is None:
            backendnew_root = os.path.dirname(os.path.dirname(__file__))
            model_path = os.path.join(backendnew_root, 'models', 'Heart Risk Predictive Model')
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.is_loaded = False

        self._load_model()

    def _load_model(self) -> bool:
        """Load the heart disease risk model and components."""
        try:
            model_file_path = os.path.join(self.model_path, 'heart-prediction.joblib')

            if not os.path.exists(model_file_path):
                raise FileNotFoundError(f"Model file not found: {model_file_path}")

            loaded_model = joblib.load(model_file_path)

            if not isinstance(loaded_model, dict):
                raise ValueError("Model file must be a dictionary containing bundled components")

            self.model = loaded_model['model']
            self.scaler = loaded_model.get('scaler')
            self.feature_names = loaded_model.get('feature_names', INPUT_FEATURES)

            print(f"✅ Loaded heart disease risk model from {model_file_path}")

            if self.scaler is not None:
                print("✅ Loaded scaler from bundled model")
            else:
                print("⚠️ Scaler not found in bundled model, predictions may not be normalized")

            if self.feature_names is not None:
                print(f"✅ Loaded feature names: {len(self.feature_names)} features")
            else:
                self.feature_names = INPUT_FEATURES
                print("⚠️ Feature names not found in bundled model, using default features")

            self.is_loaded = True
            print("✅ Heart disease risk model ready for predictions")
            return True

        except Exception as e:
            print(f"❌ Error loading heart disease model: {str(e)}")
            self.is_loaded = False
            return False

    def _validate_input(self, input_data: Dict) -> bool:
        """Validate input data for prediction."""
        if not isinstance(input_data, dict):
            logger.error("Input data must be a dictionary")
            return False

        missing_features = []
        for feature in INPUT_FEATURES:
            if feature not in input_data:
                missing_features.append(feature)

        if missing_features:
            logger.error(f"Missing required features: {missing_features}")
            return False

        for feature, value in input_data.items():
            if feature == 'Age':
                if not isinstance(value, (int, float)) or value < 0 or value > 120:
                    logger.error(f"Invalid age value: {value}")
                    return False
            elif feature == 'Gender':
                if value not in [0, 1]:
                    logger.error(f"Invalid gender value: {value} (must be 0 or 1)")
                    return False
            else:
                if value not in [0, 1]:
                    logger.error(f"Invalid binary feature value for {feature}: {value} (must be 0 or 1)")
                    return False

        return True

    def _preprocess(self, input_data: Dict) -> np.ndarray:
        """Preprocess input data for model prediction."""
        df = pd.DataFrame([input_data])
        df = df[INPUT_FEATURES]
        df = df.astype(float)

        if self.scaler is not None:
            return self.scaler.transform(df)
        else:
            return df.values

    def _process(self, processed_input: np.ndarray) -> Dict:
        """Process the preprocessed input through the model."""
        prediction = self.model.predict(processed_input)[0]
        probabilities = self.model.predict_proba(processed_input)[0]

        return {
            'prediction': int(prediction),
            'probabilities': probabilities,
            'confidence': float(probabilities[int(prediction)])
        }

    def _fallback(self, input_data: Dict) -> Dict:
        """Fallback prediction method when model is not available."""
        logger.warning("Using fallback prediction method")

        risk_score = 0
        max_score = 0

        age = input_data.get('Age', 50)
        if age > 60:
            risk_score += 2
        elif age > 45:
            risk_score += 1
        max_score += 2

        if input_data.get('Gender', 0) == 1:
            risk_score += 1
        max_score += 1

        major_factors = ['High_BP', 'High_Cholesterol', 'Diabetes', 'Smoking']
        for factor in major_factors:
            if input_data.get(factor, 0) == 1:
                risk_score += 2
            max_score += 2

        other_factors = ['Obesity', 'Sedentary_Lifestyle', 'Family_History', 'Chronic_Stress']
        for factor in other_factors:
            if input_data.get(factor, 0) == 1:
                risk_score += 1
            max_score += 1

        symptom_factors = SYMPTOM_FEATURES
        for factor in symptom_factors:
            if input_data.get(factor, 0) == 1:
                risk_score += 0.5
            max_score += 0.5

        risk_percentage = min(100, (risk_score / max_score) * 100) if max_score > 0 else 50
        prediction = 1 if risk_percentage >= 50 else 0
        risk_level = 'High Risk' if prediction == 1 else 'Low Risk'

        return {
            'success': True,
            'prediction': prediction,
            'risk_level': risk_level,
            'confidence': 0.5,
            'confidence_percent': "50.0%",
            'risk_score': risk_percentage,
            'risk_score_display': f"{risk_percentage:.1f}%",
            'severity': 'Unknown (Fallback)',
            'severity_color': 'gray',
            'probabilities': {
                'low_risk': 1 - (risk_percentage / 100),
                'high_risk': risk_percentage / 100
            },
            'fallback': True
        }

    def predict(self, input_data: Dict) -> Dict:
        """Make a heart disease risk prediction."""
        if not self.is_loaded:
            logger.warning("Model not loaded, using fallback prediction")
            return self._fallback(input_data)

        if not self._validate_input(input_data):
            return {
                'success': False,
                'error': 'Invalid input data',
                'prediction': None
            }

        try:
            processed_input = self._preprocess(input_data)
            result = self._process(processed_input)

            prediction = result['prediction']
            probabilities = result['probabilities']
            confidence = result['confidence']

            risk_level = 'High Risk' if prediction == 1 else 'Low Risk'
            risk_score = float(probabilities[1]) * 100

            if risk_score < 25:
                severity = 'Very Low'
                severity_color = 'green'
            elif risk_score < 50:
                severity = 'Low'
                severity_color = 'lightgreen'
            elif risk_score < 75:
                severity = 'Moderate'
                severity_color = 'orange'
            else:
                severity = 'High'
                severity_color = 'red'

            return {
                'success': True,
                'prediction': prediction,
                'risk_level': risk_level,
                'confidence': confidence,
                'confidence_percent': f"{confidence * 100:.1f}%",
                'risk_score': risk_score,
                'risk_score_display': f"{risk_score:.1f}%",
                'method': 'AdaBoost_Classifier',
                'severity': severity,
                'severity_color': severity_color,
                'probabilities': {
                    'low_risk': float(probabilities[0]),
                    'high_risk': float(probabilities[1])
                }
            }

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return self._fallback(input_data)

    def get_model_info(self) -> Dict:
        """Get model metadata and performance information."""
