#!/usr/bin/env python3
"""
Diabetes Model Integration for Hygieia
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
    'Age', 'Gender', 'Polyuria', 'Polydipsia', 'sudden weight loss',
    'weakness', 'Polyphagia', 'Genital thrush', 'visual blurring',
    'Itching', 'Irritability', 'delayed healing', 'partial paresis',
    'muscle stiffness', 'Alopecia', 'Obesity'
]

MODEL_INFO = {
    'name': 'Diabetes Risk Prediction',
    'id': 'diabetes-risk',
    'model_name': 'Diabetes Risk Predictive Model',
    'method': 'LightGBM_Classifier',
    'description': 'Early diabetes risk prediction using LightGBM with engineered features',
    'version': '1.0',
    'dataset': 'Early Stage Diabetes Risk Prediction - UCI',
    'modified_date': '2026-01-04',
    'author': 'Krishna Vijay G',
    'auth_url': 'https://Krishna-Vijay-G.github.io',
    'training_date': '2025-12-31',
    'performance': {
        'test_accuracy': 0.98,
        'validation_accuracy': 0.97,
        'roc_auc': 0.995,
        'f1_score': 0.98,
        'precision': 0.98,
        'recall': 0.98
    },
    'training_details': {
        'training_samples': 400,
        'validation_samples': 100,
        'test_samples': 68,
        'total_samples': 520,
        'features': 16,
        'classes': 2
    }
}

class DiabetesRiskIntegration:
    """Integration class for diabetes risk prediction model."""

    def __init__(self, model_path: str = None):
        """Initialize the diabetes risk model integration."""
        if model_path is None:
            backendnew_root = os.path.dirname(os.path.dirname(__file__))
            model_path = os.path.join(backendnew_root, 'models', 'Diabetes Risk Predictive Model')
        self.model_path = model_path
        self.model = None
        self.label_encoders = None
        self.feature_names = None
        self.is_loaded = False

        self._load_model()

    def _load_model(self) -> bool:
        """Load the diabetes risk model and components."""
        try:
            model_file_path = os.path.join(self.model_path, 'diabetes-prediction.joblib')

            if not os.path.exists(model_file_path):
                print(f"❌ Model file not found: {model_file_path}")
                return False

            model_data = joblib.load(model_file_path)

            if isinstance(model_data, dict):
                self.model = model_data.get('model')
                self.label_encoders = model_data.get('label_encoders')
                self.feature_names = model_data.get('feature_names')
            else:
                self.model = model_data

            self.is_loaded = True
            print("✅ Diabetes risk model loaded")
            return True

        except Exception as e:
            print(f"❌ Error loading diabetes model: {e}")
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

        # Validate Age
        age = input_data.get('Age')
        if not isinstance(age, (int, float)) or age < 0 or age > 120:
            logger.error(f"Invalid age value: {age}")
            return False

        # Validate Gender
        gender = input_data.get('Gender')
        if gender not in ['Male', 'Female']:
            logger.error(f"Invalid gender value: {gender} (must be 'Male' or 'Female')")
            return False

        # Validate binary features
        binary_features = [f for f in INPUT_FEATURES if f not in ['Age', 'Gender']]
        for feature in binary_features:
            value = input_data.get(feature)
            if value not in ['Yes', 'No']:
                logger.error(f"Invalid value for {feature}: {value} (must be 'Yes' or 'No')")
                return False

        return True

    def _preprocess(self, input_data: Dict) -> np.ndarray:
        """Preprocess input data for model prediction."""
        df = pd.DataFrame([input_data])

        # Apply label encoding
        if self.label_encoders:
            for col in df.columns:
                if col in self.label_encoders and col != 'Age':
                    df[col] = self.label_encoders[col].transform(df[col])

        # Get features in correct order
        if self.feature_names:
            features = df[self.feature_names].values
        else:
            features = df[INPUT_FEATURES].values

        return features

    def _process(self, processed_input: np.ndarray) -> Dict:
        """Process the preprocessed input through the model."""
        prediction = self.model.predict(processed_input)[0]

        # Get probability
        probability = 0.5
        if hasattr(self.model, 'predict_proba'):
            prob_array = self.model.predict_proba(processed_input)[0]
            probability = prob_array[1]

        return {
            'prediction': int(prediction),
            'probability': float(probability),
            'confidence': float(min(0.95, 0.7 + abs(probability - 0.5) * 0.5))
        }

    def _fallback(self, input_data: Dict) -> Dict:
        """Fallback prediction method when model is not available."""
        logger.warning("Using fallback prediction method")

        # Simple risk assessment based on key symptoms
        risk_factors = [
            'Polyuria', 'Polydipsia', 'sudden weight loss', 'weakness',
            'Polyphagia', 'Genital thrush', 'visual blurring', 'Itching'
        ]

        risk_score = 0
        for factor in risk_factors:
            if input_data.get(factor) == 'Yes':
                risk_score += 1

        # Age factor
        age = input_data.get('Age', 30)
        if age > 45:
            risk_score += 1
        elif age > 30:
            risk_score += 0.5

        probability = min(0.9, risk_score / 10)
        prediction = 1 if probability >= 0.5 else 0
        risk_level = 'High Risk' if prediction == 1 else 'Low Risk'

        return {
            'success': True,
            'prediction': prediction,
            'probability': probability,
            'risk_level': risk_level,
            'confidence': 0.5,
            'confidence_percent': "50.0%",
            'risk_score': probability * 100,
            'interpretation': 'Fallback prediction - limited accuracy',
            'risk_factors': [],
            'fallback': True
        }

    def predict(self, input_data: Dict) -> Dict:
        """Make a diabetes risk prediction."""
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
            probability = result['probability']
            confidence = result['confidence']

            # Determine risk level
            if probability < 0.3:
                risk_level = 'Low Risk'
            elif probability < 0.7:
                risk_level = 'Medium Risk'
            else:
                risk_level = 'High Risk'

            # Risk factors interpretation
            risk_factors = []
            if input_data.get('Polyuria') == 'Yes':
                risk_factors.append('Frequent urination')
            if input_data.get('Polydipsia') == 'Yes':
                risk_factors.append('Excessive thirst')
            if input_data.get('sudden weight loss') == 'Yes':
                risk_factors.append('Sudden weight loss')
            if input_data.get('weakness') == 'Yes':
                risk_factors.append('General weakness')
            if input_data.get('Age', 0) > 45:
                risk_factors.append('Age > 45')

            interpretation = "High diabetes risk detected" if prediction == 1 else "Low diabetes risk"

            return {
                'success': True,
                'prediction': prediction,
                'probability': probability,
                'risk_level': risk_level,
                'confidence': confidence,
                'confidence_percent': f"{confidence * 100:.1f}%",
                'method': 'LightGBM_Classifier',
                'risk_score': probability * 100,
                'interpretation': interpretation,
                'risk_factors': risk_factors
            }

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return self._fallback(input_data)

    def get_model_info(self) -> Dict:
        """Get model metadata and performance information."""
        return MODEL_INFO.copy()
