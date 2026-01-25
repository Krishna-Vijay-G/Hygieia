#!/usr/bin/env python3
"""
Breast Cancer Risk Predictive System for Hygieia Application

Integrates BCSC clinical risk factors model for breast cancer risk prediction.
Uses ensemble of 6 XGBoost models with calibration for clinical decision support.

Model Performance: 81.3% accuracy, 0.902 AUC-ROC, 82.1% cancer detection
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib
import logging
from typing import Dict, List, Tuple, Optional, Union

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODEL_INFO = {
    'name': 'Breast Cancer Risk Prediction',
    'id': 'breast-risk',
    'model_name': 'BC Predictive Model',
    'method': 'XGB_Ensemble',
    'description': 'Breast cancer risk prediction using ensemble of 6 XGBoost models with calibration',
    'version': '1.0',
    'dataset': 'BCSC Prediction Factors Dataset - BCSC',
    'modified_date': '2026-01-04',
    'author': 'Krishna Vijay G',
    'auth_url': 'https://Krishna-Vijay-G.github.io',
    'training_date': '2025-12-31',
    'performance': {
        'test_accuracy': 0.813,
        'validation_accuracy': 0.821,
        'roc_auc': 0.902,
        'f1_score': 0.813,
        'precision': 0.821,
        'recall': 0.821
    },
    'training_details': {
        'training_samples': 30240,
        'validation_samples': 7560,
        'test_samples': 7560,
        'total_samples': 45360,
        'features': 10,
        'classes': 2
    }
}

class BreastRiskIntegration:
    """
    Integration class for breast cancer risk prediction model in the Hygieia application

    Uses BCSC clinical risk factors (age, density, family history, etc.)
    Ensemble of 6 XGBoost models: 81.3% accuracy, 82.1% cancer detection
    """

    def __init__(self, model_path: str = None):
        """
        Initialize the breast cancer risk prediction model integration

        Args:
            model_path: Path to the models directory. If None, uses default paths.
        """
        self.base_path = model_path or os.path.dirname(os.path.dirname(__file__))

        # Model components
        self.models = None
        self.scaler = None
        self.calibrator = None
        self.threshold = 0.5

        # Model status
        self.is_loaded = False

        # Feature definitions
        self.prediction_features = [
            'age', 'density', 'race', 'bmi', 'agefirst', 'nrelbc',
            'brstproc', 'hrt', 'family_hx', 'menopaus'
        ]

        self.prediction_descriptions = {
            'age': 'Age group (1-13, where 1=18-29, 13=85+)',
            'density': 'Breast density (1=almost entirely fat, 4=extremely dense)',
            'race': 'Race/ethnicity (1=Non-Hispanic white, 2=Non-Hispanic black, etc.)',
            'bmi': 'BMI group (1=10-24.99, 2=25-29.99, 3=30-34.99, 4=35+)',
            'agefirst': 'Age at menarche (0=>14, 1=12-13, 2=<12)',
            'nrelbc': 'Age at first birth (0=<20, 1=20-24, 2=25-29, 3=>30, 4=nulliparous)',
            'brstproc': 'Previous breast biopsy (0=No, 1=Yes)',
            'hrt': 'Hormone replacement therapy (0=No, 1=Yes)',
            'family_hx': 'Family history of breast cancer (0=No, 1=Yes)',
            'menopaus': 'Menopausal status (1=pre/peri, 2=post, 3=surgical)'
        }

        # Load model
        self._load_model()

    def _load_model(self) -> bool:
        """
        Load the breast risk prediction model

        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            prediction_path = os.path.join(self.base_path, 'models', 'BC Predictive Model', 'breast-prediction.joblib')
            if os.path.exists(prediction_path):
                self.prediction_bundle = joblib.load(prediction_path)

                # Extract model components
                self.models = self.prediction_bundle['models']
                self.scaler = self.prediction_bundle['scaler']
                self.calibrator = self.prediction_bundle['calibrator']
                self.threshold = self.prediction_bundle['threshold']

                self.is_loaded = True
                logger.info("✅ Breast Risk Prediction model loaded successfully")
                return True
            else:
                logger.warning(f"❌ Breast Risk Prediction model not found at: {prediction_path}")
                self.is_loaded = False
                return False
        except Exception as e:
            logger.error(f"❌ Error loading Breast Risk Prediction model: {e}")
            self.is_loaded = False
            return False

    def _validate_input(self, data: Dict[str, Union[str, int, float]]) -> Tuple[bool, List[str]]:
        """
        Validate input data for Breast Risk Prediction model

        Args:
            data: Dictionary with prediction risk factors

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        # Check if all required features are present
        for feature in self.prediction_features:
            if feature not in data:
                errors.append(f"Missing required feature: {feature}")

        # Validate ranges for each feature (updated to match actual data ranges)
        validations = {
            'age': (1, 13),
            'density': (1, 9),
            'race': (1, 9),
            'bmi': (1, 9),
            'agefirst': (0, 9),
            'nrelbc': (0, 9),
            'brstproc': (0, 9),
            'hrt': (0, 9),
            'family_hx': (0, 9),
            'menopaus': (1, 9)
        }

        for feature, (min_val, max_val) in validations.items():
            if feature in data:
                try:
                    val = float(data[feature])
                    if val < min_val or val > max_val:
                        errors.append(f"{feature} must be between {min_val} and {max_val}")
                except (ValueError, TypeError):
                    errors.append(f"{feature} must be a valid number")

        return len(errors) == 0, errors

    def _preprocess(self, input_data: Dict[str, Union[str, int, float]]) -> np.ndarray:
        """Preprocess input data for model prediction."""
        df = pd.DataFrame([input_data])

        # Verify all required columns are present before feature engineering
        missing_cols = [col for col in self.prediction_features if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required features for prediction: {', '.join(missing_cols)}")

        X_eng = self._engineer_ultra_features(df)
        return X_eng.values

    def _engineer_ultra_features(self, df: pd.DataFrame):
        """Ultra comprehensive feature engineering for Breast Prediction model - same as training."""
        X = df[self.prediction_features].copy()

        # === TWO-WAY INTERACTIONS ===
        X['age_density'] = X['age'] * X['density']
        X['age_bmi'] = X['age'] * X['bmi']
        X['age_family_hx'] = X['age'] * X['family_hx']
        X['age_hrt'] = X['age'] * X['hrt']
        X['age_brstproc'] = X['age'] * X['brstproc']
        X['age_menopaus'] = X['age'] * X['menopaus']
        X['density_bmi'] = X['density'] * X['bmi']
        X['density_hrt'] = X['hrt'] * X['density']
        X['density_family_hx'] = X['family_hx'] * X['density']
        X['density_brstproc'] = X['brstproc'] * X['density']
        X['density_menopaus'] = X['density'] * X['menopaus']
        X['density_agefirst'] = X['density'] * X['agefirst']
        X['bmi_hrt'] = X['bmi'] * X['hrt']
        X['bmi_family_hx'] = X['bmi'] * X['family_hx']
        X['bmi_menopaus'] = X['bmi'] * X['menopaus']
        X['hrt_menopaus'] = X['hrt'] * X['menopaus']
        X['hrt_agefirst'] = X['hrt'] * X['agefirst']
        X['hrt_nrelbc'] = X['hrt'] * X['nrelbc']
        X['family_hx_hrt'] = X['family_hx'] * X['hrt']
        X['family_hx_brstproc'] = X['brstproc'] * X['family_hx']
        X['family_hx_menopaus'] = X['family_hx'] * X['menopaus']

        # === THREE-WAY INTERACTIONS ===
        X['age_density_family'] = X['age'] * X['density'] * X['family_hx']
        X['age_density_hrt'] = X['age'] * X['density'] * X['hrt']
        X['age_density_bmi'] = X['age'] * X['density'] * X['bmi']
        X['density_bmi_hrt'] = X['density'] * X['bmi'] * X['hrt']
        X['density_family_hrt'] = X['density'] * X['family_hx'] * X['hrt']
        X['age_family_hrt'] = X['age'] * X['family_hx'] * X['hrt']
        X['age_bmi_hrt'] = X['age'] * X['bmi'] * X['hrt']

        # === FOUR-WAY INTERACTIONS ===
        X['age_density_family_hrt'] = X['age'] * X['density'] * X['family_hx'] * X['hrt']
        X['age_density_bmi_hrt'] = X['age'] * X['density'] * X['bmi'] * X['hrt']

        # === POLYNOMIAL FEATURES ===
        X['age_sq'] = X['age'] ** 2
        X['age_cube'] = X['age'] ** 3
        X['density_sq'] = X['density'] ** 2
        X['density_cube'] = X['density'] ** 3
        X['bmi_sq'] = X['bmi'] ** 2
        X['bmi_cube'] = X['bmi'] ** 3

        # === LOGARITHMIC TRANSFORMATIONS ===
        X['log_age'] = np.log1p(X['age'])
        X['log_bmi'] = np.log1p(X['bmi'])
        X['log_density'] = np.log1p(X['density'])

        # === NORMALIZED FEATURES ===
        X['age_norm'] = X['age'] / 100.0
        X['density_norm'] = X['density'] / 4.0
        X['bmi_norm'] = X['bmi'] / 50.0
        X['agefirst_norm'] = X['agefirst'] / 20.0
        X['nrelbc_norm'] = X['nrelbc'] / 10.0

        # === COMPOSITE RISK SCORES ===
        X['clinical_risk'] = (
            (X['density']/4.0)*3.5 + (X['age']/100.0)*3.0 + (X['bmi']/50.0)*2.0 +
            X['hrt']*2.5 + X['brstproc']*2.5 + X['family_hx']*4.0
        )
        X['hormonal_score'] = (
            X['hrt']*3.0 + X['menopaus']*2.0 + (X['agefirst']/20.0)*1.5 + (X['nrelbc']/10.0)*1.0
        )
        X['genetic_risk'] = (
            X['family_hx']*4.0 + (X['age']/100.0)*2.0 + X['race']*0.8 + X['brstproc']*1.5
        )
        X['lifestyle_score'] = (
            (X['bmi']/50.0)*2.5 + X['hrt']*2.0 + (X['nrelbc']/10.0)*1.2 + X['menopaus']*1.0
        )
        X['density_risk'] = (
            X['density']*3.0 + (X['density']**2)*1.5 + X['density']*X['age']/50.0
        )

        # === BINARY RISK FLAGS ===
        X['very_high_density'] = (X['density'] >= 4).astype(int)
        X['high_density'] = (X['density'] >= 3).astype(int)
        X['elderly'] = (X['age'] >= 65).astype(int)
        X['senior'] = (X['age'] >= 55).astype(int)
        X['high_bmi'] = (X['bmi'] >= 30).astype(int)
        X['obese'] = (X['bmi'] >= 35).astype(int)
        X['any_family_hx'] = (X['family_hx'] > 0).astype(int)
        X['any_brstproc'] = (X['brstproc'] > 0).astype(int)
        X['on_hrt'] = (X['hrt'] > 0).astype(int)
        X['postmenopausal'] = (X['menopaus'] > 0).astype(int)

        # === COMBINED RISK FLAGS ===
        X['high_risk_combo'] = ((X['density'] >= 3) & (X['family_hx'] > 0)).astype(int)
        X['ultra_high_risk'] = ((X['density'] >= 3) & (X['family_hx'] > 0) & (X['age'] >= 50)).astype(int)
        X['age_density_risk'] = ((X['age'] >= 55) & (X['density'] >= 3)).astype(int)
        X['age_family_risk'] = ((X['age'] >= 50) & (X['family_hx'] > 0)).astype(int)
        X['density_hrt_risk'] = ((X['density'] >= 3) & (X['hrt'] > 0)).astype(int)
        X['triple_risk'] = ((X['density'] >= 3) & (X['family_hx'] > 0) & (X['hrt'] > 0)).astype(int)
        X['quad_risk'] = ((X['density'] >= 3) & (X['family_hx'] > 0) & (X['hrt'] > 0) & (X['age'] >= 50)).astype(int)

        # === RATIOS AND DIVISIONS ===
        X['age_per_density'] = X['age'] / (X['density'] + 1)
        X['bmi_per_age'] = X['bmi'] / (X['age'] + 1)
        X['density_per_bmi'] = X['density'] / (X['bmi'] + 1)

        return X

    def _process(self, processed_input: np.ndarray) -> Dict:
        """Process the preprocessed input through the model."""
        # Scale features
        X_scaled = self.scaler.transform(processed_input)

        # Get ensemble predictions
        probs = [model.predict_proba(X_scaled)[:, 1] for model in self.models]
        meta = np.vstack(probs).T
        y_proba = self.calibrator.predict_proba(meta)[:, 1][0]
        prediction = int(y_proba >= self.threshold)

        return {
            'prediction': prediction,
            'probability': float(y_proba),
            'confidence': float(max(y_proba, 1 - y_proba))
        }

    def _identify_risk_factors(self, patient_data: Dict[str, Union[str, int, float]]) -> List[str]:
        """Identify clinical risk factors from patient data"""
        risk_factors = []

        try:
            # Age-related risk
            age = patient_data.get('age', 0)
            if age and age >= 8:  # Age 55+
                risk_factors.append('Advanced age (55+ years)')

            # Breast density
            density = patient_data.get('density', 1)
            if density and density >= 4:
                risk_factors.append('Extremely dense breast tissue')
            elif density and density >= 3:
                risk_factors.append('Heterogeneously dense breast tissue')

            # Family history
            if patient_data.get('family_hx') == 1:
                risk_factors.append('Family history of breast cancer')

            # HRT usage
            if patient_data.get('hrt') == 1:
                risk_factors.append('Hormone replacement therapy use')

            # Previous biopsy
            if patient_data.get('brstproc') == 1:
                risk_factors.append('Previous breast biopsy/procedure')

            # BMI
            bmi = patient_data.get('bmi', 1)
            if bmi and bmi >= 4:
                risk_factors.append('Obesity (BMI ≥35)')
            elif bmi and bmi >= 3:
                risk_factors.append('Overweight (BMI 30-34.99)')

            # Reproductive factors
            agefirst = patient_data.get('agefirst', 0)
            if agefirst == 2:
                risk_factors.append('Early menarche (age <12)')

            nrelbc = patient_data.get('nrelbc', 0)
            if nrelbc == 4:
                risk_factors.append('Nulliparous (no children)')
            elif nrelbc == 3:
                risk_factors.append('First child after age 30')
        except (ValueError, TypeError) as e:
            logger.warning(f"Error identifying clinical risk factors: {e}")

        return risk_factors

    def _fallback(self, input_data: Dict[str, Union[str, int, float]]) -> Dict:
        """
        Fallback risk prediction using basic feature analysis.
        Used when trained model is unavailable or fails to load.
        """
        try:
            risk_score = 0
            risk_factors = []

            # Clinical risk assessment
            age = input_data.get('age', input_data.get('Age', 0))
            if age >= 8:  # 55+ years
                risk_score += 0.2
                risk_factors.append('Advanced age')

            density = input_data.get('density', 0)
            if density >= 3:
                risk_score += 0.3
                risk_factors.append('Dense breast tissue')

            if input_data.get('family_hx', 0) == 1:
                risk_score += 0.4
                risk_factors.append('Family history')

            if input_data.get('hrt', 0) == 1:
                risk_score += 0.1
                risk_factors.append('Hormone therapy')

            # Determine prediction
            if risk_score > 0.5:
                prediction = 1
                condition_name = "Malignant"
                risk_level = 'High Risk'
            else:
                prediction = 0
                condition_name = "Benign"
                risk_level = 'Low Risk'

            confidence = min(risk_score, 1.0) if prediction == 1 else 1.0 - risk_score

            return {
                'success': True,
                'prediction': prediction,
                'probability': round(float(risk_score), 3),
                'risk_level': risk_level,
                'confidence': round(float(confidence), 3),
                'condition_name': condition_name,
                'risk_factors': risk_factors,
                'interpretation': f'Rule-based clinical assessment: {risk_level} (confidence: {confidence:.1%})',
                'fallback': True
            }

        except Exception as e:
            logger.error(f"Error in fallback breast risk prediction: {e}")
            return {
                'error': f'Fallback prediction error: {str(e)}',
                'prediction': None,
                'probability': None,
                'success': False
            }

    def predict(self, input_data: Dict[str, Union[str, int, float]]) -> Dict[str, Union[int, float, str, List]]:
        """
        Predict breast cancer risk using BCSC Breast Prediction model

        Args:
            input_data: Dictionary with clinical risk factors

        Returns:
            Dictionary containing prediction results
        """
        if not self.is_loaded:
            logger.warning("Breast Risk Prediction model not loaded, using fallback prediction")
            return self._fallback(input_data)

        # Validate input
        is_valid, errors = self._validate_input(input_data)
        if not is_valid:
            logger.error(f"Validation failed for breast risk prediction: {errors}")
            return {
                'error': f"Invalid input: {'; '.join(errors)}",
                'prediction': None,
                'probability': None,
                'risk_level': None,
                'confidence': None,
                'success': False
            }

        try:
            processed_input = self._preprocess(input_data)
            result = self._process(processed_input)

            prediction = result['prediction']
            y_proba = result['probability']

            # Determine risk level based on prediction (benign/malignant)
            if prediction == 1:
                risk_level = 'High Risk'
            else:
                risk_level = 'Low Risk'

            # Generate interpretation and condition name
            if prediction == 1:
                condition_name = "Malignant"
                interpretation = f"High breast cancer risk detected (probability: {y_proba:.1%})"
            else:
                condition_name = "Benign"
                interpretation = f"Low breast cancer risk (probability: {y_proba:.1%})"

            # Identify risk factors
            risk_factors = self._identify_risk_factors(input_data)

            return {
                'success': True,
                'prediction': prediction,
                'probability': float(y_proba),
                'risk_level': risk_level,
                'confidence': float(result['confidence']),
                'interpretation': interpretation,
                'condition_name': condition_name,
                'risk_factors': risk_factors,
                'threshold_used': float(self.threshold),
                'model_type': 'Breast Risk Prediction',
                'method': 'XGB_Ensemble'
            }

        except Exception as e:
            logger.error(f"Error in breast risk prediction: {e}")
            return self._fallback(input_data)

    def get_model_info(self) -> Dict:
        """Get model metadata and information."""
        return MODEL_INFO