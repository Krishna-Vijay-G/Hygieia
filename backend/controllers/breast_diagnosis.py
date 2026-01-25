#!/usr/bin/env python3
"""
Breast Cancer Tissue Diagnostic System for Hygieia Application

Integrates Wisconsin Breast Cancer Diagnostic model for tissue analysis.
Uses FNA biopsy data with stacking ensemble of RF, XGB, GB, and SVM models.

Model Performance: 97.2% accuracy, 99.4% AUC-ROC, 97.6% malignant recall
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

# Define StackingEnsemble class globally for pickle compatibility
class StackingEnsemble:
    """Stacking ensemble class - must match training script for unpickling."""
    def __init__(self, bases, meta):
        self.bases = bases
        self.meta = meta
        self.classes_ = bases[0].classes_

    def predict_proba(self, X_scaled):
        base_probs = [b.predict_proba(X_scaled)[:, 1] for b in self.bases]
        stacked = np.column_stack(base_probs)
        meta_probs = self.meta.predict_proba(stacked)[:, 1]
        return np.column_stack([1 - meta_probs, meta_probs])

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODEL_INFO = {
    'name': 'Breast Cancer Tissue Diagnosis',
    'id': 'breast-diagnosis',
    'model_name': 'BC Diagnostic Model',
    'method': 'Stacking_Ensemble',
    'description': 'Breast cancer tissue diagnosis using FNA biopsy data with stacking ensemble of RF, XGB, GB, and SVM models',
    'version': '1.0',
    'dataset': 'Wisconsin Diagnosis Dataset - UCI',
    'modified_date': '2026-01-04',
    'author': 'Krishna Vijay G',
    'auth_url': 'https://Krishna-Vijay-G.github.io',
    'training_date': '2025-12-31',
    'performance': {
        'test_accuracy': 0.972,
        'validation_accuracy': 0.974,
        'roc_auc': 0.994,
        'f1_score': 0.972,
        'precision': 0.976,
        'recall': 0.976
    },
    'training_details': {
        'training_samples': 455,
        'validation_samples': 114,
        'test_samples': 114,
        'total_samples': 569,
        'features': 5,
        'classes': 2
    }
}

class BreastDiagnosisIntegration:
    """
    Integration class for breast cancer tissue diagnosis model in the Hygieia application

    Uses Wisconsin Breast Cancer Diagnostic dataset (FNA biopsy data)
    Stacking ensemble with 4 base models: 97.2% accuracy, 99.4% AUC-ROC
    """

    def __init__(self, model_path: str = None):
        """
        Initialize the breast cancer diagnosis model integration

        Args:
            model_path: Path to the models directory. If None, uses default paths.
        """
        self.base_path = model_path or os.path.dirname(os.path.dirname(__file__))

        # Model components
        self.model = None
        self.scaler = None
        self.model_type = None
        self.best_threshold = 0.5

        # Model status
        self.is_loaded = False

        # Feature definitions
        self.diagnosis_features = [
            'mean_radius', 'mean_texture', 'mean_perimeter', 'mean_area', 'mean_smoothness'
        ]

        self.diagnosis_descriptions = {
            'mean_radius': 'Mean radius of tumor cell nuclei (from FNA)',
            'mean_texture': 'Standard deviation of gray-scale values (surface texture)',
            'mean_perimeter': 'Mean tumor perimeter from cell nuclei',
            'mean_area': 'Mean tumor area from cell nuclei',
            'mean_smoothness': 'Mean local variation in radius lengths (smoothness)'
        }

        # Load model
        self._load_model()

    def _load_model(self) -> bool:
        """
        Load the breast diagnosis model

        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            # Make StackingEnsemble available for unpickling in multiple contexts
            import sys

            # Ensure StackingEnsemble is available in __main__ and current module
            if '__main__' not in sys.modules:
                import types
                main_module = types.ModuleType('__main__')
                sys.modules['__main__'] = main_module

            # Add StackingEnsemble to __main__ module and current module
            sys.modules['__main__'].StackingEnsemble = StackingEnsemble
            globals()['StackingEnsemble'] = StackingEnsemble

            # Also add to the breast_diagnosis module namespace
            import importlib
            current_module = importlib.import_module(__name__)
            current_module.StackingEnsemble = StackingEnsemble

            diagnosis_path = os.path.join(self.base_path, 'models', 'BC Diagnostic Model', 'breast-diagnosis.joblib')
            if os.path.exists(diagnosis_path):
                # Temporarily add StackingEnsemble to the joblib namespace
                import joblib
                original_find_class = joblib.numpy_pickle.Unpickler.find_class

                def custom_find_class(self, module, name):
                    if name == 'StackingEnsemble':
                        return StackingEnsemble
                    return original_find_class(self, module, name)

                joblib.numpy_pickle.Unpickler.find_class = custom_find_class

                try:
                    self.diagnosis_bundle = joblib.load(diagnosis_path)
                finally:
                    # Restore original find_class
                    joblib.numpy_pickle.Unpickler.find_class = original_find_class

                # Extract model components - handle different bundle structures
                if isinstance(self.diagnosis_bundle, dict):
                    self.model = self.diagnosis_bundle.get('models')  # Note: key is 'models', not 'model'
                    self.scaler = self.diagnosis_bundle.get('scaler')
                    self.model_type = self.diagnosis_bundle.get('model_type', 'stacking_ensemble')
                    self.best_threshold = self.diagnosis_bundle.get('threshold', 0.5)
                else:
                    # Assume the bundle is the model directly
                    self.model = self.diagnosis_bundle
                    self.scaler = None
                    self.model_type = 'unknown'
                    self.best_threshold = 0.5

                # Verify model was loaded
                if self.model is None:
                    logger.error("❌ Model loaded but is None")
                    self.is_loaded = False
                    return False

                self.is_loaded = True
                logger.info("✅ Breast Diagnosis model loaded successfully")
                return True
            else:
                logger.warning(f"❌ Breast Diagnosis model not found at: {diagnosis_path}")
                self.is_loaded = False
                return False
        except Exception as e:
            logger.error(f"❌ Error loading Breast Diagnosis model: {e}")
            self.is_loaded = False
            return False

    def _validate_input(self, data: Dict[str, Union[str, int, float]]) -> Tuple[bool, List[str]]:
        """
        Validate input data for Breast Diagnosis model
        
        Args:
            data: Dictionary with tissue features
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check if all required features are present
        for feature in self.diagnosis_features:
            if feature not in data:
                errors.append(f"Missing required feature: {feature}")
        
        # Validate that values are positive numbers (tissue measurements)
        for feature in self.diagnosis_features:
            if feature in data:
                try:
                    val = float(data[feature])
                    if val < 0:
                        errors.append(f"{feature} must be a positive number")
                except (ValueError, TypeError):
                    errors.append(f"{feature} must be a valid number")
        
        return len(errors) == 0, errors

    def _preprocess(self, input_data: Dict[str, Union[str, int, float]]) -> np.ndarray:
        """Preprocess input data for model prediction."""
        df = pd.DataFrame([input_data])
        
        # Verify all required columns are present
        missing_cols = [col for col in self.diagnosis_features if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required features for diagnosis: {', '.join(missing_cols)}")
        
        # Engineer features
        X_eng = self._engineer_diagnosis_features(df)
        
        # Prepare features
        X = X_eng.values
        if self.scaler is not None:
            X = self.scaler.transform(X)
        
        return X

    def _engineer_diagnosis_features(self, X: pd.DataFrame):
        """Create additional engineered features for Kaggle model - same as training."""
        X = X.copy()
        
        # Ratio features
        X['radius_to_area'] = X['mean_radius'] / (X['mean_area'] + 1)
        X['perimeter_to_area'] = X['mean_perimeter'] / (X['mean_area'] + 1)
        X['texture_density'] = X['mean_texture'] * X['mean_smoothness']
        
        # Polynomial features
        X['radius_squared'] = X['mean_radius'] ** 2
        X['area_squared'] = X['mean_area'] ** 2
        
        # Interaction features
        X['radius_x_texture'] = X['mean_radius'] * X['mean_texture']
        X['area_x_smoothness'] = X['mean_area'] * X['mean_smoothness']
        
        return X

    def _process(self, processed_input: np.ndarray) -> Dict:
        """Process the preprocessed input through the model."""
        # Get prediction
        if self.model_type == 'lightgbm':
            y_proba = self.model.predict(processed_input)[0]
        else:
            y_proba = self.model.predict_proba(processed_input)[0, 1]

        prediction = int(y_proba >= self.best_threshold)

        return {
            'prediction': prediction,
            'probability': float(y_proba),
            'confidence': float(max(y_proba, 1 - y_proba))
        }

    def _identify_risk_factors(self, tissue_data: Dict[str, Union[str, int, float]]) -> List[str]:
        """Identify tissue-based risk factors from FNA measurements (Wisconsin diagnostic criteria)"""
        risk_factors = []

        try:
            # Large cell nuclei
            radius = tissue_data.get('mean_radius', 0)
            if radius and radius > 17:
                risk_factors.append('Large cell nuclei (high radius)')
            elif radius and radius > 14:
                risk_factors.append('Moderately enlarged cell nuclei')

            # High texture variation
            texture = tissue_data.get('mean_texture', 0)
            if texture and texture > 22:
                risk_factors.append('High texture variation in cells')

            # Large perimeter
            perimeter = tissue_data.get('mean_perimeter', 0)
            if perimeter and perimeter > 110:
                risk_factors.append('Large cell perimeter')

            # Large area
            area = tissue_data.get('mean_area', 0)
            if area and area > 900:
                risk_factors.append('Large cell area')
            elif area and area > 650:
                risk_factors.append('Moderately large cell area')

            # Smoothness irregularities
            smoothness = tissue_data.get('mean_smoothness', 0)
            if smoothness and smoothness > 0.12:
                risk_factors.append('Cell surface irregularities')
        except (ValueError, TypeError) as e:
            logger.warning(f"Error identifying tissue risk factors: {e}")

        return risk_factors

    def _fallback(self, input_data: Dict[str, Union[str, int, float]]) -> Dict:
        """
        Fallback diagnosis using basic feature analysis.
        Used when trained model is unavailable or fails to load.
        """
        try:
            risk_score = 0
            risk_factors = []

            # Tissue feature assessment
            radius_mean = input_data.get('mean_radius', 0)
            area_mean = input_data.get('mean_area', 0)
            texture_mean = input_data.get('mean_texture', 0)

            if radius_mean > 15:
                risk_score += 0.3
                risk_factors.append('Large cell nuclei')
            elif radius_mean > 12:
                risk_score += 0.15

            if area_mean > 700:
                risk_score += 0.25
                risk_factors.append('Large cell area')
            elif area_mean > 500:
                risk_score += 0.1

            if texture_mean > 20:
                risk_score += 0.15
                risk_factors.append('High texture variation')

            # Determine prediction
            if risk_score > 0.4:
                prediction = 1
                condition_name = 'Malignant'
                risk_level = 'High Risk'
            else:
                prediction = 0
                condition_name = 'Benign'
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
                'interpretation': f'Rule-based tissue assessment: {condition_name} characteristics (confidence: {confidence:.1%})',
                'fallback': True
            }

        except Exception as e:
            logger.error(f"Error in fallback breast diagnosis: {e}")
            return {
                'error': f'Fallback diagnosis error: {str(e)}',
                'prediction': None,
                'probability': None,
                'success': False
            }

    def predict(self, input_data: Dict[str, Union[str, int, float]]) -> Dict[str, Union[int, float, str, List]]:
        """
        Predict breast cancer diagnosis using Wisconsin diagnostic model (97.2% accuracy, 99.4% AUC-ROC)
        Stacking ensemble with 4 base models trained on Fine Needle Aspiration data

        Args:
            input_data: Dictionary with FNA tissue measurements

        Returns:
            Dictionary containing prediction results
        """
        if not self.is_loaded:
            logger.warning("Breast Diagnosis model not loaded, using fallback prediction")
            return self._fallback(input_data)

        # Validate input
        is_valid, errors = self._validate_input(input_data)
        if not is_valid:
            logger.error(f"Validation failed for breast diagnosis: {errors}")
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

            # Determine risk level (Wisconsin model uses optimized threshold of 0.200)
            if y_proba >= 0.8:
                risk_level = 'High Risk'
            elif y_proba >= 0.5:
                risk_level = 'Moderate Risk'
            else:
                risk_level = 'Low Risk'

            # Generate interpretation
            condition_name = 'Malignant' if prediction == 1 else 'Benign'
            interpretation = f"Wisconsin diagnostic model (97.2% accuracy): {condition_name} characteristics (probability: {y_proba:.1%})"

            # Identify tissue-based risk factors
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
                'threshold_used': float(self.best_threshold),
                'model_type': 'Breast Diagnosis',
                'method': 'Stacking_Ensemble'
            }

        except Exception as e:
            logger.error(f"Error in breast diagnosis prediction: {e}")
            return self._fallback(input_data)

    def get_model_info(self) -> Dict:
        """Get model metadata and information."""
        return MODEL_INFO