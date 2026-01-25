import os
import logging
import numpy as np
import tensorflow as tf
from PIL import Image
from io import BytesIO
from typing import Dict, Any, Optional, Tuple, List
import joblib
from datetime import datetime
import warnings
import cv2
warnings.filterwarnings('ignore')

# Import calibration module for bias correction
try:
    from .skin_calibration import calibrate_prediction_result
except ImportError:
    # Fallback for dynamic imports
    import sys
    import os
    calibration_path = os.path.join(os.path.dirname(__file__), 'skin_calibration.py')
    if os.path.exists(calibration_path):
        import importlib.util
        spec = importlib.util.spec_from_file_location('skin_calibration', calibration_path)
        calibration_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(calibration_module)
        calibrate_prediction_result = calibration_module.calibrate_prediction_result
    else:
        # Fallback function if calibration module not found
        def calibrate_prediction_result(result, method='default'):
            return result

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model configuration
DERM_FOUNDATION_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'Skin Lesion Diagnostic Model')
OPTIMIZED_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'Skin Lesion Diagnostic Model', 'skin-diagnosis.joblib')
MODEL_INPUT_SIZE = (448, 448)

# Class mappings for HAM10000-style predictions
SKIN_CONDITIONS = {
    0: 'akiec',  # Actinic keratoses and intraepithelial carcinoma
    1: 'bcc',    # basal cell carcinoma
    2: 'bkl',    # benign keratosis-like lesions
    3: 'df',     # dermatofibroma
    4: 'nv',     # melanocytic nevi
    5: 'vasc',   # pyogenic granulomas and hemorrhage
    6: 'mel'     # melanoma
}

CONDITION_NAMES = {
    'akiec': 'Actinic Keratoses',
    'bcc': 'Basal Cell Carcinoma', 
    'bkl': 'Benign Keratosis',
    'df': 'Dermatofibroma',
    'nv': 'Melanocytic Nevi',
    'vasc': 'Vascular Lesions',
    'mel': 'Melanoma'
}

# Risk level mapping for each condition
RISK_LEVELS = {
    'akiec': 'Moderate Risk',  # Pre-cancerous lesions
    'bcc': 'High Risk',        # Most common skin cancer
    'bkl': 'Low Risk',         # Benign condition
    'df': 'Low Risk',          # Benign skin nodules
    'nv': 'Low Risk',          # Common moles
    'vasc': 'Low Risk',        # Vascular lesions typically benign
    'mel': 'High Risk'         # Dangerous skin cancer
}

MODEL_INFO = {
    'name': 'Skin Lesion Diagnosis',
    'id': 'skin-diagnosis',
    'model_name': 'Skin Lesion Diagnostic Model',
    'method': 'CNN_Voting_Ensemble',
    'description': 'Multi-class skin disease classification using CNN features with voting ensemble',
    'version': '1.0',
    'dataset': 'HAM10000 Dataset - ISIC',
    'modified_date': '2026-01-05',
    'author': 'Krishna Vijay G',
    'auth_url': 'https://Krishna-Vijay-G.github.io',
    'training_date': '2026-01-04',
    'performance': {
        'test_accuracy': 0.9884313969399179,
        'validation_accuracy': 0.8214939614311764,
        'roc_auc': 0.993,
        'f1_score': 0.9883062502087258,
        'precision': 0.9886283003228951,
        'recall': 0.9884313969399179
    },
    'training_details': {
        'training_samples': 8039,
        'validation_samples': 790,
        'test_samples': 1186,
        'total_samples': 10015,
        'features': 6224,
        'classes': 7
    }
}

class SkinDiagnosisIntegration:
    """
    Integration class for skin lesion diagnosis model in the Hygieia application

    Uses HAM10000 dataset with Derm Foundation model and optimized ensemble classifier
    Model v4.0: 95.9% peak accuracy, 93.9% mean production accuracy
    """

    def __init__(self, model_path: str = None):
        """
        Initialize the skin diagnosis model integration

        Args:
            model_path: Path to the models directory. If None, uses default paths.
        """
        self.base_path = model_path or os.path.dirname(os.path.dirname(__file__))

        # Model components
        self.derm_foundation_model = None
        self.optimized_classifier_data = None

        # Model status
        self.is_loaded = False

        # Load model on initialization
        self._load_model()

    def _load_model(self) -> bool:
        """
        Load the skin diagnosis models

        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            # Load Derm Foundation model
            model_path = DERM_FOUNDATION_PATH
            if not os.path.exists(model_path):
                logger.error(f"Model path not found: {model_path}")
                return False

            logger.info(f"Loading Derm Foundation model from: {model_path}")

            # Check for saved_model.pb file
            saved_model_path = os.path.join(model_path, 'saved_model.pb')
            if not os.path.exists(saved_model_path):
                logger.error(f"Model file not found: {saved_model_path}")
                logger.error("Make sure the Derm Foundation model is properly installed")
                return False

            # Try loading the model with enhanced error handling
            try:
                self.derm_foundation_model = tf.saved_model.load(model_path)

                # Verify model structure
                if 'serving_default' not in self.derm_foundation_model.signatures:
                    logger.error("Invalid model: 'serving_default' signature not found")
                    self.derm_foundation_model = None
                    return False

                # Test model structure with a basic signature check
                signature = self.derm_foundation_model.signatures["serving_default"]
                input_names = list(signature.structured_input_signature[1].keys())

                if 'inputs' not in input_names:
                    logger.error(f"Model has unexpected input signature: {input_names}")
                    self.derm_foundation_model = None
                    return False

                logger.info("✅ Derm Foundation model loaded and verified successfully")

            except tf.errors.NotFoundError as e:
                logger.error(f"TensorFlow couldn't find model components: {e}")
                return False
            except tf.errors.InvalidArgumentError as e:
                logger.error(f"TensorFlow model has invalid format: {e}")
                return False

            # Load optimized classifier
            if os.path.exists(OPTIMIZED_MODEL_PATH):
                logger.info(f"Loading optimized classifier from: {OPTIMIZED_MODEL_PATH}")
                self.optimized_classifier_data = joblib.load(OPTIMIZED_MODEL_PATH)
                accuracy = self.optimized_classifier_data.get('training_history', {}).get('accuracy', 0.0)
                logger.info(f"✅ Optimized classifier loaded (Training accuracy: {accuracy:.3f})")
            else:
                logger.warning(f"Optimized model not found: {OPTIMIZED_MODEL_PATH}")
                logger.warning("Falling back to pattern-based analysis")

            self.is_loaded = True
            return True

        except Exception as e:
            logger.error(f"Error loading skin diagnosis models: {e}")
            return False

    def _validate_input(self, image_input) -> Tuple[bool, List[str]]:
        """
        Validate input image

        Args:
            image_input: Either image file path (str) or numpy array

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        if image_input is None:
            errors.append("Image input cannot be None")
            return False, errors

        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                errors.append(f"Image file not found: {image_input}")
        elif isinstance(image_input, np.ndarray):
            if image_input.size == 0:
                errors.append("Image array is empty")
            elif len(image_input.shape) < 2 or len(image_input.shape) > 3:
                errors.append(f"Invalid image array shape: {image_input.shape}")
        else:
            errors.append("Input must be either a file path or numpy array")

        return len(errors) == 0, errors

    def _preprocess(self, image_input) -> Optional[bytes]:
        """
        Preprocess image according to Derm Foundation requirements

        Args:
            image_input: Either image file path (str) or numpy array

        Returns:
            Serialized tf.train.Example ready for model input
        """
        try:
            # Load image
            if isinstance(image_input, str):
                if not os.path.exists(image_input):
                    raise FileNotFoundError(f"Image file not found: {image_input}")
                img = Image.open(image_input)
            elif isinstance(image_input, np.ndarray):
                # Convert numpy array to PIL Image
                if image_input.dtype != np.uint8:
                    if image_input.max() <= 1.0:
                        image_input = (image_input * 255).astype(np.uint8)
                    else:
                        image_input = image_input.astype(np.uint8)
                img = Image.fromarray(image_input)
            else:
                raise ValueError("Input must be either a file path or numpy array")

            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # Resize to model input size (448x448 for Derm Foundation)
            img = img.resize(MODEL_INPUT_SIZE, Image.LANCZOS)

            # Convert to bytes
            buf = BytesIO()
            img.save(buf, format='PNG')
            image_bytes = buf.getvalue()

            # Create tf.train.Example
            input_tensor = tf.train.Example(features=tf.train.Features(
                feature={'image/encoded': tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[image_bytes]))}
            )).SerializeToString()

            return input_tensor

        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            return None

    def _process(self, image_bytes: bytes) -> Optional[np.ndarray]:
        """
        Process image bytes to generate embedding

        Args:
            image_bytes: Preprocessed image bytes

        Returns:
            6144-dimensional embedding vector or None if failed
        """
        try:
            if self.derm_foundation_model is None:
                logger.error("Derm Foundation model not loaded")
                return None

            # Convert to tensor correctly
            input_tensor = tf.constant([image_bytes])

            # Generate embedding
            infer = self.derm_foundation_model.signatures["serving_default"]
            result = infer(inputs=input_tensor)

            # Extract embedding
            embedding = result['embedding'].numpy().flatten()

            logger.debug(f"Generated embedding with shape: {embedding.shape}, norm: {np.linalg.norm(embedding):.2f}")
            return embedding

        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return None

    def _fallback(self, embedding: np.ndarray) -> Dict[str, Any]:
        """
        Fallback pattern-based analysis if optimized classifier fails

        Args:
            embedding: 6144-dimensional embedding from Derm Foundation model

        Returns:
            Dictionary containing prediction results
        """
        try:
            # Calculate comprehensive embedding statistics
            embedding_norm = np.linalg.norm(embedding)
            embedding_mean = np.mean(embedding)
            embedding_std = np.std(embedding)
            positive_ratio = np.sum(embedding > 0) / len(embedding)

            logger.info(f"Fallback analysis: norm={embedding_norm:.2f}, mean={embedding_mean:.3f}, "
                       f"std={embedding_std:.3f}, pos_ratio={positive_ratio:.3f}")

            # More balanced initial probabilities (equal for all classes)
            # akiec, bcc,  bkl,  df,   nv,   vasc, mel
            probabilities = np.array([0.14, 0.14, 0.14, 0.14, 0.15, 0.14, 0.15])

            # BALANCED NORM-BASED CLASSIFICATION
            # Reworked to be more balanced between classes and avoid overfitting to akiec
            if embedding_norm > 40:
                # Very high norm - irregular lesions (melanoma and others)
                probabilities[6] *= 1.5  # mel (melanoma)
                probabilities[0] *= 1.2  # akiec
                probabilities[1] *= 1.2  # bcc
                probabilities[4] *= 0.9  # nv
            elif embedding_norm > 37:
                # High norm - melanoma and bcc range
                probabilities[6] *= 1.4  # mel
                probabilities[1] *= 1.3  # bcc
                probabilities[0] *= 1.1  # akiec
            elif embedding_norm > 34:
                # Medium-high norm - balanced adjustments
                probabilities[4] *= 1.3  # nv
                probabilities[1] *= 1.2  # bcc
                probabilities[2] *= 1.2  # bkl
                probabilities[6] *= 1.1  # mel
            elif embedding_norm > 30:
                # Medium norm - balanced middle range
                probabilities[3] *= 1.2  # df
                probabilities[2] *= 1.2  # bkl
                probabilities[1] *= 1.1  # bcc
                probabilities[5] *= 1.1  # vasc
                probabilities[0] *= 0.9  # akiec (reduce bias)
            else:
                # Low norm - more likely benign
                probabilities[2] *= 1.3  # bkl
                probabilities[5] *= 1.2  # vasc
                probabilities[3] *= 1.2  # df
                probabilities[4] *= 1.1  # nv
                probabilities[0] *= 0.8  # akiec (reduce bias)
                probabilities[6] *= 0.8  # mel (reduce bias)

            # POSITIVE RATIO PATTERNS - More balanced
            if positive_ratio > 0.65:
                # High positive ratio - vascular, dermatofibroma
                probabilities[5] *= 1.5  # vasc
                probabilities[3] *= 1.4  # df
                probabilities[0] *= 0.9  # akiec (reduce bias)
            elif positive_ratio > 0.55:
                # Medium-high positive ratio
                probabilities[2] *= 1.3  # bkl
                probabilities[3] *= 1.2  # df
            elif positive_ratio > 0.45:
                # Medium positive ratio - balanced
                # No adjustments to maintain balance
                pass
            elif positive_ratio > 0.35:
                # Medium-low positive ratio
                probabilities[4] *= 1.3  # nv
                probabilities[1] *= 1.2  # bcc
            else:
                # Very low positive ratio - darker lesions
                probabilities[6] *= 1.3  # mel
                probabilities[4] *= 1.2  # nv
                probabilities[0] *= 0.9  # akiec (reduce bias)

            # STANDARD DEVIATION PATTERNS - More balanced
            if embedding_std > 0.50:
                # Very high std - heterogeneous lesions
                probabilities[6] *= 1.4  # mel
                probabilities[0] *= 1.3  # akiec
                probabilities[1] *= 1.2  # bcc
            elif embedding_std > 0.40:
                # High std - medium heterogeneity
                probabilities[1] *= 1.3  # bcc
                probabilities[6] *= 1.2  # mel
            elif embedding_std > 0.30:
                # Medium std - balanced
                # No adjustments to maintain balance
                pass
            else:
                # Low std - more uniform (nv, bkl)
                probabilities[4] *= 1.3  # nv
                probabilities[2] *= 1.2  # bkl
                probabilities[0] *= 0.9  # akiec (reduce bias)

            # Additional patterns based on mean value
            if embedding_mean > 0.1:
                # Higher mean - increased probability for certain classes
                probabilities[2] *= 1.2  # bkl
                probabilities[5] *= 1.1  # vasc
            elif embedding_mean < -0.1:
                # Lower mean - different pattern
                probabilities[4] *= 1.2  # nv
                probabilities[1] *= 1.1  # bcc

            # Calculate gradient pattern
            try:
                gradient = np.gradient(embedding)
                gradient_std = np.std(gradient)

                # High gradient variability suggests irregular borders
                if gradient_std > 0.2:
                    probabilities[6] *= 1.2  # mel
                    probabilities[0] *= 1.1  # akiec
                elif gradient_std < 0.05:
                    probabilities[4] *= 1.1  # nv
                    probabilities[2] *= 1.1  # bkl
            except:
                pass

            # Normalize probabilities
            probabilities = probabilities / np.sum(probabilities)

            # Smooth probabilities to avoid extreme confidence values
            smoothed_probs = 0.8 * probabilities + 0.2 * np.ones(7) / 7
            smoothed_probs = smoothed_probs / np.sum(smoothed_probs)

            # Get prediction
            predicted_class_idx = np.argmax(smoothed_probs)
            predicted_class = list(SKIN_CONDITIONS.values())[predicted_class_idx]
            confidence = float(smoothed_probs[predicted_class_idx])

            # Limit confidence for fallback method
            confidence = min(confidence, 0.70)

            # Create probabilities dictionary
            prob_dict = {}
            for i, class_name in enumerate(SKIN_CONDITIONS.values()):
                prob_dict[class_name] = float(smoothed_probs[i])

            condition_name = CONDITION_NAMES.get(predicted_class, predicted_class)
            risk_level = RISK_LEVELS.get(predicted_class, 'Unknown')

            # Log the prediction for debugging
            logger.info(f"Fallback prediction: {predicted_class} ({confidence:.3f}) - norm={embedding_norm:.2f}")

            return {
                'prediction': predicted_class,
                'condition': condition_name,  # Template expects 'condition'
                'condition_name': condition_name,  # Keep for backward compatibility
                'risk_level': risk_level,
                'confidence': confidence,
                'probabilities': prob_dict,
                'embedding_norm': embedding_norm,
                'embedding_mean': embedding_mean,
                'embedding_std': embedding_std,
                'positive_ratio': positive_ratio,
                'method': 'fallback_pattern_analysis'
            }

        except Exception as e:
            logger.error(f"Error in fallback analysis: {e}")
            # Even distribution of probabilities for unknown case
            balanced_probs = {condition: 1.0/7 for condition in SKIN_CONDITIONS.values()}
            return {
                'prediction': 'unknown',
                'condition': 'Unknown Condition',  # Template expects 'condition'
                'condition_name': 'Unknown Condition',  # Keep for backward compatibility
                'risk_level': 'Unknown',
                'confidence': 0.0,
                'probabilities': balanced_probs,
                'method': 'fallback_error',
                'error': str(e)
            }

    def predict(self, image_input) -> Dict[str, Any]:
        """
        Predict skin condition from image input
        Model v4.0: 95.9% peak accuracy, 93.9% mean production on 8,039 samples

        Args:
            image_input: Either image file path (str) or numpy array

        Returns:
            Dictionary containing prediction results with calibrated confidence scores
        """
        try:
            logger.info("🔍 Using Skin Diagnosis Model v4.0 (95.9% accuracy, HAM10000 dataset)")
            logger.info("Starting enhanced skin condition prediction")

            # Validate input
            is_valid, errors = self._validate_input(image_input)
            if not is_valid:
                logger.error(f"Validation failed: {errors}")
                return {
                    'prediction': 'error',
                    'condition': 'Error',
                    'condition_name': 'Error',
                    'risk_level': 'Unknown',
                    'confidence': 0.0,
                    'probabilities': {},
                    'method': 'validation_error',
                    'error': '; '.join(errors)
                }

            # Preprocess image
            image_bytes = self._preprocess(image_input)
            if image_bytes is None:
                raise RuntimeError("Failed to preprocess image")

            # Generate embedding
            embedding = self._process(image_bytes)
            if embedding is None:
                raise RuntimeError("Failed to generate embedding from image")

            embedding_norm = np.linalg.norm(embedding)
            logger.info(f"Generated embedding with shape: {embedding.shape}, norm: {embedding_norm:.2f}")

            # Try optimized classifier first
            if self.optimized_classifier_data is not None:
                result = self._predict_with_classifier(embedding)
            else:
                logger.warning("Optimized classifier not available, using fallback")
                result = self._fallback(embedding)

            # Add embedding norm to the result for debugging and quality assessment
            result['embedding_norm'] = float(embedding_norm)

            # If optimized classifier failed, result will be from fallback
            logger.info(f"Prediction complete: {result['condition_name']} "
                       f"(confidence: {result['confidence']:.3f}, method: {result.get('method', 'unknown')})")

            return result

        except Exception as e:
            logger.error(f"Error in image prediction: {e}")
            return {
                'prediction': 'error',
                'condition': 'Error',  # Template expects 'condition'
                'condition_name': 'Error',  # Keep for backward compatibility
                'risk_level': 'Unknown',
                'confidence': 0.0,
                'probabilities': {},
                'method': 'error',
                'error': str(e)
            }

    def _predict_with_classifier(self, embedding: np.ndarray) -> Dict[str, Any]:
        """
        Predict using the optimized ensemble classifier

        Args:
            embedding: 6144-dimensional embedding from Derm Foundation model

        Returns:
            Dictionary containing prediction results with calibrated confidence
        """
        try:
            # First, check for NaN values in the embedding
            if embedding is None or np.any(np.isnan(embedding)):
                logger.warning("Input embedding contains NaN values, attempting repair")
                if embedding is not None:
                    embedding = np.nan_to_num(embedding, nan=0.0)
                else:
                    raise ValueError("Embedding is None")

            # Extract components
            ensemble_classifier = self.optimized_classifier_data['ensemble_classifier']
            scaler = self.optimized_classifier_data['scaler']
            label_encoder = self.optimized_classifier_data['label_encoder']
            feature_selector = self.optimized_classifier_data.get('feature_selector')
            training_history = self.optimized_classifier_data.get('training_history', {})

            # Engineer enhanced features with additional error checking
            enhanced_features = self._engineer_features(embedding)
            if enhanced_features is None:
                raise ValueError("Failed to engineer features")

            # Double-check for NaN values in enhanced features
            if np.any(np.isnan(enhanced_features)):
                logger.warning("Enhanced features contain NaN values after engineering, replacing with zeros")
                enhanced_features = np.nan_to_num(enhanced_features, nan=0.0)

            # Check shape
            if enhanced_features.shape[0] != 6224:
                logger.warning(f"Feature vector has wrong shape: {enhanced_features.shape}, expected (6224,)")
                # Try to resize if possible
                if len(enhanced_features) > 0:
                    if len(enhanced_features) < 6224:
                        enhanced_features = np.pad(enhanced_features, (0, 6224 - len(enhanced_features)))
                    else:
                        enhanced_features = enhanced_features[:6224]
                else:
                    raise ValueError("Empty feature vector")

            # Transform features (same pipeline as training) with extra error checking
            try:
                X = np.array([enhanced_features], dtype=np.float64)
                # Verify the scaling won't cause NaN or infinite values
                for i in range(X.shape[1]):
                    if np.abs(X[0, i]) > 1e10:
                        logger.warning(f"Extremely large value detected at index {i}: {X[0, i]}, clamping")
                        X[0, i] = np.clip(X[0, i], -1e10, 1e10)

                # Apply scaling
                X_scaled = scaler.transform(X)

                # Check for NaN values after scaling
                if np.any(np.isnan(X_scaled)):
                    logger.warning("NaN values detected after scaling, replacing with zeros")
                    X_scaled = np.nan_to_num(X_scaled, nan=0.0)

                # Apply feature selection if configured
                if feature_selector is not None:
                    X_scaled = feature_selector.transform(X_scaled)

                    # Check again for NaN values after feature selection
                    if np.any(np.isnan(X_scaled)):
                        logger.warning("NaN values detected after feature selection, replacing with zeros")
                        X_scaled = np.nan_to_num(X_scaled, nan=0.0)

                # Make prediction with validated input
                pred_encoded = ensemble_classifier.predict(X_scaled)[0]
                pred_proba = ensemble_classifier.predict_proba(X_scaled)[0]

                # Decode prediction
                prediction = label_encoder.inverse_transform([pred_encoded])[0]
                confidence = float(np.max(pred_proba))

                # Create probabilities dictionary
                probabilities = {}
                class_names = label_encoder.classes_
                for i, class_name in enumerate(class_names):
                    probabilities[class_name] = float(pred_proba[i])

                # Get condition name and risk level
                condition_name = CONDITION_NAMES.get(prediction, prediction)
                risk_level = RISK_LEVELS.get(prediction, 'Unknown')

                # Create initial result
                result = {
                    'prediction': prediction,
                    'condition': condition_name,  # Template expects 'condition'
                    'condition_name': condition_name,  # Keep for backward compatibility
                    'risk_level': risk_level,
                    'confidence': confidence,
                    'probabilities': probabilities,
                    'method': 'CNN_Voting_Ensemble',
                    'model_info': {
                        'training_accuracy': training_history.get('accuracy', 0.724),
                        'training_samples': training_history.get('training_samples', 315),
                        'feature_count': training_history.get('feature_count', 500),
                        'cv_accuracy': training_history.get('cv_mean', 0.638),
                        'timestamp': training_history.get('timestamp', 'unknown')
                    }
                }

                # Apply calibration to reduce bias toward majority class (nv)
                calibrated_result = calibrate_prediction_result(result, 'combined')

                logger.info(f"Optimized prediction: {prediction} ({confidence:.3f} confidence) -> "
                           f"Calibrated: {calibrated_result['prediction']} ({calibrated_result['confidence']:.3f} confidence)")

                # Log final result
                print(f"📊 Skin Diagnosis Model result: {calibrated_result['prediction']} (confidence: {calibrated_result['confidence']:.3f})")

                return calibrated_result

            except Exception as e:
                logger.error(f"Error during prediction pipeline: {e}")
                # Use fallback instead of failing
                return self._fallback(embedding)

        except Exception as e:
            logger.error(f"Error in optimized prediction: {e}")
            return self._fallback(embedding)

    def _engineer_features(self, embedding):
        """
        Enhanced feature engineering on single Derm Foundation embedding
        Creates the same 6224 features used in the optimized model.
        Uses robust methods to handle edge cases and prevent NaN values.
        """
        try:
            # Safeguard against invalid inputs
            if embedding is None or len(embedding) == 0:
                logger.error("Invalid embedding input for feature engineering")
                return None

            # Check for NaN values in input and replace with zeros
            if np.any(np.isnan(embedding)):
                logger.warning("NaN values found in input embedding, replacing with zeros")
                embedding = np.nan_to_num(embedding, nan=0.0)

            # Normalize the embedding to prevent extreme values
            embedding_normalized = np.clip(embedding, -10, 10)

            # Extra safety check for NaN values after clipping
            if np.any(np.isnan(embedding_normalized)):
                logger.warning("NaN values found after normalization, replacing with zeros")
                embedding_normalized = np.nan_to_num(embedding_normalized, nan=0.0)

            # Final safety check - convert entire array to float64 to prevent numeric issues
            embedding = np.asarray(embedding, dtype=np.float64)
            embedding_normalized = np.asarray(embedding_normalized, dtype=np.float64)

            features = []

            # 1. Original embedding features (6144)
            # We use the original (not normalized) embedding here, but ensure no NaN values
            features.extend(np.nan_to_num(embedding, nan=0.0))

            # 2. Basic statistical features (25) - More robust implementation
            std_val = max(np.std(embedding_normalized), 1e-10)  # Avoid division by zero
            mean_val = np.mean(embedding_normalized)

            # Safer calculations with explicit NaN handling
            features.extend([
                float(mean_val),
                float(std_val),
                float(np.var(embedding_normalized)),
                float(np.min(embedding_normalized)),
                float(np.max(embedding_normalized)),
                float(np.median(embedding_normalized)),
                float(np.ptp(embedding_normalized)),  # peak-to-peak
                float(np.percentile(embedding_normalized, 10)),
                float(np.percentile(embedding_normalized, 25)),
                float(np.percentile(embedding_normalized, 75)),
                float(np.percentile(embedding_normalized, 90)),
                float(len(embedding_normalized[embedding_normalized > 0]) / len(embedding_normalized)),  # positive ratio
                float(len(embedding_normalized[embedding_normalized < 0]) / len(embedding_normalized)),  # negative ratio
                float(len(embedding_normalized[np.abs(embedding_normalized) < 0.1]) / len(embedding_normalized)),  # near-zero ratio
                float(np.linalg.norm(embedding_normalized)),  # L2 norm
                float(np.sum(np.abs(embedding_normalized))),  # L1 norm
                float(np.sum(embedding_normalized > mean_val) / len(embedding_normalized)),  # above-mean ratio
                float(np.sum(embedding_normalized < mean_val) / len(embedding_normalized)),  # below-mean ratio
            ])

            # Extra-safe skewness calculation
            try:
                skewness = float(np.sum((embedding_normalized - mean_val)**3) / (len(embedding_normalized) * (std_val**3)))
                skewness = np.nan_to_num(skewness, nan=0.0)
            except:
                skewness = 0.0
            features.append(skewness)

            # Extra-safe kurtosis calculation
            try:
                kurtosis = float(np.sum((embedding_normalized - mean_val)**4) / (len(embedding_normalized) * (std_val**4)))
                kurtosis = np.nan_to_num(kurtosis, nan=0.0)
            except:
                kurtosis = 0.0
            features.append(kurtosis)

            # Additional robust statistics with NaN handling
            above_75th = embedding_normalized[embedding_normalized > np.percentile(embedding_normalized, 75)]
            below_25th = embedding_normalized[embedding_normalized < np.percentile(embedding_normalized, 25)]

            features.extend([
                float(np.mean(np.abs(embedding_normalized - np.median(embedding_normalized)))),  # MAD
                float(np.percentile(embedding_normalized, 95) - np.percentile(embedding_normalized, 5)),  # 95% range
                float(np.percentile(embedding_normalized, 75) - np.percentile(embedding_normalized, 25)),  # IQR
                float(np.mean(above_75th) if len(above_75th) > 0 else 0.0),
                float(np.mean(below_25th) if len(below_25th) > 0 else 0.0)
            ])

            # 3. Segment-based features (28) - divide into 7 segments
            # Use the normalized embedding for more robust segments
            segment_size = len(embedding_normalized) // 7
            for j in range(7):
                start_idx = j * segment_size
                end_idx = start_idx + segment_size if j < 6 else len(embedding_normalized)
                segment = embedding_normalized[start_idx:end_idx]

                if len(segment) > 0:  # Safeguard against empty segments
                    seg_std = max(np.std(segment), 1e-10)  # Avoid zero std
                    features.extend([
                        float(np.mean(segment)),
                        float(seg_std),
                        float(np.min(segment)),
                        float(np.max(segment))
                    ])
                else:
                    features.extend([0.0, 0.0, 0.0, 0.0])

            # 4. Improved frequency domain features (15) with explicit error handling
            try:
                # Check for any problematic values before FFT
                if np.any(np.isnan(embedding_normalized)) or np.any(np.isinf(embedding_normalized)):
                    raise ValueError("NaN or Inf values detected before FFT")

                # Apply window function to reduce spectral leakage
                window = np.hanning(len(embedding_normalized))
                windowed_data = embedding_normalized * window

                # Calculate FFT with proper normalization
                fft_coeffs = np.fft.fft(windowed_data)
                fft_magnitude = np.abs(fft_coeffs[:len(fft_coeffs)//2])  # Use only positive frequencies

                # Check for NaN values in FFT result
                if np.any(np.isnan(fft_magnitude)):
                    raise ValueError("NaN values detected in FFT result")

                # Ensure fft_magnitude is not empty and has expected length
                if len(fft_magnitude) < 1000:
                    # Pad with zeros if too short
                    fft_magnitude = np.pad(fft_magnitude, (0, 1000 - len(fft_magnitude)))

                # Ensure there are no NaN or infinite values
                fft_magnitude = np.nan_to_num(fft_magnitude, nan=0.0, posinf=1e10, neginf=-1e10)

                # Safe frequency features with better error handling
                total_energy = max(np.sum(fft_magnitude), 1e-10)  # Avoid division by zero

                # Dominant frequency with safety check
                dom_freq_idx = np.argmax(fft_magnitude) if not np.all(fft_magnitude == 0) else 0

                low_freq_ratio = np.sum(fft_magnitude[:50]) / total_energy if total_energy > 0 else 0
                mid_freq_ratio = np.sum(fft_magnitude[50:200]) / total_energy if total_energy > 0 else 0
                high_freq_ratio = np.sum(fft_magnitude[200:]) / total_energy if total_energy > 0 else 0

                # Safe spectral centroid
                spectral_centroid = np.sum(np.arange(len(fft_magnitude)) * fft_magnitude) / total_energy if total_energy > 0 else 0

                # Extremely safe zero crossing calculation
                try:
                    zero_crossings = np.sum(np.diff(np.sign(embedding_normalized)) != 0) / max(len(embedding_normalized) - 1, 1)
                except:
                    zero_crossings = 0.0

                # Safe mean absolute difference
                try:
                    mean_abs_diff = np.mean(np.abs(np.diff(embedding_normalized)))
                except:
                    mean_abs_diff = 0.0

                # Safe high frequency content
                high_freq_content = np.sum(fft_magnitude[len(fft_magnitude)//2:]) / total_energy if total_energy > 0 else 0

                # Very safe spectral flatness calculation
                try:
                    # Add small epsilon to avoid log(0)
                    spectral_flatness = np.exp(np.mean(np.log(fft_magnitude + 1e-10))) / (np.mean(fft_magnitude) + 1e-10)
                except:
                    spectral_flatness = 0.0

                features.extend([
                    float(np.mean(fft_magnitude[:100])),
                    float(np.mean(fft_magnitude[100:500])),
                    float(np.mean(fft_magnitude[500:1000])),
                    float(np.std(fft_magnitude)),
                    float(np.max(fft_magnitude)),
                    float(dom_freq_idx),
                    float(low_freq_ratio),
                    float(mid_freq_ratio),
                    float(high_freq_ratio),
                    float(spectral_centroid),
                    float(np.sum(fft_magnitude ** 2)),
                    float(zero_crossings),
                    float(mean_abs_diff),
                    float(high_freq_content),
                    float(spectral_flatness)
                ])
            except Exception as e:
                logger.warning(f"FFT feature extraction failed: {e}")
                features.extend([0.0] * 15)

            # 5. Gradient and texture features (12) - with extra robust error handling
            try:
                # Use smoothed data for more stable gradients
                smoothed = np.convolve(embedding_normalized, np.ones(5)/5, mode='same')

                # Check for NaN values
                if np.any(np.isnan(smoothed)):
                    smoothed = np.nan_to_num(smoothed, nan=0.0)

                gradient = np.gradient(smoothed)

                # Check for NaN values in gradient
                if np.any(np.isnan(gradient)):
                    gradient = np.nan_to_num(gradient, nan=0.0)

                # Ensure gradient has enough elements
                if len(gradient) > 2:
                    # Super-safe gradient statistics
                    grad_pos = float(np.sum(gradient > 0) / len(gradient)) if len(gradient) > 0 else 0.0
                    grad_neg = float(np.sum(gradient < 0) / len(gradient)) if len(gradient) > 0 else 0.0

                    # Calculate autocorrelation very safely
                    try:
                        # Use a more robust correlation method
                        if len(embedding_normalized) > 1:
                            correlation = np.correlate(embedding_normalized[:-1], embedding_normalized[1:], mode='valid')
                            correlation_value = float(correlation[0] / (max(np.std(embedding_normalized[:-1]), 1e-10) *
                                                      max(np.std(embedding_normalized[1:]), 1e-10) *
                                                      len(embedding_normalized[:-1])))
                            if np.isnan(correlation_value):
                                correlation_value = 0.0
                        else:
                            correlation_value = 0.0
                    except:
                        correlation_value = 0.0

                    # Calculate polyfit ultra-safely
                    try:
                        trend = float(np.polyfit(np.arange(len(embedding_normalized)), embedding_normalized, 1)[0])
                        if np.isnan(trend):
                            trend = 0.0
                    except:
                        trend = 0.0

                    # Local extrema with safety
                    try:
                        peaks_ratio = float(np.sum((gradient[:-1] > 0) & (gradient[1:] < 0)) / len(gradient)) if len(gradient) > 1 else 0.0
                        valleys_ratio = float(np.sum((gradient[:-1] < 0) & (gradient[1:] > 0)) / len(gradient)) if len(gradient) > 1 else 0.0
                    except:
                        peaks_ratio = 0.0
                        valleys_ratio = 0.0

                    # Safe curvature approximation
                    try:
                        if len(smoothed) > 2:
                            curvature = float(np.mean(np.abs(smoothed[2:] - 2*smoothed[1:-1] + smoothed[:-2])))
                        else:
                            curvature = 0.0
                    except:
                        curvature = 0.0

                    features.extend([
                        float(np.mean(np.abs(gradient))),
                        float(np.std(gradient)),
                        float(np.max(np.abs(gradient))),
                        grad_pos,
                        grad_neg,
                        float(np.mean(np.abs(np.diff(gradient)))) if len(gradient) > 1 else 0.0,  # second derivative
                        float(np.std(np.diff(gradient))) if len(gradient) > 1 else 0.0,
                        correlation_value,
                        peaks_ratio,
                        valleys_ratio,
                        trend,
                        curvature
                    ])
                else:
                    features.extend([0.0] * 12)
            except Exception as e:
                logger.warning(f"Gradient feature extraction failed: {e}")
                features.extend([0.0] * 12)

            # Make sure we have exactly 6224 features
            feature_count = len(features)
            if feature_count != 6224:
                logger.warning(f"Feature count mismatch: got {feature_count}, expected 6224")
                # Pad with zeros if needed
                if feature_count < 6224:
                    features.extend([0.0] * (6224 - feature_count))
                # Truncate if too many
                if feature_count > 6224:
                    features = features[:6224]

            # Final check for NaN values in the feature vector
            features_array = np.array(features, dtype=np.float64)
            if np.any(np.isnan(features_array)):
                logger.warning(f"NaN values detected in final features, replacing with zeros")
                features_array = np.nan_to_num(features_array, nan=0.0)

            # One more check for infinite values
            if np.any(np.isinf(features_array)):
                logger.warning(f"Infinite values detected in final features, replacing with large values")
                features_array = np.nan_to_num(features_array, posinf=1e10, neginf=-1e10)

            return features_array

        except Exception as e:
            logger.error(f"Error in feature engineering: {e}")
            # Return a safe default feature vector in case of failure
            return np.zeros(6224)

    def get_model_info(self) -> Dict:
        """Get model metadata and information."""
        return MODEL_INFO
