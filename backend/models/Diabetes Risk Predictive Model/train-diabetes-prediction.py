#!/usr/bin/env python3
"""
UCI Diabetes Model Training Script

Trains a LightGBM model on the UCI Diabetes dataset
to achieve 95%+ accuracy.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from datetime import datetime
import joblib
import warnings
warnings.filterwarnings('ignore')

# LightGBM import
try:
    from lightgbm import LGBMClassifier
except ImportError:
    print("❌ LightGBM not installed. Please install with: pip install lightgbm")
    sys.exit(1)

# Scikit-learn imports
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
#fix the paths according to your project structure
DIABETES_DIR = os.path.join(os.path.dirname(__file__), 'Diabetes_Model')
DATA_PATH = os.path.join(DIABETES_DIR, 'Early Stage Diabetes Risk Prediction - UCI.csv')
MODEL_OUTPUT_PATH = os.path.join(DIABETES_DIR, 'diabetes-prediction.joblib')

# LightGBM Configuration for 95%+ accuracy
LGBM_CONFIG = {
    'test_split_ratio': 0.2,
    'cv_folds': 5,
    'random_state': 42,
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.9,
    'bagging_freq': 5,
    'verbose': -1,
    'n_estimators': 200,
    'max_depth': 6,
    'min_child_samples': 5,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'scale_pos_weight': 1.6,  # 320/200 = 1.6
}

def load_early_diabetes_data():
    """Load and preprocess early diabetes dataset"""
    try:
        if not os.path.exists(DATA_PATH):
            raise FileNotFoundError(f"Data file not found: {DATA_PATH}\nRun download_early_diabetes.py first!")

        logger.info(f"Loading data from: {DATA_PATH}")
        df = pd.read_csv(DATA_PATH)
        logger.info(f"Loaded {len(df)} samples")

        # Display feature info
        logger.info(f"Features: {list(df.columns)}")
        
        # Check target
        if 'Outcome' in df.columns:
            target_col = 'Outcome'
        elif 'class' in df.columns:
            # Convert class to binary
            df['Outcome'] = (df['class'] == 'Positive').astype(int)
            target_col = 'Outcome'
        else:
            raise ValueError("No target column found!")
        
        logger.info(f"Target distribution:\n{df[target_col].value_counts()}")

        return df

    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def preprocess_data(df):
    """Preprocess the early diabetes data"""
    logger.info("Preprocessing data...")
    
    # Separate features and target
    target_col = 'Outcome'
    
    # Remove 'class' if it exists (we use 'Outcome' instead)
    if 'class' in df.columns:
        df = df.drop('class', axis=1)
    
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    # Encode categorical variables
    label_encoders = {}
    for col in X.columns:
        if X[col].dtype == 'object':
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            label_encoders[col] = le
            logger.info(f"  Encoded {col}: {len(le.classes_)} classes")
    
    logger.info(f"Feature matrix shape: {X.shape}")
    logger.info(f"Target array shape: {y.shape}")
    logger.info(f"Feature names: {list(X.columns)}")
    
    return X, y, label_encoders

def train_lgbm_model(X, y):
    """Train LightGBM model"""
    logger.info("Training LightGBM model for 95%+ accuracy")
    
    start_time = datetime.now()
    
    # Create LightGBM classifier
    lgbm = LGBMClassifier(
        objective=LGBM_CONFIG['objective'],
        metric=LGBM_CONFIG['metric'],
        boosting_type=LGBM_CONFIG['boosting_type'],
        num_leaves=LGBM_CONFIG['num_leaves'],
        learning_rate=LGBM_CONFIG['learning_rate'],
        feature_fraction=LGBM_CONFIG['feature_fraction'],
        bagging_fraction=LGBM_CONFIG['bagging_fraction'],
        bagging_freq=LGBM_CONFIG['bagging_freq'],
        verbose=LGBM_CONFIG['verbose'],
        n_estimators=LGBM_CONFIG['n_estimators'],
        max_depth=LGBM_CONFIG['max_depth'],
        min_child_samples=LGBM_CONFIG['min_child_samples'],
        reg_alpha=LGBM_CONFIG['reg_alpha'],
        reg_lambda=LGBM_CONFIG['reg_lambda'],
        scale_pos_weight=LGBM_CONFIG['scale_pos_weight'],
        random_state=LGBM_CONFIG['random_state'],
        n_jobs=-1
    )
    
    # Cross-validation
    cv = StratifiedKFold(
        n_splits=LGBM_CONFIG['cv_folds'],
        shuffle=True,
        random_state=LGBM_CONFIG['random_state']
    )
    
    logger.info("Performing 5-fold cross-validation...")
    cv_scores = cross_val_score(lgbm, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    
    logger.info(f"Cross-validation scores: {cv_scores}")
    logger.info(f"CV Mean Accuracy: {np.mean(cv_scores):.3f} ± {np.std(cv_scores):.3f}")
    
    # Train on full dataset
    logger.info("Training final model on full dataset...")
    lgbm.fit(X, y)
    
    # Get training predictions
    y_pred = lgbm.predict(X)
    y_proba = lgbm.predict_proba(X)[:, 1]
    
    training_accuracy = accuracy_score(y, y_pred)
    training_auc = roc_auc_score(y, y_proba)
    
    # Calculate detailed metrics
    class_report = classification_report(y, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y, y_pred)
    
    training_time = (datetime.now() - start_time).total_seconds()
    
    # Prepare training history
    training_history = {
        'cv_scores': cv_scores.tolist(),
        'cv_mean': float(np.mean(cv_scores)),
        'cv_std': float(np.std(cv_scores)),
        'training_samples': len(X),
        'feature_count': X.shape[1],
        'training_time': training_time,
        'timestamp': datetime.now().isoformat(),
        'accuracy': training_accuracy,
        'auc_roc': training_auc,
        'class_performance': class_report,
        'confusion_matrix': conf_matrix.tolist(),
        'model_type': 'lightgbm_early_diabetes',
        'lgbm_params': LGBM_CONFIG
    }
    
    logger.info(f"Training Accuracy: {training_accuracy:.3f}")
    logger.info(f"Training AUC-ROC: {training_auc:.3f}")
    logger.info(f"Training Time: {training_time:.1f} seconds")
    
    return lgbm, training_history

def save_model(model, training_history, feature_names, label_encoders):
    """Save the trained model"""
    logger.info(f"Saving model to: {MODEL_OUTPUT_PATH}")
    
    os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)
    
    model_data = {
        'model': model,
        'training_history': training_history,
        'feature_names': feature_names,
        'label_encoders': label_encoders,
        'config': LGBM_CONFIG,
        'model_type': 'lightgbm_early_diabetes'
    }
    
    joblib.dump(model_data, MODEL_OUTPUT_PATH, compress=3)
    logger.info(f"Model saved successfully!")
    
    # Display summary
    logger.info("=" * 60)
    logger.info("UCI DIABETES TRAINING SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Training Accuracy: {training_history['accuracy']:.3f}")
    logger.info(f"Training AUC-ROC: {training_history['auc_roc']:.3f}")
    logger.info(f"CV Accuracy: {training_history['cv_mean']:.3f} ± {training_history['cv_std']:.3f}")
    logger.info(f"Training Samples: {training_history['training_samples']}")
    logger.info(f"Features: {training_history['feature_count']}")
    logger.info(f"Training Time: {training_history['training_time']:.1f} seconds")
    logger.info("=" * 60)

def main():
    """Main training function"""
    logger.info("=" * 60)
    logger.info("UCI DIABETES MODEL TRAINING")
    logger.info("=" * 60)
    logger.info(f"Configuration: {LGBM_CONFIG}")
    
    try:
        # Load data
        df = load_early_diabetes_data()
        
        # Preprocess
        X, y, label_encoders = preprocess_data(df)
        
        # Create train/test split for evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=LGBM_CONFIG['test_split_ratio'],
            random_state=LGBM_CONFIG['random_state'],
            stratify=y
        )
        
        logger.info(f"Train set: {len(X_train)} samples")
        logger.info(f"Test set: {len(X_test)} samples")
        
        # Train model
        model, training_history = train_lgbm_model(X_train, y_train)
        
        # Evaluate on test set
        logger.info("\nEvaluating on held-out test set...")
        y_test_pred = model.predict(X_test)
        y_test_proba = model.predict_proba(X_test)[:, 1]
        
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_auc = roc_auc_score(y_test, y_test_proba)
        
        logger.info(f"Test Accuracy: {test_accuracy:.3f}")
        logger.info(f"Test AUC-ROC: {test_auc:.3f}")
        
        # Display detailed metrics
        logger.info("\nTest Set Classification Report:")
        print(classification_report(y_test, y_test_pred, target_names=['Negative', 'Positive']))
        
        # Add test results to history
        training_history['test_accuracy'] = test_accuracy
        training_history['test_auc'] = test_auc
        
        # Save model
        save_model(model, training_history, list(X.columns), label_encoders)
        
        # Final summary
        logger.info("\n" + "=" * 60)
        if test_accuracy >= 0.95:
            logger.info("🎉 SUCCESS! ACHIEVED 95%+ ACCURACY!")
        elif test_accuracy >= 0.90:
            logger.info("✅ EXCELLENT! ACHIEVED 90%+ ACCURACY!")
        else:
            logger.info("✅ TRAINING COMPLETED")
        logger.info("=" * 60)
        logger.info(f"Final Test Accuracy: {test_accuracy:.1%}")
        logger.info(f"Final Test AUC-ROC: {test_auc:.3f}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()
