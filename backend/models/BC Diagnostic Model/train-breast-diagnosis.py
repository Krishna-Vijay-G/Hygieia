#!/usr/bin/env python3
"""
Kaggle Breast Cancer Model Training - Optimized for High Accuracy & Recall

Uses feature engineering, SMOTE for balancing, and recall-optimized threshold.
"""

import os
import numpy as np
import pandas as pd
import joblib
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report,
    confusion_matrix, precision_recall_fscore_support, make_scorer, recall_score
)

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False
    print("Warning: imbalanced-learn not installed. Install with: pip install imbalanced-learn")

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import xgboost as xgb

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

HERE = os.path.dirname(__file__)
DATA_PATH = os.path.join(HERE, 'Wisconsin Diagnosis Dataset - UCI.csv')
MODEL_PATH = os.path.join(HERE, 'breast-diagnosis.joblib')

FEATURES = ['mean_radius', 'mean_texture', 'mean_perimeter', 'mean_area', 'mean_smoothness']
TARGET = 'diagnosis'


class StackingEnsemble:
    def __init__(self, bases, meta):
        self.bases = bases
        self.meta = meta
        self.classes_ = bases[0].classes_

    def predict_proba(self, X_scaled):
        base_probs = [b.predict_proba(X_scaled)[:, 1] for b in self.bases]
        stacked = np.column_stack(base_probs)
        meta_probs = self.meta.predict_proba(stacked)[:, 1]
        return np.column_stack([1 - meta_probs, meta_probs])


def engineer_features(X):
    """Create additional engineered features."""
    X = X.copy()
    
    # Ratio features
    X['radius_to_area'] = X['mean_radius'] / (X['mean_area'] + 1)
    X['perimeter_to_area'] = X['mean_perimeter'] / (X['mean_area'] + 1)
    X['texture_density'] = X['mean_texture'] * X['mean_smoothness']
    
    # Polynomial features for key metrics
    X['radius_squared'] = X['mean_radius'] ** 2
    X['area_squared'] = X['mean_area'] ** 2
    
    # Interaction features
    X['radius_x_texture'] = X['mean_radius'] * X['mean_texture']
    X['area_x_smoothness'] = X['mean_area'] * X['mean_smoothness']
    
    return X


def load_and_prepare_data():
    """Load data with feature engineering."""
    df = pd.read_csv(DATA_PATH)
    logger.info(f"Loaded {len(df)} samples")
    
    X = df[FEATURES]
    y = df[TARGET]
    
    # Engineer features
    X_eng = engineer_features(X)
    logger.info(f"Engineered features: {X_eng.shape[1]} total features")
    
    # Split: 70% train, 15% val, 15% test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_eng, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    logger.info(f"Train class dist: Malignant={( y_train==0).sum()}, Benign={(y_train==1).sum()}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, X_eng.columns.tolist()


def apply_smote(X_train, y_train):
    """Apply SMOTE to balance classes."""
    if not HAS_SMOTE:
        logger.warning("SMOTE not available, using original data")
        return X_train, y_train
    
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    logger.info(f"After SMOTE: {len(X_resampled)} samples (was {len(X_train)})")
    logger.info(f"Class dist: Malignant={(y_resampled==0).sum()}, Benign={(y_resampled==1).sum()}")
    
    return X_resampled, y_resampled


def train_optimized_model(X_train, y_train, X_val, y_val):
    """Train optimized ensemble model."""
    logger.info("Training optimized ensemble model...")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Apply SMOTE for balanced training
    X_train_balanced, y_train_balanced = apply_smote(X_train_scaled, y_train)
    
    # Train multiple strong base models
    logger.info("Hyperparameter search: Random Forest (small grid)...")
    rf_base = RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1)
    rf_grid = {
        'n_estimators': [500, 1000],
        'max_depth': [None, 10],
        'min_samples_split': [2, 4],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 'log2']
    }
    rf_search = GridSearchCV(
        rf_base, rf_grid, scoring='f1', cv=3, n_jobs=-1, verbose=0
    )
    rf_search.fit(X_train_balanced, y_train_balanced)
    rf = rf_search.best_estimator_
    logger.info(f"RF best params: {rf_search.best_params_}")
    
    logger.info("Hyperparameter search: XGBoost (small grid)...")
    xgb_base = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',
        random_state=42,
        scale_pos_weight=1,
        use_label_encoder=False
    )
    xgb_grid = {
        'n_estimators': [500, 800],
        'max_depth': [5, 7],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'min_child_weight': [1, 2],
        'gamma': [0, 0.1]
    }
    xgb_search = GridSearchCV(
        xgb_base, xgb_grid, scoring='f1', cv=3, n_jobs=-1, verbose=0
    )
    xgb_search.fit(X_train_balanced, y_train_balanced)
    xgb_model = xgb_search.best_estimator_
    logger.info(f"XGB best params: {xgb_search.best_params_}")
    
    logger.info("Training Gradient Boosting...")
    gb = GradientBoostingClassifier(
        n_estimators=500,
        learning_rate=0.1,
        max_depth=7,
        subsample=0.8,
        min_samples_split=2,
        random_state=42
    )
    gb.fit(X_train_balanced, y_train_balanced)
    
    logger.info("Training SVM...")
    svm = SVC(
        kernel='rbf',
        C=10.0,
        gamma='scale',
        probability=True,
        class_weight='balanced',
        random_state=42
    )
    svm.fit(X_train_balanced, y_train_balanced)
    
    # Stacking meta-model (level 2)
    logger.info("Creating stacking meta-model (logistic regression)...")
    # Obtain out-of-fold like predictions from validation for meta features
    val_features = np.column_stack([
        rf.predict_proba(X_val_scaled)[:,1],
        xgb_model.predict_proba(X_val_scaled)[:,1],
        gb.predict_proba(X_val_scaled)[:,1],
        svm.predict_proba(X_val_scaled)[:,1]
    ])
    train_features = np.column_stack([
        rf.predict_proba(X_train_balanced)[:,1],
        xgb_model.predict_proba(X_train_balanced)[:,1],
        gb.predict_proba(X_train_balanced)[:,1],
        svm.predict_proba(X_train_balanced)[:,1]
    ])
    meta_clf = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    meta_clf.fit(train_features, y_train_balanced)

    ensemble = StackingEnsemble([rf, xgb_model, gb, svm], meta_clf)
    
    # Balanced threshold sweep: maximize minimum of (malignant recall, benign recall)
    y_val_proba = ensemble.predict_proba(X_val_scaled)[:, 1]
    logger.info("Sweeping thresholds to maximize balanced recall (both classes)...")
    best_threshold = 0.5
    best_balanced = -1
    thresholds = np.linspace(0.2, 0.8, 181)
    for thr in thresholds:
        y_val_pred = (y_val_proba >= thr).astype(int)
        prec, rec, f1, sup = precision_recall_fscore_support(
            y_val, y_val_pred, average=None, zero_division=0
        )
        # rec[0] malignant, rec[1] benign
        balanced_recall = min(rec[0], rec[1])
        if balanced_recall > best_balanced:
            best_balanced = balanced_recall
            best_threshold = thr
    logger.info(f"Selected threshold: {best_threshold:.3f} (Balanced recall min={best_balanced:.3f})")
    
    return ensemble, scaler, best_threshold


def evaluate_with_threshold(model, X, y, scaler, threshold=0.5):
    """Evaluate model with custom threshold."""
    X_scaled = scaler.transform(X)
    y_proba = model.predict_proba(X_scaled)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    
    acc = accuracy_score(y, y_pred)
    auc = roc_auc_score(y, y_proba)
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y, y_pred, average=None, zero_division=0
    )
    
    return {
        'accuracy': acc,
        'auc': auc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'support': support,
        'y_pred': y_pred,
        'y_proba': y_proba
    }


def main():
    logger.info("=" * 80)
    logger.info("KAGGLE BREAST CANCER - OPTIMIZED TRAINING")
    logger.info("=" * 80)
    
    # Load and engineer features
    X_train, X_val, X_test, y_train, y_val, y_test, feature_names = load_and_prepare_data()
    
    # Train optimized model
    model, scaler, best_threshold = train_optimized_model(X_train, y_train, X_val, y_val)
    
    # Evaluate
    logger.info("\n" + "=" * 80)
    logger.info("VALIDATION RESULTS")
    logger.info("=" * 80)
    val_metrics = evaluate_with_threshold(model, X_val, y_val, scaler, best_threshold)
    logger.info(f"Accuracy: {val_metrics['accuracy']:.4f}")
    logger.info(f"AUC: {val_metrics['auc']:.4f}")
    logger.info(f"Malignant - Precision: {val_metrics['precision'][0]:.4f}, Recall: {val_metrics['recall'][0]:.4f}, F1: {val_metrics['f1'][0]:.4f}")
    logger.info(f"Benign - Precision: {val_metrics['precision'][1]:.4f}, Recall: {val_metrics['recall'][1]:.4f}, F1: {val_metrics['f1'][1]:.4f}")
    
    logger.info("\n" + "=" * 80)
    logger.info("TEST RESULTS")
    logger.info("=" * 80)
    test_metrics = evaluate_with_threshold(model, X_test, y_test, scaler, best_threshold)
    logger.info(f"Accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"AUC: {test_metrics['auc']:.4f}")
    logger.info(f"Malignant - Precision: {test_metrics['precision'][0]:.4f}, Recall: {test_metrics['recall'][0]:.4f}, F1: {test_metrics['f1'][0]:.4f}")
    logger.info(f"Benign - Precision: {test_metrics['precision'][1]:.4f}, Recall: {test_metrics['recall'][1]:.4f}, F1: {test_metrics['f1'][1]:.4f}")
    
    logger.info("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, test_metrics['y_pred'])
    logger.info(f"\n{cm}")
    logger.info(f"\nMalignant Detection: {cm[0,0]}/{cm[0,0]+cm[0,1]} ({cm[0,0]/(cm[0,0]+cm[0,1])*100:.1f}%)")
    logger.info(f"Benign Detection: {cm[1,1]}/{cm[1,0]+cm[1,1]} ({cm[1,1]/(cm[1,0]+cm[1,1])*100:.1f}%)")
    
    # Save model
    model_bundle = {
        'model': model,
        'scaler': scaler,
        'model_type': 'ensemble_optimized',
        'model_name': 'Optimized Ensemble (RF+XGB+GB+SVM)',
        'feature_names': feature_names,
        'best_threshold': best_threshold,
        'metrics': {
            'test_accuracy': test_metrics['accuracy'],
            'test_auc': test_metrics['auc'],
            'test_f1': np.mean(test_metrics['f1']),
            'test_malignant_recall': test_metrics['recall'][0],
            'test_benign_recall': test_metrics['recall'][1]
        }
    }
    
    joblib.dump(model_bundle, MODEL_PATH)
    logger.info(f"\nSaved optimized model: {MODEL_PATH}")
    
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
