"""
Retrain ultra-optimized model with FN-focused optimization.
Priority: Minimize FN (missed cancers) while keeping FP reasonable.
Uses class weights heavily favoring cancer detection + custom threshold optimization.
"""

import os
import time
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import joblib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

COLUMN_MAP = {
    'age_group_5_years': 'age',
    'race_eth': 'race',
    'first_degree_hx': 'family_hx',
    'age_menarche': 'agefirst',
    'age_first_birth': 'nrelbc',
    'BIRADS_breast_density': 'density',
    'current_hrt': 'hrt',
    'bmi_group': 'bmi',
    'biophx': 'brstproc'
}

BASE_FEATURES = [
    'age', 'density', 'race', 'bmi', 'agefirst', 'nrelbc', 'brstproc', 'hrt', 'family_hx', 'menopaus'
]


def engineer_ultra_features(df: pd.DataFrame):
    """Ultra comprehensive feature engineering for maximum predictive power"""
    X = df[BASE_FEATURES].copy()
    
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
    
    feature_names = list(X.columns)
    return X.values, feature_names


def load_unique_dataset(csv_path: str):
    logger.info('[1/7] LOADING UNIQUE DATASET...')
    df = pd.read_csv(csv_path)
    logger.info(f"  ✓ Loaded {len(df):,} unique aggregated rows")
    
    y = df['breast_cancer_history'].values
    sample_weight = df['count'].values.astype(float)
    
    # BOOST CANCER SAMPLES - Give 3x weight to cancer cases
    cancer_boost = np.where(y == 1, 3.0, 1.0)
    sample_weight = sample_weight * cancer_boost
    
    X, feature_names = engineer_ultra_features(df)

    logger.info(f"  ✓ Features engineered: {len(feature_names)} total")
    logger.info(f"  ✓ Class distribution (unweighted) - No Cancer: {(y==0).sum():,}, Cancer: {(y==1).sum():,}")
    logger.info(f"  ✓ Total sample weight (with 3x cancer boost): {sample_weight.sum():,}")
    logger.info('[1/7] DATA PREP COMPLETE\n')
    return X, y, sample_weight, feature_names


def train_fn_focused_ensemble(X_train, y_train, X_val, y_val, w_train, w_val):
    logger.info('[3/7] TRAINING FN-FOCUSED ENSEMBLE (6 MODELS)...')
    
    scaler = RobustScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    n_no = (y_train==0).sum()
    n_ca = (y_train==1).sum()
    # Aggressive scale_pos_weight to prioritize cancer detection
    scale_pos_weight = (n_no / max(1, n_ca)) * 1.5  # 1.5x boost
    logger.info(f"  ✓ scale_pos_weight={scale_pos_weight:.2f} (boosted for FN reduction)")

    # All models tuned for high recall (cancer detection)
    params_configs = [
        # Model A: Very deep, conservative, high recall focus
        dict(n_estimators=1600, max_depth=12, learning_rate=0.02, subsample=0.75,
             colsample_bytree=0.75, min_child_weight=3, gamma=3, reg_alpha=0.4, reg_lambda=3,
             objective='binary:logistic', eval_metric='logloss', tree_method='hist',
             scale_pos_weight=scale_pos_weight, random_state=42, n_jobs=-1),
        
        # Model B: Deep, moderate learning, recall-optimized
        dict(n_estimators=1400, max_depth=11, learning_rate=0.025, subsample=0.78,
             colsample_bytree=0.78, min_child_weight=4, gamma=2.5, reg_alpha=0.35, reg_lambda=2.5,
             objective='binary:logistic', eval_metric='logloss', tree_method='hist',
             scale_pos_weight=scale_pos_weight, random_state=137, n_jobs=-1),
        
        # Model C: Moderate depth, faster learning
        dict(n_estimators=1200, max_depth=9, learning_rate=0.04, subsample=0.82,
             colsample_bytree=0.82, min_child_weight=5, gamma=2, reg_alpha=0.3, reg_lambda=2,
             objective='binary:logistic', eval_metric='logloss', tree_method='hist',
             scale_pos_weight=scale_pos_weight, random_state=256, n_jobs=-1),
        
        # Model D: Balanced, high estimators
        dict(n_estimators=1800, max_depth=8, learning_rate=0.03, subsample=0.85,
             colsample_bytree=0.85, min_child_weight=6, gamma=1.5, reg_alpha=0.25, reg_lambda=1.5,
             objective='binary:logistic', eval_metric='logloss', tree_method='hist',
             scale_pos_weight=scale_pos_weight, random_state=365, n_jobs=-1),
        
        # Model E: Shallow, aggressive learning
        dict(n_estimators=2000, max_depth=7, learning_rate=0.05, subsample=0.88,
             colsample_bytree=0.88, min_child_weight=7, gamma=1, reg_alpha=0.2, reg_lambda=1,
             objective='binary:logistic', eval_metric='logloss', tree_method='hist',
             scale_pos_weight=scale_pos_weight, random_state=512, n_jobs=-1),
        
        # Model F: Ultra-deep for complex patterns
        dict(n_estimators=1300, max_depth=13, learning_rate=0.018, subsample=0.72,
             colsample_bytree=0.72, min_child_weight=3, gamma=4, reg_alpha=0.5, reg_lambda=4,
             objective='binary:logistic', eval_metric='logloss', tree_method='hist',
             scale_pos_weight=scale_pos_weight, random_state=789, n_jobs=-1),
    ]

    models = []
    for i, params in enumerate(params_configs, 1):
        logger.info(f'  → Training Model {chr(64+i)} (n_estimators={params["n_estimators"]}, max_depth={params["max_depth"]})')
        model = xgb.XGBClassifier(**params)
        model.fit(X_train_s, y_train, sample_weight=w_train, eval_set=[(X_val_s, y_val)], verbose=False)
        models.append(model)
        logger.info(f'  ✓ Model {chr(64+i)} complete')

    logger.info('  → Calibrating ensemble (weighted toward recall)')
    probs = [m.predict_proba(X_val_s)[:,1] for m in models]
    meta = np.vstack(probs).T
    
    # Lighter regularization to allow aggressive predictions
    calibrator = LogisticRegression(max_iter=3000, C=0.8, penalty='l2')
    calibrator.fit(meta, y_val, sample_weight=w_val)

    logger.info('[3/7] FN-FOCUSED ENSEMBLE TRAINING COMPLETE\n')
    return models, scaler, calibrator


def evaluate(models, scaler, calibrator, X, y, w, threshold, label):
    Xs = scaler.transform(X)
    probs = [m.predict_proba(Xs)[:,1] for m in models]
    meta = np.vstack(probs).T
    p = calibrator.predict_proba(meta)[:,1]
    y_pred = (p >= threshold).astype(int)

    acc = accuracy_score(y, y_pred)
    auc = roc_auc_score(y, p)
    prec, rec, f1, _ = precision_recall_fscore_support(y, y_pred, average=None, zero_division=0)
    cm = confusion_matrix(y, y_pred)
    
    logger.info('='*80)
    logger.info(f'{label.upper()} RESULTS (threshold={threshold:.4f})')
    logger.info('='*80)
    logger.info(f'Accuracy: {acc*100:.2f}%  AUC: {auc:.4f}')
    if len(prec)==2:
        logger.info(f'No Cancer - Precision: {prec[0]*100:.2f}% Recall: {rec[0]*100:.2f}% F1: {f1[0]:.4f}')
        logger.info(f'Cancer    - Precision: {prec[1]*100:.2f}% Recall: {rec[1]*100:.2f}% F1: {f1[1]:.4f}')
    logger.info(f'Confusion Matrix:\n{cm}')
    if cm.size==4:
        tn, fp, fn, tp = cm.ravel()
        logger.info(f'TN={tn:,}  FP={fp:,}  FN={fn:,}  TP={tp:,}')
        logger.info(f'Total Errors (FP+FN): {fp+fn:,}')
    logger.info('='*80 + '\n')
    return {'accuracy':acc,'auc':auc,'cm':cm,'prec':prec,'rec':rec,'f1':f1}


def optimize_threshold_fn_focused(models, scaler, calibrator, X_val, y_val, w_val):
    """
    Optimize threshold to minimize FN primarily, then FP.
    Score = FN*3 + FP (heavily penalize false negatives)
    """
    logger.info('[4/7] FN-FOCUSED THRESHOLD OPTIMIZATION...')
    Xs = scaler.transform(X_val)
    probs = [m.predict_proba(Xs)[:,1] for m in models]
    meta = np.vstack(probs).T
    p = calibrator.predict_proba(meta)[:,1]

    thresholds = np.linspace(0.1, 0.90, 800)
    candidates = []
    
    for t in thresholds:
        y_pred = (p >= t).astype(int)
        acc = accuracy_score(y_val, y_pred)
        prec, rec, _, _ = precision_recall_fscore_support(y_val, y_pred, average=None, zero_division=0)
        
        if len(prec) == 2:
            cm = confusion_matrix(y_val, y_pred)
            tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
            
            # Score: heavily penalize FN (3x weight), then FP
            error_score = fn * 3 + fp
            
            candidates.append({
                'threshold': t,
                'accuracy': acc,
                'avg_precision': np.mean(prec),
                'avg_recall': np.mean(rec),
                'cancer_recall': rec[1],
                'cancer_precision': prec[1],
                'fp': fp,
                'fn': fn,
                'tp': tp,
                'tn': tn,
                'error_score': error_score
            })
    
    # Sort by error_score (lower is better)
    candidates.sort(key=lambda x: x['error_score'])
    
    logger.info('\n  Top 10 candidates (minimizing FN*3 + FP):')
    logger.info(f"  {'Rank':<6}{'Threshold':<12}{'Acc%':<8}{'FN':<8}{'FP':<8}{'Cancer Rec%':<12}{'Score':<10}")
    logger.info('  ' + '-'*70)
    
    for i, cand in enumerate(candidates[:10]):
        logger.info(f"  {i+1:<6}{cand['threshold']:<12.4f}{cand['accuracy']*100:<8.2f}"
                   f"{cand['fn']:<8}{cand['fp']:<8}{cand['cancer_recall']*100:<12.2f}{cand['error_score']:<10.0f}")
    
    best = candidates[0]
    
    logger.info(f"\n  ✓ Selected threshold: {best['threshold']:.4f}")
    logger.info(f"  ✓ Error score (FN*3+FP): {best['error_score']:.0f}")
    logger.info(f"  ✓ Accuracy: {best['accuracy']*100:.2f}%")
    logger.info(f"  ✓ Avg Precision: {best['avg_precision']*100:.2f}%")
    logger.info(f"  ✓ Avg Recall: {best['avg_recall']*100:.2f}%")
    logger.info(f"  ✓ Cancer Precision: {best['cancer_precision']*100:.2f}%")
    logger.info(f"  ✓ Cancer Recall: {best['cancer_recall']*100:.2f}%")
    logger.info(f"  ✓ FP={best['fp']:,}  FN={best['fn']:,}  (Total: {best['fp']+best['fn']:,})")
    logger.info('[4/7] FN-FOCUSED THRESHOLD OPTIMIZATION COMPLETE\n')
    return best['threshold']


def main():
    start = time.time()
    logger.info('='*80)
    logger.info('FN-FOCUSED OPTIMIZED MODEL')
    logger.info('Priority: Minimize False Negatives (missed cancers), then False Positives')
    logger.info('='*80+'\n')
    #fix the path according to your project structure
    X, y, sample_weight, feature_names = load_unique_dataset('BCSC Prediction Factors Dataset - BCSC.csv')

    logger.info('[2/7] SPLITTING DATA...')
    X_temp, X_test, y_temp, y_test, w_temp, w_test = train_test_split(
        X, y, sample_weight, test_size=0.15, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(
        X_temp, y_temp, w_temp, test_size=0.176, random_state=42, stratify=y_temp
    )
    logger.info(f'  ✓ Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}')
    logger.info('[2/7] SPLIT COMPLETE\n')

    models, scaler, calibrator = train_fn_focused_ensemble(X_train, y_train, X_val, y_val, w_train, w_val)

    threshold = optimize_threshold_fn_focused(models, scaler, calibrator, X_val, y_val, w_val)

    logger.info('[5/7] VALIDATION EVALUATION...')
    val_results = evaluate(models, scaler, calibrator, X_val, y_val, w_val, threshold, 'Validation')
    
    logger.info('[6/7] TEST EVALUATION...')
    test_results = evaluate(models, scaler, calibrator, X_test, y_test, w_test, threshold, 'Test')

    logger.info('[7/7] SAVING MODEL...')
    bundle = {
        'models': models,
        'scaler': scaler,
        'calibrator': calibrator,
        'threshold': threshold,
        'feature_names': feature_names,
        'training_info': {
            'unique_rows': len(X),
            'num_features': len(feature_names),
            'num_models': len(models),
            'elapsed_sec': time.time() - start,
            'approach': '6-model FN-focused ensemble + 3x cancer weighting + FN*3+FP optimization',
            'val_accuracy': float(val_results['accuracy']),
            'val_auc': float(val_results['auc']),
            'test_accuracy': float(test_results['accuracy']),
            'test_auc': float(test_results['auc'])
        }
    }
    path = 'breast-prediction.joblib'
    joblib.dump(bundle, path)
    size_mb = os.path.getsize(path) / (1024 * 1024)
    logger.info(f'  ✓ Saved {path} ({size_mb:.2f} MB)')
    logger.info('\n' + '='*80)
    logger.info('✅ FN-FOCUSED MODEL TRAINING COMPLETE')
    logger.info('='*80)

if __name__ == '__main__':
    main()
