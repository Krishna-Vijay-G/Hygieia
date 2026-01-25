#!/usr/bin/env python3
"""
Breast Cancer Risk Model Benchmark Script
Evaluates model accuracy on test data with comprehensive metrics.
"""

import os
import sys
import logging
import argparse
import pandas as pd
import numpy as np
import time
import joblib

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

try:
    import colorama
    from colorama import Fore, Style
    colorama.init()
    HAS_COLOR = True
except ImportError:
    HAS_COLOR = False
    class Fore:
        WHITE = RED = GREEN = CYAN = BLUE = YELLOW = ''
    class Style:
        NORMAL = BRIGHT = RESET_ALL = ''

from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                            roc_auc_score, roc_curve, precision_recall_curve,
                            precision_recall_fscore_support)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Initialize colorama for colored output
colorama.init()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Setup paths
MODEL_CONTROLLERS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'controllers'))
if MODEL_CONTROLLERS_PATH not in sys.path:
    sys.path.insert(0, MODEL_CONTROLLERS_PATH)

# Dynamic import
import importlib.util

# logging
import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# constants
CONFIG = {
    'use_held_out_set': True,
    'random_seed': 42,
    'test_samples': 100
}

BREAST_RISK_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BREAST_RISK_DIR, 'BCSC Prediction Factors Dataset - BCSC.csv')

class BreastRiskBenchmarker:
    """
    Breast Cancer Risk Model Benchmarking Class
    Evaluates breast cancer risk prediction model performance using BreastRiskIntegration
    """

    def __init__(self, config=None):
        """Initialize the benchmarker"""
        self.config = config or CONFIG.copy()
        self.breast_risk_model_class = None
        self.predictor = None

        # Load the breast risk model class
        self._load_model_class()

    def _load_model_class(self):
        """Load the breast risk model integration class"""
        try:
            module_file = os.path.join(MODEL_CONTROLLERS_PATH, 'breast_prediction.py')
            spec = importlib.util.spec_from_file_location('breast_prediction', module_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.breast_risk_model_class = getattr(module, 'BreastRiskIntegration', None)
            if self.breast_risk_model_class:
                logger.info("Successfully loaded BreastRiskIntegration class")
            else:
                logger.error("BreastRiskIntegration class not found")
        except Exception as e:
            logger.error(f"Failed to load breast risk model class: {e}")
            self.breast_risk_model_class = None

    def _validate_input(self, data):
        """Validate input data format and required fields (minimal validation)"""
        if not isinstance(data, dict):
            raise ValueError("Input data must be a dictionary")

        required_fields = [
            'age', 'race', 'density', 'family_hx', 'bmi', 'agefirst', 'nrelbc',
            'brstproc', 'menopaus', 'hrt'
        ]

        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")

        # Basic type validation only - let controller handle range validation
        for field in required_fields:
            if not isinstance(data[field], (int, float)):
                raise ValueError(f"{field} must be a number")

        return True

    def _load_data(self):
        """Load BCSC unique records dataset for evaluation."""
        data_path = os.path.join(os.path.dirname(__file__), 'BCSC Prediction Factors Dataset - BCSC.csv')

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")

        df = pd.read_csv(data_path)

        # Extract features and target
        y = df['breast_cancer_history'].values
        sample_weight = df['count'].values.astype(float)

        # Engineer features same as training
        X_eng = self._engineer_features(df)

        # Create test split (same split as training for consistency)
        X_temp, X_test, y_temp, y_test, w_temp, w_test = train_test_split(
            X_eng.values, y, sample_weight, test_size=0.15, random_state=42, stratify=y
        )

        print(f"INFO: Using test split with {len(X_test)} samples")
        print(f"INFO: Test class distribution - No Cancer: {(y_test==0).sum():,}, Cancer: {(y_test==1).sum():,}")
        print(f"INFO: Test sample weights sum: {w_test.sum():,.0f}")

        return X_test, y_test, w_test

    def _engineer_features(self, df):
        """Create engineered features matching the trained model."""
        BASE_FEATURES = ['age', 'race', 'density', 'family_hx', 'bmi', 'agefirst', 'nrelbc', 'brstproc', 'menopaus', 'hrt']
        X = df[BASE_FEATURES].copy()

        # Key interactions
        X['age_density'] = X['age'] * X['density']
        X['age_bmi'] = X['age'] * X['bmi']
        X['age_family_hx'] = X['age'] * X['family_hx']
        X['age_hrt'] = X['age'] * X['hrt']
        X['age_brstproc'] = X['age'] * X['brstproc']
        X['age_menopaus'] = X['age'] * X['menopaus']
        X['density_bmi'] = X['density'] * X['bmi']
        X['density_hrt'] = X['density'] * X['hrt']
        X['density_family_hx'] = X['density'] * X['family_hx']
        X['density_brstproc'] = X['density'] * X['brstproc']
        X['density_menopaus'] = X['density'] * X['menopaus']
        X['density_agefirst'] = X['density'] * X['agefirst']
        X['bmi_hrt'] = X['bmi'] * X['hrt']
        X['bmi_family_hx'] = X['bmi'] * X['family_hx']
        X['bmi_menopaus'] = X['bmi'] * X['menopaus']
        X['hrt_menopaus'] = X['hrt'] * X['menopaus']
        X['hrt_agefirst'] = X['hrt'] * X['agefirst']
        X['hrt_nrelbc'] = X['hrt'] * X['nrelbc']
        X['family_hx_hrt'] = X['family_hx'] * X['hrt']
        X['family_hx_brstproc'] = X['family_hx'] * X['brstproc']
        X['family_hx_menopaus'] = X['family_hx'] * X['menopaus']

        # Three-way interactions
        X['age_density_family'] = X['age'] * X['density'] * X['family_hx']
        X['age_density_hrt'] = X['age'] * X['density'] * X['hrt']
        X['age_density_bmi'] = X['age'] * X['density'] * X['bmi']
        X['density_bmi_hrt'] = X['density'] * X['bmi'] * X['hrt']
        X['density_family_hrt'] = X['density'] * X['family_hx'] * X['hrt']
        X['age_family_hrt'] = X['age'] * X['family_hx'] * X['hrt']
        X['age_bmi_hrt'] = X['age'] * X['bmi'] * X['hrt']

        # Four-way interactions
        X['age_density_family_hrt'] = X['age'] * X['density'] * X['family_hx'] * X['hrt']
        X['age_density_bmi_hrt'] = X['age'] * X['density'] * X['bmi'] * X['hrt']

        # Polynomial features
        X['age_sq'] = X['age'] ** 2
        X['age_cube'] = X['age'] ** 3
        X['density_sq'] = X['density'] ** 2
        X['density_cube'] = X['density'] ** 3
        X['bmi_sq'] = X['bmi'] ** 2
        X['bmi_cube'] = X['bmi'] ** 3

        # Logarithmic transformations
        X['log_age'] = np.log1p(X['age'])
        X['log_bmi'] = np.log1p(X['bmi'])
        X['log_density'] = np.log1p(X['density'])

        # Normalized features
        X['age_norm'] = X['age'] / 100.0
        X['density_norm'] = X['density'] / 4.0
        X['bmi_norm'] = X['bmi'] / 50.0
        X['agefirst_norm'] = X['agefirst'] / 20.0
        X['nrelbc_norm'] = X['nrelbc'] / 10.0

        # Composite risk scores
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

        # Binary risk flags
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

        # Combined risk flags
        X['high_risk_combo'] = ((X['density'] >= 3) & (X['family_hx'] > 0)).astype(int)
        X['ultra_high_risk'] = ((X['density'] >= 3) & (X['family_hx'] > 0) & (X['age'] >= 50)).astype(int)
        X['age_density_risk'] = ((X['age'] >= 55) & (X['density'] >= 3)).astype(int)
        X['age_family_risk'] = ((X['age'] >= 50) & (X['family_hx'] > 0)).astype(int)
        X['density_hrt_risk'] = ((X['density'] >= 3) & (X['hrt'] > 0)).astype(int)
        X['triple_risk'] = ((X['density'] >= 3) & (X['family_hx'] > 0) & (X['hrt'] > 0)).astype(int)
        X['quad_risk'] = ((X['density'] >= 3) & (X['family_hx'] > 0) & (X['hrt'] > 0) & (X['age'] >= 50)).astype(int)

        # Ratios and divisions
        X['age_per_density'] = X['age'] / (X['density'] + 1)
        X['bmi_per_age'] = X['bmi'] / (X['age'] + 1)
        X['density_per_bmi'] = X['density'] / (X['bmi'] + 1)

        return X

    def _evaluate_model(self, n_samples=None, random_seed=None):
        """Evaluate the breast risk model using the integration class"""
        if not self.breast_risk_model_class:
            raise RuntimeError("Breast risk model class not loaded")

        # Load test data
        X_test, y_test, w_test = self._load_data()

        # Limit samples if specified
        if n_samples and n_samples < len(X_test):
            indices = np.random.RandomState(random_seed or self.config['random_seed']).choice(
                len(X_test), n_samples, replace=False
            )
            X_test = X_test[indices]
            y_test = y_test[indices]
            w_test = w_test[indices]

        # Initialize predictor
        self.predictor = self.breast_risk_model_class()

        # Prepare results storage
        predictions = []
        probabilities = []
        processing_times = []

        print(colored(f"\nEvaluating {len(X_test)} test samples...", Fore.CYAN))

        # Progress bar
        iterator = tqdm(range(len(X_test)), desc="Processing") if HAS_TQDM else range(len(X_test))

        for i in iterator:
            # Convert row to dict for prediction
            row_dict = {
                'age': float(X_test[i, 0]),  # age
                'race': int(X_test[i, 1]),   # race
                'density': float(X_test[i, 2]),  # density
                'family_hx': int(X_test[i, 3]),  # family_hx
                'bmi': float(X_test[i, 4]),   # bmi
                'agefirst': float(X_test[i, 5]),  # agefirst
                'nrelbc': int(X_test[i, 6]),   # nrelbc
                'brstproc': int(X_test[i, 7]),  # brstproc
                'menopaus': int(X_test[i, 8]),  # menopaus
                'hrt': int(X_test[i, 9])       # hrt
            }

            try:
                # Validate input
                self._validate_input(row_dict)

                # Time the prediction
                start_time = time.time()
                result = self.predictor.predict(row_dict)
                end_time = time.time()

                processing_times.append(end_time - start_time)

                # Check if prediction was successful
                if result.get('success', True) and 'prediction' in result and 'probability' in result:
                    # Extract prediction and probability
                    prediction = result.get('prediction', 0)
                    probability = result.get('probability', 0.0)
                    
                    predictions.append(prediction)
                    probabilities.append(probability)
                else:
                    # Prediction failed, use defaults
                    logger.warning(f"Prediction failed for sample {i}: {result.get('error', 'Unknown error')}")
                    predictions.append(0)  # Default to no cancer
                    probabilities.append(0.0)
                    processing_times.append(0.0)

            except Exception as e:
                logger.warning(f"Prediction failed for sample {i}: {e}")
                predictions.append(0)  # Default to no cancer
                probabilities.append(0.0)
                processing_times.append(0.0)

        # Convert to numpy arrays
        y_true = np.array(y_test)
        y_pred = np.array(predictions)
        y_proba = np.array(probabilities)
        proc_times = np.array(processing_times)

        return y_true, y_pred, y_proba, proc_times

    def run_benchmark(self, n_samples=None, random_seed=None, n_runs=1):
        """Run comprehensive benchmark evaluation"""
        print(colored("\n" + "="*80, Fore.CYAN))
        print(colored(" BREAST CANCER RISK MODEL BENCHMARK ", Fore.CYAN, Style.BRIGHT))
        print(colored("="*80, Fore.CYAN))

        all_results = []

        for run in range(n_runs):
            if n_runs > 1:
                print(colored(f"\nRun {run + 1}/{n_runs}", Fore.YELLOW))
                seed = random_seed + run if random_seed else self.config['random_seed'] + run
            else:
                seed = random_seed or self.config['random_seed']

            # Evaluate model
            y_true, y_pred, y_proba, proc_times = self._evaluate_model(n_samples, seed)

            # Calculate metrics
            metrics = self._calculate_metrics(y_true, y_pred, y_proba, proc_times)

            all_results.append(metrics)

            # Print results for this run
            self._print_results(metrics, run + 1)

        # Print aggregate results if multiple runs
        if n_runs > 1:
            self._print_aggregate_results(all_results)

        return all_results

    def _calculate_metrics(self, y_true, y_pred, y_proba, proc_times):
        """Calculate comprehensive evaluation metrics"""
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        auc_roc = roc_auc_score(y_true, y_proba)

        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Processing time statistics
        avg_time = np.mean(proc_times) * 1000  # Convert to ms
        std_time = np.std(proc_times) * 1000
        min_time = np.min(proc_times) * 1000
        max_time = np.max(proc_times) * 1000

        # Clinical metrics
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) * 100 if (tn + fp) > 0 else 0
        ppv = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0  # Positive predictive value
        npv = tn / (tn + fn) * 100 if (tn + fn) > 0 else 0  # Negative predictive value

        return {
            'accuracy': accuracy,
            'auc_roc': auc_roc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': support,
            'confusion_matrix': cm,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'ppv': ppv,
            'npv': npv,
            'true_positives': tp,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn,
            'avg_processing_time': avg_time,
            'std_processing_time': std_time,
            'min_processing_time': min_time,
            'max_processing_time': max_time,
            'n_samples': len(y_true)
        }

    def _print_results(self, metrics, run_number=1):
        """Print formatted evaluation results"""
        prefix = f"Run {run_number}: " if run_number > 1 else ""

        print(colored(f"\n{prefix}PERFORMANCE METRICS", Fore.CYAN, Style.BRIGHT))
        print("-" * 50)

        # Overall metrics
        acc_pct = metrics['accuracy'] * 100
        auc_val = metrics['auc_roc']
        print(colored(f"Overall Accuracy: {acc_pct:.1f}%", Fore.GREEN, Style.BRIGHT))
        print(colored(f"AUC-ROC Score: {auc_val:.3f}", Fore.GREEN, Style.BRIGHT))

        # Processing time
        print(colored(f"\nProcessing Time: {metrics['avg_processing_time']:.2f} ± {metrics['std_processing_time']:.2f} ms per sample", Fore.BLUE))

        # Per-class performance
        print(colored("\nPer-Class Performance:", Fore.CYAN, Style.BRIGHT))
        print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
        print("-" * 61)

        class_names = ['No Cancer', 'Cancer']
        for i, name in enumerate(class_names):
            prec = metrics['precision'][i] * 100
            rec = metrics['recall'][i] * 100
            f1_val = metrics['f1'][i]
            sup = int(metrics['support'][i])

            # Color based on class (Cancer detection more important)
            if i == 1:  # Cancer class
                if rec >= 90:  # High recall for cancer is critical
                    color = Fore.GREEN
                elif rec >= 80:
                    color = Fore.YELLOW
                else:
                    color = Fore.RED
            else:  # No cancer class
                if f1_val >= 0.90:
                    color = Fore.GREEN
                elif f1_val >= 0.80:
                    color = Fore.CYAN
                else:
                    color = Fore.YELLOW

            line = f"{name:<15} {prec:>6.1f}%      {rec:>6.1f}%      {f1_val:>6.3f}      {sup:>10}"
            print(colored(line, color))

        # Confusion matrix
        cm = metrics['confusion_matrix']
        print(colored("\nConfusion Matrix:", Fore.CYAN, Style.BRIGHT))
        print(f"{'':>16} Predicted")
        print(f"{'':>14} No Cancer  Cancer")
        print("-" * 40)
        print(f"Actual No Cancer   {cm[0,0]:>8}  {cm[0,1]:>6}")
        print(f"       Cancer      {cm[1,0]:>8}  {cm[1,1]:>6}")

        # Clinical metrics
        print(colored("\nClinical Metrics:", Fore.CYAN, Style.BRIGHT))
        print(f"Sensitivity (Cancer Detection): {metrics['sensitivity']:.1f}%")
        print(f"Specificity (No Cancer Detection): {metrics['specificity']:.1f}%")
        print(f"PPV (Positive Predictive Value): {metrics['ppv']:.1f}%")
        print(f"NPV (Negative Predictive Value): {metrics['npv']:.1f}%")
        print(f"False Negatives (Missed Cancers): {metrics['false_negatives']:,}")
        print(f"False Positives (Unnecessary Alerts): {metrics['false_positives']:,}")

        # Error analysis - FN focused
        total_errors = metrics['false_positives'] + metrics['false_negatives']
        fn_ratio = metrics['false_negatives'] / total_errors * 100 if total_errors > 0 else 0
        fp_ratio = metrics['false_positives'] / total_errors * 100 if total_errors > 0 else 0

        print(colored("\nError Analysis (FN-Focused Model):", Fore.CYAN, Style.BRIGHT))
        print(f"Total Errors: {total_errors:,}")
        print(f"False Negatives: {metrics['false_negatives']:,} ({fn_ratio:.1f}% of errors)")
        print(f"False Positives: {metrics['false_positives']:,} ({fp_ratio:.1f}% of errors)")
        print(f"FN:FP Ratio: 1:{metrics['false_positives']/max(metrics['false_negatives'],1):.2f}")

        # Summary
        print(colored("\nPERFORMANCE SUMMARY", Fore.CYAN, Style.BRIGHT))
        print(colored(f"🎯 Overall Accuracy: {metrics['accuracy']:.1%}", Fore.GREEN, Style.BRIGHT))
        print(colored(f"📊 AUC-ROC: {auc_val:.3f}", Fore.GREEN, Style.BRIGHT))

        # Cancer detection is the priority
        cancer_color = Fore.GREEN if metrics['sensitivity'] >= 85 else (Fore.YELLOW if metrics['sensitivity'] >= 75 else Fore.RED)
        print(colored(f"🔍 Cancer Detection (Sensitivity): {metrics['sensitivity']:.1f}%",
                      cancer_color, Style.BRIGHT))
        print(colored(f"✅ No Cancer Detection (Specificity): {metrics['specificity']:.1f}%",
                      Fore.GREEN if metrics['specificity'] >= 70 else Fore.YELLOW, Style.BRIGHT))
        print(colored(f"⚠️  Missed Cancers (FN): {metrics['false_negatives']:,}",
                      Fore.GREEN if metrics['false_negatives'] < 10 else (Fore.YELLOW if metrics['false_negatives'] < 50 else Fore.RED), Style.BRIGHT))

        # Average F1 score
        avg_f1 = np.mean(metrics['f1'])
        print(colored(f"⚖️  Average F1-Score: {avg_f1:.3f}",
                      Fore.GREEN if avg_f1 >= 0.80 else Fore.YELLOW, Style.BRIGHT))

    def _print_aggregate_results(self, all_results):
        """Print aggregate results across multiple runs"""
        print(colored("\n" + "="*80, Fore.CYAN))
        print(colored(" AGGREGATE RESULTS ACROSS ALL RUNS ", Fore.CYAN, Style.BRIGHT))
        print(colored("="*80, Fore.CYAN))

        # Aggregate metrics
        accuracies = [r['accuracy'] for r in all_results]
        aucs = [r['auc_roc'] for r in all_results]
        sensitivities = [r['sensitivity'] for r in all_results]
        specificities = [r['specificity'] for r in all_results]
        avg_f1s = [np.mean(r['f1']) for r in all_results]
        proc_times = [r['avg_processing_time'] for r in all_results]

        print(colored("Mean ± Standard Deviation:", Fore.YELLOW, Style.BRIGHT))
        print(f"Accuracy: {np.mean(accuracies):.1%} ± {np.std(accuracies):.1%}")
        print(f"AUC-ROC: {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")
        print(f"Sensitivity: {np.mean(sensitivities):.1f}% ± {np.std(sensitivities):.1f}%")
        print(f"Specificity: {np.mean(specificities):.1f}% ± {np.std(specificities):.1f}%")
        print(f"Average F1: {np.mean(avg_f1s):.3f} ± {np.std(avg_f1s):.3f}")
        print(f"Processing Time: {np.mean(proc_times):.2f} ± {np.std(proc_times):.2f} ms")


def colored(text, color, style=''):
    """Return colored text if colorama available."""
    if HAS_COLOR:
        return f"{color}{style}{text}{Style.RESET_ALL}"
    return text


def main():
    """Main benchmarking routine."""
    parser = argparse.ArgumentParser(description='Breast Cancer Risk Model Benchmark')
    parser.add_argument('--samples', type=int, default=None,
                       help='Number of test samples to evaluate (default: all)')
    parser.add_argument('--runs', type=int, default=1,
                       help='Number of evaluation runs (default: 1)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')

    args = parser.parse_args()

    # Initialize benchmarker
    benchmarker = BreastRiskBenchmarker()

    # Run benchmark
    results = benchmarker.run_benchmark(
        n_samples=args.samples,
        random_seed=args.seed,
        n_runs=args.runs
    )

    print(colored("\nBenchmark completed successfully!", Fore.GREEN, Style.BRIGHT))


if __name__ == '__main__':
    main()