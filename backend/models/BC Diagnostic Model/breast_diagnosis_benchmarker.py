#!/usr/bin/env python3
"""
Breast Cancer Diagnosis Model Benchmark Script
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

BREAST_DIAGNOSIS_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BREAST_DIAGNOSIS_DIR, 'Breast_cancer_data.csv')

class BreastDiagnosisBenchmarker:
    """
    Breast Cancer Diagnosis Model Benchmarking Class
    Evaluates breast cancer diagnosis model performance using BreastDiagnosisIntegration
    """

    def __init__(self, config=None):
        """Initialize the benchmarker"""
        self.config = config or CONFIG.copy()
        self.breast_diagnosis_model_class = None
        self.predictor = None

        # Load the breast diagnosis model class
        self._load_model_class()

    def _load_model_class(self):
        """Load the breast diagnosis model integration class"""
        try:
            module_file = os.path.join(MODEL_CONTROLLERS_PATH, 'breast_diagnosis.py')
            spec = importlib.util.spec_from_file_location('breast_diagnosis', module_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.breast_diagnosis_model_class = getattr(module, 'BreastDiagnosisIntegration', None)
            if self.breast_diagnosis_model_class:
                logger.info("Successfully loaded BreastDiagnosisIntegration class")
            else:
                logger.error("BreastDiagnosisIntegration class not found")
        except Exception as e:
            logger.error(f"Failed to load breast diagnosis model class: {e}")
            self.breast_diagnosis_model_class = None

    def _validate_input(self, data):
        """Validate input data format and required fields"""
        if not isinstance(data, dict):
            raise ValueError("Input data must be a dictionary")

        required_fields = [
            'mean_radius', 'mean_texture', 'mean_perimeter', 'mean_area', 'mean_smoothness'
        ]

        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")

        # Validate data types and ranges
        for field in required_fields:
            if not isinstance(data[field], (int, float)):
                raise ValueError(f"{field} must be a number")
            if data[field] <= 0:
                raise ValueError(f"{field} must be positive")

        return True

    def _load_data(self):
        """Load breast cancer diagnostic dataset for evaluation."""
        data_path = os.path.join(os.path.dirname(__file__), 'Wisconsin Diagnosis Dataset - UCI.csv')

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")

        df = pd.read_csv(data_path)

        # Base features
        base_features = ['mean_radius', 'mean_texture', 'mean_perimeter', 'mean_area', 'mean_smoothness']
        X = df[base_features]
        y = df['diagnosis'].values  # 0=Malignant, 1=Benign

        # Engineer features same as training
        X_eng = self._engineer_features(X)

        print(f"INFO: Using full dataset with {len(X_eng)} samples")
        print(f"INFO: Class distribution - Malignant: {(y==0).sum():,}, Benign: {(y==1).sum():,}")

        return X_eng.values, y

    def _engineer_features(self, X):
        """Create additional engineered features - same as training."""
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

    def _evaluate_model(self, n_samples=None, random_seed=None):
        """Evaluate the breast diagnosis model using the integration class"""
        if not self.breast_diagnosis_model_class:
            raise RuntimeError("Breast diagnosis model class not loaded")

        # Load test data
        X_test, y_test = self._load_data()

        # Limit samples if specified
        if n_samples and n_samples < len(X_test):
            indices = np.random.RandomState(random_seed or self.config['random_seed']).choice(
                len(X_test), n_samples, replace=False
            )
            X_test = X_test[indices]
            y_test = y_test[indices]

        # Initialize predictor
        self.predictor = self.breast_diagnosis_model_class()

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
                'mean_radius': float(X_test[i, 0]),
                'mean_texture': float(X_test[i, 1]),
                'mean_perimeter': float(X_test[i, 2]),
                'mean_area': float(X_test[i, 3]),
                'mean_smoothness': float(X_test[i, 4])
            }

            try:
                # Validate input
                self._validate_input(row_dict)

                # Time the prediction
                start_time = time.time()
                result = self.predictor.predict(row_dict)
                end_time = time.time()

                processing_times.append(end_time - start_time)

                # Extract prediction and probability
                prediction = result.get('prediction', 0)
                probability = result.get('probability', 0.0)

                predictions.append(prediction)
                probabilities.append(probability)

            except Exception as e:
                logger.warning(f"Prediction failed for sample {i}: {e}")
                predictions.append(0)  # Default to malignant
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
        print(colored(" BREAST CANCER DIAGNOSIS MODEL BENCHMARK ", Fore.CYAN, Style.BRIGHT))
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
        sensitivity = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0  # Malignant detection rate
        specificity = tn / (tn + fp) * 100 if (tn + fp) > 0 else 0  # Benign detection rate
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

        class_names = ['Malignant', 'Benign']
        for i, name in enumerate(class_names):
            prec = metrics['precision'][i] * 100
            rec = metrics['recall'][i] * 100
            f1_val = metrics['f1'][i]
            sup = int(metrics['support'][i])

            # Color based on class (Malignant detection more important)
            if i == 0:  # Malignant class
                if rec >= 90:  # High recall for malignant is critical
                    color = Fore.GREEN
                elif rec >= 80:
                    color = Fore.YELLOW
                else:
                    color = Fore.RED
            else:  # Benign class
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
        print(f"{'':>14} Malignant  Benign")
        print("-" * 40)
        print(f"Actual Malignant   {cm[0,0]:>8}  {cm[0,1]:>6}")
        print(f"       Benign      {cm[1,0]:>8}  {cm[1,1]:>6}")

        # Clinical metrics
        print(colored("\nClinical Metrics:", Fore.CYAN, Style.BRIGHT))
        print(f"Sensitivity (Malignant Detection): {metrics['sensitivity']:.1f}%")
        print(f"Specificity (Benign Detection): {metrics['specificity']:.1f}%")
        print(f"PPV (Positive Predictive Value): {metrics['ppv']:.1f}%")
        print(f"NPV (Negative Predictive Value): {metrics['npv']:.1f}%")
        print(f"False Negatives (Missed Malignancies): {metrics['false_negatives']:,}")
        print(f"False Positives (Unnecessary Biopsies): {metrics['false_positives']:,}")

        # Error analysis - FN focused for malignant detection
        total_errors = metrics['false_positives'] + metrics['false_negatives']
        fn_ratio = metrics['false_negatives'] / total_errors * 100 if total_errors > 0 else 0
        fp_ratio = metrics['false_positives'] / total_errors * 100 if total_errors > 0 else 0

        print(colored("\nError Analysis (Malignant Detection Priority):", Fore.CYAN, Style.BRIGHT))
        print(f"Total Errors: {total_errors:,}")
        print(f"False Negatives: {metrics['false_negatives']:,} ({fn_ratio:.1f}% of errors)")
        print(f"False Positives: {metrics['false_positives']:,} ({fp_ratio:.1f}% of errors)")
        print(f"FN:FP Ratio: 1:{metrics['false_positives']/max(metrics['false_negatives'],1):.2f}")

        # Summary
        print(colored("\nPERFORMANCE SUMMARY", Fore.CYAN, Style.BRIGHT))
        print(colored(f"🎯 Overall Accuracy: {metrics['accuracy']:.1%}", Fore.GREEN, Style.BRIGHT))
        print(colored(f"📊 AUC-ROC: {auc_val:.3f}", Fore.GREEN, Style.BRIGHT))

        # Malignant detection is the priority
        malignant_color = Fore.GREEN if metrics['sensitivity'] >= 85 else (Fore.YELLOW if metrics['sensitivity'] >= 75 else Fore.RED)
        print(colored(f"🔍 Malignant Detection (Sensitivity): {metrics['sensitivity']:.1f}%",
                      malignant_color, Style.BRIGHT))
        print(colored(f"✅ Benign Detection (Specificity): {metrics['specificity']:.1f}%",
                      Fore.GREEN if metrics['specificity'] >= 70 else Fore.YELLOW, Style.BRIGHT))
        print(colored(f"⚠️  Missed Malignancies (FN): {metrics['false_negatives']:,}",
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
    parser = argparse.ArgumentParser(description='Breast Cancer Diagnosis Model Benchmark')
    parser.add_argument('--samples', type=int, default=None,
                       help='Number of test samples to evaluate (default: all)')
    parser.add_argument('--runs', type=int, default=1,
                       help='Number of evaluation runs (default: 1)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')

    args = parser.parse_args()

    # Initialize benchmarker
    benchmarker = BreastDiagnosisBenchmarker()

    # Run benchmark
    results = benchmarker.run_benchmark(
        n_samples=args.samples,
        random_seed=args.seed,
        n_runs=args.runs
    )

    print(colored("\nBenchmark completed successfully!", Fore.GREEN, Style.BRIGHT))


if __name__ == '__main__':
    main()
