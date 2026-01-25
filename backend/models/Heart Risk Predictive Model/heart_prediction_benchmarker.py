#!/usr/bin/env python3
"""
Heart Disease Model Benchmark Script
Evaluates model accuracy on test data with comprehensive metrics.
"""

import os
import sys
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

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

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
    'test_samples': 1000
}

HEART_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(HEART_DIR, 'Heart Disease Prediction Dataset - Kaggle.csv')

class HeartRiskBenchmarker:
    """
    Heart Disease Model Benchmarking Class
    Evaluates heart risk prediction model performance using HeartRiskIntegration
    """

    def __init__(self, config=None):
        """Initialize the benchmarker"""
        self.config = config or CONFIG.copy()
        self.heart_model_class = None
        self.predictor = None

        # Load the heart model class
        self._load_model_class()

    def _load_model_class(self):
        """Load the heart model integration class"""
        try:
            module_file = os.path.join(MODEL_CONTROLLERS_PATH, 'heart_prediction.py')
            spec = importlib.util.spec_from_file_location('heart_prediction', module_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.heart_model_class = getattr(module, 'HeartRiskIntegration', None)
            if self.heart_model_class:
                logger.info("Successfully loaded HeartRiskIntegration class")
            else:
                logger.error("HeartRiskIntegration class not found")
        except Exception as e:
            logger.error(f"Failed to load heart model class: {e}")
            self.heart_model_class = None

    def _validate_input(self, test_df):
        """Validate input data"""
        if test_df is None or test_df.empty:
            logger.error("Test data is None or empty")
            return False

        required_columns = ['Heart_Risk']

        # For HeartRiskIntegration, we need the symptom-based features
        symptom_features = [
            'Chest_Pain', 'Shortness_of_Breath', 'Fatigue', 'Palpitations',
            'Dizziness', 'Swelling', 'Pain_Arms_Jaw_Back', 'Cold_Sweats_Nausea'
        ]
        risk_features = [
            'High_BP', 'High_Cholesterol', 'Diabetes', 'Smoking',
            'Obesity', 'Sedentary_Lifestyle', 'Family_History', 'Chronic_Stress'
        ]
        demo_features = ['Gender', 'Age']

        required_columns.extend(symptom_features + risk_features + demo_features)

        missing_columns = [col for col in required_columns if col not in test_df.columns]
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return False

        return True

    def _preprocess(self, test_df):
        """Preprocess the test data - minimal preprocessing for HeartRiskIntegration"""
        logger.info("Preparing data for HeartRiskIntegration...")
        X = test_df.drop('Heart_Risk', axis=1)  # Remove target column
        y = test_df['Heart_Risk']

        logger.info(f"Feature matrix shape: {X.shape}")
        return X, y

    def _process(self, X_test, y_test):
        """Process predictions using HeartRiskIntegration.predict()"""
        if self.predictor is None:
            logger.error("Predictor is None")
            return None, None, None

        start_time = time.time()

        try:
            logger.info("Making predictions through HeartRiskIntegration...")
            predictions = []
            probabilities = []

            # Convert each row to dictionary format for HeartRiskIntegration
            if hasattr(X_test, 'iloc'):  # pandas DataFrame
                for idx in tqdm(range(len(X_test)), desc="Predicting", disable=not HAS_TQDM):
                    row_dict = X_test.iloc[idx].to_dict()
                    result = self.predictor.predict(row_dict)

                    if result.get('success'):
                        pred = result.get('prediction')
                        prob = result.get('risk_score', 0) / 100.0  # Convert percentage back to probability
                        predictions.append(pred)
                        probabilities.append(prob)
                    else:
                        logger.warning(f"Prediction failed for sample {idx}")
                        predictions.append(0)  # Default to low risk
                        probabilities.append(0.5)  # Default probability
            else:  # numpy array
                logger.warning("X_test is not a DataFrame, using fallback predictions")
                return self._fallback_array(X_test, y_test)

            y_pred = np.array(predictions)
            y_proba = np.array(probabilities)

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return None, None, None

        total_time = time.time() - start_time
        avg_time = total_time / len(X_test)
        print_color(f"\nProcessed {len(X_test)} samples in {total_time:.2f} seconds", Fore.CYAN)
        print_color(f"Average time: {avg_time*1000:.2f} ms per sample", Fore.CYAN)

        return y_test, y_pred, y_proba

    def _fallback_array(self, X_test, y_test):
        """Fallback method for array inputs"""
        logger.warning("Using fallback for array input")
        # Simple predictions based on basic rules
        predictions = []
        probabilities = []

        for i in range(len(X_test)):
            # Use a simple heuristic for fallback
            prob = 0.5  # Default 50% probability
            pred = 1 if prob > 0.5 else 0
            predictions.append(pred)
            probabilities.append(prob)

        return y_test, np.array(predictions), np.array(probabilities)

    def _fallback(self, test_df):
        """Fallback method for when model loading fails"""
        logger.warning("Using fallback evaluation method")
        # Simple rule-based evaluation
        predictions = []
        for _, row in test_df.iterrows():
            # Simple risk assessment based on key factors
            risk_score = 0
            if row.get('High_BP', 0): risk_score += 2
            if row.get('High_Cholesterol', 0): risk_score += 2
            if row.get('Diabetes', 0): risk_score += 2
            if row.get('Smoking', 0): risk_score += 1
            if row.get('Family_History', 0): risk_score += 1
            if row.get('Age', 50) > 60: risk_score += 1

            prediction = 1 if risk_score >= 4 else 0
            predictions.append(prediction)

        y_pred = np.array(predictions)
        y_test = test_df['Heart_Risk'].values
        y_proba = y_pred.astype(float)  # Simple binary probabilities

        return y_test, y_pred, y_proba

    def predict(self, test_df=None, use_held_out=None):
        """Main prediction method"""
        print_header("HEART MODEL EVALUATION")

        # Load model if not already loaded
        if not self._load_model():
            logger.error("Failed to load model, using fallback")
            if test_df is None:
                test_df = self._load_test_data(use_held_out)
            return self._fallback(test_df)

        # Load test data if not provided
        if test_df is None:
            test_df = self._load_test_data(use_held_out)

        if not self._validate_input(test_df):
            return None

        # Preprocess and predict
        X_test, y_test = self._preprocess(test_df)
        return self._process(X_test, y_test)

    def _load_model(self):
        """Load model instance"""
        try:
            logger.info("Loading Heart Disease model instance...")
            if self.heart_model_class is None:
                logger.error("HeartRiskIntegration not available")
                return False

            self.predictor = self.heart_model_class()
            if not self.predictor.is_loaded:
                logger.error("Model failed to load")
                return False

            logger.info("Loaded Heart model instance successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def _load_test_data(self, use_held_out=None):
        """Load test data"""
        use_held_out = use_held_out if use_held_out is not None else self.config['use_held_out_set']

        try:
            if use_held_out and os.path.exists(DATA_PATH):
                logger.info(f"Loading dataset from: {DATA_PATH}")
                df = pd.read_csv(DATA_PATH)
                _, test_df = train_test_split(df, test_size=0.2, random_state=self.config['random_seed'], stratify=df['Heart_Risk'])

                # If test_samples is specified and smaller than the split, take a subset
                if self.config['test_samples'] < len(test_df):
                    test_df = test_df.sample(n=self.config['test_samples'], random_state=self.config['random_seed'])
                    logger.info(f"Sampled {len(test_df)} test samples from held-out set")
                else:
                    logger.info(f"Loaded {len(test_df)} test samples from held-out set")

                return test_df
            else:
                logger.warning("Held-out test set not available")
                df = pd.read_csv(DATA_PATH)
                _, test_df = train_test_split(df, test_size=min(self.config['test_samples'], len(df)) / len(df), random_state=self.config['random_seed'], stratify=df['Heart_Risk'])
                logger.info(f"Using {len(test_df)} samples for testing")
                return test_df
        except Exception as e:
            logger.error(f"Failed to load test data: {e}")
            return None

def print_color(text, color=Fore.WHITE, style=Style.NORMAL, end='\n'):
    print(f"{style}{color}{text}{Style.RESET_ALL}", end=end)

def print_header(title):
    print(f"\n{'='*80}")
    print_color(f" {title.upper()} ", Fore.CYAN, Style.BRIGHT)
    print(f"{'='*80}")

def calculate_metrics(y_test, y_pred, y_proba):
    """Calculate and display performance metrics"""
    print_header("HEART PERFORMANCE METRICS")

    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    print_color(f"Overall Accuracy: {accuracy:.1%}", Fore.GREEN, Style.BRIGHT)
    print_color(f"AUC-ROC Score: {auc:.4f}", Fore.GREEN, Style.BRIGHT)

    class_report = classification_report(y_test, y_pred, target_names=['Low Risk', 'High Risk'], output_dict=True)
    print_color("\nPer-Class Performance:", Fore.CYAN, Style.BRIGHT)
    print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<8}")
    print("-" * 59)

    for class_name in ['Low Risk', 'High Risk']:
        metrics = class_report[class_name]
        color = Fore.GREEN if metrics['f1-score'] >= 0.99 else Fore.CYAN
        print_color(f"{class_name:<15} {metrics['precision']:.1%}       {metrics['recall']:.1%}       {metrics['f1-score']:.1%}       {int(metrics['support']):<8}", color)

    cm = confusion_matrix(y_test, y_pred)
    print_color("\nConfusion Matrix:", Fore.CYAN, Style.BRIGHT)
    print("                Predicted")
    print("              Low Risk  High Risk")
    print("-" * 40)
    print(f"Actual Low Risk    {cm[0,0]:>4}       {cm[0,1]:>4}")
    print(f"       High Risk   {cm[1,0]:>4}       {cm[1,1]:>4}")

    print_color("\nProbability Distribution Analysis:", Fore.CYAN, Style.BRIGHT)
    correct_mask = (y_test == y_pred)
    correct_probs = y_proba[correct_mask]
    incorrect_probs = y_proba[~correct_mask]

    print(f"Correct predictions: {len(correct_probs)}")
    print(f"  Mean probability: {np.mean(correct_probs):.1%}")
    print(f"  Std deviation: {np.std(correct_probs):.3f}")

    print(f"Incorrect predictions: {len(incorrect_probs)}")
    if len(incorrect_probs) > 0:
        print(f"  Mean probability: {np.mean(incorrect_probs):.1%}")
        print(f"  Std deviation: {np.std(incorrect_probs):.3f}")

    print_color("\nRisk Stratification:", Fore.CYAN, Style.BRIGHT)
    very_low = np.sum(y_proba < 0.25)
    low = np.sum((y_proba >= 0.25) & (y_proba < 0.5))
    moderate = np.sum((y_proba >= 0.5) & (y_proba < 0.75))
    high = np.sum(y_proba >= 0.75)

    total = len(y_proba)
    print(f"Very Low Risk (<25%): {very_low} patients ({very_low/total*100:.1f}%)")
    print(f"Low Risk (25-50%): {low} patients ({low/total*100:.1f}%)")
    print(f"Moderate Risk (50-75%): {moderate} patients ({moderate/total*100:.1f}%)")
    print(f"High Risk (>75%): {high} patients ({high/total*100:.1f}%)")

    print_header("HEART SUMMARY")
    correct_count = np.sum(y_test == y_pred)
    total_count = len(y_test)
    print_color(f"🎯 Overall Accuracy: {correct_count}/{total_count} = {accuracy:.1%}", Fore.GREEN, Style.BRIGHT)
    print_color(f"📊 AUC-ROC: {auc:.4f}", Fore.GREEN, Style.BRIGHT)

    if accuracy >= 0.99:
        msg = "🎉 OUTSTANDING: Model achieves near-perfect accuracy (99%+)"
    elif accuracy >= 0.95:
        msg = "✅ EXCELLENT: Model exceeds clinical deployment target (95%+)"
    elif accuracy >= 0.90:
        msg = "✅ VERY GOOD: Model shows strong performance (90%+)"
    elif accuracy >= 0.85:
        msg = "⚠️ GOOD: Model meets requirements but could be improved"
    else:
        msg = "❌ NEEDS IMPROVEMENT: Model performance below acceptable thresholds"

    print_color(msg, Fore.GREEN if accuracy >= 0.95 else Fore.CYAN if accuracy >= 0.90 else Fore.YELLOW if accuracy >= 0.85 else Fore.RED, Style.BRIGHT)

    return {'accuracy': accuracy, 'auc': auc, 'confusion_matrix': cm, 'classification_report': class_report}

def run_multi_seed_benchmark(seeds, use_held_out):
    """Run benchmark across multiple seeds"""
    print_header("HEART MULTI-SEED BENCHMARK ANALYSIS")

    benchmarker = HeartRiskBenchmarker()
    if not benchmarker._load_model():
        logger.error("Failed to load model")
        return

    all_results = []
    for seed in seeds:
        print_color(f"\n{'='*60}", Fore.BLUE)
        print_color(f"HEART SEED {seed}", Fore.BLUE, Style.BRIGHT)
        print_color(f"{'='*60}", Fore.BLUE)

        benchmarker.config['random_seed'] = seed
        test_df = benchmarker._load_test_data(use_held_out=use_held_out)
        if test_df is None:
            continue

        y_test, y_pred, y_proba = benchmarker.predict(test_df)
        if y_test is None:
            continue

        metrics = calculate_metrics(y_test, y_pred, y_proba)
        all_results.append({
            'seed': seed,
            'accuracy': metrics['accuracy'],
            'auc': metrics['auc'],
            'samples': len(y_test)
        })

    if len(all_results) > 1:
        print_header("HEART AGGREGATE ANALYSIS")
        accuracies = [r['accuracy'] for r in all_results]
        aucs = [r['auc'] for r in all_results]

        print_color("📊 Performance Across Seeds:", Fore.CYAN, Style.BRIGHT)
        print(f"  Mean Accuracy: {np.mean(accuracies):.1%} ± {np.std(accuracies):.1%}")
        print(f"  Mean AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
        print(f"  Min Accuracy: {np.min(accuracies):.1%}")
        print(f"  Max Accuracy: {np.max(accuracies):.1%}")

        print_color("\n📋 Per-Seed Breakdown:", Fore.CYAN, Style.BRIGHT)
        print(f"{'Seed':<8} {'Accuracy':<12} {'AUC':<10} {'Samples':<10}")
        print("-" * 40)

        for result in all_results:
            color = Fore.GREEN if result['accuracy'] >= 0.99 else Fore.CYAN
            print_color(f"{result['seed']:<8} {result['accuracy']:.1%}        {result['auc']:.4f}    {result['samples']:<10}", color)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Benchmark the Heart Disease model')
    parser.add_argument('--use-held-out', action='store_true', default=CONFIG['use_held_out_set'], help='Use held-out test set')
    parser.add_argument('--samples', type=int, default=CONFIG['test_samples'], help='Number of test samples')
    parser.add_argument('--multi-seed', type=int, nargs='+', default=None, help='Run with multiple seeds')
    parser.add_argument('--seed', type=int, default=CONFIG['random_seed'], help='Random seed')

    args = parser.parse_args()
    CONFIG['use_held_out_set'] = args.use_held_out
    CONFIG['test_samples'] = args.samples
    CONFIG['random_seed'] = args.seed

    print_header("HEART DISEASE MODEL BENCHMARK")
    print_color("Evaluating symptom and risk factor-based heart disease prediction model", Fore.CYAN)
    print_color(f"Configuration: {CONFIG}", Fore.CYAN)

    if args.multi_seed:
        run_multi_seed_benchmark(args.multi_seed, args.use_held_out)
    else:
        benchmarker = HeartRiskBenchmarker()
        y_test, y_pred, y_proba = benchmarker.predict()
        if y_test is not None:
            calculate_metrics(y_test, y_pred, y_proba)

if __name__ == "__main__":
    main()
