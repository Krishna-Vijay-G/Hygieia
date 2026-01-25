<div align="center">
  <img src="/Hero.svg" alt="Hygieia Logo" width="200"/>
</div>

<div align="center">
  <h1>Heart Risk Predictive Model</h1>
  <strong>AdaBoost Ensemble Classifier for Cardiovascular Risk Assessment</strong><br/>
  <em><a href="https://Krishna-Vijay-G.github.io">Krishna Vijay G</a> • Hygieia AI Healthcare Platform</em>
</div>
</div>

---

## 📋 Table of Contents

1. [Model Overview](#-model-overview)
2. [Performance Metrics](#-performance-metrics)
3. [Dataset Description](#-dataset-description)
4. [Architecture & Methodology](#-architecture--methodology)
5. [Training Pipeline](#-training-pipeline)
6. [Feature Engineering](#-feature-engineering)
7. [Model Comparison & Selection](#-model-comparison--selection)
8. [Trials, Errors & Learnings](#-trials-errors--learnings)
9. [Achievements](#-achievements)
10. [Technical Specifications](#-technical-specifications)
11. [Deployment Status](#-deployment-status)

---

## 📋 Table of Contents

1. [Model Overview](#-model-overview)
2. [Performance Metrics](#-performance-metrics)
3. [Dataset Description](#-dataset-description)
4. [Architecture & Methodology](#-architecture--methodology)
5. [Training Pipeline](#-training-pipeline)
6. [Feature Engineering](#-feature-engineering)
7. [Model Comparison & Selection](#-model-comparison--selection)
8. [Trials, Errors & Learnings](#-trials-errors--learnings)
9. [Achievements](#-achievements)
10. [Technical Specifications](#-technical-specifications)
11. [Deployment Status](#-deployment-status)

---

## 🎯 Model Overview

| Property | Value |
|----------|-------|
| **Model Name** | Heart Risk Predictive Model |
| **Model ID** | `heart-prediction` |
| **Algorithm** | AdaBoost Classifier |
| **Version** | 1.0 |
| **Training Date** | December 31, 2025 |
| **Modified Date** | January 4, 2026 |

### Description

The Heart Risk Predictive Model employs an **AdaBoost (Adaptive Boosting) Classifier** to assess cardiovascular disease risk using clinical symptom and risk factor data. AdaBoost combines multiple weak learners (decision stumps) sequentially, where each subsequent learner focuses on correcting the mistakes of its predecessors. This ensemble approach yields exceptional classification accuracy while maintaining interpretability.

The model processes 18 clinical features encompassing symptoms, risk factors, and demographic information to produce a binary risk classification with calibrated confidence scores.

---

## 📊 Performance Metrics

### Primary Metrics

| Metric | Value | Assessment |
|--------|-------|------------|
| **Test Accuracy** | 99.36% | Outstanding |
| **Validation Accuracy** | 99.33% | Outstanding |
| **ROC-AUC** | 99.97% | Near Perfect |
| **F1 Score** | 99.36% | Excellent |
| **Precision** | 99.36% | Excellent |
| **Recall** | 99.36% | Excellent |

### Training Statistics

| Statistic | Value |
|-----------|-------|
| Training Samples | 42,000 (60%) |
| Validation Samples | 14,000 (20%) |
| Test Samples | 14,000 (20%) |
| **Total Samples** | **70,000** |
| Features | 18 |
| Classes | 2 (Low Risk / High Risk) |
| Training Time | 4.32 seconds |

### Confusion Matrix (Test Set)

```python
Predicted →       Low Risk    High Risk
Actual ↓
Low Risk            6,970         30
High Risk              60      6,940
```

- **True Negatives**: 6,970 (Low Risk correctly identified)
- **True Positives**: 6,940 (High Risk correctly identified)  
- **False Positives**: 30 (Low Risk misclassified as High)
- **False Negatives**: 60 (High Risk misclassified as Low)

---

## 📁 Dataset Description

### Source

**Heart Disease Risk Prediction Dataset** from Kaggle (EarlyMed Clinical Data)

### Characteristics

| Property | Value |
|----------|-------|
| Total Samples | 70,000 patients |
| Features | 18 clinical indicators |
| Target | Binary (Low Risk vs High Risk) |
| Class Balance | Perfect 50/50 split (35,000 each) |
| Missing Values | None (0 across all features) |
| Data Types | All float64 |

### Feature Categories

#### 1. Symptom Features (8 binary indicators)

| Feature | Description |
|---------|-------------|
| `Chest_Pain` | Presence of chest pain or discomfort |
| `Shortness_of_Breath` | Difficulty breathing |
| `Fatigue` | Persistent unexplained tiredness |
| `Palpitations` | Irregular or rapid heartbeat |
| `Dizziness` | Episodes of lightheadedness or fainting |
| `Swelling` | Swelling in legs/ankles (fluid retention) |
| `Pain_Arms_Jaw_Back` | Pain radiating to arms, jaw, or back |
| `Cold_Sweats_Nausea` | Cold sweats and nausea (acute symptoms) |

#### 2. Risk Factor Features (8 indicators)

| Feature | Description |
|---------|-------------|
| `High_BP` | History of high blood pressure |
| `High_Cholesterol` | Elevated cholesterol levels |
| `Diabetes` | Diabetes diagnosis |
| `Smoking` | Current or past smoking history |
| `Obesity` | Obesity status (BMI > 30) |
| `Sedentary_Lifestyle` | Low physical activity |
| `Family_History` | Family history of heart disease |
| `Chronic_Stress` | High chronic stress levels |

#### 3. Demographic Features (2 indicators)

| Feature | Description |
|---------|-------------|
| `Gender` | Biological sex (0=Female, 1=Male) |
| `Age` | Patient age in years (23-83 range) |

---

## 🏗 Architecture & Methodology

### System Flow Diagram

```mermaid
graph TD
    A["🏥 RAW CLINICAL DATA<br/>70,000 × 18 features"]:::input --> B
    B["🔍 DATA QUALITY CHECK<br/>• Missing Values: ✓ None<br/>• Data Type: ✓ Float64<br/>• Range: ✓ Valid"]:::process --> C
    C["📋 FEATURE SEPARATION<br/>X: 18 Features<br/>y: Heart_Risk"]:::process --> D
    D["✂️ STRATIFIED SPLIT<br/>Train: 42,000 (60%)<br/>Val: 14,000 (20%)<br/>Test: 14,000 (20%)"]:::process --> E
    E["📊 STANDARD SCALING<br/>z = (x - μ) / σ<br/>Fit on train only"]:::modify --> F
    F["🚀 ADABOOST CLASSIFIER<br/>50 Weak Learners"]:::model --> G
    G["Iteration 1<br/>Train Weak Learner 1<br/>Update weights"]:::iteration --> H
    H["Iteration 2<br/>Focus on errors<br/>Update weights"]:::iteration --> I
    I["..."]:::iteration --> J
    J["Iteration N<br/>Final ensemble"]:::iteration --> K
    K["🗳️ WEIGHTED VOTING<br/>Σ(αᵢ × hᵢ(x))"]:::ensemble --> L
    L["📤 PREDICTION OUTPUT<br/>• Classification: Low/High Risk<br/>• Probability Scores<br/>• Confidence"]:::output

    classDef input fill:#E3F2FD,stroke:#1976D2,stroke-width:3px,color:#000
    classDef process fill:#F3E5F5,stroke:#7B1FA2,stroke-width:2px,color:#000
    classDef modify fill:#E8F5E9,stroke:#388E3C,stroke-width:2px,color:#000
    classDef model fill:#FFF3E0,stroke:#F57C00,stroke-width:3px,color:#000
    classDef iteration fill:#FCE4EC,stroke:#C2185B,stroke-width:2px,color:#000
    classDef ensemble fill:#E0F2F1,stroke:#00796B,stroke-width:2px,color:#000
    classDef output fill:#C8E6C9,stroke:#2E7D32,stroke-width:3px,color:#000
```

### AdaBoost Algorithm Details

AdaBoost operates by:

1. **Initialize** uniform sample weights: `w₁(i) = 1/N`
2. **For each iteration t = 1 to T:**
   - Train weak learner `hₜ` using weighted samples
   - Calculate weighted error: `εₜ = Σ wₜ(i) × I(yᵢ ≠ hₜ(xᵢ))`
   - Compute learner weight: `αₜ = 0.5 × ln((1-εₜ)/εₜ)`
   - Update sample weights: `wₜ₊₁(i) ∝ wₜ(i) × exp(-αₜ × yᵢ × hₜ(xᵢ))`
3. **Final prediction:** `H(x) = sign(Σ αₜ × hₜ(x))`

---

## 🔧 Training Pipeline

### Configuration

```python
AdaBoostClassifier(
    n_estimators=50,      # Number of weak learners
    learning_rate=1.0,    # Learning rate shrinkage
    algorithm='SAMME.R',  # Real-valued predictions
    random_state=42       # Reproducibility
)
```

### Training Process

1. **Data Loading**: Load 70,000 clinical records
2. **Preprocessing**: StandardScaler normalization
3. **Stratified Split**: 60/20/20 train/val/test
4. **Model Training**: AdaBoost on training set (4.32s)
5. **Validation**: Evaluate on held-out validation set
6. **Testing**: Final evaluation on test set
7. **Serialization**: Save to `heart-prediction.joblib`

### Model Bundle Contents

```python
{
    'model': AdaBoostClassifier,     # Trained model
    'scaler': StandardScaler,        # Fitted scaler
    'feature_names': [...],          # 18 feature names
    'training_info': {...}           # Performance metrics
}
```

---

## 📐 Feature Engineering

### Scaling Strategy

**StandardScaler** transforms features to zero mean and unit variance:

```python
z = (x - μ) / σ

where:
  μ = mean of training data
  σ = standard deviation of training data
```

### Feature Importance

Top features by importance (AdaBoost learner weights):

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | `Chest_Pain` | High |
| 2 | `Age` | High |
| 3 | `High_BP` | High |
| 4 | `Family_History` | Medium-High |
| 5 | `Shortness_of_Breath` | Medium-High |
| 6 | `High_Cholesterol` | Medium |
| 7 | `Diabetes` | Medium |
| 8 | `Smoking` | Medium |

---

## 🏆 Model Comparison & Selection

### Nine Models Evaluated

| Model | Test Accuracy | ROC-AUC | Training Time | Selection |
|-------|--------------|---------|---------------|-----------|
| **AdaBoost** | **99.36%** | **99.97%** | 4.32s | 🏆 **SELECTED** |
| Gradient Boosting | 99.33% | 99.97% | 4.71s | Runner-up |
| Random Forest | 99.24% | 99.94% | 1.24s | Strong |
| XGBoost | 99.19% | 99.95% | 0.63s | Fast & Good |
| LightGBM | 99.16% | 99.95% | 0.45s | Fastest |
| Logistic Regression | 99.11% | 99.95% | 0.18s | Baseline |
| Support Vector Machine | 99.09% | 99.94% | 8.23s | Slowest |
| Neural Network | 98.94% | 99.91% | 12.45s | Complex |
| K-Nearest Neighbors | 98.73% | 99.87% | 0.52s | Simple |

### Selection Rationale

**AdaBoost was selected as the champion model because:**

1. **Highest Test Accuracy**: 99.36% outperformed all competitors
2. **Best ROC-AUC**: 99.97% indicates near-perfect discrimination
3. **Fast Training**: 4.32 seconds is highly efficient
4. **Robust**: Consistent validation (99.33%) and test (99.36%) performance
5. **Interpretable**: Feature importance readily available
6. **No Overfitting**: Validation ≈ Test accuracy confirms generalization

---

## 🧪 Trials, Errors & Learnings

### Trial 1: Initial Baseline Exploration

**Approach**: Tested Logistic Regression as baseline

**Result**: 99.11% accuracy - surprisingly high baseline

**Learning**: Dataset is well-structured with clear class separation; even simple models perform excellently.

---

### Trial 2: Tree-Based Methods

**Approach**: Evaluated Random Forest, Gradient Boosting, XGBoost

**Results**: 
- All achieved >99% accuracy
- Gradient Boosting matched AdaBoost quality

**Learning**: Ensemble tree methods excel on structured clinical data with mixed binary/continuous features.

---

### Trial 3: Complex Models (SVM, Neural Network)

**Approach**: Tested SVM with RBF kernel and MLP neural network

**Results**:
- SVM: 99.09% (slower training at 8.23s)
- Neural Network: 98.94% (slowest, slightly worse)

**Learning**: Increased complexity doesn't guarantee improvement. Simpler ensemble methods outperform on this dataset.

---

### Trial 4: Hyperparameter Tuning

**Approach**: Grid search on AdaBoost parameters

**Parameters Tested**:
- `n_estimators`: [25, 50, 100, 200]
- `learning_rate`: [0.5, 1.0, 1.5]

**Result**: Default parameters (n_estimators=50, learning_rate=1.0) optimal

**Learning**: AdaBoost's adaptive weighting self-optimizes; heavy tuning unnecessary for balanced datasets.

---

### Trial 5: Feature Selection Experiments

**Approach**: Tested with symptom-only and risk-factor-only subsets

**Results**:
- Symptoms only: 97.2% accuracy
- Risk factors only: 96.8% accuracy
- All features: 99.36% accuracy

**Learning**: All 18 features contribute synergistically; no redundancy worth removing.

---

## 🏅 Achievements

### Performance Milestones

| Achievement | Details |
|-------------|---------|
| 🥇 **99.36% Test Accuracy** | Top-tier classification performance |
| 🎯 **99.97% ROC-AUC** | Near-perfect discrimination |
| ⚡ **4.32s Training** | Efficient model development |
| 🔄 **Zero Overfitting** | Validation ≈ Test performance |
| 📊 **70K Dataset** | Large-scale clinical validation |
| 🏆 **Champion Selection** | Beat 8 competing algorithms |

### Clinical Impact

- **Early Detection**: Identifies 99.36% of high-risk patients
- **Minimal False Negatives**: Only 60 high-risk patients missed in 14,000
- **Confident Predictions**: Mean confidence >99%
- **Fast Inference**: <1ms per prediction

---

## ⚙️ Technical Specifications

### Software Requirements

| Component | Version |
|-----------|---------|
| Python | 3.8+ |
| scikit-learn | 1.0+ |
| numpy | 1.20+ |
| pandas | 1.3+ |
| joblib | 1.0+ |

### Hardware Requirements

| Resource | Training | Inference |
|----------|----------|-----------|
| CPU | 2+ cores | 1 core |
| RAM | 4GB | 1GB |
| Storage | 100MB | 50MB |

### Model Artifacts

| File | Size | Description |
|------|------|-------------|
| `heart-prediction.joblib` | ~15MB | Bundled model, scaler, metadata |

### API Interface

```python
# Input format
patient_data = {
    'Chest_Pain': 1,
    'Shortness_of_Breath': 1,
    'Fatigue': 0,
    'Palpitations': 1,
    'Dizziness': 0,
    'Swelling': 0,
    'Pain_Arms_Jaw_Back': 1,
    'Cold_Sweats_Nausea': 0,
    'High_BP': 1,
    'High_Cholesterol': 1,
    'Diabetes': 0,
    'Smoking': 1,
    'Obesity': 0,
    'Sedentary_Lifestyle': 1,
    'Family_History': 1,
    'Chronic_Stress': 1,
    'Gender': 1,
    'Age': 55
}

# Output format
{
    'prediction': 'High Risk',
    'probability': 0.94,
    'confidence': 0.94,
    'risk_factors': ['Chest_Pain', 'High_BP', ...],
    'success': True
}
```

---

## 🚀 Deployment Status

### Current State

| Aspect | Status |
|--------|--------|
| **Model Training** | ✅ Complete |
| **Validation** | ✅ Passed |
| **Testing** | ✅ Passed |
| **Integration** | ✅ Deployed |
| **Production** | ✅ **LIVE** |

### Deployment Checklist

- [x] Model achieves >95% accuracy threshold
- [x] ROC-AUC exceeds 0.95 clinical requirement
- [x] Model bundled with scaler and metadata
- [x] API integration complete
- [x] Error handling implemented
- [x] Logging and monitoring active
- [x] Documentation finalized

### Production Environment

```python
Model Path: backend/models/Heart Risk Predictive Model/heart-prediction.joblib
Controller: backend/controllers/heart_prediction.py
Status: PRODUCTION READY ✅
```

---

<p align="center">
  <strong>Heart Risk Predictive Model v1.0</strong><br/>
  <em>Hygieia AI Healthcare Platform</em><br/>
  <a href="https://Krishna-Vijay-G.github.io">Krishna Vijay G</a>
</p>
