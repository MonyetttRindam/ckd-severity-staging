# CKD Severity Staging

[![Streamlit App](https://img.shields.io/badge/Streamlit-Live%20Demo-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://ckd-screening.streamlit.app/)

A cascaded machine learning pipeline for chronic kidney disease (CKD) detection and severity staging.

## Overview

- **Model 1 — XGBoost Binary Classifier**: Predicts CKD presence (CKD vs Non-CKD).
- **Model 2 — CatBoost Multiclass Classifier**: Given a positive CKD prediction, classifies severity into Early, Moderate, or Advanced stage.

The cascaded architecture means Model 2 is only invoked when Model 1 flags a patient as CKD-positive.

## 🚀 Live Demo

Try the interactive app: **[ckd-screening.streamlit.app](https://ckd-screening.streamlit.app/)**

Input patient parameters (age, blood pressure, lab values) and get:
- CKD presence prediction (Model 1)
- Severity stage classification (Model 2)
- SHAP-based explanation of the prediction

## Results

The following metrics were obtained on held-out test sets using Stratified 5-Fold Cross-Validation. ROC-AUC for the multiclass model uses One-vs-Rest (OvR) macro averaging. SHAP was used for global and local feature interpretability. Hyperparameter tuning was performed via Optuna (100 trials per model).

### Model 1 — XGBoost Binary Classifier (CKD vs Non-CKD)

| Metric | Value |
|---|---|
| Accuracy | 0.8859 |
| Macro F1 | 0.8674 |
| Cohen's Kappa | 0.7349 |
| ROC-AUC | 0.9471 |

### Model 2 — CatBoost Multiclass Staging (Early / Moderate / Advanced)

| Metric | Value |
|---|---|
| Accuracy | 0.6106 |
| Macro F1 | 0.5710 |
| Cohen's Kappa | 0.3413 |
| ROC-AUC OvR (macro) | 0.7848 |

#### Per-Class Sensitivity & Specificity (Model 2)

| Class | Sensitivity | Specificity |
|---|---|---|
| Advanced | 0.698 | 0.747 |
| Early | 0.658 | 0.877 |
| Moderate | 0.579 | 0.762 |

## Reproducibility

Results computed using Python 3.12.10, library versions as pinned in `requirements.txt`. Minor metric variation may occur across environments due to library version differences.
