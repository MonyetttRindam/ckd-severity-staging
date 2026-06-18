from pathlib import Path

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
ROOT_DIR   = Path(__file__).parent.parent
DATA_RAW   = ROOT_DIR / "data" / "raw"
MODELS_DIR = ROOT_DIR / "models"

# ------------------------------------------------------------
# Kaggle source
# ------------------------------------------------------------
KAGGLE_DATASET = "alitaqishah/ckd-nhanes-2021-2023-staged-kidney-disease"
KAGGLE_FILE    = "CKD_NHANES_2021_2023.csv"

# ------------------------------------------------------------
# Experiment settings
# ------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE    = 0.2

# ------------------------------------------------------------
# Feature lists
# ------------------------------------------------------------
BIOMARKER_COLS = [
    "serum_creatinine",
    "blood_urea_nitrogen",
    "albumin_serum",
    "urine_albumin",
    "urine_creatinine",
    "albumin_creatinine_ratio",
    "egfr",
    "participant_id",
]

SURVEY_COLS = [
    "insulin_use",
    "diabetes_pills",
    "ever_smoked",
    "current_smoker",
    "diabetes_diagnosed",
    "education_level",
]

LAB_BP_COLS = [
    "phosphorus",
    "bicarbonate",
    "calcium",
    "uric_acid",
    "bp_systolic",
    "bp_diastolic",
]

CONTINUOUS_COLS = [
    "age",
    "poverty_income_ratio",
    "height_cm",
    "phosphorus",
    "bicarbonate",
    "calcium",
    "uric_acid",
    "MAP",
]

# ------------------------------------------------------------
# Clinical winsorising bounds
# ------------------------------------------------------------
CLINICAL_BOUNDS = {
    "height_cm":    (100.0, 210.0),
    "uric_acid":    (1.5,   13.0),
    "calcium":      (7.0,   12.0),
    "bicarbonate":  (10.0,  40.0),
    "phosphorus":   (1.0,   9.0),
}

# ------------------------------------------------------------
# Encoding maps
# ------------------------------------------------------------
ORDINAL_MAPS = {
    "education_level": [1, 2, 3, 4, 5],
    "age_group":       ["child", "adult", "elderly"],
    "bmi_category":    ["underweight", "normal", "overweight", "obese"],
    "smoking_status":  ["never", "unknown", "former", "current"],
}

# CKD stage → grouped label (Model 2)
STAGE_GROUPING = {
    "Stage 1 (Kidney Damage)":       "Early",
    "Stage 2 (Mildly Decreased)":    "Moderate",
    "Stage 3a (Mild-Moderate)":      "Advanced",
    "Stage 3b (Moderate-Severe)":    "Advanced",
    "Stage 4 (Severely Decreased)":  "Advanced",
    "Stage 5 (Kidney Failure)":      "Advanced",
}

# ------------------------------------------------------------
# Model 1 — XGBoost Binary (best Optuna params, NB03 Cell 31)
# ------------------------------------------------------------
XGB_BEST_PARAMS = {
    "n_estimators":      672,
    "max_depth":         9,
    "learning_rate":     0.17132271340481314,
    "subsample":         0.8875357833693124,
    "colsample_bytree":  0.7014339417995729,
    "min_child_weight":  4,
    "gamma":             4.264887033025956,
    "reg_alpha":         0.4661602980925874,
    "reg_lambda":        2.001723487924884,
    "objective":         "binary:logistic",
    "eval_metric":       "logloss",
    "random_state":      RANDOM_STATE,
    "n_jobs":            -1,
}

# Optuna search space bounds for XGBoost
XGB_OPTUNA_SPACE = {
    "n_estimators":      (100, 800),
    "max_depth":         (3, 10),
    "learning_rate":     (0.01, 0.3),   # log=True
    "subsample":         (0.6, 1.0),
    "colsample_bytree":  (0.6, 1.0),
    "min_child_weight":  (1, 15),
    "gamma":             (0.0, 5.0),
    "reg_alpha":         (0.0, 5.0),
    "reg_lambda":        (0.0, 10.0),
}

# ------------------------------------------------------------
# Model 2 — CatBoost Multiclass (default params for quick run)
# ------------------------------------------------------------
CATBOOST_DEFAULT_PARAMS = {
    "iterations":    300,
    "depth":         4,
    "learning_rate": 0.05,
    "loss_function": "MultiClass",
    "random_seed":   RANDOM_STATE,
    "verbose":       0,
}

# Optuna search space bounds for CatBoost
CATBOOST_OPTUNA_SPACE = {
    "iterations":          (200, 1000),
    "depth":               (4, 10),
    "learning_rate":       (0.01, 0.3),    # log=True
    "l2_leaf_reg":         (1.0, 10.0),
    "random_strength":     (0.0, 10.0),
    "bagging_temperature": (0.0, 10.0),
    "border_count":        (32, 255),
}

# ------------------------------------------------------------
# Optuna trials
# ------------------------------------------------------------
OPTUNA_TRIALS_XGB = 50
OPTUNA_TRIALS_CAT = 100
