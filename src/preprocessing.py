import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, LabelEncoder

from src.config import (
    SURVEY_COLS, LAB_BP_COLS, CONTINUOUS_COLS,
    CLINICAL_BOUNDS, ORDINAL_MAPS, STAGE_GROUPING,
)


# ============================================================
# Step 1 — Survey code cleaning
# ============================================================

def replace_survey_codes(df: pd.DataFrame) -> pd.DataFrame:
    """Replace NHANES 'don't know' (7) and 'refused' (9) codes with NaN."""
    df = df.copy()
    cols = [c for c in SURVEY_COLS if c in df.columns]
    df[cols] = df[cols].replace([7, 9], np.nan)
    return df


# ============================================================
# Step 2 — Row-level filtering (task-specific)
# ============================================================

def remove_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicates (used for binary task)."""
    return df.drop_duplicates(keep="first").reset_index(drop=True)


def remove_invalid_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Remove 'Unknown' and 'No CKD' rows from ckd_stage (multiclass task)."""
    df = df[
        (df["ckd_stage"] != "Unknown") &
        (df["ckd_stage"] != "No CKD")
    ].reset_index(drop=True)
    return df


# ============================================================
# Step 3 — Imputation helpers
# ============================================================

def _fill_median_by_group(df: pd.DataFrame, col: str, group_cols: list) -> pd.DataFrame:
    """Impute col with per-subgroup median; global median as fallback."""
    group_median = df.groupby(group_cols)[col].transform("median")
    df[col] = df[col].fillna(group_median)
    df[col] = df[col].fillna(df[col].median())
    return df


def _fill_mode_by_group(df: pd.DataFrame, col: str, group_cols: list) -> pd.DataFrame:
    """Impute col with per-subgroup mode; global mode as fallback."""
    def _group_mode(x):
        m = x.mode()
        return m.iloc[0] if not m.empty else np.nan

    group_filled = df.groupby(group_cols)[col].transform(
        lambda x: x.fillna(_group_mode(x))
    )
    df[col] = group_filled
    df[col] = df[col].fillna(df[col].mode()[0])
    return df


def impute_diabetes(df: pd.DataFrame) -> pd.DataFrame:
    """Fill insulin_use and diabetes_pills NaN with 0 (no usage assumed)."""
    df = df.copy()
    for col in ["insulin_use", "diabetes_pills"]:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    return df


def impute_lab_bp(df: pd.DataFrame, group_col: str = "ckd_present") -> pd.DataFrame:
    """Fill lab + blood pressure columns.

    Imputation strategy: per-group median (grouped by ``group_col``),
    then global median as fallback.

    Parameters
    ----------
    df        : dataframe that still contains the target/group column
    group_col : column to group by.
                'ckd_present' for binary task (Model 1),
                'ckd_stage'   for multiclass task (Model 2).
    """
    df = df.copy()
    cols = [c for c in LAB_BP_COLS if c in df.columns]
    for col in cols:
        if group_col in df.columns:
            group_median = df.groupby(group_col)[col].transform("median")
            df[col] = df[col].fillna(group_median)
        df[col] = df[col].fillna(df[col].median())
    return df


# ============================================================
# Step 4 — Feature creation
# ============================================================

def create_smoking_status(df: pd.DataFrame) -> pd.DataFrame:
    """Derive smoking_status from ever_smoked + current_smoker; drop originals."""
    df = df.copy()

    def _smoking(row):
        ever    = row.get("ever_smoked")
        current = row.get("current_smoker")
        if ever == 2:
            return "never"
        elif ever == 1 and current == 1:
            return "current"
        elif ever == 1 and current == 2:
            return "former"
        return np.nan

    df["smoking_status"] = df.apply(_smoking, axis=1)
    df.drop(columns=["ever_smoked", "current_smoker"], errors="ignore", inplace=True)
    return df


def create_age_group(df: pd.DataFrame) -> pd.DataFrame:
    """Bin age into child / adult / elderly."""
    df = df.copy()

    def _age_group(age):
        if age < 18:
            return "child"
        elif age < 60:
            return "adult"
        return "elderly"

    df["age_group"] = df["age"].apply(_age_group)
    return df


def impute_body_features(df: pd.DataFrame) -> pd.DataFrame:
    """Fill bmi, weight_kg, height_cm by age_group × gender median."""
    df = df.copy()
    for col in ["bmi", "weight_kg", "height_cm"]:
        if col in df.columns:
            df = _fill_median_by_group(df, col, ["age_group", "gender"])
    return df


def impute_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """Fill education_level (mode) and poverty_income_ratio (median) by subgroup."""
    df = df.copy()
    if "education_level" in df.columns:
        df = _fill_mode_by_group(df, "education_level", ["age_group", "ethnicity"])
    if "poverty_income_ratio" in df.columns:
        df = _fill_median_by_group(df, "poverty_income_ratio", ["age_group", "ethnicity"])
    if "smoking_status" in df.columns:
        df["smoking_status"] = df["smoking_status"].fillna("unknown")
    return df


# ============================================================
# Step 5 — CKD stage grouping (multiclass only)
# ============================================================

def group_ckd_stages(df: pd.DataFrame):
    """Map granular CKD stages to Early/Moderate/Advanced and label-encode.

    Returns
    -------
    df            : dataframe with ckd_stage_grouped (int) replacing ckd_stage
    label_encoder : fitted LabelEncoder (classes_: ['Advanced','Early','Moderate'])
    """
    df = df.copy()
    df["ckd_stage_grouped"] = df["ckd_stage"].replace(STAGE_GROUPING)
    df.drop(columns=["ckd_stage"], inplace=True)

    le = LabelEncoder()
    df["ckd_stage_grouped"] = le.fit_transform(df["ckd_stage_grouped"])
    return df, le


# ============================================================
# Step 6 — Feature engineering
# ============================================================

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create MAP; derive bmi_category; drop redundant columns."""
    df = df.copy()

    # Mean Arterial Pressure
    if "bp_systolic" in df.columns and "bp_diastolic" in df.columns:
        df["MAP"] = (df["bp_systolic"] + 2 * df["bp_diastolic"]) / 3
        df.drop(columns=["bp_systolic", "bp_diastolic"], inplace=True)

    # BMI category
    def _bmi_cat(bmi):
        if pd.isna(bmi):    return np.nan
        if bmi < 18.5:      return "underweight"
        if bmi < 25.0:      return "normal"
        if bmi < 30.0:      return "overweight"
        return "obese"

    if "bmi" in df.columns:
        df["bmi_category"] = df["bmi"].apply(_bmi_cat)
        df.drop(columns=["weight_kg", "bmi"], errors="ignore", inplace=True)

    return df


# ============================================================
# Step 7 — Encoding
# ============================================================

def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ordinal, binary, and one-hot encoding for all non-target features."""
    df = df.copy()

    # Ordinal encoding
    for col, order in ORDINAL_MAPS.items():
        if col in df.columns:
            mapping = {val: idx for idx, val in enumerate(order)}
            df[col] = df[col].map(mapping)

    # Binary encoding
    if "gender" in df.columns:
        df["gender"] = df["gender"].map({"Male": 0, "Female": 1})
    if "diabetes_diagnosed" in df.columns:
        df["diabetes_diagnosed"] = df["diabetes_diagnosed"].map({1: 1, 2: 0})
    for col in ["insulin_use", "diabetes_pills"]:
        if col in df.columns:
            df[col] = df[col].astype(int)

    # One-hot encoding — ethnicity (drop_first to avoid dummy trap)
    if "ethnicity" in df.columns:
        df = pd.get_dummies(df, columns=["ethnicity"], drop_first=True, dtype=int)

    return df


# ============================================================
# Step 8 — Outlier clipping
# ============================================================

def clip_clinical_bounds(df: pd.DataFrame) -> pd.DataFrame:
    """Winsorise columns to clinically plausible ranges."""
    df = df.copy()
    for col, (lo, hi) in CLINICAL_BOUNDS.items():
        if col in df.columns:
            df[col] = df[col].clip(lower=lo, upper=hi)
    return df


# ============================================================
# Scaling
# ============================================================

def fit_scaler(X_train: pd.DataFrame, cols: list = None):
    """Fit RobustScaler on continuous columns of X_train.

    Returns
    -------
    X_train_scaled : transformed copy
    scaler         : fitted RobustScaler
    """
    if cols is None:
        cols = CONTINUOUS_COLS
    cols_present = [c for c in cols if c in X_train.columns]
    scaler = RobustScaler()
    X_out = X_train.copy()
    X_out[cols_present] = scaler.fit_transform(X_out[cols_present])
    return X_out, scaler


def apply_scaler(X: pd.DataFrame, scaler: RobustScaler, cols: list = None) -> pd.DataFrame:
    """Apply a pre-fitted scaler to continuous columns."""
    if cols is None:
        cols = CONTINUOUS_COLS
    cols_present = [c for c in cols if c in X.columns]
    X_out = X.copy()
    X_out[cols_present] = scaler.transform(X_out[cols_present])
    return X_out


# ============================================================
# Orchestration
# ============================================================

def preprocess_pipeline(
    df: pd.DataFrame,
    task: str = "binary",
    lab_bp_group_col: str = None,
):
    """Run the full preprocessing pipeline.

    Parameters
    ----------
    df               : raw dataframe (after drop_irrelevant_cols)
    task             : 'binary' or 'multiclass'
    lab_bp_group_col : column to group lab/BP imputation by.
                       Defaults to 'ckd_present' (binary) or 'ckd_stage' (multiclass).

    Returns
    -------
    df_clean      : fully preprocessed dataframe (target column still present)
    label_encoder : fitted LabelEncoder for multiclass target, else None
    """
    if task not in ("binary", "multiclass"):
        raise ValueError(f"task must be 'binary' or 'multiclass', got '{task}'")

    if lab_bp_group_col is None:
        lab_bp_group_col = "ckd_present" if task == "binary" else "ckd_stage"

    df = df.copy()

    # 1. Survey code cleaning
    df = replace_survey_codes(df)

    # 2. Task-specific row operations
    if task == "binary":
        df = remove_duplicate_rows(df)
    else:
        df = remove_invalid_rows(df)

    # 3. Imputation
    df = impute_diabetes(df)
    df = impute_lab_bp(df, group_col=lab_bp_group_col)
    df = create_smoking_status(df)
    df = create_age_group(df)
    df = impute_body_features(df)
    df = impute_categorical(df)

    # 4. Stage grouping + target encoding (multiclass only)
    label_encoder = None
    if task == "multiclass":
        df, label_encoder = group_ckd_stages(df)

    # 5. Feature engineering
    df = engineer_features(df)

    # 6. Feature encoding
    df = encode_features(df)

    # 7. Outlier clipping
    df = clip_clinical_bounds(df)

    return df, label_encoder
