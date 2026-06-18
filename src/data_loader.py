import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

from src.config import (
    KAGGLE_DATASET, KAGGLE_FILE,
    BIOMARKER_COLS, DATA_RAW,
    RANDOM_STATE, TEST_SIZE,
)

# Columns dropped on top of BIOMARKER_COLS, depending on task
_BINARY_EXTRA     = ["ckd_stage"]       # keep ckd_present as target
_MULTICLASS_EXTRA = ["ckd_present"]     # keep ckd_stage for grouping later


def load_raw(use_kaggle: bool = True, local_path: str = None) -> pd.DataFrame:
    """Load CKD NHANES dataset.

    Parameters
    ----------
    use_kaggle : bool
        If True, download via kagglehub (requires Kaggle credentials).
    local_path : str | None
        Path to local CSV. Used when use_kaggle=False, or as fallback.

    Returns
    -------
    pd.DataFrame  Raw, unprocessed dataframe.
    """
    if use_kaggle:
        try:
            import kagglehub
            from kagglehub import KaggleDatasetAdapter
            df = kagglehub.load_dataset(
                KaggleDatasetAdapter.PANDAS,
                KAGGLE_DATASET,
                KAGGLE_FILE,
            )
            return df
        except Exception as e:
            print(f"[data_loader] Kaggle download failed ({e}), falling back to local path.")

    # Local fallback
    if local_path is None:
        local_path = DATA_RAW / KAGGLE_FILE
    df = pd.read_csv(local_path)
    return df


def drop_irrelevant_cols(df: pd.DataFrame, task: str = "binary") -> pd.DataFrame:
    """Remove biomarker leakage columns and the opposite-task target column.

    Parameters
    ----------
    df   : raw dataframe
    task : 'binary'      → drop ckd_stage, keep ckd_present
           'multiclass'  → drop ckd_present, keep ckd_stage

    Returns
    -------
    pd.DataFrame with irrelevant columns removed.
    """
    if task == "binary":
        to_drop = BIOMARKER_COLS + _BINARY_EXTRA
    elif task == "multiclass":
        to_drop = BIOMARKER_COLS + _MULTICLASS_EXTRA
    else:
        raise ValueError(f"task must be 'binary' or 'multiclass', got '{task}'")

    existing = [c for c in to_drop if c in df.columns]
    return df.drop(columns=existing)


def split_data(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
):
    """Stratified train/test split.

    Parameters
    ----------
    df         : fully preprocessed dataframe (target column still present)
    target_col : name of the target column
    test_size  : fraction for test set (default 0.2)
    random_state : seed

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
