import numpy as np
import joblib
from pathlib import Path

from src.config import MODELS_DIR

# Human-readable labels keyed by LabelEncoder integer (Advanced=0, Early=1, Moderate=2)
_STAGE_LABELS = {
    0: ("Advanced",  "Stage lanjut — perlu penanganan segera oleh spesialis ginjal."),
    1: ("Early",     "Stage awal — modifikasi gaya hidup dan pantau secara rutin."),
    2: ("Moderate",  "Stage sedang — segera konsultasi ke dokter nefrologi."),
}


def load_models(
    model1_path=None,
    model2_path=None,
    scaler_path=None,
):
    """Load Model 1, Model 2, and scaler from disk.

    Defaults to models/model1_xgb.pkl, models/model2_catboost.pkl,
    and models/scaler.pkl.

    Returns
    -------
    model1, model2, scaler
    """
    model1_path = Path(model1_path) if model1_path else MODELS_DIR / "model1_xgb.pkl"
    model2_path = Path(model2_path) if model2_path else MODELS_DIR / "model2_catboost.pkl"
    scaler_path = Path(scaler_path) if scaler_path else MODELS_DIR / "scaler.pkl"

    model1 = joblib.load(model1_path)
    model2 = joblib.load(model2_path)
    scaler = joblib.load(scaler_path)
    return model1, model2, scaler


def predict_cascaded(model1, model2, X_processed) -> dict:
    """Cascaded inference: Model 1 (binary) → if CKD detected → Model 2 (multiclass).

    Model 1 screens every sample for CKD presence.
    Model 2 predicts severity stage only for CKD-positive samples.
    Non-CKD samples get None for stage fields.

    Parameters
    ----------
    model1      : fitted XGBClassifier
    model2      : fitted CatBoostClassifier
    X_processed : pd.DataFrame, fully preprocessed and scaled feature matrix.
                  Columns must match what the models were trained on.

    Returns
    -------
    dict with lists of length n_samples:
        ckd_present  : int (0 or 1)
        ckd_proba    : float, P(CKD=1) from Model 1
        stage        : int | None  (0=Advanced, 1=Early, 2=Moderate)
        stage_label  : str | None  ("Advanced" / "Early" / "Moderate")
        stage_desc   : str | None  human-readable description
        stage_proba  : np.ndarray | None  per-class probability vector from Model 2
    """
    n = len(X_processed)

    ckd_present = model1.predict(X_processed).tolist()
    ckd_proba   = model1.predict_proba(X_processed)[:, 1].tolist()

    stage       = [None] * n
    stage_label = [None] * n
    stage_desc  = [None] * n
    stage_proba = [None] * n

    ckd_idx = [i for i, v in enumerate(ckd_present) if v == 1]

    if ckd_idx:
        X_ckd         = X_processed.iloc[ckd_idx]
        stage_preds   = model2.predict(X_ckd).ravel().astype(int).tolist()
        stage_probas  = model2.predict_proba(X_ckd)

        for j, i in enumerate(ckd_idx):
            s              = stage_preds[j]
            label, desc    = _STAGE_LABELS[s]
            stage[i]       = s
            stage_label[i] = label
            stage_desc[i]  = desc
            stage_proba[i] = stage_probas[j]

    return {
        "ckd_present": ckd_present,
        "ckd_proba":   ckd_proba,
        "stage":       stage,
        "stage_label": stage_label,
        "stage_desc":  stage_desc,
        "stage_proba": stage_proba,
    }
