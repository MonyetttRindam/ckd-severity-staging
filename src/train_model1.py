import numpy as np
import joblib
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score, make_scorer

from src.config import (
    XGB_BEST_PARAMS, XGB_OPTUNA_SPACE,
    OPTUNA_TRIALS_XGB, RANDOM_STATE,
    MODELS_DIR,
)


def build_xgb(params: dict = None) -> XGBClassifier:
    """Instantiate XGBClassifier.

    Uses XGB_BEST_PARAMS (Optuna result from NB03) as base.
    Any key in ``params`` overrides the default.
    """
    p = {**XGB_BEST_PARAMS, **(params or {})}
    return XGBClassifier(**p)


def tune_xgb(X_train, y_train, n_trials: int = OPTUNA_TRIALS_XGB) -> dict:
    """Optuna hyperparameter search for XGBoost.

    Objective  : maximise macro F1 via 5-fold cross-validation.
    Search space: defined in config.XGB_OPTUNA_SPACE.

    Returns
    -------
    dict  best_params from study (tuned keys only, without fixed keys like
          objective/eval_metric which are appended in build_xgb).
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    sp = XGB_OPTUNA_SPACE
    scorer = make_scorer(f1_score, average="macro")

    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators",     *sp["n_estimators"]),
            "max_depth":        trial.suggest_int("max_depth",         *sp["max_depth"]),
            "learning_rate":    trial.suggest_float("learning_rate",   *sp["learning_rate"],    log=True),
            "subsample":        trial.suggest_float("subsample",       *sp["subsample"]),
            "colsample_bytree": trial.suggest_float("colsample_bytree",*sp["colsample_bytree"]),
            "min_child_weight": trial.suggest_int("min_child_weight",  *sp["min_child_weight"]),
            "gamma":            trial.suggest_float("gamma",           *sp["gamma"]),
            "reg_alpha":        trial.suggest_float("reg_alpha",       *sp["reg_alpha"]),
            "reg_lambda":       trial.suggest_float("reg_lambda",      *sp["reg_lambda"]),
            "objective":    "binary:logistic",
            "eval_metric":  "logloss",
            "random_state": RANDOM_STATE,
            "n_jobs":       -1,
        }
        model = XGBClassifier(**params)
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring=scorer, n_jobs=-1)
        return np.mean(scores)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    print(f"Best macro F1 : {study.best_value:.4f}")
    print("Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    return study.best_params


def train_model1(X_train, y_train, params: dict = None) -> XGBClassifier:
    """Fit the final XGBoost binary classifier.

    Parameters
    ----------
    X_train : scaled feature matrix
    y_train : binary target (ckd_present: 0/1)
    params  : dict of XGBoost params.  None → uses XGB_BEST_PARAMS (Optuna result).
    """
    model = build_xgb(params)
    model.fit(X_train, y_train)
    return model


def save_model1(
    model,
    scaler,
    model_path=None,
    scaler_path=None,
):
    """Persist Model 1 and its scaler with joblib.

    Defaults to models/model1_xgb.pkl and models/scaler.pkl.
    """
    model_path  = Path(model_path)  if model_path  else MODELS_DIR / "model1_xgb.pkl"
    scaler_path = Path(scaler_path) if scaler_path else MODELS_DIR / "scaler.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model,  model_path)
    joblib.dump(scaler, scaler_path)
    print(f"Model  saved → {model_path}")
    print(f"Scaler saved → {scaler_path}")
