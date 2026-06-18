import numpy as np
import joblib
from pathlib import Path
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import recall_score
from sklearn.utils.class_weight import compute_class_weight

from src.config import (
    CATBOOST_DEFAULT_PARAMS, CATBOOST_OPTUNA_SPACE,
    OPTUNA_TRIALS_CAT, RANDOM_STATE,
    MODELS_DIR,
)


def compute_sample_weights(y_train):
    """Compute balanced class weights from y_train.

    Returns
    -------
    sample_weights : np.ndarray, per-sample weight array (same length as y_train)
    cw_list        : list[float], per-class weights sorted by class label.
                     Passed directly to CatBoostClassifier(class_weights=cw_list).
                     Explicit float conversion avoids CatBoost/Optuna clone errors.
    """
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    cw_list = [float(w) for w in weights]
    cw_dict = dict(zip(classes.tolist(), cw_list))
    sample_weights = np.array([cw_dict[c] for c in y_train])
    return sample_weights, cw_list


def build_catboost(params: dict = None, class_weights_list: list = None) -> CatBoostClassifier:
    """Instantiate CatBoostClassifier.

    Uses CATBOOST_DEFAULT_PARAMS as base; any key in ``params`` overrides.
    ``class_weights_list`` is injected as the class_weights parameter.
    """
    p = {**CATBOOST_DEFAULT_PARAMS, **(params or {})}
    if class_weights_list is not None:
        p["class_weights"] = class_weights_list
    return CatBoostClassifier(**p)


def tune_catboost(X_train, y_train, n_trials: int = OPTUNA_TRIALS_CAT) -> dict:
    """Optuna hyperparameter search for CatBoost.

    Objective  : maximise macro recall.
    CV strategy: manual StratifiedKFold (5-fold) — NOT cross_val_score —
                 because class weights must be recomputed from each fold's
                 y_fold_train to avoid data leakage into the weight calculation.
    Search space: config.CATBOOST_OPTUNA_SPACE.

    Returns
    -------
    dict  best_params (tuned keys only; loss_function/class_weights/verbose/
          random_seed are handled by train_model2).
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    _, cw_list = compute_sample_weights(y_train)
    sp = CATBOOST_OPTUNA_SPACE

    def objective(trial):
        params = {
            "iterations":          trial.suggest_int(  "iterations",          *sp["iterations"]),
            "depth":               trial.suggest_int(  "depth",               *sp["depth"]),
            "learning_rate":       trial.suggest_float("learning_rate",        *sp["learning_rate"],       log=True),
            "l2_leaf_reg":         trial.suggest_float("l2_leaf_reg",          *sp["l2_leaf_reg"]),
            "random_strength":     trial.suggest_float("random_strength",      *sp["random_strength"]),
            "bagging_temperature": trial.suggest_float("bagging_temperature",  *sp["bagging_temperature"]),
            "border_count":        trial.suggest_int(  "border_count",         *sp["border_count"]),
            "class_weights":       cw_list,
            "loss_function":       "MultiClass",
            "verbose":             0,
            "random_seed":         RANDOM_STATE,
        }
        skf    = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        scores = []
        for tr_idx, val_idx in skf.split(X_train, y_train):
            X_f_tr,  X_f_val  = X_train.iloc[tr_idx],  X_train.iloc[val_idx]
            y_f_tr,  y_f_val  = y_train.iloc[tr_idx],  y_train.iloc[val_idx]
            model = CatBoostClassifier(**params)
            model.fit(X_f_tr, y_f_tr)
            y_pred = model.predict(X_f_val).ravel()
            scores.append(recall_score(y_f_val, y_pred, average="macro", zero_division=0))
        return np.mean(scores)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"Best macro recall : {study.best_value:.4f}")
    print("Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    return study.best_params


def train_model2(X_train, y_train, params: dict = None) -> CatBoostClassifier:
    """Fit the final CatBoost multiclass model with balanced class weights.

    Parameters
    ----------
    X_train : scaled feature matrix
    y_train : encoded multiclass target (0=Advanced, 1=Early, 2=Moderate)
    params  : Optuna best_params dict.  None → uses CATBOOST_DEFAULT_PARAMS.
    """
    _, cw_list = compute_sample_weights(y_train)
    final_params = {
        **(params or CATBOOST_DEFAULT_PARAMS),
        "class_weights": cw_list,
        "loss_function": "MultiClass",
        "verbose":       0,
        "random_seed":   RANDOM_STATE,
    }
    model = CatBoostClassifier(**final_params)
    model.fit(X_train, y_train)
    return model


def save_model2(model, path=None):
    """Persist Model 2 with joblib. Defaults to models/model2_catboost.pkl."""
    path = Path(path) if path else MODELS_DIR / "model2_catboost.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(f"Model saved → {path}")
