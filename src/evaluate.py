import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    cohen_kappa_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_class_weight

from src.config import RANDOM_STATE


# ============================================================
# Cross-validation
# ============================================================

def stratified_cv_binary(
    model,
    X,
    y,
    n_splits: int = 5,
    verbose: bool = True,
) -> dict:
    """Stratified k-fold CV for binary classification (Model 1).

    Metrics per fold: accuracy, macro_f1, ckd_recall (class 1),
    nonckd_recall (class 0), kappa.

    Returns
    -------
    dict  {metric: {"mean": float, "std": float}}
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    fold_scores = {m: [] for m in ["accuracy", "macro_f1", "ckd_recall", "nonckd_recall", "kappa"]}

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr,  X_val  = X.iloc[tr_idx],  X.iloc[val_idx]
        y_tr,  y_val  = y.iloc[tr_idx],  y.iloc[val_idx]

        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)
        report = classification_report(y_val, y_pred, output_dict=True, zero_division=0)

        acc      = accuracy_score(y_val, y_pred)
        macro_f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)
        kappa    = cohen_kappa_score(y_val, y_pred)
        ckd_r    = report.get("1", {}).get("recall", 0)
        nonckd_r = report.get("0", {}).get("recall", 0)

        fold_scores["accuracy"].append(acc)
        fold_scores["macro_f1"].append(macro_f1)
        fold_scores["ckd_recall"].append(ckd_r)
        fold_scores["nonckd_recall"].append(nonckd_r)
        fold_scores["kappa"].append(kappa)

        if verbose:
            print(
                f"Fold {fold} | Acc: {acc:.3f} | Macro F1: {macro_f1:.3f} | "
                f"CKD Recall: {ckd_r:.3f} | No-CKD Recall: {nonckd_r:.3f} | Kappa: {kappa:.3f}"
            )

    results = {k: {"mean": np.mean(v), "std": np.std(v)} for k, v in fold_scores.items()}
    if verbose:
        print(f"\n{'─'*60}")
        print("Average:")
        for m, s in results.items():
            print(f"  {m:<20}: {s['mean']:.4f} ± {s['std']:.4f}")
    return results


def stratified_cv_multiclass(
    model,
    X,
    y,
    n_splits: int = 5,
    verbose: bool = True,
) -> dict:
    """Stratified k-fold CV for multiclass (Model 2).

    Class weights are recomputed from each fold's y_fold_train and passed as
    sample_weight to model.fit — consistent with NB04 Cell 26/27.

    Metrics: accuracy, macro_f1, advanced_recall (0), early_recall (1),
    moderate_recall (2), kappa.

    Returns
    -------
    dict  {metric: {"mean": float, "std": float}}
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    keys = ["accuracy", "macro_f1", "advanced_recall", "early_recall", "moderate_recall", "kappa"]
    fold_scores = {m: [] for m in keys}

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr,  X_val  = X.iloc[tr_idx],  X.iloc[val_idx]
        y_tr,  y_val  = y.iloc[tr_idx],  y.iloc[val_idx]

        fold_classes = np.unique(y_tr)
        fold_weights = compute_class_weight(class_weight="balanced", classes=fold_classes, y=y_tr)
        fold_cw_dict = dict(zip(fold_classes.tolist(), [float(w) for w in fold_weights]))
        sample_w     = np.array([fold_cw_dict[c] for c in y_tr])

        model.fit(X_tr, y_tr, sample_weight=sample_w)
        y_pred = model.predict(X_val).ravel()
        report = classification_report(y_val, y_pred, output_dict=True, zero_division=0)

        acc      = accuracy_score(y_val, y_pred)
        macro_f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)
        kappa    = cohen_kappa_score(y_val, y_pred)
        adv_r    = report.get("0", {}).get("recall", 0)
        ear_r    = report.get("1", {}).get("recall", 0)
        mod_r    = report.get("2", {}).get("recall", 0)

        fold_scores["accuracy"].append(acc)
        fold_scores["macro_f1"].append(macro_f1)
        fold_scores["advanced_recall"].append(adv_r)
        fold_scores["early_recall"].append(ear_r)
        fold_scores["moderate_recall"].append(mod_r)
        fold_scores["kappa"].append(kappa)

        if verbose:
            print(
                f"Fold {fold} | Acc: {acc:.3f} | Macro F1: {macro_f1:.3f} | "
                f"Adv: {adv_r:.3f} | Early: {ear_r:.3f} | Mod: {mod_r:.3f} | Kappa: {kappa:.3f}"
            )

    results = {k: {"mean": np.mean(v), "std": np.std(v)} for k, v in fold_scores.items()}
    if verbose:
        print(f"\n{'─'*60}")
        print("Average:")
        for m, s in results.items():
            print(f"  {m:<22}: {s['mean']:.4f} ± {s['std']:.4f}")
    return results


# ============================================================
# Hold-out evaluation
# ============================================================

def eval_binary(model, X_test, y_test) -> dict:
    """Evaluate binary model on hold-out test set.

    Returns accuracy, macro_f1, kappa, roc_auc, and full classification_report dict.
    """
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    report  = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "macro_f1": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "kappa":    cohen_kappa_score(y_test, y_pred),
        "roc_auc":  roc_auc_score(y_test, y_proba),
        "report":   report,
    }


def eval_multiclass(model, X_test, y_test, class_names: list) -> dict:
    """Evaluate multiclass model on hold-out test set.

    Returns accuracy, macro_f1, kappa, macro ROC-AUC OvR, and classification_report.
    """
    y_pred  = model.predict(X_test).ravel()
    y_proba = model.predict_proba(X_test)
    n       = len(class_names)
    y_bin   = label_binarize(y_test, classes=list(range(n)))
    report  = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    return {
        "accuracy":    accuracy_score(y_test, y_pred),
        "macro_f1":    f1_score(y_test, y_pred, average="macro", zero_division=0),
        "kappa":       cohen_kappa_score(y_test, y_pred),
        "roc_auc_ovr": roc_auc_score(y_bin, y_proba, multi_class="ovr", average="macro"),
        "report":      report,
    }


def sensitivity_specificity(y_true, y_pred, class_names: list):
    """Print sensitivity (recall) and specificity per class using OvR binarisation."""
    n          = len(class_names)
    y_true_bin = label_binarize(y_true, classes=list(range(n)))
    y_pred_bin = label_binarize(y_pred, classes=list(range(n)))

    print(f"\n{'Class':<14} {'Sensitivity':>12} {'Specificity':>12}")
    print("─" * 40)
    for i, name in enumerate(class_names):
        TP = ((y_pred_bin[:, i] == 1) & (y_true_bin[:, i] == 1)).sum()
        TN = ((y_pred_bin[:, i] == 0) & (y_true_bin[:, i] == 0)).sum()
        FP = ((y_pred_bin[:, i] == 1) & (y_true_bin[:, i] == 0)).sum()
        FN = ((y_pred_bin[:, i] == 0) & (y_true_bin[:, i] == 1)).sum()
        sens = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        spec = TN / (TN + FP) if (TN + FP) > 0 else 0.0
        print(f"{name:<14} {sens:>12.3f} {spec:>12.3f}")


# ============================================================
# Plots
# ============================================================

def plot_confusion_matrix(
    y_true,
    y_pred,
    class_names: list,
    figsize=(14, 5),
):
    """Side-by-side count and normalised confusion matrices (NB04 Cell 31 style)."""
    cm      = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    for ax, data, fmt, title in zip(
        axes,
        [cm,        cm_norm],
        ["d",       ".2f"],
        ["Confusion Matrix (Count)", "Confusion Matrix (Normalized)"],
    ):
        sns.heatmap(
            data, annot=True, fmt=fmt, cmap="Blues",
            xticklabels=class_names, yticklabels=class_names,
            ax=ax, linewidths=0.5,
        )
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Predicted Label", fontsize=11)
        ax.set_ylabel("True Label",      fontsize=11)

    plt.tight_layout()
    return fig


def plot_roc_auc_ovr(
    model,
    X_test,
    y_test,
    class_names: list,
    figsize=(16, 6),
):
    """Per-class and macro-average ROC curves (OvR) — NB04 Cell 32 style.

    Prints AUC per class and macro average. Returns matplotlib Figure.
    """
    n_classes = len(class_names)
    y_proba   = model.predict_proba(X_test)
    y_bin     = label_binarize(y_test, classes=list(range(n_classes)))
    colors    = ["#C44E52", "#4C72B0", "#55A868"][:n_classes]

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # ── Per-class ──
    auc_scores = []
    for i, (name, color) in enumerate(zip(class_names, colors)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        roc_auc_val = auc(fpr, tpr)
        auc_scores.append(roc_auc_val)
        axes[0].plot(fpr, tpr, color=color, linewidth=2, label=f"{name} (AUC={roc_auc_val:.3f})")
    axes[0].plot([0, 1], [0, 1], "k--", linewidth=1, label="Random Classifier")
    axes[0].set_title("ROC Curve — Per Class (One-vs-Rest)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("False Positive Rate"); axes[0].set_ylabel("True Positive Rate")
    axes[0].legend(fontsize=10); axes[0].grid(alpha=0.3)
    axes[0].set_xlim(0, 1); axes[0].set_ylim(0, 1.02)

    # ── Macro average ──
    all_fpr  = np.unique(np.concatenate([
        roc_curve(y_bin[:, i], y_proba[:, i])[0] for i in range(n_classes)
    ]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        mean_tpr += np.interp(all_fpr, fpr, tpr)
    mean_tpr  /= n_classes
    macro_auc  = auc(all_fpr, mean_tpr)

    axes[1].plot(all_fpr, mean_tpr, color="navy", linewidth=2.5,
                 label=f"Macro Average (AUC={macro_auc:.3f})")
    for i, (name, color) in enumerate(zip(class_names, colors)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        axes[1].plot(fpr, tpr, color=color, linewidth=1.2, alpha=0.5, linestyle="--", label=name)
    axes[1].plot([0, 1], [0, 1], "k--", linewidth=1, label="Random Classifier")
    axes[1].set_title("ROC Curve — Macro Average", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("False Positive Rate"); axes[1].set_ylabel("True Positive Rate")
    axes[1].legend(fontsize=10); axes[1].grid(alpha=0.3)
    axes[1].set_xlim(0, 1); axes[1].set_ylim(0, 1.02)

    plt.tight_layout()

    print("\nROC-AUC per class:")
    for name, s in zip(class_names, auc_scores):
        print(f"  {name:<12}: {s:.4f}")
    print(f"  Macro Avg   : {macro_auc:.4f}")

    return fig
