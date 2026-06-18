import numpy as np
import matplotlib.pyplot as plt
import shap


def get_shap_values(model, X):
    """Compute SHAP values with TreeExplainer.

    Parameters
    ----------
    model : fitted tree model (XGBClassifier or CatBoostClassifier)
    X     : pd.DataFrame, the dataset to explain (typically X_test)

    Returns
    -------
    explainer   : shap.TreeExplainer
    shap_values : np.ndarray
                  Binary     → shape (n_samples, n_features)
                  Multiclass → shape (n_samples, n_features, n_classes)
    """
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    return explainer, np.array(shap_values)


def plot_shap_bar(shap_values, X, title: str = "SHAP Feature Importance"):
    """Global mean |SHAP| bar chart.

    Works for binary (2-D shap_values) and multiclass (3-D).
    """
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.title(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_shap_beeswarm(
    shap_values,
    X,
    class_idx: int = None,
    class_name: str = None,
):
    """Beeswarm (dot) summary plot showing feature-value direction.

    Parameters
    ----------
    shap_values : 2-D for binary, 3-D for multiclass
    class_idx   : required for multiclass — which output class to visualise
    class_name  : optional display label used in title
    """
    if shap_values.ndim == 3:
        if class_idx is None:
            raise ValueError("class_idx is required for multiclass shap_values (shape n×f×c)")
        sv    = shap_values[:, :, class_idx]
        title = f"SHAP Beeswarm — Class: {class_name if class_name is not None else class_idx}"
    else:
        sv    = shap_values
        title = "SHAP Beeswarm"

    shap.summary_plot(sv, X, plot_type="dot", show=False)
    plt.title(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_shap_force(
    explainer,
    shap_values,
    X,
    idx: int = 0,
    class_idx: int = None,
    class_name: str = None,
):
    """Force plot for a single sample (matplotlib mode).

    Parameters
    ----------
    idx       : row index in X to explain
    class_idx : required for multiclass
    class_name: optional display label
    """
    if shap_values.ndim == 3:
        if class_idx is None:
            raise ValueError("class_idx is required for multiclass shap_values")
        sv       = shap_values[idx][:, class_idx]   # shape (n_features,)
        base_val = explainer.expected_value[class_idx]
        title    = f"SHAP Force — Sample {idx} | Class: {class_name if class_name is not None else class_idx}"
    else:
        sv       = shap_values[idx]                  # shape (n_features,)
        base_val = explainer.expected_value
        title    = f"SHAP Force — Sample {idx}"

    shap.force_plot(base_val, sv, X.iloc[idx], matplotlib=True, show=False)
    plt.title(title, pad=20, fontsize=11)
    plt.tight_layout()
    plt.show()
