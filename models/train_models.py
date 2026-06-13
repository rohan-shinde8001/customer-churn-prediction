"""
models/train_models.py
Trains Logistic Regression, Decision Tree, Random Forest, and XGBoost.
Includes cross-validation, hyperparameter tuning, and SHAP explainability.
"""

import os, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model   import LogisticRegression
from sklearn.tree           import DecisionTreeClassifier
from sklearn.ensemble       import RandomForestClassifier
from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                      cross_val_score, RandomizedSearchCV)
from sklearn.metrics        import (accuracy_score, precision_score,
                                     recall_score, f1_score, roc_auc_score,
                                     confusion_matrix, roc_curve,
                                     classification_report)

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

warnings.filterwarnings("ignore")

REPORTS_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")
MODELS_DIR  = os.path.dirname(__file__)
os.makedirs(REPORTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _evaluate(name, model, X_test, y_test) -> dict:
    y_pred  = model.predict(X_test)
    y_prob  = model.predict_proba(X_test)[:, 1]
    metrics = {
        "model"    : name,
        "accuracy" : round(accuracy_score (y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall"   : round(recall_score   (y_test, y_pred), 4),
        "f1"       : round(f1_score       (y_test, y_pred), 4),
        "roc_auc"  : round(roc_auc_score  (y_test, y_prob), 4),
    }
    print(f"  {name:<25}  acc={metrics['accuracy']}  f1={metrics['f1']}  auc={metrics['roc_auc']}")
    return metrics


def _plot_confusion(name, model, X_test, y_test):
    cm = confusion_matrix(y_test, model.predict(X_test))
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Churn", "Churn"],
                yticklabels=["No Churn", "Churn"], ax=ax)
    ax.set_title(f"Confusion Matrix – {name}")
    ax.set_ylabel("Actual"); ax.set_xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, f"cm_{name.replace(' ','_')}.png"), dpi=120)
    plt.close()


def _plot_roc(results, X_test, y_test, models_dict):
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, model in models_dict.items():
        fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
        auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0,1],[0,1],"k--", label="Random")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves – All Models")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "roc_all_models.png"), dpi=120)
    plt.close()


def _plot_feature_importance(model, feat_cols, name):
    if hasattr(model, "feature_importances_"):
        imp = pd.Series(model.feature_importances_, index=feat_cols).nlargest(15)
    elif hasattr(model, "coef_"):
        imp = pd.Series(np.abs(model.coef_[0]), index=feat_cols).nlargest(15)
    else:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    imp.sort_values().plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title(f"Feature Importance – {name}")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, f"feat_imp_{name.replace(' ','_')}.png"), dpi=120)
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# SHAP explainability
# ─────────────────────────────────────────────────────────────────────────────
def shap_analysis(model, X_test, feat_cols, model_name="XGBoost"):
    if not SHAP_AVAILABLE:
        print("[shap]  SHAP not installed – skipping.")
        return
    try:
        explainer = shap.Explainer(model)
        shap_values = explainer(X_test[:200])

        # Summary plot
        fig = plt.figure(figsize=(9, 6))
        shap.summary_plot(shap_values, X_test[:200],
                          feature_names=feat_cols, show=False)
        plt.title(f"SHAP Summary – {model_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(REPORTS_DIR, "shap_summary.png"), dpi=120)
        plt.close()

        # Bar plot
        fig = plt.figure(figsize=(8, 5))
        shap.summary_plot(shap_values, X_test[:200],
                          feature_names=feat_cols,
                          plot_type="bar", show=False)
        plt.title(f"SHAP Feature Importance – {model_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(REPORTS_DIR, "shap_bar.png"), dpi=120)
        plt.close()
        print("[shap]  Plots saved.")
    except Exception as e:
        print(f"[shap]  Error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────────────────────────────────────
def train_all(X_scaled, y, feat_cols):
    X = X_scaled.values if hasattr(X_scaled, "values") else X_scaled
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n[split]  train={len(y_tr)}  test={len(y_te)}")

    # ── Base models ──────────────────────────────────────────────────────────
    base_models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree"      : DecisionTreeClassifier(max_depth=6, random_state=42),
        "Random Forest"      : RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    }
    if XGB_AVAILABLE:
        base_models["XGBoost"] = XGBClassifier(
            n_estimators=100, use_label_encoder=False,
            eval_metric="logloss", random_state=42
        )

    print("\n── Base model evaluation ──────────────────────────────────────────")
    results, trained = [], {}
    for name, model in base_models.items():
        model.fit(X_tr, y_tr)
        results.append(_evaluate(name, model, X_te, y_te))
        _plot_confusion(name, model, X_te, y_te)
        _plot_feature_importance(model, feat_cols, name)
        trained[name] = model

    _plot_roc(results, X_te, y_te, trained)

    # ── Hyperparameter tuning (Random Forest) ────────────────────────────────
    print("\n── Hyperparameter tuning: Random Forest ──────────────────────────")
    param_grid = {
        "n_estimators"     : [100, 200, 300],
        "max_depth"        : [None, 6, 10, 15],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf" : [1, 2, 4],
    }
    rf_tuned = RandomizedSearchCV(
        RandomForestClassifier(random_state=42, n_jobs=-1),
        param_distributions=param_grid,
        n_iter=15, cv=3, scoring="roc_auc",
        random_state=42, verbose=0, n_jobs=-1
    )
    rf_tuned.fit(X_tr, y_tr)
    best_rf = rf_tuned.best_estimator_
    print(f"  Best params: {rf_tuned.best_params_}")
    results.append(_evaluate("RF (Tuned)", best_rf, X_te, y_te))
    trained["RF (Tuned)"] = best_rf

    # ── Cross-validation ─────────────────────────────────────────────────────
    print("\n── 5-Fold CV AUC ──────────────────────────────────────────────────")
    for name, model in {"Random Forest": trained["Random Forest"],
                         "RF (Tuned)"  : best_rf}.items():
        cv_scores = cross_val_score(model, X_tr, y_tr, cv=5,
                                     scoring="roc_auc", n_jobs=-1)
        print(f"  {name:<20}  mean={cv_scores.mean():.4f}  std={cv_scores.std():.4f}")

    # ── SHAP (best model) ────────────────────────────────────────────────────
    best_name = "XGBoost" if XGB_AVAILABLE else "RF (Tuned)"
    shap_analysis(trained[best_name], X_te, feat_cols, best_name)

    # ── Save artefacts ────────────────────────────────────────────────────────
    best_model = trained[best_name]
    with open(os.path.join(MODELS_DIR, "best_model.pkl"), "wb") as f:
        pickle.dump(best_model, f)
    print(f"\n[save]  best_model.pkl  ({best_name})")

    # Comparison chart
    res_df = pd.DataFrame(results)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    metrics_to_plot = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    res_df.set_index("model")[metrics_to_plot].T.plot(
        kind="bar", ax=axes[0], colormap="tab10")
    axes[0].set_title("Model Comparison – All Metrics")
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=30, ha="right")
    axes[0].set_ylim(0, 1.05); axes[0].legend(fontsize=7)

    res_df.set_index("model")["roc_auc"].plot(
        kind="barh", ax=axes[1], color="steelblue")
    axes[1].set_title("ROC-AUC Comparison")
    axes[1].set_xlim(0.5, 1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "model_comparison.png"), dpi=120)
    plt.close()

    res_df.to_csv(os.path.join(REPORTS_DIR, "model_metrics.csv"), index=False)
    print("[save]  model_metrics.csv + model_comparison.png")

    return trained, best_model, results, X_te, y_te
