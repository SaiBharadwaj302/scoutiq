"""
models/xg/train.py

Trains the xG 2.0 stacked ensemble:
  Base model:  LightGBM (captures non-linear 360 spatial patterns)
  Meta model:  Logistic Regression (calibrates probability output)

Every run is logged to MLflow with params, metrics, SHAP plots,
and the model artifact registered in the MLflow model registry.

Run directly:
    python -m models.xg.train
"""
import os
import warnings
from typing import Any

import mlflow
import mlflow.sklearn
import pandas as pd
import shap
import matplotlib.pyplot as plt
from loguru import logger
from sklearn.calibration import calibration_curve
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, brier_score_loss,
    average_precision_score, log_loss,
)
import lightgbm as lgb
from dotenv import load_dotenv

from models.xg.features import load_training_data, get_feature_names

warnings.filterwarnings("ignore")
load_dotenv()

# ── MLflow Setup ──────────────────────────────────────────────────────────────
MLFLOW_URI      = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
EXPERIMENT_NAME = "scoutiq_xg"
MODEL_NAME      = "scoutiq_xg"

# ── Hyperparameters ───────────────────────────────────────────────────────────
LGBM_PARAMS: dict[str, Any] = {
    "n_estimators":    500,
    "learning_rate":   0.05,
    "num_leaves":      63,
    "max_depth":       -1,
    "min_child_samples": 20,
    "subsample":       0.8,
    "colsample_bytree": 0.8,
    "reg_alpha":       0.1,
    "reg_lambda":      0.1,
    "class_weight":    "balanced",
    "random_state":    42,
    "n_jobs":          -1,
    "verbose":         -1,
}

META_PARAMS = {
    "C":           1.0,
    "max_iter":    1000,
    "random_state": 42,
}

CV_FOLDS = 5


# ── Helpers ───────────────────────────────────────────────────────────────────

def build_model() -> StackingClassifier:
    """Build the stacked ensemble."""
    base = lgb.LGBMClassifier(**LGBM_PARAMS)
    meta = LogisticRegression(**META_PARAMS)
    return StackingClassifier(
        estimators=[("lgbm", base)],
        final_estimator=meta,
        cv=CV_FOLDS,
        passthrough=True,   # pass original features to meta learner too
        n_jobs=-1,
    )


def train_val_split(
    X: pd.DataFrame,
    y: pd.Series,
    val_size: float = 0.2,
    random_state: int = 42,
) -> tuple:
    """Stratified train/validation split."""
    from sklearn.model_selection import train_test_split
    return train_test_split(
        X, y,
        test_size=val_size,
        stratify=y,
        random_state=random_state,
    )


def evaluate_model(model, X_val: pd.DataFrame, y_val: pd.Series) -> tuple[dict, Any]:
    """Compute all evaluation metrics."""
    probs = model.predict_proba(X_val)[:, 1]

    metrics = {
        "auc_roc":          roc_auc_score(y_val, probs),
        "brier_score":      brier_score_loss(y_val, probs),
        "avg_precision":    average_precision_score(y_val, probs),
        "log_loss":         log_loss(y_val, probs),
        "val_goal_rate":    float(y_val.mean()),
        "val_size":         len(y_val),
    }
    return metrics, probs


def plot_calibration(y_val, probs, run_id: str) -> str:
    """Plot reliability diagram and save to temp file."""
    fraction_pos, mean_pred = calibration_curve(y_val, probs, n_bins=10)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.plot(mean_pred, fraction_pos, "b-o", label="xG Model")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives (actual goal rate)")
    ax.set_title("xG Model Calibration Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)

    path = f"calibration_{run_id[:8]}.png"
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_shap(model, X_val: pd.DataFrame, run_id: str) -> str | None:
    """Generate SHAP summary plot for the base LightGBM model."""
    try:
        # Extract base LightGBM from stacking classifier
        lgbm_model = model.estimators_[0]
        explainer   = shap.TreeExplainer(lgbm_model)

        # Use a sample of val set for SHAP (faster)
        sample = X_val.sample(min(500, len(X_val)), random_state=42)
        shap_values = explainer.shap_values(sample)

        # For binary classification, shap_values is a list — take class 1
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values,    sample,
            plot_type="bar",    
            show=False,
        ) 
        plt.title("xG Model — Feature Importance (SHAP)")
        fig = plt.gcf()

        path = f"shap_{run_id[:8]}.png"
        fig.savefig(path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        return path
    except Exception as e:
        logger.warning(f"SHAP plot failed: {e}")
        return None


# ── Main Training Function ────────────────────────────────────────────────────

def train_xg_model(
    use_360_only: bool = False,
    register: bool = True,
) -> dict:
    """
    Full training pipeline:
    1. Load features from DB
    2. Split train/val
    3. Train stacked ensemble
    4. Evaluate + log to MLflow
    5. Register model if metrics pass threshold

    Returns: dict of metrics
    """
    # Setup MLflow
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    logger.info("═══ xG Model Training Starting ═══")

    # Load data
    X, y = load_training_data(use_360_only=use_360_only)
    X_train, X_val, y_train, y_val = train_val_split(X, y)

    logger.info(f"Train: {len(X_train)} shots | Val: {len(X_val)} shots")
    logger.info(f"Goal rate — Train: {y_train.mean():.3f} | Val: {y_val.mean():.3f}")

    with mlflow.start_run(run_name="xg_stacked_360_v1") as run:
        run_id = run.info.run_id
        logger.info(f"MLflow run: {run_id}")

        # Log parameters
        mlflow.log_params({
            "base_model":    "lightgbm",
            "meta_model":    "logistic_regression",
            "cv_folds":      CV_FOLDS,
            "feature_set":   "360_spatial_v1",
            "n_features":    X.shape[1],
            "train_size":    len(X_train),
            "val_size":      len(X_val),
            "use_360_only":  use_360_only,
            **{f"lgbm_{k}": v for k, v in LGBM_PARAMS.items()},
        })

        # Train
        logger.info("Training stacked ensemble...")
        model = build_model()
        model.fit(X_train, y_train)
        logger.info("Training complete")

        # Evaluate
        metrics, probs = evaluate_model(model, X_val, y_val)
        mlflow.log_metrics(metrics)

        logger.info(f"AUC-ROC:     {metrics['auc_roc']:.4f}  (target > 0.78)")
        logger.info(f"Brier Score: {metrics['brier_score']:.4f}  (target < 0.07)")
        logger.info(f"Avg Precision: {metrics['avg_precision']:.4f}")
        logger.info(f"Log Loss:    {metrics['log_loss']:.4f}")

        # Calibration plot
        cal_path = plot_calibration(y_val, probs, run_id)
        mlflow.log_artifact(cal_path)
        os.remove(cal_path)

        # SHAP plot
        shap_path = plot_shap(model, X_val, run_id)
        if shap_path:
            mlflow.log_artifact(shap_path)
            os.remove(shap_path)

        # Log feature names
        mlflow.log_param("features", ",".join(get_feature_names()))

        # Register model if AUC passes threshold
        if register and metrics["auc_roc"] > 0.72:
            logger.info("AUC threshold passed — registering model")
            mlflow.sklearn.log_model(
                model,
                artifact_path="xg_model",
                registered_model_name=MODEL_NAME,
                input_example=X_val.head(3),
            )
            logger.info(f"Model registered as '{MODEL_NAME}'")
        else:
            mlflow.sklearn.log_model(model, artifact_path="xg_model")
            if metrics["auc_roc"] <= 0.72:
                logger.warning(f"AUC {metrics['auc_roc']:.4f} below threshold — not registering")

    logger.info("═══ Training Complete ═══")
    return metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--360-only", action="store_true",
                        help="Train only on shots with 360 data")
    parser.add_argument("--no-register", action="store_true",
                        help="Skip model registry")
    args = parser.parse_args()

    metrics = train_xg_model(
        use_360_only=getattr(args, "360_only"),
        register=not args.no_register,
    )
    print(f"\n{'='*50}")
    print(f"AUC-ROC:     {metrics['auc_roc']:.4f}")
    print(f"Brier Score: {metrics['brier_score']:.4f}")
    print(f"Avg Precision: {metrics['avg_precision']:.4f}")
    print(f"{'='*50}")
