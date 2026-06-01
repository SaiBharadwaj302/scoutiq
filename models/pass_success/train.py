"""
models/pass_success/train.py

Trains the Pass Success Probability model:
  Algorithm: LightGBM with Platt scaling (calibrated probabilities)

Why LightGBM here (not stacked ensemble like xG)?
  Pass success is a high-volume prediction (~800K passes).
  LightGBM alone with calibration gives excellent results and
  faster inference — critical for a real-time API endpoint.

Run directly:
    python -m models.pass_success.train
"""
import os
import warnings
warnings.filterwarnings("ignore")

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from loguru import logger
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    roc_auc_score, brier_score_loss,
    average_precision_score, log_loss,
)
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from dotenv import load_dotenv

from models.pass_success.features import load_training_data, get_feature_names

load_dotenv()

MLFLOW_URI      = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
EXPERIMENT_NAME = "scoutiq_pass_success"
MODEL_NAME      = "scoutiq_pass"

LGBM_PARAMS = {
    "n_estimators":      500,
    "learning_rate":     0.05,
    "num_leaves":        63,
    "max_depth":         -1,
    "min_child_samples": 50,   # higher than xG — more data, avoid overfit
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "reg_alpha":         0.1,
    "reg_lambda":        0.1,
    "random_state":      42,
    "n_jobs":            -1,
    "verbose":           -1,
}


def build_model():
    """LightGBM with Platt scaling for calibrated probabilities."""
    base = lgb.LGBMClassifier(**LGBM_PARAMS)
    # Platt scaling = sigmoid calibration on held-out CV folds
    return CalibratedClassifierCV(base, method="sigmoid", cv=5)


def evaluate_model(model, X_val, y_val):
    probs = model.predict_proba(X_val)[:, 1]
    return {
        "auc_roc":       roc_auc_score(y_val, probs),
        "brier_score":   brier_score_loss(y_val, probs),
        "avg_precision": average_precision_score(y_val, probs),
        "log_loss":      log_loss(y_val, probs),
        "val_completion_rate": float(y_val.mean()),
        "val_size":      len(y_val),
    }, probs


def plot_calibration(y_val, probs, run_id):
    fraction_pos, mean_pred = calibration_curve(y_val, probs, n_bins=10)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.plot(mean_pred, fraction_pos, "b-o", label="Pass Success Model")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Actual completion rate")
    ax.set_title("Pass Success Model — Calibration Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    path = f"pass_calibration_{run_id[:8]}.png"
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return path


def train_pass_model(
    use_360_only: bool = False,
    register: bool = True,
) -> dict:
    """Full training pipeline for pass success model."""
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    logger.info("═══ Pass Success Model Training Starting ═══")

    X, y = load_training_data(use_360_only=use_360_only)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    logger.info(f"Train: {len(X_train)} passes | Val: {len(X_val)} passes")
    logger.info(f"Completion rate — Train: {y_train.mean():.3f} | Val: {y_val.mean():.3f}")

    with mlflow.start_run(run_name="pass_lgbm_platt_v1") as run:
        run_id = run.info.run_id

        mlflow.log_params({
            "algorithm":     "lightgbm_platt_scaling",
            "calibration":   "sigmoid",
            "cv_folds":      5,
            "feature_set":   "360_spatial_v1",
            "n_features":    X.shape[1],
            "train_size":    len(X_train),
            "val_size":      len(X_val),
            "use_360_only":  use_360_only,
            **{f"lgbm_{k}": v for k, v in LGBM_PARAMS.items()},
        })

        logger.info("Training LightGBM with Platt scaling...")
        model = build_model()
        model.fit(X_train, y_train)
        logger.info("Training complete")

        metrics, probs = evaluate_model(model, X_val, y_val)
        mlflow.log_metrics(metrics)

        logger.info(f"AUC-ROC:     {metrics['auc_roc']:.4f}  (target > 0.72)")
        logger.info(f"Brier Score: {metrics['brier_score']:.4f}")
        logger.info(f"Avg Precision: {metrics['avg_precision']:.4f}")
        logger.info(f"Log Loss:    {metrics['log_loss']:.4f}")

        cal_path = plot_calibration(y_val, probs, run_id)
        mlflow.log_artifact(cal_path)
        os.remove(cal_path)

        mlflow.log_param("features", ",".join(get_feature_names()))

        if register and metrics["auc_roc"] > 0.72:
            logger.info("AUC threshold passed — registering model")
            mlflow.sklearn.log_model(
                model,
                artifact_path="pass_model",
                registered_model_name=MODEL_NAME,
                input_example=X_val.head(3),
            )
            logger.info(f"Model registered as '{MODEL_NAME}'")
        else:
            mlflow.sklearn.log_model(model, artifact_path="pass_model")

    logger.info("═══ Training Complete ═══")
    return metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--360-only", action="store_true")
    parser.add_argument("--no-register", action="store_true")
    args = parser.parse_args()

    metrics = train_pass_model(
        use_360_only=getattr(args, "360_only"),
        register=not args.no_register,
    )
    print(f"\n{'='*50}")
    print(f"AUC-ROC:       {metrics['auc_roc']:.4f}")
    print(f"Brier Score:   {metrics['brier_score']:.4f}")
    print(f"Avg Precision: {metrics['avg_precision']:.4f}")
    print(f"{'='*50}")
