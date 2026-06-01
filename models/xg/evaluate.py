"""
models/xg/evaluate.py

Standalone evaluation script — loads the registered Production model
from MLflow and runs a full evaluation report.

Run directly:
    python -m models.xg.evaluate
"""
import os
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from loguru import logger
from sklearn.metrics import (
    roc_auc_score, brier_score_loss,
    RocCurveDisplay, PrecisionRecallDisplay,
)
from sklearn.calibration import calibration_curve
from dotenv import load_dotenv

from models.xg.features import load_training_data

load_dotenv()

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = "scoutiq_xg"


def load_production_model():
    """Load the Production model from MLflow registry."""
    mlflow.set_tracking_uri(MLFLOW_URI)
    model_uri = f"models:/{MODEL_NAME}/Production"
    try:
        model = mlflow.sklearn.load_model(model_uri)
        logger.info(f"Loaded Production model: {model_uri}")
        return model
    except Exception:
        # Fall back to latest version if no Production alias
        model_uri = f"models:/{MODEL_NAME}/latest"
        model = mlflow.sklearn.load_model(model_uri)
        logger.info(f"Loaded latest model: {model_uri}")
        return model


def run_evaluation():
    """Full evaluation report on held-out validation set."""
    X, y = load_training_data()

    # Use last 20% as eval set (same split as training)
    from sklearn.model_selection import train_test_split
    _, X_val, _, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = load_production_model()
    probs = model.predict_proba(X_val)[:, 1]

    auc   = roc_auc_score(y_val, probs)
    brier = brier_score_loss(y_val, probs)

    logger.info(f"AUC-ROC:     {auc:.4f}")
    logger.info(f"Brier Score: {brier:.4f}")

    # Plot ROC + Calibration + PR curve
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ROC
    RocCurveDisplay.from_predictions(y_val, probs, ax=axes[0], name="xG Model")
    axes[0].set_title(f"ROC Curve (AUC = {auc:.3f})")

    # Calibration
    fraction_pos, mean_pred = calibration_curve(y_val, probs, n_bins=10)
    axes[1].plot([0, 1], [0, 1], "k--", label="Perfect")
    axes[1].plot(mean_pred, fraction_pos, "b-o", label=f"xG Model (Brier={brier:.3f})")
    axes[1].set_xlabel("Mean predicted xG")
    axes[1].set_ylabel("Actual goal rate")
    axes[1].set_title("Calibration Curve")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Precision-Recall
    PrecisionRecallDisplay.from_predictions(y_val, probs, ax=axes[2], name="xG Model")
    axes[2].set_title("Precision-Recall Curve")

    plt.tight_layout()
    plt.savefig("evaluation_report.png", dpi=120, bbox_inches="tight")
    logger.info("Saved evaluation_report.png")
    plt.show()

    return {"auc": auc, "brier": brier}


if __name__ == "__main__":
    run_evaluation()
