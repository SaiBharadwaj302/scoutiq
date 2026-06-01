"""
models/xg/train_v2.py

xG 3.0 — Modern Ensemble with Automated Hyperparameter Tuning

Architecture upgrade over train.py (v1):
  v1: StackingClassifier( LightGBM → LogisticRegression )
  v3: SoftVotingEnsemble( LightGBM + CatBoost + XGBoost )
      → Isotonic calibration
      → Optuna Bayesian hyperparameter search (50 trials per base model)

Why this stack (2024-25 state of the art for tabular sports data):
  ─ LightGBM   : leaf-wise trees, fastest on large data, native NaN handling
  ─ CatBoost   : ordered boosting (prevents target leakage), best on
                 categorically-rich data, no pre-encoding needed internally
  ─ XGBoost 2  : histogram method, GPU-ready, default-direction NaN routing —
                 strong diversity vs LGBM despite similar algorithm class
  ─ Isotonic   : non-parametric calibration — outperforms Platt/sigmoid on
                 datasets > ~5 000 samples (Zadrozny & Elkan, 2002)
  ─ Optuna     : Tree-structured Parzen Estimator (TPE) Bayesian search;
                 far more sample-efficient than grid/random search

Every run is tracked in MLflow: Optuna study, per-model params, ensemble
metrics, SHAP bar chart, calibration curve, and model artifact.

Run directly:
    python -m models.xg.train_v2
    python -m models.xg.train_v2 --trials 30 --no-register
"""
from __future__ import annotations

import os
import warnings
warnings.filterwarnings("ignore")

import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier
from dotenv import load_dotenv
from loguru import logger
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
import lightgbm as lgb
import xgboost as xgb

from models.xg.features import load_training_data, get_feature_names

load_dotenv()

# ── MLflow ────────────────────────────────────────────────────────────────────
MLFLOW_URI      = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = "scoutiq_xg_v3"
MODEL_NAME      = "scoutiq_xg"

# ── Tuning ────────────────────────────────────────────────────────────────────
DEFAULT_TRIALS  = 50      # Optuna trials per base model (reduce for fast testing)
CV_FOLDS        = 5       # inner CV folds used during Optuna search
OPTUNA_TIMEOUT  = 600     # seconds — hard limit per model tuning phase

# ── Optuna search spaces ──────────────────────────────────────────────────────

def _lgbm_space(trial: optuna.Trial) -> dict:
    return {
        "n_estimators":       trial.suggest_int("lgbm_n_est",   200, 1000),
        "learning_rate":      trial.suggest_float("lgbm_lr",    1e-3, 0.15, log=True),
        "num_leaves":         trial.suggest_int("lgbm_leaves",  31, 255),
        "max_depth":          trial.suggest_int("lgbm_depth",   4, 12),
        "min_child_samples":  trial.suggest_int("lgbm_min_child", 10, 100),
        "subsample":          trial.suggest_float("lgbm_sub",   0.5, 1.0),
        "colsample_bytree":   trial.suggest_float("lgbm_col",   0.5, 1.0),
        "reg_alpha":          trial.suggest_float("lgbm_alpha", 1e-4, 10.0, log=True),
        "reg_lambda":         trial.suggest_float("lgbm_lambda",1e-4, 10.0, log=True),
        "class_weight":       "balanced",
        "random_state":       42,
        "n_jobs":             -1,
        "verbose":            -1,
    }


def _xgb_space(trial: optuna.Trial) -> dict:
    return {
        "n_estimators":     trial.suggest_int("xgb_n_est",   200, 1000),
        "learning_rate":    trial.suggest_float("xgb_lr",    1e-3, 0.15, log=True),
        "max_depth":        trial.suggest_int("xgb_depth",   3, 12),
        "min_child_weight": trial.suggest_int("xgb_mcw",     1, 20),
        "subsample":        trial.suggest_float("xgb_sub",   0.5, 1.0),
        "colsample_bytree": trial.suggest_float("xgb_col",   0.5, 1.0),
        "gamma":            trial.suggest_float("xgb_gamma", 0.0, 5.0),
        "reg_alpha":        trial.suggest_float("xgb_alpha", 1e-4, 10.0, log=True),
        "reg_lambda":       trial.suggest_float("xgb_lambda",1e-4, 10.0, log=True),
        "scale_pos_weight": trial.suggest_float("xgb_spw",   1.0, 20.0),
        "tree_method":      "hist",
        "eval_metric":      "logloss",
        "random_state":     42,
        "n_jobs":           -1,
    }


def _catboost_space(trial: optuna.Trial) -> dict:
    return {
        "iterations":        trial.suggest_int("cb_iter",    200, 1000),
        "learning_rate":     trial.suggest_float("cb_lr",    1e-3, 0.15, log=True),
        "depth":             trial.suggest_int("cb_depth",   4, 10),
        "l2_leaf_reg":       trial.suggest_float("cb_l2",    1e-4, 10.0, log=True),
        "bagging_temperature":trial.suggest_float("cb_bag",  0.0, 1.0),
        "random_strength":   trial.suggest_float("cb_rs",    0.0, 1.0),
        "border_count":      trial.suggest_int("cb_border",  32, 255),
        "auto_class_weights": "Balanced",
        "random_seed":       42,
        "verbose":           0,
        "allow_writing_files": False,
    }


# ── Optuna objective factory ───────────────────────────────────────────────────

def _make_objective(model_cls, space_fn, X, y, cv_folds: int):
    """Return an Optuna objective function for a given model class."""
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    def objective(trial: optuna.Trial) -> float:
        params = space_fn(trial)
        model  = model_cls(**params)
        scores = cross_val_score(
            model, X, y,
            cv=skf,
            scoring="roc_auc",
            n_jobs=-1,
            error_score=0.0,
        )
        return float(scores.mean())

    return objective


def tune_model(
    model_cls,
    space_fn,
    X: pd.DataFrame,
    y: pd.Series,
    n_trials: int = DEFAULT_TRIALS,
    model_name: str = "model",
) -> dict:
    """
    Run Optuna Bayesian search for one base model.
    Returns the best hyperparameter dict found.
    """
    logger.info(f"Tuning {model_name} — {n_trials} trials (TPE)...")
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        direction="maximize",
        study_name=f"xg_{model_name}",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
    )
    study.optimize(
        _make_objective(model_cls, space_fn, X, y, CV_FOLDS),
        n_trials=n_trials,
        timeout=OPTUNA_TIMEOUT,
        show_progress_bar=False,
    )

    best = study.best_params
    logger.info(
        f"{model_name} best AUC (CV): {study.best_value:.4f} "
        f"after {len(study.trials)} trials"
    )
    return best, study


# ── Model builders ─────────────────────────────────────────────────────────────

def build_lgbm(params: dict) -> lgb.LGBMClassifier:
    # Strip Optuna prefixes from param names before passing to constructor
    clean = {k.replace("lgbm_", ""): v for k, v in params.items()
             if not k.startswith(("xgb_", "cb_"))}
    return lgb.LGBMClassifier(**clean)


def build_xgb(params: dict) -> xgb.XGBClassifier:
    clean = {k.replace("xgb_", ""): v for k, v in params.items()
             if not k.startswith(("lgbm_", "cb_"))}
    return xgb.XGBClassifier(**clean)


def build_catboost(params: dict) -> CatBoostClassifier:
    clean = {k.replace("cb_", ""): v for k, v in params.items()
             if not k.startswith(("lgbm_", "xgb_"))}
    return CatBoostClassifier(**clean)


def build_ensemble(
    lgbm_params: dict,
    xgb_params:  dict,
    cb_params:   dict,
) -> CalibratedClassifierCV:
    """
    Soft-voting ensemble of the three base learners,
    wrapped in isotonic calibration.
    """
    base = VotingClassifier(
        estimators=[
            ("lgbm", build_lgbm(lgbm_params)),
            ("xgb",  build_xgb(xgb_params)),
            ("cb",   build_catboost(cb_params)),
        ],
        voting="soft",
        n_jobs=-1,
    )
    # Isotonic calibration — non-parametric, works best with ≥5 000 samples
    return CalibratedClassifierCV(base, method="isotonic", cv=3)


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate_model(model, X_val: pd.DataFrame, y_val: pd.Series) -> tuple[dict, np.ndarray]:
    probs = model.predict_proba(X_val)[:, 1]
    metrics = {
        "auc_roc":       roc_auc_score(y_val, probs),
        "brier_score":   brier_score_loss(y_val, probs),
        "avg_precision": average_precision_score(y_val, probs),
        "log_loss":      log_loss(y_val, probs),
        "val_goal_rate": float(y_val.mean()),
        "val_size":      int(len(y_val)),
    }
    return metrics, probs


# ── Plotting helpers ───────────────────────────────────────────────────────────

def plot_calibration(y_val, probs, run_id: str) -> str:
    frac_pos, mean_pred = calibration_curve(y_val, probs, n_bins=10)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.plot(mean_pred, frac_pos, "b-o", label="xG 3.0 Ensemble")
    ax.set_xlabel("Mean predicted xG")
    ax.set_ylabel("Actual goal rate")
    ax.set_title("xG 3.0 — Calibration Curve")
    ax.legend(); ax.grid(True, alpha=0.3)
    path = f"xg_calibration_{run_id[:8]}.png"
    fig.savefig(path, dpi=100, bbox_inches="tight"); plt.close(fig)
    return path


def plot_shap(model, X_val: pd.DataFrame, run_id: str) -> str | None:
    """SHAP summary using the LightGBM sub-model (fastest explainer)."""
    try:
        # Reach inside: CalibratedClassifierCV → VotingClassifier → LGBM
        voting  = model.calibrated_classifiers_[0].estimator
        lgbm_m  = dict(voting.named_estimators_)["lgbm"]
        sample  = X_val.sample(min(500, len(X_val)), random_state=42)
        explainer   = shap.TreeExplainer(lgbm_m)
        shap_vals   = explainer.shap_values(sample)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_vals, sample, plot_type="bar", show=False)
        plt.title("xG 3.0 — Feature Importance (SHAP via LightGBM)")
        path = f"xg_shap_{run_id[:8]}.png"
        plt.gcf().savefig(path, dpi=100, bbox_inches="tight"); plt.close()
        return path
    except Exception as e:
        logger.warning(f"SHAP plot skipped: {e}")
        return None


# ── Main training function ─────────────────────────────────────────────────────

def train_xg_model_v2(
    use_360_only: bool = False,
    register:     bool = True,
    n_trials:     int  = DEFAULT_TRIALS,
) -> dict:
    """
    Full xG 3.0 training pipeline:

    1. Load features from DB
    2. Stratified train / val split
    3. Optuna TPE search for each base model (LightGBM, XGBoost, CatBoost)
    4. Build soft-voting ensemble + isotonic calibration
    5. Evaluate + log everything to MLflow
    6. Register model if AUC ≥ 0.75

    Args:
        use_360_only : only use shots with StatsBomb 360 freeze-frame data
        register     : register model in MLflow model registry when AUC passes
        n_trials     : Optuna trials per base model

    Returns:
        dict of final evaluation metrics
    """
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    logger.info("═══ xG 3.0 Training Starting ═══")

    # 1. Data
    X, y = load_training_data(use_360_only=use_360_only)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    logger.info(f"Train: {len(X_train)} shots | Val: {len(X_val)} shots")
    logger.info(f"Goal rate — Train: {y_train.mean():.3f} | Val: {y_val.mean():.3f}")

    # 2. Tune base models
    lgbm_best, lgbm_study = tune_model(
        lgb.LGBMClassifier, _lgbm_space, X_train, y_train, n_trials, "LightGBM"
    )
    xgb_best, xgb_study = tune_model(
        xgb.XGBClassifier, _xgb_space, X_train, y_train, n_trials, "XGBoost"
    )
    cb_best, cb_study = tune_model(
        CatBoostClassifier, _catboost_space, X_train, y_train, n_trials, "CatBoost"
    )

    # 3. Train final ensemble
    with mlflow.start_run(run_name="xg_ensemble_v3") as run:
        run_id = run.info.run_id
        logger.info(f"MLflow run: {run_id}")

        # Log all params
        mlflow.log_params({
            "architecture":       "lgbm+xgb+catboost_soft_voting",
            "calibration":        "isotonic",
            "n_trials_per_model": n_trials,
            "cv_folds":           CV_FOLDS,
            "feature_set":        "360_spatial_v1",
            "n_features":         X.shape[1],
            "train_size":         len(X_train),
            "val_size":           len(X_val),
            "use_360_only":       use_360_only,
            **{f"lgbm_best_{k}": v for k, v in lgbm_best.items()},
            **{f"xgb_best_{k}":  v for k, v in xgb_best.items()},
            **{f"cb_best_{k}":   v for k, v in cb_best.items()},
        })

        logger.info("Training final ensemble with best params...")
        model = build_ensemble(lgbm_best, xgb_best, cb_best)
        model.fit(X_train, y_train)
        logger.info("Ensemble training complete")

        # Evaluate
        metrics, probs = evaluate_model(model, X_val, y_val)
        mlflow.log_metrics(metrics)

        logger.info(f"AUC-ROC:       {metrics['auc_roc']:.4f}  (target > 0.78)")
        logger.info(f"Brier Score:   {metrics['brier_score']:.4f}  (target < 0.07)")
        logger.info(f"Avg Precision: {metrics['avg_precision']:.4f}")
        logger.info(f"Log Loss:      {metrics['log_loss']:.4f}")

        # Log Optuna best CV scores
        mlflow.log_metrics({
            "optuna_lgbm_best_auc": lgbm_study.best_value,
            "optuna_xgb_best_auc":  xgb_study.best_value,
            "optuna_cb_best_auc":   cb_study.best_value,
        })

        # Artifacts
        cal_path = plot_calibration(y_val, probs, run_id)
        mlflow.log_artifact(cal_path); os.remove(cal_path)

        shap_path = plot_shap(model, X_val, run_id)
        if shap_path:
            mlflow.log_artifact(shap_path); os.remove(shap_path)

        mlflow.log_param("features", ",".join(get_feature_names()))

        # Register
        if register and metrics["auc_roc"] >= 0.75:
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
            if metrics["auc_roc"] < 0.75:
                logger.warning(
                    f"AUC {metrics['auc_roc']:.4f} below 0.75 threshold — not registering"
                )

    logger.info("═══ xG 3.0 Training Complete ═══")
    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train xG 3.0 ensemble model")
    parser.add_argument("--360-only",    action="store_true",
                        help="Use only shots with StatsBomb 360 data")
    parser.add_argument("--no-register", action="store_true",
                        help="Skip MLflow model registry")
    parser.add_argument("--trials",      type=int, default=DEFAULT_TRIALS,
                        help=f"Optuna trials per base model (default: {DEFAULT_TRIALS})")
    args = parser.parse_args()

    results = train_xg_model_v2(
        use_360_only=getattr(args, "360_only"),
        register=not args.no_register,
        n_trials=args.trials,
    )

    print(f"\n{'='*50}")
    print(f"xG 3.0 Final Results")
    print(f"{'='*50}")
    print(f"AUC-ROC:       {results['auc_roc']:.4f}")
    print(f"Brier Score:   {results['brier_score']:.4f}")
    print(f"Avg Precision: {results['avg_precision']:.4f}")
    print(f"Log Loss:      {results['log_loss']:.4f}")
    print(f"{'='*50}")
