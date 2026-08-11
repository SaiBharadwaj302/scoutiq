"""
models/pass_success/train_v2.py

Pass Success 3.0 — Modern Ensemble with Automated Hyperparameter Tuning

Architecture upgrade over train.py (v1):
  v1: CalibratedClassifierCV( LightGBM, method="sigmoid", cv=5 )
  v3: SoftVotingEnsemble( LightGBM + CatBoost + XGBoost )
      → Isotonic calibration
      → Optuna Bayesian hyperparameter search (50 trials per base model)

Pass success is a high-volume task (~800K passes/season) so we keep the same
fast gradient-boosting family — the upgrade is in diversity (3 models with
different bias-variance profiles), better calibration (isotonic), and
automated tuning (Optuna TPE) rather than hand-tuned knobs.

Run directly:
    python -m models.pass_success.train_v2
    python -m models.pass_success.train_v2 --trials 20 --no-register
"""
from __future__ import annotations

import os
import warnings
warnings.filterwarnings("ignore")

import mlflow
import mlflow.sklearn
import optuna
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

from models.pass_success.features import load_training_data, get_feature_names

load_dotenv()

# ── MLflow ────────────────────────────────────────────────────────────────────
MLFLOW_URI      = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
EXPERIMENT_NAME = "scoutiq_pass_success_v3"
MODEL_NAME      = "scoutiq_pass"

# ── Tuning ────────────────────────────────────────────────────────────────────
DEFAULT_TRIALS  = 50
CV_FOLDS        = 5
OPTUNA_TIMEOUT  = 600     # seconds per base model


# ── Search spaces ─────────────────────────────────────────────────────────────
# Pass success has many more samples than xG, so we push min_child_samples
# higher for LGBM and min_child_weight higher for XGB to avoid overfitting.

def _lgbm_space(trial: optuna.Trial) -> dict:
    return {
        "n_estimators":      trial.suggest_int("lgbm_n_est",   300, 1200),
        "learning_rate":     trial.suggest_float("lgbm_lr",    1e-3, 0.12, log=True),
        "num_leaves":        trial.suggest_int("lgbm_leaves",  31, 255),
        "max_depth":         trial.suggest_int("lgbm_depth",   4, 12),
        "min_child_samples": trial.suggest_int("lgbm_min_child", 50, 300),
        "subsample":         trial.suggest_float("lgbm_sub",   0.5, 1.0),
        "colsample_bytree":  trial.suggest_float("lgbm_col",   0.5, 1.0),
        "reg_alpha":         trial.suggest_float("lgbm_alpha", 1e-4, 10.0, log=True),
        "reg_lambda":        trial.suggest_float("lgbm_lambda",1e-4, 10.0, log=True),
        "random_state":      42,
        "n_jobs":            -1,
        "verbose":           -1,
    }


def _xgb_space(trial: optuna.Trial) -> dict:
    return {
        "n_estimators":     trial.suggest_int("xgb_n_est",   300, 1200),
        "learning_rate":    trial.suggest_float("xgb_lr",    1e-3, 0.12, log=True),
        "max_depth":        trial.suggest_int("xgb_depth",   3, 10),
        "min_child_weight": trial.suggest_int("xgb_mcw",     5, 50),
        "subsample":        trial.suggest_float("xgb_sub",   0.5, 1.0),
        "colsample_bytree": trial.suggest_float("xgb_col",   0.5, 1.0),
        "gamma":            trial.suggest_float("xgb_gamma", 0.0, 5.0),
        "reg_alpha":        trial.suggest_float("xgb_alpha", 1e-4, 10.0, log=True),
        "reg_lambda":       trial.suggest_float("xgb_lambda",1e-4, 10.0, log=True),
        "tree_method":      "hist",
        "eval_metric":      "logloss",
        "random_state":     42,
        "n_jobs":           -1,
    }


def _catboost_space(trial: optuna.Trial) -> dict:
    return {
        "iterations":         trial.suggest_int("cb_iter",    300, 1200),
        "learning_rate":      trial.suggest_float("cb_lr",    1e-3, 0.12, log=True),
        "depth":              trial.suggest_int("cb_depth",   4, 10),
        "l2_leaf_reg":        trial.suggest_float("cb_l2",    1e-4, 10.0, log=True),
        "bagging_temperature":trial.suggest_float("cb_bag",   0.0, 1.0),
        "random_strength":    trial.suggest_float("cb_rs",    0.0, 1.0),
        "border_count":       trial.suggest_int("cb_border",  32, 255),
        "random_seed":        42,
        "verbose":            0,
        "allow_writing_files":False,
    }


# ── Optuna objective factory ───────────────────────────────────────────────────

def _make_objective(model_cls, space_fn, X, y, cv_folds: int):
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    def objective(trial: optuna.Trial) -> float:
        params = space_fn(trial)
        model  = model_cls(**params)
        scores = cross_val_score(
            model, X, y, cv=skf, scoring="roc_auc", n_jobs=-1, error_score=0.0
        )
        return float(scores.mean())

    return objective


def tune_model(model_cls, space_fn, X, y, n_trials: int, model_name: str):
    logger.info(f"Tuning {model_name} — {n_trials} trials (TPE)...")
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        direction="maximize",
        study_name=f"pass_{model_name}",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
    )
    study.optimize(
        _make_objective(model_cls, space_fn, X, y, CV_FOLDS),
        n_trials=n_trials,
        timeout=OPTUNA_TIMEOUT,
        show_progress_bar=False,
    )
    logger.info(
        f"{model_name} best AUC (CV): {study.best_value:.4f} "
        f"after {len(study.trials)} trials"
    )
    return study.best_params, study


# ── Ensemble ──────────────────────────────────────────────────────────────────

def _clean(params: dict, prefix: str) -> dict:
    """Strip Optuna name prefixes, keep only params for this model."""
    others = [p for p in ("lgbm_", "xgb_", "cb_") if p != prefix]
    return {k.replace(prefix, ""): v
            for k, v in params.items()
            if not any(k.startswith(o) for o in others)}


def build_ensemble(lgbm_params: dict, xgb_params: dict, cb_params: dict):
    voting = VotingClassifier(
        estimators=[
            ("lgbm", lgb.LGBMClassifier(**_clean(lgbm_params, "lgbm_"))),
            ("xgb",  xgb.XGBClassifier(**_clean(xgb_params,  "xgb_"))),
            ("cb",   CatBoostClassifier(**_clean(cb_params,   "cb_"))),
        ],
        voting="soft",
        n_jobs=-1,
    )
    return CalibratedClassifierCV(voting, method="isotonic", cv=3)


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate_model(model, X_val, y_val):
    probs = model.predict_proba(X_val)[:, 1]
    metrics = {
        "auc_roc":              roc_auc_score(y_val, probs),
        "brier_score":          brier_score_loss(y_val, probs),
        "avg_precision":        average_precision_score(y_val, probs),
        "log_loss":             log_loss(y_val, probs),
        "val_completion_rate":  float(y_val.mean()),
        "val_size":             int(len(y_val)),
    }
    return metrics, probs


def plot_calibration(y_val, probs, run_id: str) -> str:
    frac_pos, mean_pred = calibration_curve(y_val, probs, n_bins=10)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.plot(mean_pred, frac_pos, "b-o", label="Pass 3.0 Ensemble")
    ax.set_xlabel("Mean predicted pass success prob")
    ax.set_ylabel("Actual completion rate")
    ax.set_title("Pass Success 3.0 — Calibration Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    path = f"pass_calibration_{run_id[:8]}.png"
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return path


# ── Main training function ─────────────────────────────────────────────────────

def train_pass_model_v2(
    use_360_only: bool = False,
    register:     bool = True,
    n_trials:     int  = DEFAULT_TRIALS,
    sample_size:  int | None = None,
) -> dict:
    """
    Full Pass Success 3.0 training pipeline.

    Steps:
    1. Load features from feature store (optional sample cap for dev)
    2. Stratified train / val split
    3. Optuna TPE search for LightGBM, XGBoost, CatBoost
    4. Soft-voting ensemble + isotonic calibration
    5. Evaluate and log to MLflow
    6. Register if AUC ≥ 0.72

    Args:
        use_360_only : only use passes with StatsBomb 360 data
        register     : register in MLflow model registry on success
        n_trials     : Optuna trials per base model
        sample_size  : cap training data (useful for fast dev iterations)

    Returns:
        dict of final evaluation metrics
    """
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    logger.info("═══ Pass Success 3.0 Training Starting ═══")

    X, y = load_training_data(use_360_only=use_360_only, sample_size=sample_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    logger.info(f"Train: {len(X_train)} passes | Val: {len(X_val)} passes")
    logger.info(f"Completion rate — Train: {y_train.mean():.3f} | Val: {y_val.mean():.3f}")

    # Tune
    lgbm_best, lgbm_study = tune_model(
        lgb.LGBMClassifier, _lgbm_space, X_train, y_train, n_trials, "LightGBM"
    )
    xgb_best, xgb_study = tune_model(
        xgb.XGBClassifier, _xgb_space, X_train, y_train, n_trials, "XGBoost"
    )
    cb_best, cb_study = tune_model(
        CatBoostClassifier, _catboost_space, X_train, y_train, n_trials, "CatBoost"
    )

    with mlflow.start_run(run_name="pass_ensemble_v3") as run:
        run_id = run.info.run_id
        logger.info(f"MLflow run: {run_id}")

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

        logger.info("Training final pass ensemble...")
        model = build_ensemble(lgbm_best, xgb_best, cb_best)
        model.fit(X_train, y_train)
        logger.info("Ensemble training complete")

        metrics, probs = evaluate_model(model, X_val, y_val)
        mlflow.log_metrics(metrics)

        logger.info(f"AUC-ROC:       {metrics['auc_roc']:.4f}  (target > 0.72)")
        logger.info(f"Brier Score:   {metrics['brier_score']:.4f}")
        logger.info(f"Avg Precision: {metrics['avg_precision']:.4f}")
        logger.info(f"Log Loss:      {metrics['log_loss']:.4f}")

        mlflow.log_metrics({
            "optuna_lgbm_best_auc": lgbm_study.best_value,
            "optuna_xgb_best_auc":  xgb_study.best_value,
            "optuna_cb_best_auc":   cb_study.best_value,
        })

        cal_path = plot_calibration(y_val, probs, run_id)
        mlflow.log_artifact(cal_path)
        os.remove(cal_path)

        mlflow.log_param("features", ",".join(get_feature_names()))

        if register and metrics["auc_roc"] >= 0.72:
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

    logger.info("═══ Pass Success 3.0 Training Complete ═══")
    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Pass Success 3.0 ensemble model")
    parser.add_argument("--360-only",    action="store_true")
    parser.add_argument("--no-register", action="store_true")
    parser.add_argument("--trials",      type=int, default=DEFAULT_TRIALS,
                        help=f"Optuna trials per base model (default: {DEFAULT_TRIALS})")
    parser.add_argument("--sample",      type=int, default=None,
                        help="Cap training data size (e.g. 100000 for fast dev run)")
    args = parser.parse_args()

    results = train_pass_model_v2(
        use_360_only=getattr(args, "360_only"),
        register=not args.no_register,
        n_trials=args.trials,
        sample_size=args.sample,
    )

    print(f"\n{'='*50}")
    print("Pass Success 3.0 Final Results")
    print(f"{'='*50}")
    print(f"AUC-ROC:       {results['auc_roc']:.4f}")
    print(f"Brier Score:   {results['brier_score']:.4f}")
    print(f"Avg Precision: {results['avg_precision']:.4f}")
    print(f"Log Loss:      {results['log_loss']:.4f}")
    print(f"{'='*50}")
