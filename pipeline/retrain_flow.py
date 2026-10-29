"""
pipeline/retrain_flow.py

Drift-triggered retraining with gated promotion.

Trigger: for each model, Evidently computes per-feature PSI between the
reference window and the current window of its feature table (shots → xG,
passes → pass success; see monitoring.drift_report.compute_psi_scores).
A model is retrained when ANY of its monitored features has PSI > 0.2.

Promotion: the retrained model's validation AUC is compared with the AUC
logged on the MLflow run behind the current `@Production` alias.
  new AUC >= production AUC  → `Production` alias moves to the new version
  new AUC <  production AUC  → new version gets the `Staging` alias, warning logged
The FastAPI server loads `models:/<name>@Production`, so only promoted
versions are served (after an API restart).

Every run — drift scores, retrain decision, AUCs, promotion outcome — is
logged as one run in the MLflow experiment `scoutiq_retraining`.

Run once:
    python -m pipeline.retrain_flow
    python -m pipeline.retrain_flow --force xg --pass-sample-size 200000
Schedule weekly: python -m pipeline.retrain_deploy
"""
from __future__ import annotations

import os

# Training scripts plot calibration/SHAP charts; Prefect runs tasks in worker
# threads, where a GUI backend (Tk on Windows) crashes. Headless = Agg.
os.environ.setdefault("MPLBACKEND", "Agg")

import mlflow  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402
from prefect import flow, get_run_logger, task  # noqa: E402

from monitoring.drift_report import MODEL_DATASETS, PSI_THRESHOLD, compute_psi_scores  # noqa: E402

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
RETRAIN_EXPERIMENT = "scoutiq_retraining"
PROD_ALIAS, STAGING_ALIAS = "Production", "Staging"


@task(name="compute-psi")
def task_compute_psi(psi_threshold: float) -> dict[str, dict]:
    return compute_psi_scores(psi_threshold)


@task(name="retrain-model")
def task_retrain(model_name: str, pass_sample_size: int | None) -> dict:
    """Run the existing training script; return its metrics + the registered version."""
    if model_name == "scoutiq_xg":
        from models.xg.train import train_xg_model
        metrics = train_xg_model()
    else:
        from models.pass_success.train import train_pass_model
        metrics = train_pass_model(sample_size=pass_sample_size)

    # The training scripts register inside their own run; find the version it created.
    run_id = mlflow.last_active_run().info.run_id
    versions = MlflowClient().search_model_versions(f"name='{model_name}' and run_id='{run_id}'")
    return {
        "run_id": run_id,
        "auc": float(metrics["auc_roc"]),
        "version": versions[0].version if versions else None,  # None = below register threshold
    }


@task(name="evaluate-and-promote")
def task_promote(model_name: str, new: dict) -> dict:
    log = get_run_logger()
    client = MlflowClient()
    if new["version"] is None:
        log.warning(f"{model_name}: retrained model was not registered (AUC {new['auc']:.4f})")
        return {"outcome": "not_registered", "prod_auc": None}

    try:
        prod = client.get_model_version_by_alias(model_name, PROD_ALIAS)
        prod_auc = client.get_run(prod.run_id).data.metrics["auc_roc"]
    except Exception:  # no Production model yet → promote whatever we have
        prod, prod_auc = None, float("-inf")

    if new["auc"] >= prod_auc:
        client.set_registered_model_alias(model_name, PROD_ALIAS, new["version"])
        log.info(f"{model_name}: v{new['version']} promoted to {PROD_ALIAS} "
                 f"(AUC {new['auc']:.4f} >= {prod_auc:.4f})")
        outcome = "promoted"
    else:
        client.set_registered_model_alias(model_name, STAGING_ALIAS, new["version"])
        log.warning(f"{model_name}: v{new['version']} kept in {STAGING_ALIAS} — "
                    f"AUC {new['auc']:.4f} < production v{prod.version} {prod_auc:.4f}")
        outcome = "staging"
    return {"outcome": outcome, "prod_auc": None if prod is None else prod_auc,
            "prod_version": None if prod is None else prod.version}


@flow(name="scoutiq-drift-retrain", log_prints=True)
def retrain_flow(
    psi_threshold: float = PSI_THRESHOLD,
    force_retrain: list[str] | None = None,
    pass_sample_size: int | None = None,
) -> dict:
    """
    psi_threshold:    retrain a model when any of its features exceeds this PSI
    force_retrain:    model names to retrain regardless of drift (e.g. ["scoutiq_xg"])
    pass_sample_size: cap pass training rows (None = full table, ~3.4M rows)
    """
    log = get_run_logger()
    mlflow.set_tracking_uri(MLFLOW_URI)
    force_retrain = force_retrain or []

    psi = task_compute_psi(psi_threshold)
    results: dict[str, dict] = {}
    for model_name in MODEL_DATASETS:
        drift = psi[model_name]
        retrain = drift["drift_detected"] or model_name in force_retrain
        reason = ("drift" if drift["drift_detected"] else "forced") if retrain else "no_drift"
        log.info(f"{model_name}: max PSI {drift['max_psi']:.4f} → {reason}")
        results[model_name] = {"drift": drift, "retrained": retrain, "reason": reason}
        if retrain:
            new = task_retrain(model_name, pass_sample_size)
            results[model_name] |= {"new": new, **task_promote(model_name, new)}

    # One summary run per flow execution, separate from the training experiments
    mlflow.set_experiment(RETRAIN_EXPERIMENT)
    with mlflow.start_run(run_name="drift_retrain"):
        mlflow.log_param("psi_threshold", psi_threshold)
        for model_name, r in results.items():
            short = model_name.removeprefix("scoutiq_")
            mlflow.log_metric(f"{short}_max_psi", r["drift"]["max_psi"])
            mlflow.log_metrics({f"{short}_psi_{f}": v for f, v in r["drift"]["psi"].items()})
            mlflow.set_tag(f"{short}_decision", r["reason"])
            if r["retrained"]:
                mlflow.log_metric(f"{short}_new_auc", r["new"]["auc"])
                if r.get("prod_auc") is not None:
                    mlflow.log_metric(f"{short}_prod_auc", r["prod_auc"])
                mlflow.set_tag(f"{short}_new_version", r["new"]["version"])
                mlflow.set_tag(f"{short}_promotion", r["outcome"])
        mlflow.log_dict(results, "retrain_summary.json")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Drift-triggered retraining")
    parser.add_argument("--threshold", type=float, default=PSI_THRESHOLD)
    parser.add_argument("--force", nargs="*", default=[], choices=["xg", "pass"],
                        help="retrain these models even without drift")
    parser.add_argument("--pass-sample-size", type=int, default=None)
    args = parser.parse_args()

    retrain_flow(
        psi_threshold=args.threshold,
        force_retrain=[f"scoutiq_{m}" for m in args.force],
        pass_sample_size=args.pass_sample_size,
    )
