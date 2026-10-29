"""
pipeline/retrain_deploy.py

Weekly schedule for the drift-triggered retraining flow.

    prefect server start              # scheduler — ephemeral mode can't fire schedules
    python -m pipeline.retrain_deploy  # second shell, PREFECT_API_URL=http://127.0.0.1:4200/api

`serve()` registers the deployment with the Prefect API and keeps this process
running as the worker that executes scheduled runs — no work pool needed.
Every Monday 06:00 UTC. Trigger an ad-hoc run from another shell with:
    prefect deployment run "scoutiq-drift-retrain/weekly-drift-retrain"
"""
from pipeline.retrain_flow import retrain_flow

if __name__ == "__main__":
    retrain_flow.serve(
        name="weekly-drift-retrain",
        cron="0 6 * * 1",
        tags=["scoutiq", "retraining"],
        description="Weekly PSI drift check; retrain + gated promotion on drift.",
    )
