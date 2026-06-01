"""
pipeline/flows.py

Prefect orchestration flows for ScoutIQ.
Defines the weekly pipeline: ingest → build features → drift check → retrain if needed.
"""
from prefect import flow, task
from loguru import logger

from pipeline.ingest import run_full_ingest, ingest_competitions, ingest_matches, ingest_events_for_match


# ── Tasks ──────────────────────────────────────────────────────────────────────

@task(name="ingest-competitions", retries=2, retry_delay_seconds=30)
def task_ingest_competitions():
    return ingest_competitions()


@task(name="ingest-matches", retries=2, retry_delay_seconds=30)
def task_ingest_matches(competition_id: int, season_id: int):
    return ingest_matches(competition_id, season_id)


@task(name="ingest-match-events", retries=3, retry_delay_seconds=10)
def task_ingest_match(match_id: int):
    shots, passes = ingest_events_for_match(match_id)
    return {"match_id": match_id, "shots": shots, "passes": passes}


@task(name="check-drift")
def task_check_drift():
    """Check for data drift using Evidently AI. Returns True if drift detected."""
    try:
        from monitoring.drift_report import check_drift
        return check_drift()
    except Exception as e:
        logger.warning(f"Drift check skipped: {e}")
        return False


@task(name="retrain-models")
def task_retrain_models():
    """Trigger model retraining via MLflow."""
    logger.info("Drift detected — triggering model retraining")
    # Import lazily to avoid heavy imports when not needed
    from models.xg.train import train_xg_model
    from models.pass_success.train import train_pass_model
    train_xg_model()
    train_pass_model()


# ── Flows ──────────────────────────────────────────────────────────────────────

@flow(name="scoutiq-full-ingest", log_prints=True)
def full_ingest_flow(
    competition_ids: list[int] | None = None,
    max_matches: int | None = None,
):
    """
    Full data ingestion flow.
    Runs on first setup or when loading a new competition.
    """
    stats = run_full_ingest(
        competition_ids=competition_ids,
        max_matches=max_matches,
    )
    logger.info(f"Full ingest complete: {stats}")
    return stats


@flow(name="scoutiq-weekly-pipeline", log_prints=True)
def weekly_pipeline_flow():
    """
    Weekly orchestration:
    1. Refresh competitions list
    2. Check for new matches and ingest events
    3. Run drift detection
    4. Retrain models if drift detected
    """
    logger.info("Weekly pipeline starting")

    # Step 1: refresh competition metadata
    comps = task_ingest_competitions()

    # Step 2: process each competition
    total_shots = total_passes = 0
    for _, comp in comps.iterrows():
        matches_df = task_ingest_matches(
            int(comp["competition_id"]), int(comp["season_id"])
        )
        for _, match_row in matches_df.iterrows():
            result = task_ingest_match(int(match_row["match_id"]))
            total_shots  += result["shots"]
            total_passes += result["passes"]

    logger.info(f"Ingested {total_shots} shots, {total_passes} passes this run")

    # Step 3: drift check
    drift_detected = task_check_drift()

    # Step 4: retrain if drift found
    if drift_detected:
        logger.warning("Data drift detected — initiating retraining")
        task_retrain_models()
    else:
        logger.info("No significant drift — skipping retraining")

    return {
        "shots": total_shots,
        "passes": total_passes,
        "drift_detected": drift_detected,
    }


if __name__ == "__main__":
    # Quick test: ingest first 5 matches from La Liga 2020/21
    # La Liga = competition_id 11, 2020/21 = season_id 42
    full_ingest_flow(competition_ids=[11], max_matches=5)
