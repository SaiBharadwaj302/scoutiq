"""
monitoring/drift_report.py

Data drift detection for the ScoutIQ ML pipeline.
Uses Evidently AI to compare a reference snapshot (training data)
against the latest production data from PostgreSQL.

Called by the Prefect weekly_pipeline_flow — returns True when drift
exceeds the threshold, which triggers model retraining.

Run directly for a one-off report:
    python -m monitoring.drift_report
    python -m monitoring.drift_report --report-only
"""
from __future__ import annotations

import os
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from loguru import logger

# Evidently v0.4+ API
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.metrics import DatasetDriftMetric

from db.store import get_connection

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
REPORTS_DIR         = Path(os.getenv("DRIFT_REPORTS_DIR", "reports/drift"))
DRIFT_THRESHOLD     = float(os.getenv("DRIFT_THRESHOLD", "0.2"))   # share of drifted features
REFERENCE_DAYS      = int(os.getenv("DRIFT_REFERENCE_DAYS", "30")) # how old the reference window is
CURRENT_DAYS        = int(os.getenv("DRIFT_CURRENT_DAYS",   "7"))  # recency of current window
FALLBACK_ROW_LIMIT  = int(os.getenv("DRIFT_FALLBACK_ROW_LIMIT", "100000"))  # cap on full-table fallback loads


# ── Features to monitor ───────────────────────────────────────────────────────
SHOT_MONITOR_COLS = [
    "distance_to_goal", "angle_to_goal",
    "under_pressure", "first_time", "follows_dribble",
    "open_goal", "deflected", "is_goal",
]

PASS_MONITOR_COLS = [
    "length", "under_pressure", "is_switch",
    "is_through_ball", "is_cross", "is_complete",
]


# ── Data loaders ──────────────────────────────────────────────────────────────

def _load_shot_window(days_ago_start: int, days_ago_end: int) -> pd.DataFrame:
    """Load shot features ingested within a time window."""
    query = f"""
        SELECT {', '.join(SHOT_MONITOR_COLS)}
        FROM shot_features
        WHERE ingested_at >= NOW() - INTERVAL '{days_ago_start} days'
          AND ingested_at <  NOW() - INTERVAL '{days_ago_end} days'
    """
    with get_connection() as conn:
        return pd.read_sql(query, conn)


def _load_pass_window(days_ago_start: int, days_ago_end: int) -> pd.DataFrame:
    """Load pass features ingested within a time window."""
    query = f"""
        SELECT {', '.join(PASS_MONITOR_COLS)}
        FROM pass_features
        WHERE ingested_at >= NOW() - INTERVAL '{days_ago_start} days'
          AND ingested_at <  NOW() - INTERVAL '{days_ago_end} days'
    """
    with get_connection() as conn:
        return pd.read_sql(query, conn)


def _load_all_shots() -> pd.DataFrame:
    """Fallback: load a bounded, temporally-ordered sample when time windows are too small.

    Capped at FALLBACK_ROW_LIMIT rows — pulling the entire table (millions of rows
    for pass_features) into a pandas object array can exhaust available memory.
    """
    query = f"""
        SELECT {', '.join(SHOT_MONITOR_COLS)} FROM shot_features
        ORDER BY ingested_at
        LIMIT {FALLBACK_ROW_LIMIT}
    """
    with get_connection() as conn:
        return pd.read_sql(query, conn)


def _load_all_passes() -> pd.DataFrame:
    query = f"""
        SELECT {', '.join(PASS_MONITOR_COLS)} FROM pass_features
        ORDER BY ingested_at
        LIMIT {FALLBACK_ROW_LIMIT}
    """
    with get_connection() as conn:
        return pd.read_sql(query, conn)


# ── Report generation ─────────────────────────────────────────────────────────

def _run_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    dataset_name: str,
    report_path: Path,
) -> dict:
    """
    Run Evidently drift + quality report.
    Returns dict with drift_detected, drifted_features, share_drifted.
    """
    if reference.empty or current.empty:
        logger.warning(f"{dataset_name}: empty window — skipping drift check")
        return {"drift_detected": False, "drifted_features": [], "share_drifted": 0.0}

    # Cast booleans to int so Evidently handles them correctly
    for col in reference.columns:
        if reference[col].dtype == bool:
            reference = reference.copy()
            current   = current.copy()
            reference[col] = reference[col].astype(int)
            current[col]   = current[col].astype(int)

    report = Report(metrics=[
        DatasetDriftMetric(stattest_threshold=0.05),
        DataDriftPreset(),
        DataQualityPreset(),
    ])

    report.run(reference_data=reference, current_data=current)

    # Save HTML report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report.save_html(str(report_path))
    logger.info(f"Drift report saved: {report_path}")

    # Extract summary
    result = report.as_dict()
    metrics_list = result.get("metrics", [])

    drift_detected   = False
    share_drifted    = 0.0
    drifted_features = []

    for m in metrics_list:
        if m.get("metric") == "DatasetDriftMetric":
            r = m.get("result", {})
            drift_detected = r.get("dataset_drift", False)
            share_drifted  = r.get("share_of_drifted_columns", 0.0)
        if m.get("metric") == "DataDriftTable":
            for col_info in m.get("result", {}).get("drift_by_columns", {}).values():
                if col_info.get("drift_detected", False):
                    drifted_features.append(col_info.get("column_name", "?"))

    logger.info(
        f"{dataset_name}: drift_detected={drift_detected}, "
        f"share={share_drifted:.2%}, features={drifted_features}"
    )
    return {
        "drift_detected":    drift_detected,
        "share_drifted":     share_drifted,
        "drifted_features":  drifted_features,
        "report_path":       str(report_path),
    }


def _save_summary(summary: dict) -> None:
    """Persist a JSON summary alongside the HTML reports for dashboarding."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = REPORTS_DIR / "latest_summary.json"
    summary["generated_at"] = datetime.utcnow().isoformat()
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary written to {summary_path}")


# ── Public API ────────────────────────────────────────────────────────────────

def check_drift() -> bool:
    """
    Entry point called by the Prefect pipeline.

    Strategy:
    - Reference = data ingested 30–60 days ago (stable distribution)
    - Current   = data ingested in the last 7 days (live window)
    - Fallback  = split entire dataset in half if windows are empty

    Returns True when ANY dataset exceeds DRIFT_THRESHOLD share of drifted features.
    """
    logger.info("═══ Drift Detection Starting ═══")
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    # ── Shot drift ──────────────────────────────────────────────────────────
    ref_shots = _load_shot_window(REFERENCE_DAYS + 30, REFERENCE_DAYS)
    cur_shots = _load_shot_window(CURRENT_DAYS, 0)

    # Fallback: if windows are too small, use a temporal split of all data
    if len(ref_shots) < 100 or len(cur_shots) < 100:
        logger.warning("Shot windows too small — using 80/20 temporal split of all data")
        all_shots = _load_all_shots()
        split = int(len(all_shots) * 0.8)
        ref_shots, cur_shots = all_shots.iloc[:split], all_shots.iloc[split:]

    shot_result = _run_drift_report(
        reference=ref_shots,
        current=cur_shots,
        dataset_name="shots",
        report_path=REPORTS_DIR / f"shot_drift_{ts}.html",
    )

    # ── Pass drift ──────────────────────────────────────────────────────────
    ref_passes = _load_pass_window(REFERENCE_DAYS + 30, REFERENCE_DAYS)
    cur_passes = _load_pass_window(CURRENT_DAYS, 0)

    if len(ref_passes) < 500 or len(cur_passes) < 500:
        logger.warning("Pass windows too small — using 80/20 temporal split of all data")
        all_passes = _load_all_passes()
        split = int(len(all_passes) * 0.8)
        ref_passes, cur_passes = all_passes.iloc[:split], all_passes.iloc[split:]

    pass_result = _run_drift_report(
        reference=ref_passes,
        current=cur_passes,
        dataset_name="passes",
        report_path=REPORTS_DIR / f"pass_drift_{ts}.html",
    )

    # ── Aggregate decision ──────────────────────────────────────────────────
    shot_exceeded = shot_result["share_drifted"] >= DRIFT_THRESHOLD
    pass_exceeded = pass_result["share_drifted"] >= DRIFT_THRESHOLD
    overall_drift = shot_exceeded or pass_exceeded

    summary = {
        "drift_detected": overall_drift,
        "threshold":      DRIFT_THRESHOLD,
        "shots": shot_result,
        "passes": pass_result,
    }
    _save_summary(summary)

    if overall_drift:
        logger.warning(
            f"DRIFT DETECTED — shots: {shot_result['share_drifted']:.1%} | "
            f"passes: {pass_result['share_drifted']:.1%} | "
            f"threshold: {DRIFT_THRESHOLD:.1%}"
        )
    else:
        logger.info(
            f"No significant drift — shots: {shot_result['share_drifted']:.1%} | "
            f"passes: {pass_result['share_drifted']:.1%}"
        )

    logger.info("═══ Drift Detection Complete ═══")
    return overall_drift


def load_latest_summary() -> dict | None:
    """Load the most recent drift summary JSON (used by the dashboard)."""
    summary_path = REPORTS_DIR / "latest_summary.json"
    if not summary_path.exists():
        return None
    with open(summary_path) as f:
        return json.load(f)


def list_drift_reports() -> list[dict]:
    """List all saved drift report HTML files with metadata."""
    if not REPORTS_DIR.exists():
        return []
    reports = []
    for p in sorted(REPORTS_DIR.glob("*.html"), reverse=True):
        reports.append({
            "filename":   p.name,
            "path":       str(p),
            "dataset":    "shots" if "shot" in p.name else "passes",
            "created_at": datetime.fromtimestamp(p.stat().st_mtime).isoformat(),
            "size_kb":    round(p.stat().st_size / 1024, 1),
        })
    return reports


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run ScoutIQ drift detection")
    parser.add_argument("--report-only", action="store_true",
                        help="Generate report without triggering retrain")
    parser.add_argument("--threshold", type=float, default=DRIFT_THRESHOLD,
                        help=f"Drift threshold (default: {DRIFT_THRESHOLD})")
    args = parser.parse_args()

    if args.threshold != DRIFT_THRESHOLD:
        DRIFT_THRESHOLD = args.threshold

    drift = check_drift()
    print(f"\nDrift detected: {drift}")

    summary = load_latest_summary()
    if summary:
        print(f"Shot drift:  {summary['shots']['share_drifted']:.1%}")
        print(f"Pass drift:  {summary['passes']['share_drifted']:.1%}")
        print(f"Report dir:  {REPORTS_DIR}")
