"""
dashboard/pages/4_Monitoring.py

MLOps Monitoring view:
  - Latest drift detection summary (shots + passes)
  - Per-feature drift indicators
  - Pipeline run history
  - Links to full Evidently HTML reports
  - Manual drift check trigger
"""
import sys; sys.path.insert(0, ".")

import json
from pathlib import Path
from datetime import datetime

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from utils import load_drift_summary, list_drift_reports, get_pipeline_runs, api_health

st.set_page_config(page_title="Monitoring · ScoutIQ", page_icon="📊", layout="wide")

with st.sidebar:
    st.markdown("## 📊 Monitoring")
    st.markdown("Track data drift and pipeline health.")

# ── Header ────────────────────────────────────────────────────────────────────
st.title("📊 MLOps Monitoring")
st.markdown("Powered by **Evidently AI** · Drift threshold: **20% of features**")

# API health badge
healthy = api_health()
st.markdown(
    f"**API Status:** {'🟢 Online' if healthy else '🔴 Offline'}"
)
st.markdown("---")

# ── Drift Summary ─────────────────────────────────────────────────────────────
st.subheader("Latest Drift Report")

summary = load_drift_summary()

if summary is None:
    st.info(
        "No drift report found yet. Run the pipeline or press the button below "
        "to generate the first report."
    )
else:
    generated = summary.get("generated_at", "unknown")
    st.caption(f"Generated: {generated}")

    overall_drift = summary.get("drift_detected", False)
    drift_col, *_ = st.columns([1, 3])
    drift_col.metric(
        "Drift detected",
        "YES ⚠️" if overall_drift else "NO ✅",
        delta=None,
    )

    threshold = summary.get("threshold", 0.2)
    st.markdown(f"**Threshold:** {threshold*100:.0f}% of features must drift to trigger retraining")

    # Per-dataset drift breakdown
    dcol1, dcol2 = st.columns(2)

    for col, key, title in [(dcol1, "shots", "Shot Features"), (dcol2, "passes", "Pass Features")]:
        ds = summary.get(key, {})
        share = ds.get("share_drifted", 0.0)
        drifted = ds.get("drifted_features", [])
        ds_drift = ds.get("drift_detected", False)

        with col:
            st.markdown(f"**{title}**")
            st.progress(min(share, 1.0), text=f"{share*100:.1f}% features drifted")
            if ds_drift:
                st.warning(f"⚠️ Drift detected — {', '.join(drifted) if drifted else 'multiple features'}")
            else:
                st.success("✅ No significant drift")

            if ds.get("report_path"):
                report_path = Path(ds["report_path"])
                if report_path.exists():
                    with open(report_path) as f:
                        html_content = f.read()
                    with st.expander(f"View full {title} drift report"):
                        components.html(html_content, height=600, scrolling=True)

# ── Manual drift trigger ──────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Run Drift Check Manually")
st.markdown(
    "This runs Evidently on the most recent week of data vs a 30-day "
    "reference window. If drift exceeds the threshold, the Prefect pipeline "
    "will schedule model retraining."
)

if st.button("🔄 Run Drift Check Now", type="primary"):
    with st.spinner("Running Evidently drift analysis…"):
        try:
            from monitoring.drift_report import check_drift
            drift_found = check_drift()
            if drift_found:
                st.warning("⚠️ Drift detected — retraining will be scheduled on next Prefect run.")
            else:
                st.success("✅ No significant drift detected.")
            st.rerun()
        except Exception as e:
            st.error(f"Drift check failed: {e}")
            st.code(str(e))

# ── All drift report links ────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Drift Report History")
reports = list_drift_reports()
if not reports:
    st.info("No drift reports saved yet.")
else:
    reports_df = pd.DataFrame(reports)[["file", "dataset", "created_at"]]
    reports_df.columns = ["Filename", "Dataset", "Created"]
    st.dataframe(reports_df, use_container_width=True, hide_index=True)

# ── Pipeline run history ──────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Pipeline Run History")
try:
    runs = get_pipeline_runs(n=25)
    if runs.empty:
        st.info("No pipeline runs recorded yet.")
    else:
        def _status_icon(s):
            return {"success": "✅", "failed": "❌", "running": "🔄"}.get(s, "❓")

        runs["Status"] = runs["status"].apply(lambda s: _status_icon(s) + " " + s)
        runs["Rows"] = runs["rows_processed"].fillna(0).astype(int).apply(lambda r: f"{r:,}")

        display = runs[["started_at", "flow_name", "Status", "Rows", "error_message"]].rename(columns={
            "started_at":    "Started",
            "flow_name":     "Flow",
            "error_message": "Error",
        })
        st.dataframe(display, use_container_width=True, hide_index=True)
except Exception as e:
    st.warning(f"Could not load pipeline history: {e}")

# ── Model versions ────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Model Registry")
if healthy:
    try:
        import requests
        resp = requests.get(f"http://localhost:8000/health", timeout=3)
        if resp.ok:
            data = resp.json()
            models = data.get("models", {})
            model_rows = [{"Model": k, "Version": v} for k, v in models.items()]
            st.dataframe(pd.DataFrame(model_rows), use_container_width=True, hide_index=True)
    except Exception:
        st.info("Could not fetch model versions from API.")
else:
    st.info("Start the API to see loaded model versions.")

# ── Architecture info ─────────────────────────────────────────────────────────
st.markdown("---")
with st.expander("📐 MLOps Architecture"):
    st.markdown("""
    **Training pipeline** (Prefect weekly flow):
    1. `ingest_competitions` → `ingest_matches` → `ingest_events_for_match`
    2. `check_drift` — Evidently compares last 7 days vs 30-day reference
    3. If drift > 20% of features → `retrain_models` (xG + Pass Success)
    4. New models auto-registered in MLflow registry

    **Model serving** (FastAPI):
    - Models loaded from MLflow registry at startup (cached in-process)
    - `POST /v1/predict/xg` — xG prediction
    - `POST /v1/predict/pass` — pass success probability
    - `GET /v1/players/similar/{id}` — PCA cosine similarity

    **Monitoring** (Evidently AI):
    - `DatasetDriftMetric` — overall drift flag
    - `DataDriftPreset` — per-feature drift (K-S test for continuous, χ² for categorical)
    - `DataQualityPreset` — missing values, distribution shifts
    - Reports saved as HTML + JSON summary to `reports/drift/`
    """)
