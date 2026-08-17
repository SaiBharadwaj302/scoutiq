"""
scripts/export_dashboard_bundle.py

One-off export: freezes the current xG model, pass model, and the full
player_per90_features table into plain, portable files under
streamlit_app/models_export/ — so the standalone dashboard (deployed to
Streamlit Community Cloud) never needs a live Postgres connection or
MLflow's local file-store registry, both of which only exist on this
machine.

Run locally (with Postgres up and models trained/registered) whenever the
models are retrained or the player data changes:
    python -m scripts.export_dashboard_bundle
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import pandas as pd
from loguru import logger

from api.dependencies import get_xg_model, get_pass_model
from db.store import get_connection

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "streamlit_app", "models_export")


def export_models():
    os.makedirs(OUT_DIR, exist_ok=True)

    logger.info("Loading xG model from MLflow registry...")
    xg_model = get_xg_model()
    joblib.dump(xg_model, os.path.join(OUT_DIR, "xg_model.joblib"))
    logger.info("Saved xg_model.joblib")

    logger.info("Loading pass model from MLflow registry...")
    pass_model = get_pass_model()
    joblib.dump(pass_model, os.path.join(OUT_DIR, "pass_model.joblib"))
    logger.info("Saved pass_model.joblib")


def export_players():
    query = """
        SELECT player_id, player_name, goals_per90, passes_per90,
               pass_completion_pct, minutes_played, embedding
        FROM player_per90_features
    """
    with get_connection() as conn:
        df = pd.read_sql(query, conn)

    # Postgres double precision[] comes back as Python lists — parquet
    # (pyarrow) handles list columns natively, no extra encoding needed.
    path = os.path.join(OUT_DIR, "players.parquet")
    df.to_parquet(path, index=False)
    logger.info(f"Saved players.parquet — {len(df)} rows")


if __name__ == "__main__":
    export_models()
    export_players()
    logger.info(f"═══ Export complete: {OUT_DIR} ═══")
