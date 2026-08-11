"""
dashboard/utils.py

Shared helpers for the ScoutIQ Streamlit dashboard:
  - Database queries (read-only, cached with st.cache_data)
  - Pitch drawing (StatsBomb coordinate system, 120×80)
  - API client for real-time predictions
  - Colour palettes and formatting helpers
"""
from __future__ import annotations

import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
import requests
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

# ── API & DB config ───────────────────────────────────────────────────────────
API_BASE   = os.getenv("API_BASE_URL", "http://localhost:8000")
REPORTS_DIR = Path(os.getenv("DRIFT_REPORTS_DIR", "reports/drift"))

def _db_url() -> str:
    return (
        f"postgresql+psycopg2://"
        f"{os.getenv('POSTGRES_USER', 'scoutiq_user')}:"
        f"{os.getenv('POSTGRES_PASSWORD', 'scoutiq_pass')}@"
        f"{os.getenv('POSTGRES_HOST', 'localhost')}:"
        f"{os.getenv('POSTGRES_PORT', '5432')}/"
        f"{os.getenv('POSTGRES_DB', 'scoutiq')}"
    )

@st.cache_resource
def _engine():
    return create_engine(_db_url(), pool_pre_ping=True)

def _query(sql: str, params: dict | None = None) -> pd.DataFrame:
    with _engine().connect() as conn:
        return pd.read_sql(text(sql), conn, params=params)


# ── Cached DB queries ─────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def get_matches() -> pd.DataFrame:
    return _query("""
        SELECT m.match_id, m.match_date, m.home_team_name, m.away_team_name,
               m.home_score, m.away_score, c.competition_name, c.season_name
        FROM matches m
        JOIN competitions c USING (competition_id, season_id)
        ORDER BY m.match_date DESC
    """)

@st.cache_data(ttl=300)
def get_shots_for_match(match_id: int) -> pd.DataFrame:
    return _query("""
        SELECT player_name, team_id, location_x, location_y,
               distance_to_goal, angle_to_goal, is_goal,
               outcome_name, technique_name, body_part_name,
               minute, period, under_pressure, first_time
        FROM shot_features
        WHERE match_id = :mid
        ORDER BY minute
    """, {"mid": match_id})

@st.cache_data(ttl=300)
def get_passes_for_match(match_id: int) -> pd.DataFrame:
    return _query("""
        SELECT player_name, team_id, start_x, start_y, end_x, end_y,
               length, is_complete, is_cross, is_through_ball,
               under_pressure, minute, period
        FROM pass_features
        WHERE match_id = :mid
        ORDER BY minute
    """, {"mid": match_id})

@st.cache_data(ttl=300)
def get_players() -> pd.DataFrame:
    return _query("""
        SELECT DISTINCT player_id, player_name, team_name,
                        position_group, minutes_played
        FROM player_per90_features
        WHERE minutes_played >= 90
        ORDER BY player_name
    """)

@st.cache_data(ttl=300)
def get_player_stats(player_id: int) -> pd.Series | None:
    df = _query("""
        SELECT * FROM player_per90_features
        WHERE player_id = :pid
        ORDER BY computed_at DESC LIMIT 1
    """, {"pid": player_id})
    if df.empty:
        return None
    row = df.iloc[0]
    return row

@st.cache_data(ttl=300)
def get_similar_players(player_id: int, top_n: int = 5) -> pd.DataFrame:
    """Call similarity API endpoint."""
    try:
        r = requests.get(
            f"{API_BASE}/v1/players/similar/{player_id}",
            params={"top_n": top_n},
            timeout=5,
        )
        r.raise_for_status()
        data = r.json()
        return pd.DataFrame(data["similar_players"])
    except Exception:
        # Fallback: cosine similarity directly in Python
        return _similarity_fallback(player_id, top_n)

@st.cache_data(ttl=60)
def get_pipeline_runs(n: int = 20) -> pd.DataFrame:
    return _query(f"""
        SELECT flow_name, status, rows_processed, error_message,
               started_at, finished_at
        FROM pipeline_runs
        ORDER BY started_at DESC
        LIMIT {n}
    """)


def _similarity_fallback(player_id: int, top_n: int) -> pd.DataFrame:
    """Compute cosine similarity in-process when API is offline."""
    df = _query("""
        SELECT player_id, player_name, embedding
        FROM player_per90_features
        WHERE embedding IS NOT NULL
    """)
    if df.empty or player_id not in df["player_id"].values:
        return pd.DataFrame()
    embs = np.array(df["embedding"].tolist(), dtype=float)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normed = embs / norms
    idx = df.index[df["player_id"] == player_id][0]
    sims = normed @ normed[idx]
    sims[idx] = -1
    top = np.argsort(sims)[::-1][:top_n]
    return pd.DataFrame({
        "player_id":   df.iloc[top]["player_id"].values,
        "player_name": df.iloc[top]["player_name"].values,
        "similarity":  sims[top].round(4),
    })


# ── Pitch drawing ──────────────────────────────────────────────────────────────

PITCH_COLOR  = "#1a6b3c"
LINE_COLOR   = "white"
GOAL_COLOR   = "#f0f0f0"

def draw_pitch(ax: plt.Axes, half: str = "full") -> None:
    """
    Draw a StatsBomb pitch (120×80) on ax.
    half: "full" | "attack" (shows only final 60m)
    """
    ax.set_facecolor(PITCH_COLOR)
    kw = dict(color=LINE_COLOR, linewidth=1.5)

    if half == "attack":
        ax.set_xlim(60, 122)
        ax.set_ylim(-2, 82)
    else:
        ax.set_xlim(-2, 122)
        ax.set_ylim(-2, 82)

    ax.set_aspect("equal")
    ax.axis("off")

    # Pitch outline
    ax.plot([0,120,120,0,0], [0,0,80,80,0], **kw)
    # Halfway line
    if half == "full":
        ax.plot([60,60], [0,80], **kw)
        # Centre circle
        centre = plt.Circle((60,40), 10, fill=False, **kw)
        ax.add_patch(centre)
        ax.plot(60, 40, "o", color=LINE_COLOR, ms=3)

    # Penalty areas
    for x0, x1 in [(0,18), (102,120)]:
        ax.plot([x0,x0,x1,x1], [18,62,62,18], **kw)      # 18-yard box
    for x0, x1 in [(0,6), (114,120)]:
        ax.plot([x0,x0,x1,x1], [30,50,50,30], **kw)       # 6-yard box

    # Penalty spots
    ax.plot(12, 40, "o", color=LINE_COLOR, ms=3)
    ax.plot(108, 40, "o", color=LINE_COLOR, ms=3)

    # Penalty arcs
    for cx, theta1, theta2 in [(12, -53, 53), (108, 127, 233)]:
        arc = Arc((cx,40), 20, 20, angle=0,
                  theta1=theta1, theta2=theta2, **kw)
        ax.add_patch(arc)

    # Goals
    for x in [0, 120]:
        gx = x - 2 if x == 0 else x
        ax.plot([gx, gx if x==0 else x+2, gx if x==0 else x+2, gx],
                [36, 36, 44, 44], color=GOAL_COLOR, linewidth=2)


# ── Colour helpers ────────────────────────────────────────────────────────────

def xg_color(xg: float) -> str:
    """Red-yellow-green gradient mapped to xG value."""
    xg = max(0, min(1, xg))
    if xg < 0.1:
        return "#4ade80"   # green
    if xg < 0.2:
        return "#a3e635"
    if xg < 0.35:
        return "#facc15"   # yellow
    if xg < 0.5:
        return "#fb923c"   # orange
    return "#f87171"                   # red


def risk_color(prob: float) -> str:
    """Green (safe) → Red (risky) for pass completion probability."""
    prob = max(0, min(1, prob))
    if prob > 0.85:
        return "#22c55e"
    if prob > 0.65:
        return "#84cc16"
    if prob > 0.45:
        return "#eab308"
    if prob > 0.25:
        return "#f97316"
    return "#ef4444"


# ── API client ────────────────────────────────────────────────────────────────

def predict_xg(payload: dict) -> float | None:
    try:
        r = requests.post(f"{API_BASE}/v1/predict/xg", json=payload, timeout=5)
        r.raise_for_status()
        return r.json()["xg"]
    except Exception:
        return None

def predict_pass(payload: dict) -> float | None:
    try:
        r = requests.post(f"{API_BASE}/v1/predict/pass", json=payload, timeout=5)
        r.raise_for_status()
        return r.json()["completion_probability"]
    except Exception:
        return None

def api_health() -> bool:
    try:
        r = requests.get(f"{API_BASE}/health", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


# ── Radar chart ───────────────────────────────────────────────────────────────

RADAR_STATS = {
    "Goals p90":       "goals_per90",
    "Shots p90":       "shots_per90",
    "xG p90":          "xg_per90",
    "Passes p90":      "passes_per90",
    "Pass%":           "pass_completion_pct",
    "Prog passes p90": "progressive_passes_per90",
    "Key passes p90":  "key_passes_per90",
    "Assists p90":     "assists_per90",
}

def draw_radar(
    ax: plt.Axes,
    stats_a: pd.Series,
    stats_b: pd.Series | None,
    label_a: str,
    label_b: str = "",
    max_vals: pd.Series | None = None,
) -> None:
    """Draw a radar/spider chart comparing up to 2 players."""
    labels = list(RADAR_STATS.keys())
    cols   = list(RADAR_STATS.values())
    N      = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]   # close polygon

    def _vals(s: pd.Series) -> list[float]:
        raw = [float(s.get(c, 0) or 0) for c in cols]
        if max_vals is not None:
            normed = [v / max(float(max_vals.get(c, 1) or 1), 1e-9) for v, c in zip(raw, cols)]
        else:
            normed = [min(v, 1.0) for v in raw]
        return normed + normed[:1]

    ax.set_facecolor("#0f172a")
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=8, color="white")
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25%","50%","75%","100%"], fontsize=6, color="#94a3b8")
    ax.set_ylim(0, 1)
    ax.spines["polar"].set_color("#334155")
    ax.grid(color="#334155", linewidth=0.8)

    # Player A (blue)
    va = _vals(stats_a)
    ax.plot(angles, va, color="#60a5fa", linewidth=2, label=label_a)
    ax.fill(angles, va, color="#60a5fa", alpha=0.25)

    # Player B (orange) — optional
    if stats_b is not None:
        vb = _vals(stats_b)
        ax.plot(angles, vb, color="#fb923c", linewidth=2, label=label_b)
        ax.fill(angles, vb, color="#fb923c", alpha=0.25)

    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1),
              fontsize=8, frameon=False, labelcolor="white")


# ── Drift summary ─────────────────────────────────────────────────────────────

def load_drift_summary() -> dict | None:
    p = REPORTS_DIR / "latest_summary.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)

def list_drift_reports() -> list[dict]:
    if not REPORTS_DIR.exists():
        return []
    out = []
    for p in sorted(REPORTS_DIR.glob("*.html"), reverse=True):
        out.append({
            "file":    p.name,
            "dataset": "Shots" if "shot" in p.name else "Passes",
            "path":    str(p),
        })
    return out
