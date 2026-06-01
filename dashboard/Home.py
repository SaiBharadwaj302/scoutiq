"""
dashboard/Home.py

ScoutIQ — Home page.
Shows high-level KPIs from the feature store and links to all four analysis views.

Run:
    cd scoutiq
    streamlit run dashboard/Home.py
"""
import streamlit as st
from utils import get_matches, get_players, api_health, _query

st.set_page_config(
    page_title="ScoutIQ",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stSidebar"] { background-color: #0f172a; }
    .metric-card {
        background: #1e293b; border-radius: 10px;
        padding: 18px 22px; text-align: center;
    }
    .metric-val { font-size: 2rem; font-weight: 800; color: #60a5fa; }
    .metric-lbl { font-size: 0.85rem; color: #94a3b8; margin-top: 4px; }
    .pill {
        display:inline-block; padding: 3px 10px; border-radius: 99px;
        font-size: 0.75rem; font-weight: 600;
    }
    .pill-green { background:#dcfce7; color:#15803d; }
    .pill-red   { background:#fee2e2; color:#b91c1c; }
</style>
""", unsafe_allow_html=True)

st.title("⚽ ScoutIQ")
st.markdown("**Production ML Football Analytics Platform** — StatsBomb open data")

# ── API status ────────────────────────────────────────────────────────────────
healthy = api_health()
status_pill = (
    '<span class="pill pill-green">● API Online</span>'
    if healthy
    else '<span class="pill pill-red">● API Offline</span>'
)
st.markdown(status_pill, unsafe_allow_html=True)
st.markdown("---")

# ── KPI cards ─────────────────────────────────────────────────────────────────
try:
    kpis = _query("""
        SELECT
            (SELECT COUNT(*) FROM shot_features)       AS total_shots,
            (SELECT SUM(is_goal::int) FROM shot_features) AS total_goals,
            (SELECT COUNT(*) FROM pass_features)       AS total_passes,
            (SELECT COUNT(*) FROM matches)             AS total_matches,
            (SELECT COUNT(DISTINCT player_id) FROM player_per90_features) AS total_players,
            (SELECT AVG(is_goal::float) FROM shot_features) AS overall_xg_rate
    """).iloc[0]

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    def _card(col, val, label):
        col.markdown(
            f'<div class="metric-card"><div class="metric-val">{val}</div>'
            f'<div class="metric-lbl">{label}</div></div>',
            unsafe_allow_html=True,
        )

    _card(c1, f"{int(kpis.total_shots):,}",   "Total Shots")
    _card(c2, f"{int(kpis.total_goals):,}",   "Goals in Dataset")
    _card(c3, f"{int(kpis.total_passes)/1e6:.1f}M", "Total Passes")
    _card(c4, f"{int(kpis.total_matches):,}", "Matches")
    _card(c5, f"{int(kpis.total_players):,}", "Players")
    _card(c6, f"{kpis.overall_xg_rate*100:.1f}%", "Overall Goal Rate")

except Exception as e:
    st.warning(f"Could not load KPIs — is the database running? ({e})")

st.markdown("---")

# ── Navigation cards ──────────────────────────────────────────────────────────
st.subheader("Explore the platform")

nav1, nav2, nav3, nav4 = st.columns(4)

with nav1:
    st.markdown("### 📈 Match Analysis")
    st.markdown(
        "Select any match to see an interactive shot map, xG timeline, "
        "and per-team expected goals breakdown."
    )

with nav2:
    st.markdown("### 🗺 Pass Heatmap")
    st.markdown(
        "Visualise pass risk across the pitch. Every arrow shows the "
        "completion probability predicted by the model."
    )

with nav3:
    st.markdown("### 🔍 Player Scouting")
    st.markdown(
        "Find tactically similar players using PCA embeddings + cosine "
        "similarity. Compare stats with a radar chart."
    )

with nav4:
    st.markdown("### 📊 Monitoring")
    st.markdown(
        "Track data drift over time with Evidently AI reports. See pipeline "
        "run history and retrain logs."
    )

st.markdown("---")

# ── Recent matches ────────────────────────────────────────────────────────────
st.subheader("Recent matches in dataset")
try:
    matches = get_matches().head(10)
    display = matches[[
        "match_date", "competition_name", "season_name",
        "home_team_name", "home_score", "away_score", "away_team_name"
    ]].rename(columns={
        "match_date": "Date",
        "competition_name": "Competition",
        "season_name": "Season",
        "home_team_name": "Home",
        "home_score": "HG",
        "away_score": "AG",
        "away_team_name": "Away",
    })
    st.dataframe(display, use_container_width=True, hide_index=True)
except Exception as e:
    st.warning(f"Could not load matches: {e}")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ScoutIQ")
    st.markdown("**v3.0** · LightGBM + XGBoost + CatBoost")
    st.markdown("---")
    st.markdown("**Models**")
    st.markdown("⚽ xG — Ensemble + Optuna")
    st.markdown("🎯 Pass Success — Ensemble + Optuna")
    st.markdown("👤 Player Similarity — PCA 32-dim")
    st.markdown("---")
    st.markdown("**Data**")
    st.markdown("StatsBomb Open Data")
    st.markdown("PostgreSQL feature store")
    st.markdown("MLflow experiment tracking")
    st.markdown("Prefect orchestration")
