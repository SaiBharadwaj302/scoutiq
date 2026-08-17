"""
streamlit_app/app.py

ScoutIQ — standalone demo dashboard.

Unlike the root app.py (which talks to a live Postgres DB + MLflow model
registry — the "production" setup, deployed via Docker on EC2), this is a
self-contained build meant for free hosting on Streamlit Community Cloud:
models and player data are frozen into streamlit_app/models_export/ by
scripts/export_dashboard_bundle.py, so this app has zero external
dependencies at runtime. Re-run that export script and commit the refreshed
files whenever the models are retrained.

Run with: streamlit run streamlit_app/app.py
"""
import os
import math

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import plotly.graph_objects as go

from features import build_xg_features, build_pass_features
from similarity import find_similar

# ── Palette ──────────────────────────────────────────────────────────────────
BG          = "#0b1120"
SURFACE     = "#141b2d"
BORDER      = "#293347"
TEXT        = "#e2e8f0"
TEXT_MUTED  = "#94a3b8"
ACCENT      = "#22c55e"
ACCENT_SOFT = "#16a34a"
ACCENT_2    = "#22d3ee"
INFO        = "#38bdf8"
WARNING     = "#f59e0b"
DANGER      = "#ef4444"
PITCH_DARK  = "#146334"
PITCH_LIGHT = "#1a7a41"

BUNDLE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models_export")

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ScoutIQ — Football Analytics",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_css():
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }}

    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header[data-testid="stHeader"] {{background: transparent;}}

    .block-container {{
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }}

    @keyframes drift {{
        0%   {{ background-position: 0% 0%, 100% 100%, 0 0; }}
        50%  {{ background-position: 100% 40%, 10% 60%, 0 0; }}
        100% {{ background-position: 0% 0%, 100% 100%, 0 0; }}
    }}
    .stApp {{
        background:
            radial-gradient(circle at 15% 20%, rgba(34,197,94,0.14) 0%, transparent 42%),
            radial-gradient(circle at 85% 75%, rgba(34,211,238,0.12) 0%, transparent 42%),
            {BG};
        background-size: 180% 180%, 180% 180%, 100% 100%;
        animation: drift 26s ease-in-out infinite;
    }}

    @keyframes fadeInUp {{
        from {{ opacity: 0; transform: translateY(14px); }}
        to   {{ opacity: 1; transform: translateY(0); }}
    }}
    @keyframes pulseDot {{
        0%, 100% {{ box-shadow: 0 0 0 0 rgba(34,197,94,0.55); }}
        50%      {{ box-shadow: 0 0 0 7px rgba(34,197,94,0); }}
    }}

    .scoutiq-hero {{
        background: rgba(20,27,45,0.55);
        backdrop-filter: blur(18px);
        -webkit-backdrop-filter: blur(18px);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 20px;
        padding: 32px 36px;
        margin-bottom: 26px;
        position: relative;
        overflow: hidden;
        box-shadow: 0 8px 32px rgba(0,0,0,0.35);
        animation: fadeInUp 0.5s ease both;
    }}
    .scoutiq-hero::before {{
        content: "";
        position: absolute; top: -50%; right: -10%;
        width: 340px; height: 340px; border-radius: 50%;
        background: radial-gradient(circle, rgba(34,197,94,0.22) 0%, transparent 70%);
        pointer-events: none;
    }}
    .scoutiq-hero::after {{
        content: "";
        position: absolute; bottom: -60%; left: -5%;
        width: 280px; height: 280px; border-radius: 50%;
        background: radial-gradient(circle, rgba(34,211,238,0.16) 0%, transparent 70%);
        pointer-events: none;
    }}
    .scoutiq-hero h1 {{
        font-size: 2.1rem; font-weight: 900; margin: 0 0 8px 0;
        letter-spacing: -0.02em;
        background: linear-gradient(90deg, #ffffff 10%, {ACCENT} 55%, {ACCENT_2} 100%);
        -webkit-background-clip: text; background-clip: text; color: transparent;
        position: relative; z-index: 1;
    }}
    .scoutiq-hero p {{
        font-size: 1rem; color: {TEXT_MUTED}; margin: 0; max-width: 640px;
        position: relative; z-index: 1;
    }}
    .hero-pills {{ margin-top: 16px; display: flex; gap: 8px; flex-wrap: wrap; position: relative; z-index: 1; }}
    .hero-pill {{
        background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.12);
        padding: 5px 13px; border-radius: 999px; font-size: 0.75rem;
        color: {TEXT_MUTED}; font-weight: 500;
    }}

    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, rgba(11,17,32,0.99), rgba(11,17,32,0.94));
        border-right: 1px solid rgba(255,255,255,0.06);
    }}
    .sidebar-brand {{ display: flex; align-items: center; gap: 10px; padding: 4px 0 6px 0; }}
    .sidebar-brand .badge {{
        width: 42px; height: 42px; border-radius: 12px;
        background: linear-gradient(135deg, {ACCENT} 0%, {ACCENT_2} 100%);
        display: flex; align-items: center; justify-content: center; font-size: 21px;
        box-shadow: 0 4px 18px rgba(34,197,94,0.4);
    }}
    .sidebar-brand .name {{ font-weight: 800; font-size: 1.18rem; color: {TEXT}; line-height: 1.1; }}
    .sidebar-brand .tag {{
        font-size: 0.68rem; color: {TEXT_MUTED}; text-transform: uppercase;
        letter-spacing: 0.06em; display: flex; align-items: center; gap: 5px;
    }}
    .live-dot {{
        width: 6px; height: 6px; border-radius: 50%;
        background: {ACCENT}; display: inline-block; animation: pulseDot 2s infinite;
    }}

    .model-chip {{
        display: flex; align-items: center; justify-content: space-between;
        background: rgba(255,255,255,0.03); border: 1px solid {BORDER}; border-radius: 10px;
        padding: 9px 12px; margin-bottom: 8px; font-size: 0.82rem;
        transition: border-color 0.2s ease, background 0.2s ease;
    }}
    .model-chip:hover {{ border-color: rgba(34,197,94,0.4); background: rgba(34,197,94,0.06); }}
    .model-chip .dot {{
        width: 7px; height: 7px; border-radius: 50%;
        background: {ACCENT}; display: inline-block; margin-right: 8px;
        animation: pulseDot 2.4s infinite;
    }}
    .model-chip .label {{ color: {TEXT_MUTED}; }}
    .model-chip .val {{ color: {ACCENT}; font-weight: 700; }}

    section[data-testid="stSidebar"] [role="radiogroup"] label {{
        border-radius: 12px; padding: 9px 14px !important; margin-bottom: 5px;
        border: 1px solid transparent; transition: all 0.2s ease;
    }}
    section[data-testid="stSidebar"] [role="radiogroup"] label:hover {{
        background: rgba(34,197,94,0.08); border-color: rgba(34,197,94,0.25);
    }}

    div[data-testid="stVerticalBlockBorderWrapper"] {{
        border-radius: 18px !important;
        border-color: rgba(255,255,255,0.08) !important;
        background: rgba(20,27,45,0.55) !important;
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        box-shadow: 0 4px 24px rgba(0,0,0,0.22);
        transition: box-shadow 0.25s ease, transform 0.25s ease, border-color 0.25s ease;
    }}
    div[data-testid="stVerticalBlockBorderWrapper"]:hover {{
        box-shadow: 0 10px 34px rgba(34,197,94,0.14);
        border-color: rgba(34,197,94,0.25) !important;
    }}

    [data-testid="stMetric"] {{
        background: linear-gradient(160deg, rgba(30,41,59,0.85), rgba(20,27,45,0.85));
        border: 1px solid {BORDER}; border-top: 3px solid {ACCENT}; border-radius: 14px;
        padding: 15px 18px 11px 18px; box-shadow: 0 4px 18px rgba(0,0,0,0.28);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }}
    [data-testid="stMetric"]:hover {{
        transform: translateY(-3px); box-shadow: 0 12px 30px rgba(34,197,94,0.18);
    }}
    [data-testid="stMetricLabel"] {{
        color: {TEXT_MUTED} !important; font-size: 0.76rem !important;
        text-transform: uppercase; letter-spacing: 0.05em;
    }}
    [data-testid="stMetricValue"] {{
        background: linear-gradient(90deg, #ffffff, {ACCENT});
        -webkit-background-clip: text; background-clip: text; color: transparent !important;
        font-weight: 800 !important;
    }}

    .verdict {{
        border-radius: 14px; padding: 16px 20px; margin-top: 12px;
        font-size: 0.95rem; line-height: 1.6; border-left: 4px solid;
        backdrop-filter: blur(10px); animation: fadeInUp 0.4s ease both;
    }}
    .verdict-good {{ background: rgba(34,197,94,0.09); border-color: {ACCENT}; color: #86efac; box-shadow: 0 4px 22px rgba(34,197,94,0.08); }}
    .verdict-mid  {{ background: rgba(56,189,248,0.09); border-color: {INFO};   color: #7dd3fc; box-shadow: 0 4px 22px rgba(56,189,248,0.08); }}
    .verdict-bad  {{ background: rgba(239,68,68,0.09);  border-color: {DANGER}; color: #fca5a5; box-shadow: 0 4px 22px rgba(239,68,68,0.08); }}

    [data-testid="stSlider"] [role="slider"] {{ box-shadow: 0 0 0 6px rgba(34,197,94,0.16) !important; }}
    [data-baseweb="select"] > div {{
        background: rgba(30,41,59,0.6) !important; border-color: {BORDER} !important; border-radius: 10px !important;
    }}

    h2, h3 {{ color: {TEXT}; font-weight: 700; border-left: 3px solid {ACCENT}; padding-left: 10px; }}
    hr {{ border-color: {BORDER}; }}

    ::-webkit-scrollbar {{ width: 10px; height: 10px; }}
    ::-webkit-scrollbar-track {{ background: {BG}; }}
    ::-webkit-scrollbar-thumb {{ background: linear-gradient(180deg, {ACCENT}, {ACCENT_SOFT}); border-radius: 10px; }}
    ::-webkit-scrollbar-thumb:hover {{ background: {ACCENT_2}; }}
    </style>
    """, unsafe_allow_html=True)


def hero(title: str, subtitle: str, pills=None):
    pills_html = ""
    if pills:
        chips = "".join(f'<span class="hero-pill">{p}</span>' for p in pills)
        pills_html = f'<div class="hero-pills">{chips}</div>'
    st.markdown(f"""
    <div class="scoutiq-hero">
        <h1>{title}</h1>
        <p>{subtitle}</p>
        {pills_html}
    </div>
    """, unsafe_allow_html=True)


def verdict_box(kind: str, text: str):
    cls = {"good": "verdict-good", "mid": "verdict-mid", "bad": "verdict-bad"}[kind]
    st.markdown(f'<div class="verdict {cls}">{text}</div>', unsafe_allow_html=True)


# ── Load bundled models & data (cached) ───────────────────────────────────────
@st.cache_resource
def load_models():
    xg_model   = joblib.load(os.path.join(BUNDLE_DIR, "xg_model.joblib"))
    pass_model = joblib.load(os.path.join(BUNDLE_DIR, "pass_model.joblib"))
    return xg_model, pass_model


@st.cache_data
def load_players_bundle():
    return pd.read_parquet(os.path.join(BUNDLE_DIR, "players.parquet"))


@st.cache_data
def load_player_list():
    """Mirrors the original SQL: named players with >200 minutes, top 500 by goals/90."""
    df = load_players_bundle()
    mask = (
        df["player_name"].notna()
        & (df["player_name"] != "")
        & (df["player_name"] != "0")
        & (df["minutes_played"] > 200)
    )
    return (df[mask]
            .sort_values("goals_per90", ascending=False, na_position="last")
            .head(500))


@st.cache_resource
def load_embeddings():
    """Mirrors the original SQL: all players with a non-null embedding, ordered by id."""
    df = load_players_bundle()
    emb_df = df[df["embedding"].notna()].sort_values("player_id")
    embeddings = np.array([
        e if isinstance(e, list) else list(e) for e in emb_df["embedding"]
    ], dtype=float)
    player_ids   = emb_df["player_id"].astype(int).tolist()
    player_names = emb_df["player_name"].fillna("Unknown").tolist()
    return embeddings, player_ids, player_names


# ── Pitch Drawing ─────────────────────────────────────────────────────────────

def draw_pitch(ax, linecolor="#f8fafc"):
    """Draw a StatsBomb pitch (120 x 80) with a broadcast-style mown-stripe turf."""
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 80)
    ax.set_aspect("equal")
    ax.axis("off")

    n_stripes = 12
    stripe_w = 120 / n_stripes
    for i in range(n_stripes):
        shade = PITCH_DARK if i % 2 == 0 else PITCH_LIGHT
        ax.add_patch(patches.Rectangle((i * stripe_w, 0), stripe_w, 80,
                     facecolor=shade, edgecolor="none", zorder=0))

    lc, lw, alpha = linecolor, 1.6, 0.9

    ax.add_patch(patches.Rectangle((0, 0), 120, 80,
                 fill=False, edgecolor=lc, linewidth=lw, alpha=alpha, zorder=1))
    ax.plot([60, 60], [0, 80], color=lc, linewidth=lw, alpha=alpha, zorder=1)
    centre = plt.Circle((60, 40), 10, fill=False, color=lc, linewidth=lw, alpha=alpha, zorder=1)
    ax.add_patch(centre)
    ax.plot(60, 40, "o", color=lc, markersize=3, alpha=alpha, zorder=1)

    ax.add_patch(patches.Rectangle((0, 18), 18, 44,
                 fill=False, edgecolor=lc, linewidth=lw, alpha=alpha, zorder=1))
    ax.add_patch(patches.Rectangle((102, 18), 18, 44,
                 fill=False, edgecolor=lc, linewidth=lw, alpha=alpha, zorder=1))
    ax.add_patch(patches.Rectangle((0, 30), 6, 20,
                 fill=False, edgecolor=lc, linewidth=lw, alpha=alpha, zorder=1))
    ax.add_patch(patches.Rectangle((114, 30), 6, 20,
                 fill=False, edgecolor=lc, linewidth=lw, alpha=alpha, zorder=1))
    ax.add_patch(patches.Rectangle((-2, 36), 2, 8,
                 fill=True, facecolor=lc, edgecolor=lc, linewidth=lw, alpha=alpha, zorder=1))
    ax.add_patch(patches.Rectangle((120, 36), 2, 8,
                 fill=True, facecolor=lc, edgecolor=lc, linewidth=lw, alpha=alpha, zorder=1))
    ax.plot(12, 40, "o", color=lc, markersize=3, alpha=alpha, zorder=1)
    ax.plot(108, 40, "o", color=lc, markersize=3, alpha=alpha, zorder=1)


def glow_marker(ax, x, y, color, size=18, zorder=5):
    for mult, a in [(2.6, 0.10), (1.9, 0.16), (1.3, 0.28)]:
        ax.plot(x, y, "o", color=color, markersize=size * mult, alpha=a, zorder=zorder - 1)
    ax.plot(x, y, "o", color=color, markersize=size, alpha=0.95, zorder=zorder)
    ax.plot(x, y, "o", color="white", markersize=size * 0.32, zorder=zorder + 1)


def xg_color(xg: float) -> str:
    if xg < 0.05:
        return INFO
    if xg < 0.15:
        return ACCENT
    if xg < 0.30:
        return WARNING
    return DANGER


# ── Prediction Helpers ────────────────────────────────────────────────────────

GOAL_CENTRE = (120.0, 40.0)

def dist_to_goal(x, y):
    return math.sqrt((x - GOAL_CENTRE[0])**2 + (y - GOAL_CENTRE[1])**2)

def angle_to_goal(x, y):
    dx1, dy1 = GOAL_CENTRE[0] - x, 36.0 - y
    dx2, dy2 = GOAL_CENTRE[0] - x, 44.0 - y
    cross = abs(dx1 * dy2 - dy1 * dx2)
    dot   = dx1 * dx2 + dy1 * dy2
    return math.atan2(cross, dot)


def predict_xg(model, loc_x, loc_y, technique, body_part,
               play_pattern, under_pressure, first_time, period):
    df = pd.DataFrame([{
        "distance_to_goal":           dist_to_goal(loc_x, loc_y),
        "angle_to_goal":              angle_to_goal(loc_x, loc_y),
        "under_pressure":             under_pressure,
        "first_time":                 first_time,
        "follows_dribble":            False,
        "open_goal":                  False,
        "deflected":                  False,
        "technique_name":             technique,
        "body_part_name":             body_part,
        "play_pattern_name":          play_pattern,
        "period":                     period,
        "defenders_in_2m_cone":       None,
        "defenders_in_5m_cone":       None,
        "gk_distance_to_goal_centre": None,
        "gk_x": None, "gk_y": None,
        "attackers_in_box":           None,
        "defenders_in_box":           None,
        "numerical_advantage":        None,
        "nearest_defender_distance":  None,
    }])
    features = build_xg_features(df)
    return float(model.predict_proba(features)[0][1])


def predict_pass(model, sx, sy, ex, ey, height, body_part,
                 under_pressure, is_cross, is_switch, period):
    length = math.sqrt((ex - sx)**2 + (ey - sy)**2)
    angle  = math.atan2(ey - sy, ex - sx)
    df = pd.DataFrame([{
        "length": length, "angle": angle,
        "under_pressure": under_pressure,
        "is_switch": is_switch, "is_through_ball": False,
        "is_cut_back": False, "is_cross": is_cross,
        "goal_assist": False, "shot_assist": False,
        "start_x": sx, "start_y": sy, "end_x": ex, "end_y": ey,
        "height_name": height, "body_part_name": body_part,
        "play_pattern_name": "Regular Play", "period": period,
        "defenders_in_pass_lane": None, "defender_density_endpoint": None,
        "nearest_defender_distance": None, "nearest_opponent_at_end": None,
    }])
    features = build_pass_features(df)
    return float(model.predict_proba(features)[0][1])


# ── Main App ──────────────────────────────────────────────────────────────────

def main():
    inject_css()

    st.sidebar.markdown("""
    <div class="sidebar-brand">
        <div class="badge">⚽</div>
        <div>
            <div class="name">ScoutIQ</div>
            <div class="tag"><span class="live-dot"></span> Football Analytics — Demo</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    page = st.sidebar.radio(
        "Navigate",
        ["🎯 xG Predictor", "🔄 Pass Analyzer", "🔍 Player Similarity"],
        label_visibility="collapsed",
    )

    st.sidebar.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    st.sidebar.markdown('<div style="color:#94a3b8;font-size:0.72rem;text-transform:uppercase;'
                         'letter-spacing:0.06em;margin-bottom:8px;">Models</div>',
                         unsafe_allow_html=True)
    for label, val in [("xG Model", "AUC 0.808"),
                        ("Pass Model", "AUC 0.905"),
                        ("Similarity", "5,653 players")]:
        st.sidebar.markdown(f"""
        <div class="model-chip">
            <span class="label"><span class="dot"></span>{label}</span>
            <span class="val">{val}</span>
        </div>
        """, unsafe_allow_html=True)

    st.sidebar.caption("Trained on StatsBomb open data — 88K shots · 3.4M passes")
    st.sidebar.caption("📦 Standalone demo build — frozen model snapshot, no live database.")

    with st.spinner("Loading models..."):
        try:
            xg_model, pass_model = load_models()
            models_loaded = True
        except Exception as e:
            st.error(f"Model loading failed: {e}")
            models_loaded = False

    # ── xG PAGE ──────────────────────────────────────────────────────────────
    if page == "🎯 xG Predictor":
        hero("🎯 Expected Goals (xG) Predictor",
             "Place a shot on the pitch and get the model's goal probability instantly.",
             pills=["⚡ Real-time inference", "🌲 LightGBM + Logistic stacking", "📊 AUC 0.808"])

        col1, col2 = st.columns([1, 1])

        with col1:
            with st.container(border=True):
                st.subheader("Shot Details")
                loc_x = st.slider("Shot X (distance along pitch)", 60.0, 120.0, 105.0, 0.5)
                loc_y = st.slider("Shot Y (width)", 0.0, 80.0, 38.0, 0.5)
                technique    = st.selectbox("Technique", ["Normal", "Volley", "Half Volley", "Lob", "Backheel"])
                body_part    = st.selectbox("Body Part", ["Right Foot", "Left Foot", "Head"])
                play_pattern = st.selectbox("Play Pattern", ["Regular Play", "From Counter", "From Corner", "From Free Kick", "From Throw In"])
                period       = st.radio("Period", [1, 2], horizontal=True)
                under_pressure = st.checkbox("Under Pressure")
                first_time     = st.checkbox("First Time Shot")

        with col2:
            with st.container(border=True):
                st.subheader("Pitch Visualization")
                fig, ax = plt.subplots(figsize=(8, 5.5))
                fig.patch.set_facecolor(SURFACE)
                draw_pitch(ax)

                if models_loaded:
                    xg = predict_xg(xg_model, loc_x, loc_y, technique,
                                    body_part, play_pattern, under_pressure,
                                    first_time, period)
                    color = xg_color(xg)
                    glow_marker(ax, loc_x, loc_y, color, size=16)
                    ax.annotate("", xy=(120, 40), xytext=(loc_x, loc_y),
                        arrowprops=dict(arrowstyle="->", color="white",
                                        lw=1.5, alpha=0.55))
                    ax.text(loc_x - 2, loc_y + 4,
                            f"xG = {xg:.3f}", color="white",
                            fontsize=11, fontweight="bold",
                            bbox=dict(boxstyle="round,pad=0.35",
                                      facecolor=color, edgecolor="none", alpha=0.92))

                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

        if models_loaded:
            dist  = dist_to_goal(loc_x, loc_y)
            angle = math.degrees(angle_to_goal(loc_x, loc_y))

            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("xG", f"{xg:.3f}")
            m2.metric("Distance", f"{dist:.1f}m")
            m3.metric("Angle", f"{angle:.1f}°")
            m4.metric("Big Chance", "Yes ✅" if xg > 0.3 else "No ❌")

            if xg > 0.3:
                verdict_box("bad", f"🔥 <b>Big Chance!</b> xG of {xg:.3f} — the shooter is expected to score roughly 1 in {int(1/xg)} shots from here.")
            elif xg > 0.1:
                verdict_box("mid", f"⚡ <b>Decent chance.</b> xG of {xg:.3f} — about 1 in {int(1/xg)} shots from this position result in a goal.")
            else:
                verdict_box("good", f"📉 <b>Low probability shot.</b> xG of {xg:.3f} — this is a difficult chance.")

    # ── PASS PAGE ─────────────────────────────────────────────────────────────
    elif page == "🔄 Pass Analyzer":
        hero("🔄 Pass Success Analyzer",
             "Define a pass and see the model's predicted completion probability.",
             pills=["⚡ Real-time inference", "🌲 LightGBM + Platt scaling", "📊 AUC 0.905"])

        col1, col2 = st.columns([1, 1])

        with col1:
            with st.container(border=True):
                st.subheader("Pass Details")
                st.markdown("**Start Position**")
                sx = st.slider("Start X", 0.0, 120.0, 50.0, 0.5)
                sy = st.slider("Start Y", 0.0, 80.0, 40.0, 0.5)
                st.markdown("**End Position**")
                ex = st.slider("End X", 0.0, 120.0, 80.0, 0.5)
                ey = st.slider("End Y", 0.0, 80.0, 35.0, 0.5)
                height       = st.selectbox("Height", ["Ground Pass", "Low Pass", "High Pass"])
                body_part    = st.selectbox("Body Part", ["Right Foot", "Left Foot", "Head"])
                period       = st.radio("Period", [1, 2], horizontal=True)
                under_pressure = st.checkbox("Under Pressure")
                is_cross       = st.checkbox("Cross")
                is_switch      = st.checkbox("Switch")

        with col2:
            with st.container(border=True):
                st.subheader("Pitch Visualization")
                fig, ax = plt.subplots(figsize=(8, 5.5))
                fig.patch.set_facecolor(SURFACE)
                draw_pitch(ax)

                if models_loaded:
                    prob = predict_pass(pass_model, sx, sy, ex, ey,
                                        height, body_part, under_pressure,
                                        is_cross, is_switch, period)
                    color = ACCENT if prob > 0.7 else WARNING if prob > 0.5 else DANGER

                    ax.annotate("", xy=(ex, ey), xytext=(sx, sy),
                        arrowprops=dict(arrowstyle="->", color=color, lw=2.8,
                                        alpha=0.95))
                    glow_marker(ax, sx, sy, "white", size=11)
                    ax.plot(ex, ey, "D", color=color, markersize=11, zorder=6,
                            markeredgecolor="white", markeredgewidth=1.2)

                    mid_x = (sx + ex) / 2
                    mid_y = (sy + ey) / 2
                    ax.text(mid_x, mid_y + 2,
                            f"{prob*100:.0f}%", color="white",
                            fontsize=11, fontweight="bold", ha="center",
                            bbox=dict(boxstyle="round,pad=0.35",
                                      facecolor=color, edgecolor="none", alpha=0.92))

                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

        if models_loaded:
            length = math.sqrt((ex - sx)**2 + (ey - sy)**2)
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Completion %", f"{prob*100:.1f}%")
            m2.metric("Pass Length", f"{length:.1f}m")
            m3.metric("Risk", "High 🔴" if prob < 0.5 else "Medium 🟡" if prob < 0.7 else "Low 🟢")
            m4.metric("Forward Pass", "Yes" if ex > sx else "No")

            if prob > 0.8:
                verdict_box("good", f"✅ <b>High completion probability</b> ({prob*100:.0f}%) — safe pass.")
            elif prob > 0.5:
                verdict_box("mid", f"⚡ <b>Moderate risk</b> ({prob*100:.0f}%) — playable but needs precision.")
            else:
                verdict_box("bad", f"🚨 <b>High risk pass</b> ({prob*100:.0f}%) — likely to be intercepted.")

    # ── SIMILARITY PAGE ───────────────────────────────────────────────────────
    elif page == "🔍 Player Similarity":
        hero("🔍 Player Similarity Engine",
             "Find tactically similar players using PCA embeddings and cosine similarity.",
             pills=["🧬 32-dim PCA embeddings", "📐 Cosine similarity", "👥 5,653 players"])

        try:
            players_df = load_player_list()
            embeddings, player_ids, player_names = load_embeddings()
        except Exception as e:
            st.error(f"Could not load player data: {e}")
            return

        col1, col2 = st.columns([1, 2])

        with col1:
            with st.container(border=True):
                st.subheader("Search")
                player_options = players_df["player_name"].dropna().tolist()
                selected_name  = st.selectbox("Select a player", player_options)
                top_n = st.slider("Number of similar players", 3, 15, 5)
                st.markdown("---")

                sel_row = players_df[players_df["player_name"] == selected_name]
                if not sel_row.empty:
                    row = sel_row.iloc[0]
                    st.markdown(f"**{selected_name}**")
                    st.metric("Goals/90", f"{row.get('goals_per90', 0):.2f}")
                    st.metric("Passes/90", f"{row.get('passes_per90', 0):.0f}")
                    st.metric("Pass Completion", f"{row.get('pass_completion_pct', 0)*100:.1f}%")
                    st.metric("Minutes Played", f"{row.get('minutes_played', 0):.0f}")

        with col2:
            with st.container(border=True):
                st.subheader("Similar Players")
                sel_row = players_df[players_df["player_name"] == selected_name]
                if not sel_row.empty:
                    pid = int(sel_row.iloc[0]["player_id"])
                    try:
                        results = find_similar(
                            player_id=pid,
                            top_n=top_n,
                            embeddings=embeddings,
                            player_ids=player_ids,
                            player_names=player_names,
                        )

                        results_df = pd.DataFrame(results)
                        results_df["similarity_pct"] = (results_df["similarity"] * 100).round(1)
                        results_df = results_df[results_df["player_name"].astype(str) != "0.0"]
                        results_df = results_df.sort_values("similarity_pct")

                        fig = go.Figure(go.Bar(
                            x=results_df["similarity_pct"],
                            y=results_df["player_name"].astype(str),
                            orientation="h",
                            marker=dict(
                                color=results_df["similarity_pct"],
                                colorscale=[[0, "#4ade80"], [1, "#15803d"]],
                                line=dict(width=1, color="rgba(255,255,255,0.15)"),
                            ),
                            text=[f"{v:.1f}%" for v in results_df["similarity_pct"]],
                            textposition="outside",
                            textfont=dict(color=TEXT, size=13),
                            hovertemplate="<b>%{y}</b><br>Similarity: %{x:.1f}%<extra></extra>",
                        ))
                        fig.update_layout(
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            font=dict(color=TEXT, family="Inter, sans-serif"),
                            title=dict(text=f"Players most similar to {selected_name}",
                                       font=dict(size=15, color=TEXT)),
                            xaxis=dict(range=[0, 108], gridcolor=BORDER,
                                       title="Similarity Score (%)", zeroline=False),
                            yaxis=dict(gridcolor=BORDER),
                            margin=dict(l=10, r=30, t=50, b=40),
                            height=max(280, 46 * len(results_df)),
                            showlegend=False,
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        st.dataframe(
                            results_df.sort_values("similarity_pct", ascending=False)
                                [["player_name", "similarity_pct"]].rename(
                                    columns={"player_name": "Player", "similarity_pct": "Similarity (%)"}
                                ),
                            use_container_width=True,
                            hide_index=True,
                        )
                    except Exception as e:
                        st.error(f"Similarity search failed: {e}")

    st.markdown("---")
    st.caption("ScoutIQ — Built on StatsBomb Open Data · "
               "88K shots · 3.4M passes · 5,653 players · "
               "Models: LightGBM + Stacked Ensemble + PCA Embeddings")


if __name__ == "__main__":
    main()
