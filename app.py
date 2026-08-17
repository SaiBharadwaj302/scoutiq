"""
app.py

ScoutIQ Streamlit Dashboard
Visual interface for all three ML models.
Run with: streamlit run app.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import math
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import plotly.graph_objects as go

# ── Palette ──────────────────────────────────────────────────────────────────
# Single source of truth for color across CSS, matplotlib, and Plotly so the
# dashboard reads as one system instead of three mismatched chart libraries.
BG          = "#0b1120"   # app background
SURFACE     = "#141b2d"   # card / pitch-card surface
SURFACE_2   = "#1b2436"   # slightly lighter surface (hover, alt rows)
BORDER      = "#293347"
TEXT        = "#e2e8f0"
TEXT_MUTED  = "#94a3b8"
ACCENT      = "#22c55e"   # emerald — brand / pitch green
ACCENT_SOFT = "#16a34a"
INFO        = "#38bdf8"
WARNING     = "#f59e0b"
DANGER      = "#ef4444"
PITCH_GREEN = "#0f3d24"

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
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

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

    /* ── Hero banner ─────────────────────────────────────────────────────── */
    .scoutiq-hero {{
        background: linear-gradient(135deg, {SURFACE} 0%, {BG} 100%);
        border: 1px solid {BORDER};
        border-radius: 16px;
        padding: 28px 32px;
        margin-bottom: 24px;
        position: relative;
        overflow: hidden;
    }}
    .scoutiq-hero::before {{
        content: "";
        position: absolute;
        top: -40%; right: -8%;
        width: 320px; height: 320px;
        background: radial-gradient(circle, rgba(34,197,94,0.18) 0%, transparent 70%);
        pointer-events: none;
    }}
    .scoutiq-hero h1 {{
        font-size: 1.9rem;
        font-weight: 800;
        margin: 0 0 6px 0;
        color: {TEXT};
        letter-spacing: -0.02em;
    }}
    .scoutiq-hero p {{
        font-size: 1rem;
        color: {TEXT_MUTED};
        margin: 0;
        max-width: 640px;
    }}

    /* ── Sidebar branding ────────────────────────────────────────────────── */
    .sidebar-brand {{
        display: flex; align-items: center; gap: 10px;
        padding: 4px 0 14px 0;
    }}
    .sidebar-brand .badge {{
        width: 40px; height: 40px; border-radius: 10px;
        background: linear-gradient(135deg, {ACCENT} 0%, {ACCENT_SOFT} 100%);
        display: flex; align-items: center; justify-content: center;
        font-size: 20px;
        box-shadow: 0 4px 14px rgba(34,197,94,0.35);
    }}
    .sidebar-brand .name {{ font-weight: 800; font-size: 1.15rem; color: {TEXT}; line-height: 1.1; }}
    .sidebar-brand .tag {{ font-size: 0.72rem; color: {TEXT_MUTED}; text-transform: uppercase; letter-spacing: 0.06em; }}

    .model-chip {{
        display: flex; align-items: center; justify-content: space-between;
        background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 10px;
        padding: 9px 12px; margin-bottom: 8px; font-size: 0.82rem;
    }}
    .model-chip .dot {{
        width: 7px; height: 7px; border-radius: 50%;
        background: {ACCENT}; display: inline-block; margin-right: 8px;
        box-shadow: 0 0 6px {ACCENT};
    }}
    .model-chip .label {{ color: {TEXT_MUTED}; }}
    .model-chip .val {{ color: {ACCENT}; font-weight: 700; }}

    /* ── Metric cards (restyle native st.metric) ─────────────────────────── */
    [data-testid="stMetric"] {{
        background: {SURFACE};
        border: 1px solid {BORDER};
        border-top: 3px solid {ACCENT};
        border-radius: 12px;
        padding: 14px 16px 10px 16px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.25);
    }}
    [data-testid="stMetricLabel"] {{
        color: {TEXT_MUTED} !important; font-size: 0.78rem !important;
        text-transform: uppercase; letter-spacing: 0.04em;
    }}
    [data-testid="stMetricValue"] {{ color: {TEXT} !important; font-weight: 800 !important; }}

    /* ── Verdict banners (replace st.success/info/warning walls of color) ─── */
    .verdict {{
        border-radius: 12px; padding: 14px 18px; margin-top: 10px;
        font-size: 0.95rem; border: 1px solid; line-height: 1.5;
    }}
    .verdict-good {{ background: rgba(34,197,94,0.10);  border-color: rgba(34,197,94,0.35);  color: #86efac; }}
    .verdict-mid  {{ background: rgba(56,189,248,0.10);  border-color: rgba(56,189,248,0.35); color: #7dd3fc; }}
    .verdict-bad  {{ background: rgba(239,68,68,0.10);  border-color: rgba(239,68,68,0.35);  color: #fca5a5; }}

    /* ── Sidebar nav ─────────────────────────────────────────────────────── */
    section[data-testid="stSidebar"] {{
        background: {BG};
        border-right: 1px solid {BORDER};
    }}
    section[data-testid="stSidebar"] [role="radiogroup"] label {{
        border-radius: 10px; padding: 8px 12px !important; margin-bottom: 4px;
        transition: background 0.15s ease;
    }}
    section[data-testid="stSidebar"] [role="radiogroup"] label:hover {{
        background: {SURFACE};
    }}

    /* ── Bordered containers as cards ────────────────────────────────────── */
    div[data-testid="stVerticalBlockBorderWrapper"] {{
        border-radius: 14px !important;
        border-color: {BORDER} !important;
        background: {SURFACE};
    }}

    h2, h3 {{ color: {TEXT}; font-weight: 700; }}
    hr {{ border-color: {BORDER}; }}
    </style>
    """, unsafe_allow_html=True)


def hero(title: str, subtitle: str):
    """Static gradient banner — safe to render as raw HTML (no widgets inside)."""
    st.markdown(f"""
    <div class="scoutiq-hero">
        <h1>{title}</h1>
        <p>{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)


def verdict_box(kind: str, text: str):
    cls = {"good": "verdict-good", "mid": "verdict-mid", "bad": "verdict-bad"}[kind]
    st.markdown(f'<div class="verdict {cls}">{text}</div>', unsafe_allow_html=True)


# ── Load Models (cached) ──────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    from api.dependencies import get_xg_model, get_pass_model, get_similarity_data
    xg_model   = get_xg_model()
    pass_model = get_pass_model()
    emb, pids, pnames = get_similarity_data()
    return xg_model, pass_model, emb, pids, pnames


@st.cache_data
def load_player_list():
    from db.store import get_connection
    query = """
        SELECT player_id, player_name, goals_per90, passes_per90,
               pass_completion_pct, minutes_played
        FROM player_per90_features
        WHERE player_name IS NOT NULL
          AND player_name != ''
          AND player_name != '0'
          AND minutes_played > 200
        ORDER BY goals_per90 DESC NULLS LAST
        LIMIT 500
    """
    with get_connection() as conn:
        return pd.read_sql(query, conn)


# ── Pitch Drawing ─────────────────────────────────────────────────────────────

def draw_pitch(ax, color=PITCH_GREEN, linecolor="#e2e8f0"):
    """Draw a StatsBomb pitch (120 x 80)."""
    ax.set_facecolor(color)
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 80)
    ax.set_aspect("equal")
    ax.axis("off")

    lc = linecolor
    lw = 1.5
    alpha = 0.85

    # Pitch outline
    ax.add_patch(patches.Rectangle((0, 0), 120, 80,
                 fill=False, edgecolor=lc, linewidth=lw, alpha=alpha))
    # Halfway line
    ax.plot([60, 60], [0, 80], color=lc, linewidth=lw, alpha=alpha)
    # Centre circle
    centre = plt.Circle((60, 40), 10, fill=False, color=lc, linewidth=lw, alpha=alpha)
    ax.add_patch(centre)
    ax.plot(60, 40, "o", color=lc, markersize=3, alpha=alpha)

    # Penalty boxes
    ax.add_patch(patches.Rectangle((0, 18), 18, 44,
                 fill=False, edgecolor=lc, linewidth=lw, alpha=alpha))
    ax.add_patch(patches.Rectangle((102, 18), 18, 44,
                 fill=False, edgecolor=lc, linewidth=lw, alpha=alpha))
    # Six yard boxes
    ax.add_patch(patches.Rectangle((0, 30), 6, 20,
                 fill=False, edgecolor=lc, linewidth=lw, alpha=alpha))
    ax.add_patch(patches.Rectangle((114, 30), 6, 20,
                 fill=False, edgecolor=lc, linewidth=lw, alpha=alpha))
    # Goals
    ax.add_patch(patches.Rectangle((-2, 36), 2, 8,
                 fill=True, facecolor=lc, edgecolor=lc, linewidth=lw, alpha=alpha))
    ax.add_patch(patches.Rectangle((120, 36), 2, 8,
                 fill=True, facecolor=lc, edgecolor=lc, linewidth=lw, alpha=alpha))
    # Penalty spots
    ax.plot(12, 40, "o", color=lc, markersize=3, alpha=alpha)
    ax.plot(108, 40, "o", color=lc, markersize=3, alpha=alpha)


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
    from models.xg.features import build_features
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
    features = build_features(df)
    return float(model.predict_proba(features)[0][1])


def predict_pass(model, sx, sy, ex, ey, height, body_part,
                 under_pressure, is_cross, is_switch, period):
    from models.pass_success.features import build_features
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
    features = build_features(df)
    return float(model.predict_proba(features)[0][1])


# ── Main App ──────────────────────────────────────────────────────────────────

def main():
    inject_css()

    # ── Sidebar ──────────────────────────────────────────────────────────────
    st.sidebar.markdown("""
    <div class="sidebar-brand">
        <div class="badge">⚽</div>
        <div>
            <div class="name">ScoutIQ</div>
            <div class="tag">Football Analytics</div>
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

    # Load models
    with st.spinner("Loading models..."):
        try:
            xg_model, pass_model, embeddings, player_ids, player_names = load_models()
            models_loaded = True
        except Exception as e:
            st.error(f"Model loading failed: {e}")
            models_loaded = False

    # ── xG PAGE ──────────────────────────────────────────────────────────────
    if page == "🎯 xG Predictor":
        hero("🎯 Expected Goals (xG) Predictor",
             "Place a shot on the pitch and get the model's goal probability instantly.")

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
                    ax.plot(loc_x, loc_y, "o", color=color,
                            markersize=18, alpha=0.9, zorder=5)
                    ax.plot(loc_x, loc_y, "o", color="white",
                            markersize=6, zorder=6)
                    # Arrow to goal
                    ax.annotate("", xy=(120, 40), xytext=(loc_x, loc_y),
                        arrowprops=dict(arrowstyle="->", color="white",
                                        lw=1.5, alpha=0.5))
                    ax.text(loc_x - 2, loc_y + 3,
                            f"xG = {xg:.3f}", color="white",
                            fontsize=11, fontweight="bold",
                            bbox=dict(boxstyle="round,pad=0.35",
                                      facecolor=color, edgecolor="none", alpha=0.9))

                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

        # Metrics
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
             "Define a pass and see the model's predicted completion probability.")

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

                    # Draw pass line
                    ax.annotate("", xy=(ex, ey), xytext=(sx, sy),
                        arrowprops=dict(arrowstyle="->", color=color, lw=2.5))
                    ax.plot(sx, sy, "o", color="white", markersize=10, zorder=5)
                    ax.plot(ex, ey, "D", color=color, markersize=10, zorder=5)

                    mid_x = (sx + ex) / 2
                    mid_y = (sy + ey) / 2
                    ax.text(mid_x, mid_y + 2,
                            f"{prob*100:.0f}%", color="white",
                            fontsize=11, fontweight="bold", ha="center",
                            bbox=dict(boxstyle="round,pad=0.35",
                                      facecolor=color, edgecolor="none", alpha=0.9))

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
             "Find tactically similar players using PCA embeddings and cosine similarity.")

        try:
            players_df = load_player_list()
        except Exception as e:
            st.error(f"Could not load player list: {e}")
            return

        col1, col2 = st.columns([1, 2])

        with col1:
            with st.container(border=True):
                st.subheader("Search")
                player_options = players_df["player_name"].dropna().tolist()
                selected_name  = st.selectbox("Select a player", player_options)
                top_n = st.slider("Number of similar players", 3, 15, 5)
                st.markdown("---")

                # Show selected player stats
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
                if models_loaded:
                    sel_row = players_df[players_df["player_name"] == selected_name]
                    if not sel_row.empty:
                        pid = int(sel_row.iloc[0]["player_id"])
                        try:
                            from models.similarity.query import find_similar
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
                            # Ascending so the highest-similarity player sits at the top
                            # of a horizontal bar chart (Plotly draws bottom-up).
                            results_df = results_df.sort_values("similarity_pct")

                            fig = go.Figure(go.Bar(
                                x=results_df["similarity_pct"],
                                y=results_df["player_name"].astype(str),
                                orientation="h",
                                marker=dict(
                                    color=results_df["similarity_pct"],
                                    colorscale=[[0, "#4ade80"], [1, "#15803d"]],
                                    line=dict(width=0),
                                ),
                                text=[f"{v:.1f}%" for v in results_df["similarity_pct"]],
                                textposition="outside",
                                textfont=dict(color=TEXT, size=13),
                                hovertemplate="<b>%{y}</b><br>Similarity: %{x:.1f}%<extra></extra>",
                            ))
                            fig.update_layout(
                                paper_bgcolor=SURFACE,
                                plot_bgcolor=SURFACE,
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

                            # Table
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

    # Footer
    st.markdown("---")
    st.caption("ScoutIQ — Built on StatsBomb Open Data · "
               "88K shots · 3.4M passes · 5,653 players · "
               "Models: LightGBM + Stacked Ensemble + PCA Embeddings")


if __name__ == "__main__":
    main()
