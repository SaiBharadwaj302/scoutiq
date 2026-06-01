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
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ScoutIQ — Football Analytics",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

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

def draw_pitch(ax, color="white", linecolor="#333333"):
    """Draw a StatsBomb pitch (120 x 80)."""
    ax.set_facecolor(color)
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 80)
    ax.set_aspect("equal")
    ax.axis("off")

    lc = linecolor
    lw = 1.5

    # Pitch outline
    ax.add_patch(patches.Rectangle((0, 0), 120, 80,
                 fill=False, edgecolor=lc, linewidth=lw))
    # Halfway line
    ax.plot([60, 60], [0, 80], color=lc, linewidth=lw)
    # Centre circle
    centre = plt.Circle((60, 40), 10, fill=False, color=lc, linewidth=lw)
    ax.add_patch(centre)
    ax.plot(60, 40, "o", color=lc, markersize=3)

    # Penalty boxes
    ax.add_patch(patches.Rectangle((0, 18), 18, 44,
                 fill=False, edgecolor=lc, linewidth=lw))
    ax.add_patch(patches.Rectangle((102, 18), 18, 44,
                 fill=False, edgecolor=lc, linewidth=lw))
    # Six yard boxes
    ax.add_patch(patches.Rectangle((0, 30), 6, 20,
                 fill=False, edgecolor=lc, linewidth=lw))
    ax.add_patch(patches.Rectangle((114, 30), 6, 20,
                 fill=False, edgecolor=lc, linewidth=lw))
    # Goals
    ax.add_patch(patches.Rectangle((-2, 36), 2, 8,
                 fill=True, facecolor=lc, edgecolor=lc, linewidth=lw))
    ax.add_patch(patches.Rectangle((120, 36), 2, 8,
                 fill=True, facecolor=lc, edgecolor=lc, linewidth=lw))
    # Penalty spots
    ax.plot(12, 40, "o", color=lc, markersize=3)
    ax.plot(108, 40, "o", color=lc, markersize=3)


def xg_color(xg: float) -> str:
    if xg < 0.05:   return "#3498db"
    if xg < 0.15:   return "#2ecc71"
    if xg < 0.30:   return "#f39c12"
    return "#e74c3c"


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
    # Sidebar
    st.sidebar.image("https://img.icons8.com/emoji/96/soccer-ball-emoji.png", width=60)
    st.sidebar.title("ScoutIQ")
    st.sidebar.caption("Production ML Football Analytics")
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Navigate",
        ["🎯 xG Predictor", "🔄 Pass Analyzer", "🔍 Player Similarity"],
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Models**")
    st.sidebar.success("xG Model — AUC 0.808")
    st.sidebar.success("Pass Model — AUC 0.905")
    st.sidebar.success("Similarity — 5,653 players")
    st.sidebar.caption("Trained on StatsBomb open data\n88K shots · 3.4M passes")

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
        st.title("🎯 Expected Goals (xG) Predictor")
        st.markdown("Place a shot on the pitch and get the xG probability instantly.")

        col1, col2 = st.columns([1, 1])

        with col1:
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
            st.subheader("Pitch Visualization")
            fig, ax = plt.subplots(figsize=(8, 5.5))
            fig.patch.set_facecolor("#1a1a2e")
            draw_pitch(ax, color="#1a472a", linecolor="white")

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
                        bbox=dict(boxstyle="round,pad=0.3",
                                  facecolor=color, alpha=0.8))

            st.pyplot(fig)
            plt.close(fig)

        # Metrics
        if models_loaded:
            dist  = dist_to_goal(loc_x, loc_y)
            angle = math.degrees(angle_to_goal(loc_x, loc_y))

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("xG", f"{xg:.3f}")
            m2.metric("Distance", f"{dist:.1f}m")
            m3.metric("Angle", f"{angle:.1f}°")
            m4.metric("Big Chance", "Yes ✅" if xg > 0.3 else "No ❌")

            st.markdown("---")
            if xg > 0.3:
                st.success(f"🔥 Big Chance! xG of {xg:.3f} — shooter is expected to score roughly 1 in {int(1/xg)} shots from here.")
            elif xg > 0.1:
                st.info(f"⚡ Decent chance. xG of {xg:.3f} — about 1 in {int(1/xg)} shots from this position result in a goal.")
            else:
                st.warning(f"📉 Low probability shot. xG of {xg:.3f} — this is a difficult chance.")

    # ── PASS PAGE ─────────────────────────────────────────────────────────────
    elif page == "🔄 Pass Analyzer":
        st.title("🔄 Pass Success Analyzer")
        st.markdown("Define a pass and see the predicted completion probability.")

        col1, col2 = st.columns([1, 1])

        with col1:
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
            st.subheader("Pitch Visualization")
            fig, ax = plt.subplots(figsize=(8, 5.5))
            fig.patch.set_facecolor("#1a1a2e")
            draw_pitch(ax, color="#1a472a", linecolor="white")

            if models_loaded:
                prob = predict_pass(pass_model, sx, sy, ex, ey,
                                    height, body_part, under_pressure,
                                    is_cross, is_switch, period)
                color = "#2ecc71" if prob > 0.7 else "#f39c12" if prob > 0.5 else "#e74c3c"

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
                        bbox=dict(boxstyle="round,pad=0.3",
                                  facecolor=color, alpha=0.8))

            st.pyplot(fig)
            plt.close(fig)

        if models_loaded:
            length = math.sqrt((ex - sx)**2 + (ey - sy)**2)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Completion %", f"{prob*100:.1f}%")
            m2.metric("Pass Length", f"{length:.1f}m")
            m3.metric("Risk", "High 🔴" if prob < 0.5 else "Medium 🟡" if prob < 0.7 else "Low 🟢")
            m4.metric("Forward Pass", "Yes" if ex > sx else "No")

            st.markdown("---")
            if prob > 0.8:
                st.success(f"✅ High completion probability ({prob*100:.0f}%) — safe pass.")
            elif prob > 0.5:
                st.info(f"⚡ Moderate risk ({prob*100:.0f}%) — playable but needs precision.")
            else:
                st.error(f"🚨 High risk pass ({prob*100:.0f}%) — likely to be intercepted.")

    # ── SIMILARITY PAGE ───────────────────────────────────────────────────────
    elif page == "🔍 Player Similarity":
        st.title("🔍 Player Similarity Engine")
        st.markdown("Find tactically similar players using PCA embeddings and cosine similarity.")

        try:
            players_df = load_player_list()
        except Exception as e:
            st.error(f"Could not load player list: {e}")
            return

        col1, col2 = st.columns([1, 2])

        with col1:
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
            st.subheader("Similar Players")
            if models_loaded:
                # Get player_id for selected name
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

                        # Bar chart
                        fig, ax = plt.subplots(figsize=(8, 4))
                        fig.patch.set_facecolor("#1a1a2e")
                        ax.set_facecolor("#1a1a2e")
                        colors = plt.cm.RdYlGn(
                            np.linspace(0.4, 0.9, len(results_df))
                        )[::-1]
                        bars = ax.barh(
                            results_df["player_name"].astype(str),
                            results_df["similarity_pct"],
                            color=colors, edgecolor="none"
                        )
                        for bar, val in zip(bars, results_df["similarity_pct"]):
                            ax.text(bar.get_width() - 1, bar.get_y() + bar.get_height()/2,
                                    f"{val:.1f}%", va="center", ha="right",
                                    color="white", fontweight="bold", fontsize=10)
                        ax.set_xlabel("Similarity Score (%)", color="white")
                        ax.set_title(f"Players most similar to {selected_name}",
                                     color="white", fontsize=12)
                        ax.tick_params(colors="white")
                        ax.spines[["top","right","bottom"]].set_visible(False)
                        ax.spines["left"].set_color("#444")
                        ax.set_xlim(0, 105)
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)

                        # Table
                        st.dataframe(
                            results_df[["player_name","similarity_pct"]].rename(
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
