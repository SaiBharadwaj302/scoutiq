"""
dashboard/pages/1_Match_Analysis.py

Match Analysis view:
  - Select a match from the database
  - Interactive shot map (circle size = xG, star = goal, colour = xG value)
  - Cumulative xG timeline (both teams)
  - Per-team stat table
"""
import sys
sys.path.insert(0, ".")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st

from utils import (
    get_matches, get_shots_for_match,
    draw_pitch, xg_color,
)

st.set_page_config(page_title="Match Analysis · ScoutIQ", page_icon="📈", layout="wide")

# ── Sidebar — match selector ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚽ Match Analysis")
    st.markdown("Select a match to analyse.")

try:
    matches = get_matches()
except Exception as e:
    st.error(f"Database unavailable: {e}")
    st.stop()

if matches.empty:
    st.warning("No matches found — run the ingestion pipeline first.")
    st.stop()

matches["label"] = (
    matches["match_date"].astype(str) + " · " +
    matches["home_team_name"] + " " +
    matches["home_score"].astype(str) + "–" +
    matches["away_score"].astype(str) + " " +
    matches["away_team_name"] + " (" +
    matches["competition_name"] + ")"
)

selected_label = st.sidebar.selectbox("Match", matches["label"].tolist())
row = matches[matches["label"] == selected_label].iloc[0]
match_id = int(row["match_id"])

# ── Load shots ────────────────────────────────────────────────────────────────
try:
    shots = get_shots_for_match(match_id)
except Exception as e:
    st.error(f"Could not load shots: {e}")
    st.stop()

home_team = row["home_team_name"]
away_team = row["away_team_name"]

# Determine which team_id is home (use mode of team_id by home team name)
if not shots.empty:
    team_ids = shots[["team_id", "player_name"]].drop_duplicates()
    home_id = shots[shots["player_name"].notna()]["team_id"].value_counts().idxmax()
else:
    home_id = None

def team_label(tid):
    if home_id and tid == home_id:
        return home_team
    return away_team

# ── Header ────────────────────────────────────────────────────────────────────
st.title(f"📈 {home_team} vs {away_team}")
st.markdown(
    f"**{row['competition_name']} · {row['season_name']} · {row['match_date']}**"
)
score_col, *_ = st.columns([1, 5])
score_col.metric("Score", f"{int(row['home_score'])} – {int(row['away_score'])}")

if shots.empty:
    st.warning("No shots recorded for this match.")
    st.stop()

# ── xG Summary table ──────────────────────────────────────────────────────────
# We need xG per shot — build a quick local estimate (distance + angle proxy)
# This avoids needing the API for every shot, using a simple formula as fallback
def local_xg_estimate(distance: float, angle: float, is_header: bool) -> float:
    """Simple xG proxy for visualisation only (actual model runs via API)."""
    base = np.exp(-0.11 * distance) * (angle / np.pi)
    if is_header:
        base *= 0.6
    return float(np.clip(base, 0.01, 0.99))

shots["xg_est"] = shots.apply(
    lambda r: local_xg_estimate(
        r["distance_to_goal"] or 20,
        r["angle_to_goal"] or 0.3,
        "head" in str(r.get("body_part_name", "")).lower()
    ),
    axis=1,
)
shots["team_label"] = shots["team_id"].apply(team_label)

home_shots = shots[shots["team_label"] == home_team]
away_shots = shots[shots["team_label"] == away_team]

c1, c2, c3, c4 = st.columns(4)
c1.metric(f"{home_team} shots",  len(home_shots))
c2.metric(f"{home_team} xG",     f"{home_shots['xg_est'].sum():.2f}")
c3.metric(f"{away_team} shots",  len(away_shots))
c4.metric(f"{away_team} xG",     f"{away_shots['xg_est'].sum():.2f}")

st.markdown("---")

# ── Two-panel layout: Shot map + xG timeline ──────────────────────────────────
col_map, col_tl = st.columns([3, 2])

# ── Shot Map ──────────────────────────────────────────────────────────────────
with col_map:
    st.subheader("Shot Map")
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor("#0f172a")
    draw_pitch(ax, half="attack")

    for _, s in shots.iterrows():
        x, y = s["location_x"], s["location_y"]
        if x is None or y is None:
            continue
        xg  = s["xg_est"]
        col = xg_color(xg)
        size = 80 + xg * 400   # bigger circle = higher xG

        if s["is_goal"]:
            # Star marker for goals
            ax.plot(x, y, marker="*", color=col,
                    markersize=14, zorder=5,
                    markeredgecolor="white", markeredgewidth=0.8)
        else:
            # Home team = filled, away = hollow
            if s["team_label"] == home_team:
                ax.scatter(x, y, s=size, c=col, alpha=0.85,
                           edgecolors="white", linewidths=0.5, zorder=4)
            else:
                # Mirror away team shots (flip y for visual clarity)
                ax.scatter(120 - x, 80 - y, s=size, c=col, alpha=0.55,
                           edgecolors="white", linewidths=0.5,
                           marker="^", zorder=4)

    # Legend
    legend_els = [
        mpatches.Patch(color=xg_color(0.05), label="Low xG (<0.1)"),
        mpatches.Patch(color=xg_color(0.2),  label="Med xG (0.1–0.3)"),
        mpatches.Patch(color=xg_color(0.5),  label="High xG (>0.3)"),
        plt.Line2D([0],[0], marker="*", color="w", markersize=10,
                   label="Goal", markerfacecolor="gold"),
        plt.Line2D([0],[0], marker="^", color="white", markersize=8,
                   label=f"{away_team} (mirrored)", linestyle="None"),
    ]
    ax.legend(handles=legend_els, loc="lower left",
              fontsize=8, framealpha=0.3, labelcolor="white",
              facecolor="#1e293b", edgecolor="none")
    ax.set_title(f"{home_team} (circles) vs {away_team} (triangles)",
                 color="white", pad=8, fontsize=10)

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

# ── xG Timeline ──────────────────────────────────────────────────────────────
with col_tl:
    st.subheader("Cumulative xG Timeline")
    fig2, ax2 = plt.subplots(figsize=(6, 6))
    fig2.patch.set_facecolor("#0f172a")
    ax2.set_facecolor("#0f172a")

    for team_df, color, label in [
        (home_shots, "#60a5fa", home_team),
        (away_shots, "#fb923c", away_team),
    ]:
        td = team_df.sort_values("minute")
        minutes = td["minute"].tolist()
        cumxg   = td["xg_est"].cumsum().tolist()
        # Step plot
        ax2.step([0] + minutes, [0] + cumxg,
                 where="post", color=color, linewidth=2.5, label=label)
        # Goal markers
        goals = td[td["is_goal"]]
        for _, g in goals.iterrows():
            xg_at = float(td[td["minute"] <= g["minute"]]["xg_est"].sum())
            ax2.plot(g["minute"], xg_at, marker="*", color=color,
                     markersize=14, zorder=5,
                     markeredgecolor="white", markeredgewidth=0.5)

    # Halftime line
    ax2.axvline(45, color="#94a3b8", linestyle="--", linewidth=1, alpha=0.6)
    ax2.text(46, ax2.get_ylim()[1] * 0.95 if ax2.get_ylim()[1] > 0 else 0.5,
             "HT", color="#94a3b8", fontsize=8)

    ax2.set_xlabel("Minute", color="white", fontsize=10)
    ax2.set_ylabel("Cumulative xG", color="white", fontsize=10)
    ax2.tick_params(colors="white")
    ax2.spines[["top","right"]].set_visible(False)
    for sp in ["bottom","left"]:
        ax2.spines[sp].set_color("#334155")
    ax2.grid(axis="y", color="#334155", linewidth=0.8, alpha=0.5)
    ax2.legend(fontsize=9, frameon=False, labelcolor="white")

    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)

# ── Shot log table ────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Shot Log")
display_cols = ["minute", "team_label", "player_name", "technique_name",
                "body_part_name", "distance_to_goal", "xg_est", "outcome_name"]
display = shots[display_cols].rename(columns={
    "minute": "Min", "team_label": "Team", "player_name": "Player",
    "technique_name": "Technique", "body_part_name": "Body Part",
    "distance_to_goal": "Distance (m)", "xg_est": "xG (est.)",
    "outcome_name": "Outcome",
})
display["Distance (m)"] = display["Distance (m)"].round(1)
display["xG (est.)"]    = display["xG (est.)"].round(3)
st.dataframe(
    display.sort_values("Min"),
    use_container_width=True,
    hide_index=True,
)
