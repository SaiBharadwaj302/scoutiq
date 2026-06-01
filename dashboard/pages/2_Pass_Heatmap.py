"""
dashboard/pages/2_Pass_Heatmap.py

Pass Heatmap view:
  - Select match + team filter
  - Draw pass arrows on pitch, coloured by completion status
  - Zone heatmap of pass origins
  - Stats: completion rate by zone and pass type
"""
import sys; sys.path.insert(0, ".")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyArrowPatch
import streamlit as st

from utils import get_matches, get_passes_for_match, draw_pitch

st.set_page_config(page_title="Pass Heatmap · ScoutIQ", page_icon="🗺", layout="wide")

with st.sidebar:
    st.markdown("## 🗺 Pass Heatmap")
    st.markdown("Visualise passing patterns and completion risk.")

# ── Match selector ────────────────────────────────────────────────────────────
try:
    matches = get_matches()
except Exception as e:
    st.error(f"Database unavailable: {e}")
    st.stop()

if matches.empty:
    st.warning("No matches in database.")
    st.stop()

matches["label"] = (
    matches["match_date"].astype(str) + " · " +
    matches["home_team_name"] + " vs " +
    matches["away_team_name"]
)

selected_label = st.sidebar.selectbox("Match", matches["label"].tolist())
row = matches[matches["label"] == selected_label].iloc[0]
match_id = int(row["match_id"])

# ── Filters ───────────────────────────────────────────────────────────────────
pass_type = st.sidebar.multiselect(
    "Pass type",
    ["All", "Short (<10m)", "Medium (10–32m)", "Long (>32m)", "Cross", "Through ball"],
    default=["All"],
)
show_incomplete = st.sidebar.checkbox("Show incomplete passes", value=True)
max_arrows = st.sidebar.slider("Max arrows displayed", 50, 500, 200, step=50)

# ── Load data ─────────────────────────────────────────────────────────────────
try:
    passes = get_passes_for_match(match_id)
except Exception as e:
    st.error(f"Could not load passes: {e}")
    st.stop()

if passes.empty:
    st.warning("No pass data for this match.")
    st.stop()

# Apply filters
if "All" not in pass_type:
    mask = passes["length"].isna()   # start with all-False
    if "Short (<10m)" in pass_type:
        mask = mask | (passes["length"] < 10)
    if "Medium (10–32m)" in pass_type:
        mask = mask | ((passes["length"] >= 10) & (passes["length"] <= 32))
    if "Long (>32m)" in pass_type:
        mask = mask | (passes["length"] > 32)
    if "Cross" in pass_type:
        mask = mask | passes["is_cross"].fillna(False)
    if "Through ball" in pass_type:
        mask = mask | passes["is_through_ball"].fillna(False)
    passes = passes[mask]

if not show_incomplete:
    passes = passes[passes["is_complete"]]

# Sample if too many
if len(passes) > max_arrows:
    passes = passes.sample(max_arrows, random_state=42)

# ── Header ────────────────────────────────────────────────────────────────────
st.title(f"🗺 {row['home_team_name']} vs {row['away_team_name']}")
st.markdown(f"**{row['competition_name']} · {row['match_date']}**")

# ── KPIs ──────────────────────────────────────────────────────────────────────
total_passes = get_passes_for_match(match_id)   # unfiltered
comp_rate = total_passes["is_complete"].mean() if not total_passes.empty else 0.0
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total passes", f"{len(total_passes):,}")
k2.metric("Completion rate", f"{comp_rate*100:.1f}%")
k3.metric("Through balls", int(total_passes["is_through_ball"].sum()))
k4.metric("Crosses", int(total_passes["is_cross"].sum()))

st.markdown("---")
col_map, col_stats = st.columns([3, 2])

# ── Pass arrow map ────────────────────────────────────────────────────────────
with col_map:
    st.subheader("Pass Map")
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor("#0f172a")
    draw_pitch(ax, half="full")

    complete_color   = "#22c55e"
    incomplete_color = "#ef4444"

    for _, p in passes.iterrows():
        sx, sy = p.get("start_x"), p.get("start_y")
        ex, ey = p.get("end_x"), p.get("end_y")
        if any(v is None for v in [sx, sy, ex, ey]):
            continue
        color   = complete_color if p["is_complete"] else incomplete_color
        alpha   = 0.55 if p["is_complete"] else 0.4
        lw      = 1.2 if p.get("is_through_ball") or p.get("is_cross") else 0.7
        ax.annotate(
            "", xy=(ex, ey), xytext=(sx, sy),
            arrowprops=dict(
                arrowstyle="-|>",
                color=color, lw=lw, alpha=alpha,
                mutation_scale=6,
            ),
        )

        import matplotlib.lines as mlines
    legend_els = [
        mlines.Line2D([],[],color=complete_color, linewidth=2, label="Complete"),
        mlines.Line2D([],[],color=incomplete_color, linewidth=2, label="Incomplete"),
    ]
    ax.legend(handles=legend_els, loc="upper left", fontsize=9,
              framealpha=0.4, labelcolor="white", facecolor="#1e293b", edgecolor="none")
    ax.set_title(f"Showing {len(passes)} passes (filtered)", color="white", fontsize=10)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

# ── Zone heatmap of pass origins ──────────────────────────────────────────────
with col_stats:
    st.subheader("Pass Origin Density")
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    fig2.patch.set_facecolor("#0f172a")
    draw_pitch(ax2, half="full")

    sx_vals = passes["start_x"].dropna().values
    sy_vals = passes["start_y"].dropna().values

    if len(sx_vals) > 5:
        h = ax2.hist2d(
            sx_vals, sy_vals,
            bins=[24, 16],
            range=[[0, 120], [0, 80]],
            cmap="hot",
            alpha=0.65,
        )
        plt.colorbar(h[3], ax=ax2, fraction=0.03, pad=0.02,
                     label="Pass count").ax.yaxis.label.set_color("white")

    ax2.set_title("Pass Origins", color="white", fontsize=10)
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)

    # Completion by zone table
    st.subheader("Completion Rate by Zone")
    all_p = get_passes_for_match(match_id)
    if not all_p.empty:
        all_p["zone"] = pd.cut(
            all_p["start_x"],
            bins=[0, 40, 80, 120],
            labels=["Defensive (0–40m)", "Middle (40–80m)", "Attacking (80–120m)"],
        )
        zone_stats = (
            all_p.groupby("zone", observed=True)["is_complete"]
            .agg(passes="count", completion=lambda x: f"{x.mean()*100:.1f}%")
            .reset_index()
            .rename(columns={"zone": "Zone", "passes": "Passes", "completion": "Completion %"})
        )
        st.dataframe(zone_stats, use_container_width=True, hide_index=True)

import pandas as pd
