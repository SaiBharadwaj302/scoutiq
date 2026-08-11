"""
dashboard/pages/3_Player_Scouting.py

Player Scouting view:
  - Search for any player in the dataset
  - Show their per-90 stats
  - Find top-5 most similar players (deep metric learning + cosine similarity)
  - Radar chart comparing query player vs best match
  - Sortable comparison table
"""
import sys
sys.path.insert(0, ".")

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from utils import get_players, get_player_stats, get_similar_players, draw_radar, RADAR_STATS, _query

st.set_page_config(page_title="Player Scouting · ScoutIQ", page_icon="🔍", layout="wide")

with st.sidebar:
    st.markdown("## 🔍 Player Scouting")
    st.markdown("Find tactically similar players using deep metric learning embeddings and cosine similarity.")

# ── Load player list ──────────────────────────────────────────────────────────
try:
    players_df = get_players()
except Exception as e:
    st.error(f"Database unavailable: {e}")
    st.stop()

if players_df.empty:
    st.warning("No players found — run the similarity model first.")
    st.stop()

# ── Player search ─────────────────────────────────────────────────────────────
search = st.sidebar.text_input("Search player", placeholder="e.g. Messi")
filtered = players_df[
    players_df["player_name"].str.contains(search, case=False, na=False)
] if search else players_df

if filtered.empty:
    st.warning(f"No players matching '{search}'.")
    st.stop()

player_options = filtered["player_name"].tolist()
selected_name  = st.sidebar.selectbox("Select player", player_options)
selected_row   = filtered[filtered["player_name"] == selected_name].iloc[0]
player_id      = int(selected_row["player_id"])

top_n = st.sidebar.slider("Similar players to return", 3, 15, 5)

# ── Header ────────────────────────────────────────────────────────────────────
st.title(f"🔍 {selected_name}")
meta_cols = st.columns(3)
meta_cols[0].markdown(f"**Team:** {selected_row.get('team_name', 'Unknown')}")
meta_cols[1].markdown(f"**Position:** {selected_row.get('position_group', 'Unknown')}")
meta_cols[2].markdown(f"**Minutes played:** {int(selected_row.get('minutes_played', 0)):,}")

# ── Per-90 stats ──────────────────────────────────────────────────────────────
query_stats = get_player_stats(player_id)

if query_stats is None:
    st.warning("Detailed stats not found for this player.")
    st.stop()

st.markdown("---")
st.subheader("Per-90 Statistics")

stat_cols = st.columns(4)
stat_display = [
    ("Goals", "goals_per90"),
    ("Assists", "assists_per90"),
    ("Shots", "shots_per90"),
    ("xG", "xg_per90"),
    ("Passes", "passes_per90"),
    ("Pass %", "pass_completion_pct"),
    ("Prog passes", "progressive_passes_per90"),
    ("Key passes", "key_passes_per90"),
]
for i, (label, col) in enumerate(stat_display):
    val = query_stats.get(col, 0) or 0
    if col == "pass_completion_pct":
        fmt = f"{val*100:.1f}%"
    else:
        fmt = f"{val:.2f}"
    stat_cols[i % 4].metric(label, fmt)

# ── Find similar players ──────────────────────────────────────────────────────
st.markdown("---")
st.subheader(f"Most similar players (top {top_n})")

with st.spinner("Computing similarity…"):
    similar_df = get_similar_players(player_id, top_n=top_n)

if similar_df.empty:
    st.warning("No similar players found.")
    st.stop()

# ── Radar comparison ──────────────────────────────────────────────────────────
col_radar, col_table = st.columns([2, 3])

# Compare query player vs top-1 similar
top_match_id   = int(similar_df.iloc[0]["player_id"])
top_match_name = similar_df.iloc[0]["player_name"]
top_match_stats = get_player_stats(top_match_id)

# Build normalisation max values from all player stats (for the radar)
try:
    all_stats = _query(f"SELECT {', '.join(RADAR_STATS.values())} FROM player_per90_features")
    max_vals  = all_stats.quantile(0.95)    # use 95th pct as "max" for normalisation
except Exception:
    max_vals = None

with col_radar:
    st.markdown(f"**{selected_name}** (blue) vs **{top_match_name}** (orange)")
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("#0f172a")
    draw_radar(
        ax,
        stats_a=query_stats,
        stats_b=top_match_stats,
        label_a=selected_name,
        label_b=top_match_name,
        max_vals=max_vals,
    )
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

# ── Similarity table ──────────────────────────────────────────────────────────
with col_table:
    st.markdown("**Similarity ranking**")
    similar_df["similarity_pct"] = (similar_df["similarity"] * 100).round(1).astype(str) + "%"

    # Enrich with position and team
    enriched_rows = []
    for _, r in similar_df.iterrows():
        s = get_player_stats(int(r["player_id"]))
        enriched_rows.append({
            "Rank":        len(enriched_rows) + 1,
            "Player":      r["player_name"],
            "Team":        s.get("team_name", "—") if s is not None else "—",
            "Position":    s.get("position_group", "—") if s is not None else "—",
            "Minutes":     f"{int(s.get('minutes_played', 0) or 0):,}" if s is not None else "—",
            "Similarity":  r["similarity_pct"],
        })

    enriched_df = pd.DataFrame(enriched_rows)
    st.dataframe(enriched_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("**How similarity works**")
    st.markdown(
        "Player stats are aggregated per-90 minutes, normalised with StandardScaler, "
        "then mapped to a 32-dimensional embedding by a PyTorch network trained with "
        "triplet margin loss (same position_group = pulled together, different = pushed "
        "apart). Cosine similarity in that space reflects tactical and statistical "
        "profile similarity — not just raw output."
    )
