"""
models/similarity/features.py

Shared per-90 feature aggregation for the player similarity engine.
Joins shot_features + pass_features into per-90 player stats and
attaches position_group from the players table, independent of
whichever embedding model (PCA, deep metric net, ...) consumes them.
"""
import numpy as np
import pandas as pd
from loguru import logger

from db.store import get_connection

MIN_MINUTES = 200  # minimum minutes played to include player

EMBEDDING_FEATURES = [
    "shots_p90",
    "goals_p90",
    "shots_on_target_p90",
    "avg_shot_distance",
    "avg_shot_angle",
    "passes_p90",
    "pass_completion_pct",
    "shot_conversion_pct",
    "shots_on_target_pct",
    "switches_p90",
    "through_balls_p90",
    "crosses_p90",
    "goal_assists_p90",
    "shot_assists_p90",
    "passes_under_pressure_p90",
    "progressive_passes_p90",
    "passes_into_box_p90",
    "avg_pass_length",
]


# ── Per-90 Aggregation ────────────────────────────────────────────────────────

def aggregate_player_stats() -> pd.DataFrame:
    """
    Aggregate shot + pass features into per-90 player stats.
    Joins shot_features and pass_features by player_id, and attaches
    position_group from the players table.
    """
    logger.info("Aggregating player stats from feature store...")

    shot_query = """
        SELECT
            player_id,
            player_name,
            COUNT(*)                                    AS shots,
            SUM(CASE WHEN is_goal THEN 1 ELSE 0 END)   AS goals,
            SUM(CASE WHEN outcome_name = 'Saved'
                     THEN 1 ELSE 0 END)                AS shots_on_target,
            AVG(distance_to_goal)                       AS avg_shot_distance,
            AVG(angle_to_goal)                          AS avg_shot_angle,
            COUNT(DISTINCT match_id)                    AS matches_from_shots
        FROM shot_features
        WHERE player_id IS NOT NULL
        GROUP BY player_id, player_name
    """

    pass_query = """
        SELECT
            player_id,
            COUNT(*)                                        AS passes,
            SUM(CASE WHEN is_complete THEN 1 ELSE 0 END)   AS passes_complete,
            SUM(CASE WHEN is_switch THEN 1 ELSE 0 END)     AS switches,
            SUM(CASE WHEN is_through_ball THEN 1 ELSE 0 END) AS through_balls,
            SUM(CASE WHEN is_cross THEN 1 ELSE 0 END)      AS crosses,
            SUM(CASE WHEN goal_assist THEN 1 ELSE 0 END)   AS goal_assists,
            SUM(CASE WHEN shot_assist THEN 1 ELSE 0 END)   AS shot_assists,
            SUM(CASE WHEN under_pressure THEN 1 ELSE 0 END) AS passes_under_pressure,
            SUM(CASE WHEN end_x > start_x
                      AND SQRT(POWER(120 - end_x, 2) + POWER(40 - end_y, 2))
                        < 0.9 * SQRT(POWER(120 - start_x, 2) + POWER(40 - start_y, 2))
                     THEN 1 ELSE 0 END)                    AS progressive_passes,
            SUM(CASE WHEN end_x >= 102
                      AND end_y >= 18
                      AND end_y <= 62
                     THEN 1 ELSE 0 END)                    AS passes_into_box,
            AVG(length)                                     AS avg_pass_length,
            COUNT(DISTINCT match_id)                        AS matches_from_passes
        FROM pass_features
        WHERE player_id IS NOT NULL
        GROUP BY player_id
    """

    position_query = "SELECT player_id, position_group, team_name FROM players"

    with get_connection() as conn:
        shots_df    = pd.read_sql(shot_query, conn)
        passes_df   = pd.read_sql(pass_query, conn)
        try:
            positions_df = pd.read_sql(position_query, conn)
        except Exception:
            positions_df = pd.DataFrame(columns=["player_id", "position_group", "team_name"])

    logger.info(f"Shot aggregates: {len(shots_df)} players")
    logger.info(f"Pass aggregates: {len(passes_df)} players")

    # Merge on player_id
    df = shots_df.merge(passes_df, on="player_id", how="outer")
    df = df.merge(positions_df, on="player_id", how="left")

    # Estimate total matches (take max of shot/pass match counts)
    df["matches_played"] = df[["matches_from_shots", "matches_from_passes"]].max(axis=1)
    # Preserve string columns before filling numeric NaNs with 0
    _player_name = df.get("player_name")
    _position_group = df.get("position_group")
    _team_name = df.get("team_name")
    df = df.fillna(0)
    if _player_name is not None:
        df["player_name"] = _player_name
    df["position_group"] = _position_group.fillna("Unknown") if _position_group is not None else "Unknown"
    df["team_name"] = _team_name.fillna("") if _team_name is not None else ""

    # Rough minutes estimate: assume 85 mins average per match
    df["minutes_played"] = df["matches_played"] * 85

    # Filter minimum playing time
    df = df[df["minutes_played"] >= MIN_MINUTES].copy()
    logger.info(f"Players with {MIN_MINUTES}+ minutes: {len(df)}")

    return df


def compute_per90(df: pd.DataFrame) -> pd.DataFrame:
    """Convert raw counts to per-90-minute rates."""
    per90 = df.copy()
    mins  = df["minutes_played"].clip(lower=1)
    factor = 90 / mins

    count_cols = [
        "shots", "goals", "shots_on_target",
        "switches", "through_balls", "crosses",
        "goal_assists", "shot_assists",
        "passes_under_pressure", "progressive_passes",
        "passes_into_box", "passes", "passes_complete",
    ]

    for col in count_cols:
        if col in df.columns:
            per90[f"{col}_p90"] = df[col] * factor

    # Rates (not per90)
    per90["pass_completion_pct"] = np.where(
        df["passes"] > 0, df["passes_complete"] / df["passes"], 0
    )
    per90["shot_conversion_pct"] = np.where(
        df["shots"] > 0, df["goals"] / df["shots"], 0
    )
    per90["shots_on_target_pct"] = np.where(
        df["shots"] > 0, df["shots_on_target"] / df["shots"], 0
    )

    return per90


def build_feature_matrix(per90_df: pd.DataFrame) -> pd.DataFrame:
    """Extract and clean the feature matrix, indexed by player_id."""
    available = [f for f in EMBEDDING_FEATURES if f in per90_df.columns]
    X = per90_df[available].copy()
    X.index = per90_df["player_id"].astype(int).values
    X = X.fillna(0)
    # Clip extreme outliers (> 99th percentile)
    for col in X.columns:
        cap = X[col].quantile(0.99)
        X[col] = X[col].clip(upper=cap)
    return X
