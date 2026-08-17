"""
streamlit_app/features.py

Standalone copies of the xG and pass-success feature builders, decoupled
from models/xg/features.py and models/pass_success/features.py.

Why duplicated rather than imported: the originals import `db.store` at
module scope (for load_training_data, unused here), which pulls in
sqlalchemy/psycopg2 and — more importantly — ties this file's importability
to the parent repo's package layout and working directory. This dashboard
is deployed standalone (Streamlit Community Cloud only reads this
subdirectory), so it carries its own copy of the pure-pandas transform
logic with zero cross-package dependencies. Keep in sync with the
originals if the feature engineering changes.
"""
import numpy as np
import pandas as pd

# ── xG ─────────────────────────────────────────────────────────────────────

XG_SPATIAL_FEATURES_360 = [
    "defenders_in_2m_cone",
    "defenders_in_5m_cone",
    "gk_distance_to_goal_centre",
    "attackers_in_box",
    "defenders_in_box",
    "numerical_advantage",
    "nearest_defender_distance",
]


def build_xg_features(df: pd.DataFrame) -> pd.DataFrame:
    """Transform raw shot rows into the xG model's feature matrix."""
    feat = pd.DataFrame(index=df.index)

    _dist_median = df["distance_to_goal"].median()
    _ang_median  = df["angle_to_goal"].median()
    _dist_fill = float(_dist_median) if pd.notna(_dist_median) else 20.0
    _ang_fill  = float(_ang_median)  if pd.notna(_ang_median)  else 0.35
    feat["distance_to_goal"] = df["distance_to_goal"].fillna(_dist_fill)
    feat["angle_to_goal"]    = df["angle_to_goal"].fillna(_ang_fill)

    for col in ["under_pressure", "first_time", "follows_dribble", "open_goal", "deflected"]:
        feat[col] = df[col].fillna(False).astype(int)

    tech = df["technique_name"].fillna("Normal").str.lower()
    feat["technique_normal"]      = (tech == "normal").astype(int)
    feat["technique_header"]      = tech.str.contains("head", na=False).astype(int)
    feat["technique_volley"]      = (tech == "volley").astype(int)
    feat["technique_half_volley"] = (tech == "half volley").astype(int)

    body = df["body_part_name"].fillna("Right Foot").str.lower()
    feat["body_part_head"]       = (body == "head").astype(int)
    feat["body_part_left_foot"]  = (body == "left foot").astype(int)
    feat["body_part_right_foot"] = (body == "right foot").astype(int)

    pattern = df["play_pattern_name"].fillna("Regular Play").str.lower()
    feat["play_pattern_regular"]   = (pattern == "regular play").astype(int)
    feat["play_pattern_counter"]   = pattern.str.contains("counter", na=False).astype(int)
    feat["play_pattern_set_piece"] = pattern.str.contains("set piece|free kick|corner|throw", na=False).astype(int)
    feat["play_pattern_corner"]    = pattern.str.contains("corner", na=False).astype(int)
    feat["play_pattern_free_kick"] = pattern.str.contains("free kick", na=False).astype(int)

    feat["period_2"] = (df["period"] == 2).astype(int)

    dist = feat["distance_to_goal"]
    ang  = feat["angle_to_goal"]
    feat["big_chance"] = ((dist < 10) & (ang > 0.5)).astype(int)
    feat["long_shot"]  = (dist > 25).astype(int)

    for col in XG_SPATIAL_FEATURES_360:
        if col in df.columns:
            feat[col] = pd.to_numeric(df[col], errors="coerce").fillna(-1.0)
        else:
            feat[col] = -1.0

    return feat


# ── Pass success ─────────────────────────────────────────────────────────────

PASS_SPATIAL_FEATURES_360 = [
    "defenders_in_pass_lane",
    "defender_density_endpoint",
    "nearest_defender_distance",
    "nearest_opponent_at_end",
]


def build_pass_features(df: pd.DataFrame) -> pd.DataFrame:
    """Transform raw pass rows into the pass-success model's feature matrix."""
    feat = pd.DataFrame(index=df.index)

    feat["length"] = df["length"].fillna(df["length"].median())

    angle = df["angle"].fillna(0)
    feat["angle_sin"] = np.sin(angle)
    feat["angle_cos"] = np.cos(angle)

    for col in ["under_pressure", "is_switch", "is_through_ball",
                "is_cut_back", "is_cross", "goal_assist", "shot_assist"]:
        feat[col] = df[col].fillna(False).astype(int)

    sx = df["start_x"].fillna(60)
    feat["start_zone_defensive"] = (sx < 40).astype(int)
    feat["start_zone_middle"]    = ((sx >= 40) & (sx < 80)).astype(int)
    feat["start_zone_attacking"] = (sx >= 80).astype(int)

    ex = df["end_x"].fillna(60)
    ey = df["end_y"].fillna(40)
    feat["end_zone_attacking"] = (ex >= 80).astype(int)
    feat["end_zone_box"]       = ((ex >= 102) & (ey >= 18) & (ey <= 62)).astype(int)

    height = df["height_name"].fillna("Ground Pass").str.lower()
    feat["height_ground"] = (height == "ground pass").astype(int)
    feat["height_low"]    = (height == "low pass").astype(int)
    feat["height_high"]   = (height == "high pass").astype(int)

    body = df["body_part_name"].fillna("Right Foot").str.lower()
    feat["body_head"]  = (body == "head").astype(int)
    feat["body_left"]  = (body == "left foot").astype(int)
    feat["body_right"] = (body == "right foot").astype(int)

    pattern = df["play_pattern_name"].fillna("Regular Play").str.lower()
    feat["play_pattern_regular"]   = (pattern == "regular play").astype(int)
    feat["play_pattern_counter"]   = pattern.str.contains("counter", na=False).astype(int)
    feat["play_pattern_set_piece"] = pattern.str.contains(
        "set piece|free kick|corner|throw", na=False).astype(int)

    feat["is_forward_pass"] = (ex > sx).astype(int)
    feat["is_long_pass"]    = (feat["length"] > 32).astype(int)
    feat["is_short_pass"]   = (feat["length"] < 10).astype(int)

    start_dist = np.sqrt((120 - sx)**2 + (40 - df["start_y"].fillna(40))**2)
    end_dist   = np.sqrt((120 - ex)**2 + (40 - ey)**2)
    feat["progressive"] = (end_dist < 0.9 * start_dist).astype(int)

    feat["period_2"] = (df["period"] == 2).astype(int)

    for col in PASS_SPATIAL_FEATURES_360:
        if col in df.columns:
            feat[col] = pd.to_numeric(df[col], errors="coerce").fillna(-1.0)
        else:
            feat[col] = -1.0

    return feat
