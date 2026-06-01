"""
models/xg/features.py

Feature engineering for the xG 2.0 model.
Pulls from shot_features table and builds the final feature matrix.
"""
import numpy as np
import pandas as pd
from loguru import logger

from db.store import get_shot_features

# ── Feature columns used by the model ────────────────────────────────────────

# Base features (available for ALL shots)
BASE_FEATURES = [
    "distance_to_goal",
    "angle_to_goal",
    "under_pressure",
    "first_time",
    "follows_dribble",
    "open_goal",
    "deflected",
    # Encoded categoricals (added during build)
    "technique_normal",
    "technique_header",
    "technique_volley",
    "technique_half_volley",
    "body_part_head",
    "body_part_left_foot",
    "body_part_right_foot",
    "play_pattern_regular",
    "play_pattern_counter",
    "play_pattern_set_piece",
    "play_pattern_corner",
    "play_pattern_free_kick",
    "period_2",       # second half flag
    "big_chance",     # distance < 10 and angle > 0.5
    "long_shot",      # distance > 25
]

# 360 spatial features (only available for ~30% of shots)
SPATIAL_FEATURES_360 = [
    "defenders_in_2m_cone",
    "defenders_in_5m_cone",
    "gk_distance_to_goal_centre",
    "attackers_in_box",
    "defenders_in_box",
    "numerical_advantage",
    "nearest_defender_distance",
]

ALL_FEATURES    = BASE_FEATURES + SPATIAL_FEATURES_360
TARGET          = "is_goal"


# ── Feature Builder ───────────────────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw shot_features rows into model-ready feature matrix.
    Handles missing values and encodes categoricals.
    """
    feat = pd.DataFrame(index=df.index)

    # ── Continuous base features ──────────────────────────────────────────
    # Hard fallbacks guard against single-row batches where median is NaN
    _dist_median = df["distance_to_goal"].median()
    _ang_median  = df["angle_to_goal"].median()
    _dist_fill = float(_dist_median) if pd.notna(_dist_median) else 20.0
    _ang_fill  = float(_ang_median)  if pd.notna(_ang_median)  else 0.35
    feat["distance_to_goal"] = df["distance_to_goal"].fillna(_dist_fill)
    feat["angle_to_goal"]    = df["angle_to_goal"].fillna(_ang_fill)

    # ── Boolean features ──────────────────────────────────────────────────
    for col in ["under_pressure", "first_time", "follows_dribble", "open_goal", "deflected"]:
        feat[col] = df[col].fillna(False).astype(int)

    # ── Technique encoding ────────────────────────────────────────────────
    tech = df["technique_name"].fillna("Normal").str.lower()
    feat["technique_normal"]      = (tech == "normal").astype(int)
    feat["technique_header"]      = tech.str.contains("head", na=False).astype(int)
    feat["technique_volley"]      = (tech == "volley").astype(int)
    feat["technique_half_volley"] = (tech == "half volley").astype(int)

    # ── Body part encoding ────────────────────────────────────────────────
    body = df["body_part_name"].fillna("Right Foot").str.lower()
    feat["body_part_head"]       = (body == "head").astype(int)
    feat["body_part_left_foot"]  = (body == "left foot").astype(int)
    feat["body_part_right_foot"] = (body == "right foot").astype(int)

    # ── Play pattern encoding ─────────────────────────────────────────────
    pattern = df["play_pattern_name"].fillna("Regular Play").str.lower()
    feat["play_pattern_regular"]   = (pattern == "regular play").astype(int)
    feat["play_pattern_counter"]   = pattern.str.contains("counter", na=False).astype(int)
    feat["play_pattern_set_piece"] = pattern.str.contains("set piece|free kick|corner|throw", na=False).astype(int)
    feat["play_pattern_corner"]    = pattern.str.contains("corner", na=False).astype(int)
    feat["play_pattern_free_kick"] = pattern.str.contains("free kick", na=False).astype(int)

    # ── Period ────────────────────────────────────────────────────────────
    feat["period_2"] = (df["period"] == 2).astype(int)

    # ── Derived spatial context features ─────────────────────────────────
    dist = feat["distance_to_goal"]
    ang  = feat["angle_to_goal"]
    feat["big_chance"] = ((dist < 10) & (ang > 0.5)).astype(int)
    feat["long_shot"]  = (dist > 25).astype(int)

    # ── 360 spatial features ──────────────────────────────────────────────
    # Fill NaN with -1 as a sentinel value (model learns "no 360 data" pattern)
    for col in SPATIAL_FEATURES_360:
        if col in df.columns:
            feat[col] = pd.to_numeric(df[col], errors="coerce").fillna(-1.0)
        else:
            feat[col] = -1.0

    return feat


def load_training_data(
    min_shots: int = 100,
    use_360_only: bool = False,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load shot features from DB and return (X, y) ready for training.

    Args:
        min_shots:    minimum shots required (raises if fewer found)
        use_360_only: if True, only use shots with 360 data

    Returns:
        X: feature DataFrame
        y: target Series (is_goal)
    """
    logger.info("Loading shot features from feature store...")
    raw = get_shot_features(min_has_360=use_360_only)

    if len(raw) < min_shots:
        raise ValueError(f"Only {len(raw)} shots found, need at least {min_shots}")

    logger.info(f"Loaded {len(raw)} shots | Goals: {raw['is_goal'].sum()} "
                f"({raw['is_goal'].mean()*100:.1f}%)")

    X = build_features(raw)
    y = raw[TARGET].astype(int)

    logger.info(f"Feature matrix: {X.shape[0]} rows × {X.shape[1]} features")
    return X, y


def get_feature_names() -> list[str]:
    return ALL_FEATURES
