"""
models/pass_success/features.py

Feature engineering for the Pass Success Probability model.
Pulls from pass_features table and builds the final feature matrix.
"""
import numpy as np
import pandas as pd
from loguru import logger

from db.store import get_pass_features

# ── Feature columns ───────────────────────────────────────────────────────────

BASE_FEATURES = [
    "length",
    "angle_sin",          # sin(angle) — circular encoding
    "angle_cos",          # cos(angle) — circular encoding
    "under_pressure",
    "is_switch",
    "is_through_ball",
    "is_cut_back",
    "is_cross",
    "goal_assist",
    "shot_assist",
    # Location zones
    "start_zone_defensive",   # own half
    "start_zone_middle",      # middle third
    "start_zone_attacking",   # final third
    "end_zone_attacking",     # pass ends in final third
    "end_zone_box",           # pass ends in penalty box
    # Encoded categoricals
    "height_ground",
    "height_low",
    "height_high",
    "body_head",
    "body_left",
    "body_right",
    "play_pattern_regular",
    "play_pattern_counter",
    "play_pattern_set_piece",
    # Derived
    "is_forward_pass",        # end_x > start_x
    "is_long_pass",           # length > 32m
    "is_short_pass",          # length < 10m
    "progressive",            # advances ball significantly toward goal
    "period_2",
]

SPATIAL_FEATURES_360 = [
    "defenders_in_pass_lane",
    "defender_density_endpoint",
    "nearest_defender_distance",
    "nearest_opponent_at_end",
]

ALL_FEATURES = BASE_FEATURES + SPATIAL_FEATURES_360
TARGET       = "is_complete"


# ── Feature Builder ───────────────────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    feat = pd.DataFrame(index=df.index)

    # ── Pass geometry ─────────────────────────────────────────────────────
    feat["length"] = df["length"].fillna(df["length"].median())

    # Circular encoding for angle (avoids discontinuity at ±π)
    angle = df["angle"].fillna(0)
    feat["angle_sin"] = np.sin(angle)
    feat["angle_cos"] = np.cos(angle)

    # ── Boolean features ──────────────────────────────────────────────────
    for col in ["under_pressure", "is_switch", "is_through_ball",
                "is_cut_back", "is_cross", "goal_assist", "shot_assist"]:
        feat[col] = df[col].fillna(False).astype(int)

    # ── Start location zones (StatsBomb pitch: 0-120 length) ─────────────
    sx = df["start_x"].fillna(60)
    feat["start_zone_defensive"] = (sx < 40).astype(int)
    feat["start_zone_middle"]    = ((sx >= 40) & (sx < 80)).astype(int)
    feat["start_zone_attacking"] = (sx >= 80).astype(int)

    # ── End location zones ────────────────────────────────────────────────
    ex = df["end_x"].fillna(60)
    ey = df["end_y"].fillna(40)
    feat["end_zone_attacking"] = (ex >= 80).astype(int)
    feat["end_zone_box"]       = ((ex >= 102) & (ey >= 18) & (ey <= 62)).astype(int)

    # ── Height encoding ───────────────────────────────────────────────────
    height = df["height_name"].fillna("Ground Pass").str.lower()
    feat["height_ground"] = (height == "ground pass").astype(int)
    feat["height_low"]    = (height == "low pass").astype(int)
    feat["height_high"]   = (height == "high pass").astype(int)

    # ── Body part encoding ────────────────────────────────────────────────
    body = df["body_part_name"].fillna("Right Foot").str.lower()
    feat["body_head"]  = (body == "head").astype(int)
    feat["body_left"]  = (body == "left foot").astype(int)
    feat["body_right"] = (body == "right foot").astype(int)

    # ── Play pattern ──────────────────────────────────────────────────────
    pattern = df["play_pattern_name"].fillna("Regular Play").str.lower()
    feat["play_pattern_regular"]   = (pattern == "regular play").astype(int)
    feat["play_pattern_counter"]   = pattern.str.contains("counter", na=False).astype(int)
    feat["play_pattern_set_piece"] = pattern.str.contains(
        "set piece|free kick|corner|throw", na=False).astype(int)

    # ── Derived features ──────────────────────────────────────────────────
    feat["is_forward_pass"] = (ex > sx).astype(int)
    feat["is_long_pass"]    = (feat["length"] > 32).astype(int)
    feat["is_short_pass"]   = (feat["length"] < 10).astype(int)

    # Progressive pass: end is at least 10% closer to goal than start
    start_dist = np.sqrt((120 - sx)**2 + (40 - df["start_y"].fillna(40))**2)
    end_dist   = np.sqrt((120 - ex)**2 + (40 - ey)**2)
    feat["progressive"] = (end_dist < 0.9 * start_dist).astype(int)

    feat["period_2"] = (df["period"] == 2).astype(int)

    # ── 360 spatial features ──────────────────────────────────────────────
    for col in SPATIAL_FEATURES_360:
        if col in df.columns:
            feat[col] = pd.to_numeric(df[col], errors="coerce").fillna(-1.0)
        else:
            feat[col] = -1.0

    return feat


def load_training_data(
    min_passes: int = 1000,
    use_360_only: bool = False,
    sample_size: int | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load pass features and return (X, y) ready for training.

    Args:
        min_passes:   minimum passes required
        use_360_only: only use passes with 360 data
        sample_size:  cap dataset size (for fast testing)
    """
    logger.info("Loading pass features from feature store...")
    raw = get_pass_features(min_has_360=use_360_only)

    if len(raw) < min_passes:
        raise ValueError(f"Only {len(raw)} passes found, need at least {min_passes}")

    if sample_size and len(raw) > sample_size:
        original_size = len(raw)
        raw = raw.sample(sample_size, random_state=42)
        logger.info(f"Sampled {sample_size} passes from {original_size} total")

    logger.info(f"Loaded {len(raw)} passes | "
                f"Complete: {raw['is_complete'].sum()} "
                f"({raw['is_complete'].mean()*100:.1f}%)")

    X = build_features(raw)
    y = raw[TARGET].astype(int)

    logger.info(f"Feature matrix: {X.shape[0]} rows × {X.shape[1]} features")
    return X, y


def get_feature_names() -> list[str]:
    return ALL_FEATURES
