"""
tests/test_ingest.py
Unit tests for ingestion helpers — no DB required, pure logic.
"""
from pipeline.ingest import (
    distance_to_goal,
    angle_to_goal,
    extract_360_shot_features,
    extract_360_pass_features,
)


# ── Geometry tests ─────────────────────────────────────────────────────────────

def test_distance_to_goal_penalty_spot():
    """Penalty spot is at x=108, y=40 — should be ~12m from goal."""
    d = distance_to_goal(108, 40)
    assert 11 < d < 13


def test_distance_to_goal_centre_circle():
    """Centre circle (60, 40) is far from goal."""
    d = distance_to_goal(60, 40)
    assert d > 50


def test_angle_to_goal_central_position():
    """Directly in front of goal should have wider angle than acute angle."""
    central = angle_to_goal(110, 40)   # central, close
    wide    = angle_to_goal(105, 40)   # slightly further
    assert central > wide


def test_angle_to_goal_wide_position():
    """Wide position at same distance should have narrower angle."""
    central = angle_to_goal(110, 40)
    wide    = angle_to_goal(110, 10)   # very wide
    assert central > wide


# ── 360 shot feature tests ─────────────────────────────────────────────────────

SAMPLE_FREEZE_FRAMES = [
    {"location": [115, 38], "teammate": False, "position": {"name": "Centre-Back"}},
    {"location": [116, 42], "teammate": False, "position": {"name": "Goalkeeper"}},
    {"location": [112, 35], "teammate": True,  "position": {"name": "Centre-Forward"}},
]

def test_360_shot_features_populated():
    feats = extract_360_shot_features(SAMPLE_FREEZE_FRAMES, 110, 40)
    assert feats["has_360"] is True
    assert feats["defenders_in_2m_cone"] is not None
    assert feats["defenders_in_5m_cone"] is not None


def test_360_shot_no_frames():
    feats = extract_360_shot_features([], 110, 40)
    assert feats["has_360"] is False
    assert feats["defenders_in_2m_cone"] is None


def test_360_gk_detected():
    feats = extract_360_shot_features(SAMPLE_FREEZE_FRAMES, 110, 40)
    # Goalkeeper at [116, 42] should be detected
    assert feats["gk_x"] == 116
    assert feats["gk_y"] == 42
    assert feats["gk_distance_to_goal_centre"] is not None


def test_numerical_advantage_calculated():
    feats = extract_360_shot_features(SAMPLE_FREEZE_FRAMES, 110, 40)
    # 1 attacker in box, 1 defender (GK not counted as outfield defender)
    assert feats["numerical_advantage"] is not None


# ── 360 pass feature tests ─────────────────────────────────────────────────────

PASS_FREEZE_FRAMES = [
    {"location": [55, 35], "teammate": False},  # defender in pass lane
    {"location": [65, 42], "teammate": False},  # defender near endpoint
    {"location": [70, 40], "teammate": True},   # teammate (ignored)
]

def test_360_pass_features_populated():
    feats = extract_360_pass_features(PASS_FREEZE_FRAMES, 50, 40, 70, 40)
    assert feats["has_360"] is True
    assert feats["defenders_in_pass_lane"] is not None


def test_360_pass_no_frames():
    feats = extract_360_pass_features([], 50, 40, 70, 40)
    assert feats["has_360"] is False


def test_defender_density_endpoint():
    """Defender at [65,42] is ~5.4m from endpoint [70,40] — outside 3m threshold."""
    feats = extract_360_pass_features(PASS_FREEZE_FRAMES, 50, 40, 70, 40)
    assert feats["defender_density_endpoint"] == 0

def test_defender_density_endpoint_within_threshold():
    """Defender placed within 3m of endpoint should be counted."""
    close_frames = [{"location": [68, 40], "teammate": False}]  # 2m from endpoint
    feats = extract_360_pass_features(close_frames, 50, 40, 70, 40)
    assert feats["defender_density_endpoint"] >= 1
