"""
tests/test_features.py

Unit tests for feature engineering functions.
Tests run without a live database — all inputs are synthetic DataFrames.
"""
import math
import numpy as np
import pandas as pd
import pytest

from models.xg.features import build_features as build_xg_features, BASE_FEATURES, SPATIAL_FEATURES_360
from models.pass_success.features import build_features as build_pass_features


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_shot_row(**overrides) -> pd.DataFrame:
    """Return a one-row shot DataFrame with sensible defaults."""
    defaults = {
        "distance_to_goal":  15.0,
        "angle_to_goal":     0.45,
        "under_pressure":    False,
        "first_time":        False,
        "follows_dribble":   False,
        "open_goal":         False,
        "deflected":         False,
        "technique_name":    "Normal",
        "body_part_name":    "Right Foot",
        "play_pattern_name": "Regular Play",
        "period":            1,
        # 360 features absent by default
    }
    defaults.update(overrides)
    return pd.DataFrame([defaults])


def _make_pass_row(**overrides) -> pd.DataFrame:
    """Return a one-row pass DataFrame with sensible defaults."""
    defaults = {
        "length":           25.0,
        "angle":            0.1,
        "under_pressure":   False,
        "is_switch":        False,
        "is_through_ball":  False,
        "is_cut_back":      False,
        "is_cross":         False,
        "goal_assist":      False,
        "shot_assist":      False,
        "start_x":          60.0,
        "start_y":          40.0,
        "end_x":            85.0,
        "end_y":            35.0,
        "height_name":      "Ground Pass",
        "body_part_name":   "Right Foot",
        "play_pattern_name":"Regular Play",
        "period":           1,
    }
    defaults.update(overrides)
    return pd.DataFrame([defaults])


# ── xG feature tests ──────────────────────────────────────────────────────────

class TestXGFeatures:
    def test_output_columns_present(self):
        feat = build_xg_features(_make_shot_row())
        for col in BASE_FEATURES:
            assert col in feat.columns, f"Missing column: {col}"

    def test_output_shape(self):
        df = pd.concat([_make_shot_row(), _make_shot_row(distance_to_goal=30.0)])
        feat = build_xg_features(df)
        assert feat.shape[0] == 2
        assert feat.shape[1] == len(BASE_FEATURES) + len(SPATIAL_FEATURES_360)

    def test_distance_passthrough(self):
        feat = build_xg_features(_make_shot_row(distance_to_goal=20.5))
        assert abs(feat["distance_to_goal"].iloc[0] - 20.5) < 1e-9

    def test_missing_distance_imputed(self):
        """NaN distance is filled: no NaN should survive in the output feature matrix.
        This verifies the fallback logic works in a realistic multi-row batch."""
        import numpy as np
        two_rows = pd.concat([
            _make_shot_row(distance_to_goal=22.0),
            _make_shot_row(distance_to_goal=float("nan")),
        ], ignore_index=True)
        # Sanity-check the input contains one valid value
        assert not np.isnan(two_rows["distance_to_goal"].iloc[0])
        assert np.isnan(two_rows["distance_to_goal"].iloc[1])
        feat = build_xg_features(two_rows)
        # Filled row must not be NaN
        assert not np.isnan(float(feat["distance_to_goal"].iloc[1])), (
            f"Expected imputed value, got NaN. "
            f"Input distance col: {two_rows['distance_to_goal'].tolist()}, "
            f"Output: {feat['distance_to_goal'].tolist()}"
        )

    def test_boolean_encoded_as_int(self):
        feat = build_xg_features(_make_shot_row(under_pressure=True))
        assert feat["under_pressure"].iloc[0] == 1
        feat2 = build_xg_features(_make_shot_row(under_pressure=False))
        assert feat2["under_pressure"].iloc[0] == 0

    def test_technique_encoding_normal(self):
        feat = build_xg_features(_make_shot_row(technique_name="Normal"))
        assert feat["technique_normal"].iloc[0] == 1
        assert feat["technique_header"].iloc[0] == 0

    def test_technique_encoding_header(self):
        feat = build_xg_features(_make_shot_row(technique_name="Header"))
        assert feat["technique_header"].iloc[0] == 1
        assert feat["technique_normal"].iloc[0] == 0

    def test_body_part_encoding(self):
        feat = build_xg_features(_make_shot_row(body_part_name="Left Foot"))
        assert feat["body_part_left_foot"].iloc[0] == 1
        assert feat["body_part_right_foot"].iloc[0] == 0
        assert feat["body_part_head"].iloc[0] == 0

    def test_big_chance_flag(self):
        # distance < 10 and angle > 0.5 → big_chance = 1
        feat = build_xg_features(_make_shot_row(distance_to_goal=8.0, angle_to_goal=0.6))
        assert feat["big_chance"].iloc[0] == 1

    def test_not_big_chance_flag(self):
        feat = build_xg_features(_make_shot_row(distance_to_goal=20.0, angle_to_goal=0.3))
        assert feat["big_chance"].iloc[0] == 0

    def test_long_shot_flag(self):
        feat = build_xg_features(_make_shot_row(distance_to_goal=28.0))
        assert feat["long_shot"].iloc[0] == 1

    def test_period_2_flag(self):
        feat1 = build_xg_features(_make_shot_row(period=1))
        feat2 = build_xg_features(_make_shot_row(period=2))
        assert feat1["period_2"].iloc[0] == 0
        assert feat2["period_2"].iloc[0] == 1

    def test_missing_360_filled_with_sentinel(self):
        """360 features missing → filled with -1 sentinel."""
        feat = build_xg_features(_make_shot_row())   # no 360 cols
        for col in SPATIAL_FEATURES_360:
            assert feat[col].iloc[0] == -1

    def test_play_pattern_counter(self):
        feat = build_xg_features(_make_shot_row(play_pattern_name="From Counter"))
        assert feat["play_pattern_counter"].iloc[0] == 1

    def test_no_nulls_in_output(self):
        feat = build_xg_features(_make_shot_row())
        assert not feat.isnull().any().any(), "Output contains NaN values"


# ── Pass feature tests ────────────────────────────────────────────────────────

class TestPassFeatures:
    def test_output_columns_present(self):
        from models.pass_success.features import BASE_FEATURES as PBF
        feat = build_pass_features(_make_pass_row())
        for col in PBF:
            assert col in feat.columns, f"Missing pass column: {col}"

    def test_length_passthrough(self):
        feat = build_pass_features(_make_pass_row(length=30.0))
        assert abs(feat["length"].iloc[0] - 30.0) < 1e-9

    def test_angle_circular_encoding(self):
        feat = build_pass_features(_make_pass_row(angle=math.pi / 4))
        # sin(π/4) ≈ 0.707
        assert abs(feat["angle_sin"].iloc[0] - math.sin(math.pi / 4)) < 1e-9
        assert abs(feat["angle_cos"].iloc[0] - math.cos(math.pi / 4)) < 1e-9

    def test_forward_pass_flag(self):
        feat = build_pass_features(_make_pass_row(start_x=50.0, end_x=70.0))
        assert feat["is_forward_pass"].iloc[0] == 1

    def test_backward_pass_flag(self):
        feat = build_pass_features(_make_pass_row(start_x=70.0, end_x=50.0))
        assert feat["is_forward_pass"].iloc[0] == 0

    def test_long_pass_flag(self):
        feat = build_pass_features(_make_pass_row(length=35.0))
        assert feat["is_long_pass"].iloc[0] == 1
        assert feat["is_short_pass"].iloc[0] == 0

    def test_short_pass_flag(self):
        feat = build_pass_features(_make_pass_row(length=8.0))
        assert feat["is_short_pass"].iloc[0] == 1
        assert feat["is_long_pass"].iloc[0] == 0

    def test_end_zone_box(self):
        # end inside box: x>=102, 18<=y<=62
        feat = build_pass_features(_make_pass_row(end_x=110.0, end_y=40.0))
        assert feat["end_zone_box"].iloc[0] == 1

    def test_end_zone_not_box(self):
        feat = build_pass_features(_make_pass_row(end_x=50.0, end_y=40.0))
        assert feat["end_zone_box"].iloc[0] == 0

    def test_height_encoding(self):
        feat_ground = build_pass_features(_make_pass_row(height_name="Ground Pass"))
        assert feat_ground["height_ground"].iloc[0] == 1
        assert feat_ground["height_high"].iloc[0] == 0

        feat_high = build_pass_features(_make_pass_row(height_name="High Pass"))
        assert feat_high["height_high"].iloc[0] == 1
        assert feat_high["height_ground"].iloc[0] == 0

    def test_progressive_pass(self):
        # Pass that moves significantly closer to goal
        feat = build_pass_features(_make_pass_row(
            start_x=60.0, start_y=40.0,
            end_x=100.0, end_y=40.0,
        ))
        assert feat["progressive"].iloc[0] == 1

    def test_no_nulls_in_output(self):
        feat = build_pass_features(_make_pass_row())
        assert not feat.isnull().any().any(), "Pass features contain NaN values"
