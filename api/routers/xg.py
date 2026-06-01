"""
api/routers/xg.py

POST /v1/predict/xg
Given a shot event, returns the expected goals probability.
"""
import math
import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException

from api.schemas import ShotRequest, XGResponse
from api.dependencies import get_xg_model
from models.xg.features import build_features

router = APIRouter(prefix="/v1/predict", tags=["xG Model"])

GOAL_CENTRE = (120.0, 40.0)


def _distance_to_goal(x: float, y: float) -> float:
    return math.sqrt((x - GOAL_CENTRE[0])**2 + (y - GOAL_CENTRE[1])**2)


def _angle_to_goal(x: float, y: float) -> float:
    dx1, dy1 = GOAL_CENTRE[0] - x, 36.0 - y
    dx2, dy2 = GOAL_CENTRE[0] - x, 44.0 - y
    cross = abs(dx1 * dy2 - dy1 * dx2)
    dot   = dx1 * dx2 + dy1 * dy2
    return math.atan2(cross, dot)


def _shot_request_to_df(shot: ShotRequest) -> pd.DataFrame:
    """Convert a ShotRequest into a DataFrame matching shot_features schema."""
    return pd.DataFrame([{
        "distance_to_goal":           _distance_to_goal(shot.location_x, shot.location_y),
        "angle_to_goal":              _angle_to_goal(shot.location_x, shot.location_y),
        "under_pressure":             shot.under_pressure,
        "first_time":                 shot.first_time,
        "follows_dribble":            shot.follows_dribble,
        "open_goal":                  shot.open_goal,
        "deflected":                  shot.deflected,
        "technique_name":             shot.technique_name,
        "body_part_name":             shot.body_part_name,
        "play_pattern_name":          shot.play_pattern_name,
        "period":                     shot.period,
        "defenders_in_2m_cone":       shot.defenders_in_2m_cone,
        "defenders_in_5m_cone":       shot.defenders_in_5m_cone,
        "gk_distance_to_goal_centre": shot.gk_distance_to_goal_centre,
        "gk_x":                       None,
        "gk_y":                       None,
        "attackers_in_box":           shot.attackers_in_box,
        "defenders_in_box":           shot.defenders_in_box,
        "numerical_advantage":        shot.numerical_advantage,
        "nearest_defender_distance":  shot.nearest_defender_distance,
    }])


@router.post("/xg", response_model=XGResponse, summary="Predict Expected Goals")
async def predict_xg(
    shot: ShotRequest,
    model=Depends(get_xg_model),
):
    """
    Predict the probability that a shot results in a goal (xG).

    - **location_x / location_y**: shot coordinates on StatsBomb pitch (0-120, 0-80)
    - **360 features**: optional — include for higher accuracy when available
    - Returns **xg** probability between 0 and 1
    """
    try:
        raw_df   = _shot_request_to_df(shot)
        features = build_features(raw_df)
        prob     = float(model.predict_proba(features)[0][1])
        dist     = _distance_to_goal(shot.location_x, shot.location_y)
        angle    = _angle_to_goal(shot.location_x, shot.location_y)

        return XGResponse(
            xg=round(prob, 4),
            is_big_chance=prob > 0.3,
            distance_to_goal=round(dist, 2),
            angle_to_goal=round(angle, 4),
            model_version="scoutiq_xg/1",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
