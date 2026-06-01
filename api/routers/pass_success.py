"""
api/routers/pass_success.py

POST /v1/predict/pass
Given a pass attempt, returns the completion probability.
"""
import math
import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException

from api.schemas import PassRequest, PassResponse
from api.dependencies import get_pass_model
from models.pass_success.features import build_features

router = APIRouter(prefix="/v1/predict", tags=["Pass Success Model"])


def _pass_request_to_df(p: PassRequest) -> pd.DataFrame:
    """Convert PassRequest into a DataFrame matching pass_features schema."""
    length = p.length
    angle  = p.angle
    if length is None:
        length = math.sqrt((p.end_x - p.start_x)**2 + (p.end_y - p.start_y)**2)
    if angle is None:
        angle = math.atan2(p.end_y - p.start_y, p.end_x - p.start_x)

    return pd.DataFrame([{
        "length":                    length,
        "angle":                     angle,
        "under_pressure":            p.under_pressure,
        "is_switch":                 p.is_switch,
        "is_through_ball":           p.is_through_ball,
        "is_cut_back":               p.is_cut_back,
        "is_cross":                  p.is_cross,
        "goal_assist":               p.goal_assist,
        "shot_assist":               p.shot_assist,
        "start_x":                   p.start_x,
        "start_y":                   p.start_y,
        "end_x":                     p.end_x,
        "end_y":                     p.end_y,
        "height_name":               p.height_name,
        "body_part_name":            p.body_part_name,
        "play_pattern_name":         p.play_pattern_name,
        "period":                    p.period,
        "defenders_in_pass_lane":    p.defenders_in_pass_lane,
        "defender_density_endpoint": p.defender_density_endpoint,
        "nearest_defender_distance": p.nearest_defender_distance,
        "nearest_opponent_at_end":   p.nearest_opponent_at_end,
    }])


@router.post("/pass", response_model=PassResponse, summary="Predict Pass Completion")
async def predict_pass(
    pass_event: PassRequest,
    model=Depends(get_pass_model),
):
    """
    Predict the probability that a pass is completed successfully.

    - **start_x/y**: pass origin coordinates
    - **end_x/y**: pass destination coordinates
    - **360 features**: optional — include defender positions for higher accuracy
    - Returns **completion_probability** between 0 and 1
    """
    try:
        raw_df   = _pass_request_to_df(pass_event)
        features = build_features(raw_df)
        prob     = float(model.predict_proba(features)[0][1])
        length   = raw_df["length"].iloc[0]

        return PassResponse(
            completion_probability=round(prob, 4),
            is_high_risk=prob < 0.5,
            pass_length=round(float(length), 2),
            model_version="scoutiq_pass/1",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
