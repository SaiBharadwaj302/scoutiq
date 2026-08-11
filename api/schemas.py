"""
api/schemas.py

Pydantic request/response models for all ScoutIQ endpoints.
These define exactly what the API accepts and returns —
invalid inputs are rejected automatically before hitting the model.
"""
from pydantic import BaseModel, ConfigDict, Field
from typing import Optional


# ── xG Endpoint ───────────────────────────────────────────────────────────────

class ShotRequest(BaseModel):
    """Input for POST /v1/predict/xg"""
    location_x:        float = Field(..., ge=0, le=120, description="Shot x-coordinate (0-120)")
    location_y:        float = Field(..., ge=0, le=80,  description="Shot y-coordinate (0-80)")
    technique_name:    Optional[str] = Field("Normal", description="Shot technique")
    body_part_name:    Optional[str] = Field("Right Foot", description="Body part used")
    play_pattern_name: Optional[str] = Field("Regular Play", description="Play pattern")
    under_pressure:    bool = False
    first_time:        bool = False
    follows_dribble:   bool = False
    open_goal:         bool = False
    deflected:         bool = False
    period:            int  = Field(1, ge=1, le=5)
    # Optional 360 spatial features
    defenders_in_2m_cone:       Optional[float] = None
    defenders_in_5m_cone:       Optional[float] = None
    gk_distance_to_goal_centre: Optional[float] = None
    attackers_in_box:           Optional[float] = None
    defenders_in_box:           Optional[float] = None
    numerical_advantage:        Optional[float] = None
    nearest_defender_distance:  Optional[float] = None

    model_config = {"json_schema_extra": {"example": {
        "location_x": 105.0, "location_y": 38.0,
        "technique_name": "Normal", "body_part_name": "Right Foot",
        "play_pattern_name": "Regular Play",
        "under_pressure": False, "first_time": True,
        "period": 1
    }}}


class XGResponse(BaseModel):
    """Output for POST /v1/predict/xg"""
    model_config = ConfigDict(protected_namespaces=())

    xg:              float = Field(..., description="Expected goals probability (0-1)")
    is_big_chance:   bool  = Field(..., description="xG > 0.3")
    distance_to_goal: float
    angle_to_goal:   float
    model_version:   str
    model_name:      str = "scoutiq_xg"


# ── Pass Success Endpoint ──────────────────────────────────────────────────────

class PassRequest(BaseModel):
    """Input for POST /v1/predict/pass"""
    start_x:          float = Field(..., ge=0, le=120)
    start_y:          float = Field(..., ge=0, le=80)
    end_x:            float = Field(..., ge=0, le=120)
    end_y:            float = Field(..., ge=0, le=80)
    length:           Optional[float] = None
    angle:            Optional[float] = None
    height_name:      Optional[str]  = "Ground Pass"
    body_part_name:   Optional[str]  = "Right Foot"
    play_pattern_name: Optional[str] = "Regular Play"
    under_pressure:   bool = False
    is_switch:        bool = False
    is_through_ball:  bool = False
    is_cut_back:      bool = False
    is_cross:         bool = False
    goal_assist:      bool = False
    shot_assist:      bool = False
    period:           int  = Field(1, ge=1, le=5)
    # Optional 360 features
    defenders_in_pass_lane:    Optional[float] = None
    defender_density_endpoint: Optional[float] = None
    nearest_defender_distance: Optional[float] = None
    nearest_opponent_at_end:   Optional[float] = None

    model_config = {"json_schema_extra": {"example": {
        "start_x": 60.0, "start_y": 40.0,
        "end_x": 85.0,   "end_y": 35.0,
        "height_name": "Ground Pass",
        "body_part_name": "Right Foot",
        "under_pressure": False
    }}}


class PassResponse(BaseModel):
    """Output for POST /v1/predict/pass"""
    model_config = ConfigDict(protected_namespaces=())

    completion_probability: float = Field(..., description="Pass success probability (0-1)")
    is_high_risk:           bool  = Field(..., description="Completion probability < 0.5")
    pass_length:            float
    model_version:          str
    model_name:             str = "scoutiq_pass"


# ── Player Similarity Endpoint ────────────────────────────────────────────────

class SimilarPlayer(BaseModel):
    player_id:   int
    player_name: str
    similarity:  float = Field(..., description="Cosine similarity score (0-1)")


class SimilarityResponse(BaseModel):
    """Output for GET /v1/players/similar/{player_id}"""
    model_config = ConfigDict(protected_namespaces=())

    query_player_id:   int
    query_player_name: str
    similar_players:   list[SimilarPlayer]
    model_name:        str = "scoutiq_similarity"


# ── Health Check ──────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status:  str
    models:  dict[str, str]
    version: str = "1.0.0"
