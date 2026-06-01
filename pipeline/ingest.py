"""
pipeline/ingest.py

Ingests StatsBomb open data into PostgreSQL.
Compatible with statsbombpy >= 1.17.0 which returns flattened DataFrames.
"""
import math
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from loguru import logger
from statsbombpy import sb

from db.store import init_schema, log_pipeline_run, upsert_dataframe

load_dotenv()

# StatsBomb pitch dimensions
PITCH_LENGTH  = 120.0
PITCH_WIDTH   = 80.0
GOAL_CENTRE   = (120.0, 40.0)


# ── Geometry Helpers ──────────────────────────────────────────────────────────

def distance_to_goal(x: float, y: float) -> float:
    return math.sqrt((x - GOAL_CENTRE[0]) ** 2 + (y - GOAL_CENTRE[1]) ** 2)


def angle_to_goal(x: float, y: float) -> float:
    GOAL_Y_LEFT, GOAL_Y_RIGHT = 36.0, 44.0
    dx1, dy1 = GOAL_CENTRE[0] - x, GOAL_Y_LEFT - y
    dx2, dy2 = GOAL_CENTRE[0] - x, GOAL_Y_RIGHT - y
    cross = abs(dx1 * dy2 - dy1 * dx2)
    dot   = dx1 * dx2 + dy1 * dy2
    return math.atan2(cross, dot)


def safe_get(obj: Any, key: str, default=None) -> Any:
    """Safely get a value from a dict, or return default if not a dict."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return default


def safe_bool(val) -> bool:
    if isinstance(val, bool):
        return val
    if isinstance(val, float):
        return False
    return bool(val)


def safe_location(loc: Any) -> tuple[float | None, float | None]:
    """Extract x,y from a location value that may be a list, string, or NaN."""
    if isinstance(loc, list) and len(loc) >= 2:
        return loc[0], loc[1]
    if isinstance(loc, str):
        # statsbombpy sometimes returns "[x, y]" as a string
        try:
            import ast
            parsed = ast.literal_eval(loc)
            if isinstance(parsed, list) and len(parsed) >= 2:
                return parsed[0], parsed[1]
        except Exception:
            pass
    return None, None


# ── 360 Spatial Features ──────────────────────────────────────────────────────

def extract_360_shot_features(
    freeze_frames: list[dict],
    shot_x: float,
    shot_y: float,
) -> dict[str, Any]:
    if not freeze_frames:
        return {
            "defenders_in_2m_cone": None, "defenders_in_5m_cone": None,
            "gk_distance_to_goal_centre": None, "gk_x": None, "gk_y": None,
            "attackers_in_box": None, "defenders_in_box": None,
            "numerical_advantage": None, "nearest_defender_distance": None,
            "has_360": False,
        }

    goal_vec_x = GOAL_CENTRE[0] - shot_x
    goal_vec_y = GOAL_CENTRE[1] - shot_y
    goal_vec_len = math.sqrt(goal_vec_x**2 + goal_vec_y**2) or 1

    defenders_2m = defenders_5m = attackers_box = defenders_box = 0
    gk_x = gk_y = None
    nearest_def_dist = float("inf")

    for frame in freeze_frames:
        if not isinstance(frame, dict):
            continue
        loc = frame.get("location", [])
        fx, fy = safe_location(loc)
        if fx is None:
            continue

        teammate  = frame.get("teammate", False)
        position  = frame.get("position", {})
        is_keeper = safe_get(position, "name", "") == "Goalkeeper"
        dist      = math.sqrt((fx - shot_x)**2 + (fy - shot_y)**2)

        if not teammate:
            nearest_def_dist = min(nearest_def_dist, dist)
            px, py = fx - shot_x, fy - shot_y
            proj = (px * goal_vec_x + py * goal_vec_y) / (goal_vec_len**2)
            if 0 < proj < 1:
                perp = math.sqrt((px - proj*goal_vec_x)**2 + (py - proj*goal_vec_y)**2)
                if perp < 2: defenders_2m += 1
                if perp < 5: defenders_5m += 1
            if fx > 102 and 18 < fy < 62:
                if is_keeper:
                    gk_x, gk_y = fx, fy
                else:
                    defenders_box += 1
        else:
            if fx > 102 and 18 < fy < 62:
                attackers_box += 1

    gk_dist = math.sqrt((gk_x - GOAL_CENTRE[0])**2 + (gk_y - GOAL_CENTRE[1])**2) if gk_x else None

    return {
        "defenders_in_2m_cone":       defenders_2m,
        "defenders_in_5m_cone":       defenders_5m,
        "gk_distance_to_goal_centre": gk_dist,
        "gk_x": gk_x, "gk_y": gk_y,
        "attackers_in_box":           attackers_box,
        "defenders_in_box":           defenders_box,
        "numerical_advantage":        attackers_box - defenders_box,
        "nearest_defender_distance":  nearest_def_dist if nearest_def_dist < float("inf") else None,
        "has_360": True,
    }


def extract_360_pass_features(
    freeze_frames: list[dict],
    start_x: float, start_y: float,
    end_x: float,   end_y: float,
) -> dict[str, Any]:
    if not freeze_frames:
        return {
            "defenders_in_pass_lane": None, "defender_density_endpoint": None,
            "nearest_defender_distance": None, "nearest_opponent_at_end": None,
            "has_360": False,
        }

    pass_vec_x = end_x - start_x
    pass_vec_y = end_y - start_y
    pass_len   = math.sqrt(pass_vec_x**2 + pass_vec_y**2) or 1

    defenders_lane = defenders_end = 0
    nearest_lane = nearest_end = float("inf")

    for frame in freeze_frames:
        if not isinstance(frame, dict):
            continue
        loc = frame.get("location", [])
        fx, fy = safe_location(loc)
        if fx is None or frame.get("teammate", False):
            continue

        dist_end = math.sqrt((fx - end_x)**2 + (fy - end_y)**2)
        nearest_end = min(nearest_end, dist_end)
        if dist_end < 3:
            defenders_end += 1

        px, py = fx - start_x, fy - start_y
        proj = (px * pass_vec_x + py * pass_vec_y) / (pass_len**2)
        if 0 < proj < 1:
            perp = math.sqrt((px - proj*pass_vec_x)**2 + (py - proj*pass_vec_y)**2)
            nearest_lane = min(nearest_lane, perp)
            if perp < 1.5:
                defenders_lane += 1

    return {
        "defenders_in_pass_lane":    defenders_lane,
        "defender_density_endpoint": defenders_end,
        "nearest_defender_distance": nearest_lane if nearest_lane < float("inf") else None,
        "nearest_opponent_at_end":   nearest_end  if nearest_end  < float("inf") else None,
        "has_360": True,
    }


# ── Ingestion Functions ───────────────────────────────────────────────────────

def ingest_competitions() -> pd.DataFrame:
    logger.info("Fetching competitions...")
    comps = sb.competitions()
    cols  = ["competition_id","season_id","competition_name",
             "country_name","season_name","competition_gender"]
    comps = comps[[c for c in cols if c in comps.columns]]
    upsert_dataframe(comps, "competitions",
                     conflict_columns=["competition_id","season_id"])
    logger.info(f"Ingested {len(comps)} competition-seasons")
    return comps


def ingest_matches(competition_id: int, season_id: int) -> pd.DataFrame:
    logger.info(f"Fetching matches: comp={competition_id} season={season_id}")
    try:
        matches = sb.matches(competition_id=competition_id, season_id=season_id)
    except Exception as e:
        logger.warning(f"Could not fetch matches: {e}")
        return pd.DataFrame()

    if matches is None or (isinstance(matches, pd.DataFrame) and matches.empty):
        return pd.DataFrame()

    rows = []
    for _, m in matches.iterrows():
        # statsbombpy 1.17 returns team names as strings directly
        home_team = m.get("home_team", "")
        away_team = m.get("away_team", "")
        if isinstance(home_team, dict):
            home_team_name = home_team.get("home_team_name", "")
            home_team_id   = home_team.get("home_team_id")
        else:
            home_team_name = str(home_team)
            home_team_id   = m.get("home_team_id")

        if isinstance(away_team, dict):
            away_team_name = away_team.get("away_team_name", "")
            away_team_id   = away_team.get("away_team_id")
        else:
            away_team_name = str(away_team)
            away_team_id   = m.get("away_team_id")

        stadium  = m.get("stadium", {})
        referee  = m.get("referee", {})

        rows.append({
            "match_id":       int(m["match_id"]),
            "competition_id": competition_id,
            "season_id":      season_id,
            "match_date":     m.get("match_date"),
            "kick_off":       m.get("kick_off"),
            "home_team_id":   home_team_id,
            "home_team_name": home_team_name,
            "away_team_id":   away_team_id,
            "away_team_name": away_team_name,
            "home_score":     m.get("home_score"),
            "away_score":     m.get("away_score"),
            "match_week":     m.get("match_week"),
            "stadium_name":   safe_get(stadium, "name") if isinstance(stadium, dict) else None,
            "referee_name":   safe_get(referee, "name") if isinstance(referee, dict) else None,
        })

    df = pd.DataFrame(rows)
    upsert_dataframe(df, "matches", conflict_columns=["match_id"])
    return df


def _parse_nested(val: Any, key: str, default=None):
    """Handle both dict and flat string values from statsbombpy."""
    if isinstance(val, dict):
        return val.get(key, default)
    if val is not None and not (isinstance(val, float) and math.isnan(val)):
        return str(val)
    return default


def ingest_events_for_match(match_id: int) -> tuple[int, int]:
    logger.debug(f"Processing match {match_id}")

    try:
        # flatten_attrs=True gives us a flat DataFrame (statsbombpy 1.17 default)
        events = sb.events(match_id=match_id)
    except Exception as e:
        logger.error(f"Failed to fetch events for match {match_id}: {e}")
        return 0, 0

    if events is None or events.empty:
        return 0, 0

    # Try to load 360 freeze frames
    freeze_frame_map: dict[str, list] = {}
    try:
        frames_360 = sb.frames(match_id=match_id)
        if frames_360 is not None and not frames_360.empty:
            id_col = next((c for c in ["id","event_uuid","event_id"]
                           if c in frames_360.columns), None)
            if id_col:
                for _, row in frames_360.iterrows():
                    ff = row.get("freeze_frame", [])
                    if isinstance(ff, list):
                        freeze_frame_map[str(row[id_col])] = ff
    except Exception:
        pass

    shot_rows = []
    pass_rows = []

    for _, ev in events.iterrows():
        # statsbombpy 1.17 flattens columns: type becomes "type" string directly
        type_val  = ev.get("type", "")
        type_name = type_val if isinstance(type_val, str) else safe_get(type_val, "name", "")

        event_id      = str(ev.get("id", ""))
        freeze_frames = freeze_frame_map.get(event_id, [])

        # Location — may be list, string, or NaN
        loc_raw = ev.get("location")
        loc_x, loc_y = safe_location(loc_raw)

        # Player / team — statsbombpy ≥1.17 puts player name directly in "player"
        # column as a string; older format uses {"id": ..., "name": ...} dict.
        _player_raw = ev.get("player")
        player_id   = ev.get("player_id") or safe_get(_player_raw, "id")
        player_name = (ev.get("player_name")
                       or (_player_raw if isinstance(_player_raw, str) else None)
                       or safe_get(_player_raw, "name"))
        team_id     = ev.get("team_id") or (safe_get(ev.get("team"), "id"))

        try:
            player_id = int(player_id) if player_id and not (isinstance(player_id, float) and math.isnan(player_id)) else None
            team_id   = int(team_id)   if team_id   and not (isinstance(team_id,   float) and math.isnan(team_id))   else None
        except (ValueError, TypeError):
            player_id = team_id = None

        under_pressure = ev.get("under_pressure", False)
        if isinstance(under_pressure, float):
            under_pressure = False

        # ── SHOTS ──────────────────────────────────────────────────────────
        if type_name == "Shot":
            # Flattened columns: shot_outcome_name, shot_technique_name, etc.
            outcome_name  = ev.get("shot_outcome_name") or _parse_nested(ev.get("shot_outcome"), "name")
            technique_name = ev.get("shot_technique_name") or _parse_nested(ev.get("shot_technique"), "name")
            body_part_name = ev.get("shot_body_part_name") or _parse_nested(ev.get("shot_body_part"), "name")
            play_pattern   = ev.get("play_pattern_name") or _parse_nested(ev.get("play_pattern"), "name")

            row = {
                "event_id":        event_id,
                "match_id":        int(match_id),
                "player_id":       player_id,
                "player_name":     player_name,
                "team_id":         team_id,
                "minute":          int(ev.get("minute", 0)),
                "period":          int(ev.get("period", 0)),
                "location_x":      loc_x,
                "location_y":      loc_y,
                "distance_to_goal": distance_to_goal(loc_x, loc_y) if loc_x and loc_y else None,
                "angle_to_goal":   angle_to_goal(loc_x, loc_y)    if loc_x and loc_y else None,
                "technique_name":  technique_name,
                "body_part_name":  body_part_name,
                "play_pattern_name": play_pattern,
                "first_time":      safe_bool(ev.get("shot_first_time")),
                "follows_dribble": safe_bool(ev.get("shot_follows_dribble")),
                "open_goal":       safe_bool(ev.get("shot_open_goal")),
                "deflected":       safe_bool(ev.get("shot_deflected")),
                "under_pressure":  safe_bool(under_pressure),
                "outcome_name":    outcome_name,
                "is_goal":         outcome_name == "Goal",
            }
            spatial = extract_360_shot_features(freeze_frames, loc_x or 0, loc_y or 0)
            row.update(spatial)
            shot_rows.append(row)

        # ── PASSES ─────────────────────────────────────────────────────────
        elif type_name == "Pass":
            end_loc_raw = ev.get("pass_end_location")
            end_x, end_y = safe_location(end_loc_raw)

            outcome_name   = ev.get("pass_outcome_name") or _parse_nested(ev.get("pass_outcome"), "name")
            height_name    = ev.get("pass_height_name")  or _parse_nested(ev.get("pass_height"),  "name")
            technique_name = ev.get("pass_technique_name") or _parse_nested(ev.get("pass_technique"), "name")
            body_part_name = ev.get("pass_body_part_name") or _parse_nested(ev.get("pass_body_part"), "name")
            play_pattern   = ev.get("play_pattern_name")   or _parse_nested(ev.get("play_pattern"), "name")

            row = {
                "event_id":        event_id,
                "match_id":        int(match_id),
                "player_id":       player_id,
                "player_name":     player_name,
                "team_id":         team_id,
                "minute":          int(ev.get("minute", 0)),
                "period":          int(ev.get("period", 0)),
                "start_x":         loc_x,
                "start_y":         loc_y,
                "end_x":           end_x,
                "end_y":           end_y,
                "length":          ev.get("pass_length"),
                "angle":           ev.get("pass_angle"),
                "height_name":     height_name,
                "technique_name":  technique_name,
                "body_part_name":  body_part_name,
                "play_pattern_name": play_pattern,
                "is_switch":       safe_bool(ev.get("pass_switch")),
                "is_through_ball": safe_bool(ev.get("pass_through_ball")),
                "is_cut_back":     safe_bool(ev.get("pass_cut_back")),
                "is_cross":        safe_bool(ev.get("pass_cross")),
                "goal_assist":     safe_bool(ev.get("pass_goal_assist")),
                "shot_assist":     safe_bool(ev.get("pass_shot_assist")),
                "under_pressure":  safe_bool(under_pressure),
                "outcome_name":    outcome_name,
                "is_complete":     outcome_name is None or (isinstance(outcome_name, float) and math.isnan(outcome_name)),
            }

            if loc_x and loc_y and end_x and end_y:
                spatial = extract_360_pass_features(freeze_frames, loc_x, loc_y, end_x, end_y)
            else:
                spatial = {"defenders_in_pass_lane": None, "defender_density_endpoint": None,
                           "nearest_defender_distance": None, "nearest_opponent_at_end": None,
                           "has_360": False}
            row.update(spatial)
            pass_rows.append(row)

    shots_count = passes_count = 0
    if shot_rows:
        shots_count = upsert_dataframe(
            pd.DataFrame(shot_rows), "shot_features", conflict_columns=["event_id"]
        )
    if pass_rows:
        passes_count = upsert_dataframe(
            pd.DataFrame(pass_rows), "pass_features", conflict_columns=["event_id"]
        )

    return shots_count, passes_count


# ── Main Entry Point ──────────────────────────────────────────────────────────

def run_full_ingest(
    competition_ids: list[int] | None = None,
    max_matches: int | None = None,
) -> dict:
    logger.info("═══ ScoutIQ Ingestion Pipeline Starting ═══")
    init_schema()

    stats = {"competitions": 0, "matches": 0, "shots": 0, "passes": 0, "errors": 0}

    try:
        comps_df = ingest_competitions()
        stats["competitions"] = len(comps_df)

        if competition_ids:
            comps_df = comps_df[comps_df["competition_id"].isin(competition_ids)]

        match_count = 0
        for _, comp in comps_df.iterrows():
            cid, sid = int(comp["competition_id"]), int(comp["season_id"])
            matches_df = ingest_matches(cid, sid)
            if matches_df.empty:
                continue

            for _, match_row in matches_df.iterrows():
                if max_matches and match_count >= max_matches:
                    logger.info(f"Reached max_matches={max_matches}, stopping")
                    break
                try:
                    s, p = ingest_events_for_match(int(match_row["match_id"]))
                    stats["shots"]  += s
                    stats["passes"] += p
                    match_count     += 1
                except Exception as e:
                    logger.error(f"Error processing match {match_row['match_id']}: {e}")
                    stats["errors"] += 1

        stats["matches"] = match_count
        log_pipeline_run("full_ingest", "success",
                         rows_processed=stats["shots"] + stats["passes"])

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        log_pipeline_run("full_ingest", "failed", error_message=str(e))
        raise

    logger.info(f"═══ Ingestion Complete: {stats} ═══")
    return stats


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--competitions", nargs="+", type=int)
    parser.add_argument("--max-matches",  type=int)
    args = parser.parse_args()

    result = run_full_ingest(
        competition_ids=args.competitions,
        max_matches=args.max_matches,
    )
    print(f"\nIngestion complete: {result}")