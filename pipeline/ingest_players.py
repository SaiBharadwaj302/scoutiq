"""
pipeline/ingest_players.py

Pulls player names, positions, and league data from StatsBomb lineups.
Populates the players and player_appearances tables, then back-fills
player_name in shot_features and pass_features.

Run after pipeline/ingest.py:
    python -m pipeline.ingest_players
    python -m pipeline.ingest_players --max-matches 50
"""
import math
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from loguru import logger
from sqlalchemy import text
from statsbombpy import sb

from db.store import get_connection, upsert_dataframe, log_pipeline_run

load_dotenv()

# Map StatsBomb position names to compact position groups
POSITION_GROUPS: dict[str, str] = {
    "Goalkeeper":               "GK",
    "Right Back":               "FB",
    "Left Back":                "FB",
    "Right Wing Back":          "FB",
    "Left Wing Back":           "FB",
    "Right Center Back":        "CB",
    "Left Center Back":         "CB",
    "Center Back":              "CB",
    "Right Midfield":           "CM",
    "Left Midfield":            "CM",
    "Right Center Midfield":    "CM",
    "Left Center Midfield":     "CM",
    "Center Midfield":          "CM",
    "Right Defensive Midfield": "CM",
    "Left Defensive Midfield":  "CM",
    "Center Defensive Midfield": "CM",
    "Right Attacking Midfield": "AM",
    "Left Attacking Midfield":  "AM",
    "Center Attacking Midfield": "AM",
    "Right Wing":               "W",
    "Left Wing":                "W",
    "Right Center Forward":     "ST",
    "Left Center Forward":      "ST",
    "Center Forward":           "ST",
    "Secondary Striker":        "ST",
}


def _ensure_tables() -> None:
    """Create players and player_appearances tables if they don't exist."""
    ddl = text("""
        CREATE TABLE IF NOT EXISTS players (
            player_id        INTEGER PRIMARY KEY,
            player_name      TEXT NOT NULL,
            player_nickname  TEXT,
            position_name    TEXT,
            position_group   TEXT,
            country_name     TEXT
        );

        CREATE TABLE IF NOT EXISTS player_appearances (
            player_id        INTEGER NOT NULL,
            match_id         INTEGER NOT NULL,
            competition_id   INTEGER,
            season_id        INTEGER,
            competition_name TEXT,
            team_name        TEXT,
            PRIMARY KEY (player_id, match_id)
        );
    """)
    with get_connection() as conn:
        conn.execute(ddl)
        conn.commit()
    logger.info("players and player_appearances tables ensured")


def _get_match_meta(max_matches: int | None = None) -> tuple[list[int], dict, dict]:
    """Return match_ids in DB with competition/season metadata."""
    with get_connection() as conn:
        matches_df = pd.read_sql(
            "SELECT match_id, competition_id, season_id FROM matches", conn
        )
        comps_df = pd.read_sql(
            "SELECT competition_id, season_id, competition_name FROM competitions", conn
        )

    match_ids = matches_df["match_id"].astype(int).tolist()
    if max_matches:
        match_ids = match_ids[:max_matches]

    match_meta: dict[int, dict] = {
        int(r["match_id"]): {
            "competition_id": int(r["competition_id"]),
            "season_id":      int(r["season_id"]),
        }
        for _, r in matches_df.iterrows()
    }

    comp_names: dict[tuple, str] = {
        (int(r["competition_id"]), int(r["season_id"])): r["competition_name"]
        for _, r in comps_df.iterrows()
    }

    return match_ids, match_meta, comp_names


def _safe_str(val: Any, default: str = "") -> str:
    """Return val as a clean string, or default if it's NaN / None."""
    if val is None:
        return default
    if isinstance(val, float) and math.isnan(val):
        return default
    return str(val).strip()


def _parse_lineup_row(
    player_row: pd.Series,
    match_id: int,
    team_name: str,
    competition_id: int,
    season_id: int,
    competition_name: str,
) -> tuple[dict | None, dict | None]:
    """Parse a single lineup row into (player_dict, appearance_dict)."""
    raw_pid = player_row.get("player_id")
    if raw_pid is None or (isinstance(raw_pid, float) and math.isnan(raw_pid)):
        return None, None

    try:
        pid = int(raw_pid)
    except (ValueError, TypeError):
        return None, None

    pname = _safe_str(player_row.get("player_name"))
    if not pname:
        return None, None

    nickname = _safe_str(player_row.get("player_nickname")) or None

    # Position — statsbombpy returns a list of dicts: [{"position_id": 1, "position": "Goalkeeper", ...}]
    positions = player_row.get("positions", [])
    position_name = ""
    if isinstance(positions, list) and positions:
        first = positions[0]
        if isinstance(first, dict):
            position_name = _safe_str(first.get("position", ""))
        else:
            position_name = _safe_str(first)
    position_group = POSITION_GROUPS.get(position_name, "Unknown")

    # Country
    country = player_row.get("country", {})
    if isinstance(country, dict):
        country_name = _safe_str(country.get("name", ""))
    else:
        country_name = _safe_str(country)

    player_dict = {
        "player_id":       pid,
        "player_name":     pname,
        "player_nickname": nickname,
        "position_name":   position_name,
        "position_group":  position_group,
        "country_name":    country_name,
    }

    appearance_dict = {
        "player_id":       pid,
        "match_id":        match_id,
        "competition_id":  competition_id,
        "season_id":       season_id,
        "competition_name": competition_name,
        "team_name":       team_name,
    }

    return player_dict, appearance_dict


def ingest_players(max_matches: int | None = None) -> dict:
    """
    Pull player info from StatsBomb lineups for all ingested matches.

    1. Reads all match_ids from the matches table.
    2. Fetches lineups from StatsBomb for each match.
    3. Upserts into players and player_appearances.
    4. Back-fills player_name in shot_features and pass_features.
    """
    logger.info("═══ Player Ingestion Starting ═══")
    _ensure_tables()

    match_ids, match_meta, comp_names = _get_match_meta(max_matches)
    if not match_ids:
        logger.warning("No matches found in DB — run pipeline/ingest.py first")
        return {"players": 0, "appearances": 0, "errors": 0}

    logger.info(f"Processing {len(match_ids)} matches for player data")

    player_rows: dict[int, dict] = {}
    appearance_rows: list[dict] = []
    errors = 0

    for match_id in match_ids:
        try:
            lineups = sb.lineups(match_id=match_id)
            if lineups is None:
                continue

            meta = match_meta.get(match_id, {})
            cid  = meta.get("competition_id", 0)
            sid  = meta.get("season_id", 0)
            comp_name = comp_names.get((cid, sid), "")

            # lineups is a dict: {team_name: DataFrame}
            for team_name, team_df in lineups.items():
                if team_df is None or (isinstance(team_df, pd.DataFrame) and team_df.empty):
                    continue
                if not isinstance(team_df, pd.DataFrame):
                    continue

                for _, prow in team_df.iterrows():
                    p_dict, a_dict = _parse_lineup_row(
                        prow, match_id, str(team_name), cid, sid, comp_name
                    )
                    if p_dict is None:
                        continue

                    pid = p_dict["player_id"]
                    # Keep existing record only if it lacks position info
                    if pid not in player_rows or (
                        not player_rows[pid]["position_name"]
                        and p_dict["position_name"]
                    ):
                        player_rows[pid] = p_dict

                    appearance_rows.append(a_dict)

        except Exception as e:
            logger.error(f"Failed to process lineups for match {match_id}: {e}")
            errors += 1

    # Upsert players
    n_players = 0
    if player_rows:
        players_df = pd.DataFrame(list(player_rows.values()))
        n_players = upsert_dataframe(
            players_df, "players", conflict_columns=["player_id"]
        )
        logger.info(f"Upserted {n_players} players")
    else:
        logger.warning("No player records collected — check lineup API response")

    # Upsert appearances (deduplicate first)
    n_appearances = 0
    if appearance_rows:
        appearances_df = pd.DataFrame(appearance_rows).drop_duplicates(
            subset=["player_id", "match_id"]
        )
        n_appearances = upsert_dataframe(
            appearances_df, "player_appearances",
            conflict_columns=["player_id", "match_id"]
        )
        logger.info(f"Upserted {n_appearances} player appearances")

    # Back-fill player_name in shot_features and pass_features
    logger.info("Back-filling player_name in shot_features and pass_features...")
    backfill_sql = text("""
        UPDATE shot_features sf
           SET player_name = p.player_name
          FROM players p
         WHERE sf.player_id = p.player_id
           AND (sf.player_name IS NULL
                OR sf.player_name = ''
                OR sf.player_name = '0');

        UPDATE pass_features pf
           SET player_name = p.player_name
          FROM players p
         WHERE pf.player_id = p.player_id
           AND (pf.player_name IS NULL
                OR pf.player_name = ''
                OR pf.player_name = '0');
    """)
    with get_connection() as conn:
        conn.execute(backfill_sql)
        conn.commit()
    logger.info("player_name back-fill complete")

    try:
        log_pipeline_run(
            "ingest_players", "success",
            rows_processed=n_players + n_appearances,
        )
    except Exception as e:
        logger.warning(f"Could not log pipeline run: {e}")

    logger.info(
        f"═══ Player Ingestion Complete: {n_players} players, "
        f"{n_appearances} appearances, {errors} errors ═══"
    )
    return {"players": n_players, "appearances": n_appearances, "errors": errors}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest player data from StatsBomb lineups")
    parser.add_argument(
        "--max-matches", type=int,
        help="Limit to first N matches (useful for testing)"
    )
    args = parser.parse_args()

    result = ingest_players(max_matches=args.max_matches)
    print(f"\nPlayer ingestion complete: {result}")
