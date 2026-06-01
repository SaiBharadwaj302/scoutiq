-- ScoutIQ Feature Store Schema
-- Run once to initialize the database

-- ─────────────────────────────────────────
-- COMPETITIONS & MATCHES  (reference tables)
-- ─────────────────────────────────────────

CREATE TABLE IF NOT EXISTS competitions (
    competition_id      INTEGER NOT NULL,
    season_id           INTEGER NOT NULL,
    competition_name    TEXT NOT NULL,
    country_name        TEXT,
    season_name         TEXT NOT NULL,
    competition_gender  TEXT,
    ingested_at         TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (competition_id, season_id)
);

CREATE TABLE IF NOT EXISTS matches (
    match_id            INTEGER PRIMARY KEY,
    competition_id      INTEGER NOT NULL,
    season_id           INTEGER NOT NULL,
    match_date          DATE,
    kick_off            TIME,
    home_team_id        INTEGER,
    home_team_name      TEXT,
    away_team_id        INTEGER,
    away_team_name      TEXT,
    home_score          INTEGER,
    away_score          INTEGER,
    match_week          INTEGER,
    stadium_name        TEXT,
    referee_name        TEXT,
    ingested_at         TIMESTAMPTZ DEFAULT NOW(),
    FOREIGN KEY (competition_id, season_id)
        REFERENCES competitions(competition_id, season_id)
);

-- ─────────────────────────────────────────
-- RAW EVENTS  (one row per StatsBomb event)
-- ─────────────────────────────────────────

CREATE TABLE IF NOT EXISTS events (
    event_id            TEXT PRIMARY KEY,   -- StatsBomb UUID
    match_id            INTEGER NOT NULL REFERENCES matches(match_id),
    index               INTEGER,            -- event order within match
    period              INTEGER,            -- 1=first half, 2=second, etc.
    timestamp           INTERVAL,           -- time within period
    minute              INTEGER,
    second              INTEGER,
    type_id             INTEGER,
    type_name           TEXT,               -- 'Shot', 'Pass', 'Pressure', etc.
    possession          INTEGER,
    possession_team_id  INTEGER,
    possession_team_name TEXT,
    play_pattern_name   TEXT,
    team_id             INTEGER,
    team_name           TEXT,
    player_id           INTEGER,
    player_name         TEXT,
    position_name       TEXT,
    location_x          FLOAT,
    location_y          FLOAT,
    duration            FLOAT,
    under_pressure      BOOLEAN,
    off_camera          BOOLEAN,
    out                 BOOLEAN,
    -- raw JSON for type-specific fields (shot, pass, carry etc.)
    -- we parse these into dedicated feature tables below
    extra               JSONB,
    ingested_at         TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_events_match   ON events(match_id);
CREATE INDEX IF NOT EXISTS idx_events_player  ON events(player_id);
CREATE INDEX IF NOT EXISTS idx_events_type    ON events(type_name);

-- ─────────────────────────────────────────
-- SHOT FEATURES  (Model 1 — xG)
-- ─────────────────────────────────────────

CREATE TABLE IF NOT EXISTS shot_features (
    event_id                    TEXT PRIMARY KEY,
    match_id                    INTEGER NOT NULL,
    player_id                   INTEGER NOT NULL,
    player_name                 TEXT,
    team_id                     INTEGER,
    minute                      INTEGER,
    period                      INTEGER,

    -- Location
    location_x                  FLOAT,
    location_y                  FLOAT,
    distance_to_goal            FLOAT,   -- Euclidean distance to goal centre
    angle_to_goal               FLOAT,   -- radians

    -- Shot type
    technique_name              TEXT,    -- 'Normal', 'Half Volley', 'Volley', etc.
    body_part_name              TEXT,    -- 'Head', 'Right Foot', 'Left Foot'
    play_pattern_name           TEXT,
    first_time                  BOOLEAN,
    follows_dribble             BOOLEAN,
    open_goal                   BOOLEAN,
    deflected                   BOOLEAN,
    under_pressure              BOOLEAN,

    -- Outcome
    outcome_name                TEXT,    -- 'Goal', 'Saved', 'Off T', 'Blocked', etc.
    is_goal                     BOOLEAN, -- TARGET for xG model

    -- 360 spatial features (populated separately when 360 data available)
    defenders_in_2m_cone        INTEGER,
    defenders_in_5m_cone        INTEGER,
    gk_distance_to_goal_centre  FLOAT,
    gk_x                        FLOAT,
    gk_y                        FLOAT,
    attackers_in_box            INTEGER,
    defenders_in_box            INTEGER,
    numerical_advantage         INTEGER, -- attackers - defenders in shot zone
    nearest_defender_distance   FLOAT,
    has_360                     BOOLEAN DEFAULT FALSE,

    ingested_at                 TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_shot_features_player ON shot_features(player_id);
CREATE INDEX IF NOT EXISTS idx_shot_features_match  ON shot_features(match_id);

-- ─────────────────────────────────────────
-- PASS FEATURES  (Model 2 — Pass Success)
-- ─────────────────────────────────────────

CREATE TABLE IF NOT EXISTS pass_features (
    event_id                    TEXT PRIMARY KEY,
    match_id                    INTEGER NOT NULL,
    player_id                   INTEGER NOT NULL,
    player_name                 TEXT,
    team_id                     INTEGER,
    minute                      INTEGER,
    period                      INTEGER,

    -- Pass geometry
    start_x                     FLOAT,
    start_y                     FLOAT,
    end_x                       FLOAT,
    end_y                       FLOAT,
    length                      FLOAT,
    angle                       FLOAT,   -- radians from horizontal

    -- Pass type
    height_name                 TEXT,    -- 'Ground Pass', 'Low Pass', 'High Pass'
    technique_name              TEXT,
    body_part_name              TEXT,
    play_pattern_name           TEXT,
    is_switch                   BOOLEAN,
    is_through_ball             BOOLEAN,
    is_cut_back                 BOOLEAN,
    is_cross                    BOOLEAN,
    goal_assist                 BOOLEAN,
    shot_assist                 BOOLEAN,
    under_pressure              BOOLEAN,

    -- Outcome
    outcome_name                TEXT,    -- NULL = complete, else 'Incomplete', 'Out' etc.
    is_complete                 BOOLEAN, -- TARGET for pass success model

    -- 360 spatial features
    defenders_in_pass_lane      INTEGER, -- defenders intersecting straight line to endpoint
    defender_density_endpoint   FLOAT,   -- defenders within 3m of endpoint
    nearest_defender_distance   FLOAT,   -- distance to closest defender along lane
    nearest_opponent_at_end     FLOAT,   -- closest opponent to recipient location
    has_360                     BOOLEAN DEFAULT FALSE,

    ingested_at                 TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_pass_features_player ON pass_features(player_id);
CREATE INDEX IF NOT EXISTS idx_pass_features_match  ON pass_features(match_id);

-- ─────────────────────────────────────────
-- PLAYER PER-90 FEATURES  (Model 3 — Similarity)
-- ─────────────────────────────────────────

CREATE TABLE IF NOT EXISTS player_per90_features (
    player_id               INTEGER NOT NULL,
    competition_id          INTEGER NOT NULL,
    season_id               INTEGER NOT NULL,
    player_name             TEXT,
    team_name               TEXT,
    position_group          TEXT,   -- 'GK', 'CB', 'FB', 'CM', 'AM', 'W', 'ST'
    minutes_played          FLOAT,
    matches_played          INTEGER,

    -- Attacking
    goals_per90             FLOAT,
    assists_per90           FLOAT,
    shots_per90             FLOAT,
    shots_on_target_per90   FLOAT,
    xg_per90                FLOAT,
    xg_overperformance      FLOAT,   -- goals - xG (positive = clinical)

    -- Passing
    passes_per90            FLOAT,
    pass_completion_pct     FLOAT,
    progressive_passes_per90 FLOAT,
    key_passes_per90        FLOAT,
    switches_per90          FLOAT,

    -- Defending
    pressures_per90         FLOAT,
    pressure_success_pct    FLOAT,
    tackles_per90           FLOAT,
    interceptions_per90     FLOAT,
    blocks_per90            FLOAT,

    -- Carrying
    carries_per90           FLOAT,
    progressive_carries_per90 FLOAT,
    carries_into_final_third_per90 FLOAT,
    dribbles_completed_per90 FLOAT,

    -- Embedding (populated after PCA)
    embedding               FLOAT[],   -- 32-dim PCA vector

    computed_at             TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (player_id, competition_id, season_id)
);

CREATE INDEX IF NOT EXISTS idx_per90_player   ON player_per90_features(player_id);
CREATE INDEX IF NOT EXISTS idx_per90_position ON player_per90_features(position_group);

-- ─────────────────────────────────────────
-- PLAYERS & APPEARANCES  (from lineups)
-- ─────────────────────────────────────────

CREATE TABLE IF NOT EXISTS players (
    player_id        INTEGER PRIMARY KEY,
    player_name      TEXT NOT NULL,
    player_nickname  TEXT,
    position_name    TEXT,
    position_group   TEXT,   -- 'GK', 'CB', 'FB', 'CM', 'AM', 'W', 'ST'
    country_name     TEXT
);

CREATE INDEX IF NOT EXISTS idx_players_name ON players(player_name);

CREATE TABLE IF NOT EXISTS player_appearances (
    player_id        INTEGER NOT NULL,
    match_id         INTEGER NOT NULL,
    competition_id   INTEGER,
    season_id        INTEGER,
    competition_name TEXT,
    team_name        TEXT,
    PRIMARY KEY (player_id, match_id)
);

CREATE INDEX IF NOT EXISTS idx_appearances_player ON player_appearances(player_id);

-- ─────────────────────────────────────────
-- PIPELINE AUDIT LOG
-- ─────────────────────────────────────────

CREATE TABLE IF NOT EXISTS pipeline_runs (
    run_id          SERIAL PRIMARY KEY,
    flow_name       TEXT NOT NULL,
    status          TEXT NOT NULL,   -- 'success', 'failed', 'running'
    rows_processed  INTEGER,
    error_message   TEXT,
    started_at      TIMESTAMPTZ DEFAULT NOW(),
    finished_at     TIMESTAMPTZ
);
