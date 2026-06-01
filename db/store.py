"""
db/store.py
Database connection pool and read/write helpers for the ScoutIQ feature store.
"""
import os
from contextlib import contextmanager

import pandas as pd
from dotenv import load_dotenv
from loguru import logger
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool

load_dotenv()


# ── Connection ────────────────────────────────────────────────────────────────

def _build_url() -> str:
    return (
        f"postgresql+psycopg2://"
        f"{os.getenv('POSTGRES_USER', 'scoutiq_user')}:"
        f"{os.getenv('POSTGRES_PASSWORD', 'scoutiq_pass')}@"
        f"{os.getenv('POSTGRES_HOST', 'localhost')}:"
        f"{os.getenv('POSTGRES_PORT', '5432')}/"
        f"{os.getenv('POSTGRES_DB', 'scoutiq')}"
    )


_engine = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = create_engine(
            _build_url(),
            poolclass=QueuePool,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,   # auto-reconnect on stale connections
        )
        logger.info("Database engine initialised")
    return _engine


@contextmanager
def get_connection():
    """Context manager for a raw SQLAlchemy connection."""
    engine = get_engine()
    with engine.connect() as conn:
        yield conn


# ── Schema Initialization ─────────────────────────────────────────────────────

def init_schema(schema_path: str = "db/schema.sql") -> None:
    """Run schema.sql to create all tables if they don't exist."""
    with open(schema_path) as f:
        sql = f.read()
    with get_connection() as conn:
        conn.execute(text(sql))
        conn.commit()
    logger.info("Schema initialized successfully")


# ── Generic Upsert ────────────────────────────────────────────────────────────

def upsert_dataframe(
    df: pd.DataFrame,
    table: str,
    conflict_columns: list[str],
    chunk_size: int = 1000,
) -> int:
    """
    Upsert a DataFrame into a PostgreSQL table using INSERT ... ON CONFLICT DO UPDATE.
    Returns total rows upserted.
    """
    if df.empty:
        logger.warning(f"Empty DataFrame, skipping upsert to {table}")
        return 0

    total = 0

    for i in range(0, len(df), chunk_size):
        chunk = df.iloc[i : i + chunk_size]
        cols = list(chunk.columns)
        conflict_str = ", ".join(conflict_columns)
        update_str = ", ".join(
            f"{c} = EXCLUDED.{c}"
            for c in cols
            if c not in conflict_columns
        )
        placeholders = ", ".join(f":{c}" for c in cols)
        col_str = ", ".join(cols)

        sql = text(f"""
            INSERT INTO {table} ({col_str})
            VALUES ({placeholders})
            ON CONFLICT ({conflict_str})
            DO UPDATE SET {update_str}
        """)

        with get_connection() as conn:
            conn.execute(sql, chunk.to_dict(orient="records"))
            conn.commit()

        total += len(chunk)

    logger.info(f"Upserted {total} rows into {table}")
    return total



# ── Read Helpers ──────────────────────────────────────────────────────────────

def read_table(table: str, where: str | None = None) -> pd.DataFrame:
    """Read an entire table (or filtered subset) into a DataFrame."""
    query = f"SELECT * FROM {table}"
    if where:
        query += f" WHERE {where}"
    with get_connection() as conn:
        return pd.read_sql(query, conn)


def get_shot_features(min_has_360: bool = False) -> pd.DataFrame:
    """Load shot features for xG model training."""
    where = "has_360 = TRUE" if min_has_360 else None
    df = read_table("shot_features", where)
    logger.info(f"Loaded {len(df)} shots (360_only={min_has_360})")
    return df


def get_pass_features(min_has_360: bool = False) -> pd.DataFrame:
    """Load pass features for pass success model training."""
    where = "has_360 = TRUE" if min_has_360 else None
    df = read_table("pass_features", where)
    logger.info(f"Loaded {len(df)} passes (360_only={min_has_360})")
    return df


def get_player_per90(position_group: str | None = None) -> pd.DataFrame:
    """Load per-90 player features for similarity model."""
    with get_connection() as conn:
        if position_group:
            query = text("SELECT * FROM player_per90_features WHERE position_group = :pg")
            df = pd.read_sql(query, conn, params={"pg": position_group})
        else:
            df = pd.read_sql("SELECT * FROM player_per90_features", conn)
    logger.info(f"Loaded {len(df)} player-season records")
    return df


def get_player_history(player_id: int, n: int = 10) -> pd.DataFrame:
    """Load last N matches for a player (used by forecast API)."""
    query = text("""
        SELECT * FROM player_per90_features
        WHERE player_id = :player_id
        ORDER BY computed_at DESC
        LIMIT :n
    """)
    with get_connection() as conn:
        return pd.read_sql(query, conn, params={"player_id": player_id, "n": n})


# ── Pipeline Audit ────────────────────────────────────────────────────────────

def log_pipeline_run(
    flow_name: str,
    status: str,
    rows_processed: int = 0,
    error_message: str | None = None,
) -> None:
    sql = text("""
        INSERT INTO pipeline_runs (flow_name, status, rows_processed, error_message, finished_at)
        VALUES (:flow_name, :status, :rows_processed, :error_message, NOW())
    """)
    with get_connection() as conn:
        conn.execute(sql, {
            "flow_name": flow_name,
            "status": status,
            "rows_processed": rows_processed,
            "error_message": error_message,
        })
        conn.commit()
