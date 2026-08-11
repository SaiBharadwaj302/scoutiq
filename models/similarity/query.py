"""
models/similarity/query.py

Runtime similarity query — loads embeddings from PostgreSQL
and returns top-N similar players via cosine similarity.

Used by the FastAPI endpoint at /v1/players/similar/{player_id}
"""
import numpy as np
import pandas as pd
from loguru import logger

from db.store import get_connection


def load_embeddings() -> tuple[np.ndarray, list[int], list[str]]:
    """
    Load all player embeddings from PostgreSQL.
    Returns (embeddings_matrix, player_ids, player_names).
    """
    query = """
        SELECT player_id, player_name, embedding
        FROM player_per90_features
        WHERE embedding IS NOT NULL
        ORDER BY player_id
    """
    with get_connection() as conn:
        df = pd.read_sql(query, conn)

    if df.empty:
        raise ValueError("No embeddings found — run build_embeddings.py first")

    # Parse embedding arrays
    embeddings = np.array([
        e if isinstance(e, list) else list(e)
        for e in df["embedding"]
    ], dtype=float)

    player_ids   = df["player_id"].astype(int).tolist()
    player_names = df["player_name"].fillna("Unknown").tolist()

    logger.debug(f"Loaded {len(player_ids)} player embeddings")
    return embeddings, player_ids, player_names


def find_similar(
    player_id: int,
    top_n: int = 5,
    embeddings: np.ndarray | None = None,
    player_ids: list[int] | None = None,
    player_names: list[str] | None = None,
) -> list[dict]:
    """
    Find top-N most similar players to a given player_id.

    Args:
        player_id:    ID of the query player
        top_n:        number of results to return
        embeddings:   pre-loaded embedding matrix (pass to avoid DB hit)
        player_ids:   pre-loaded player ID list
        player_names: pre-loaded player name list

    Returns:
        list of dicts with player_id, player_name, similarity score
    """
    if embeddings is None or player_ids is None or player_names is None:
        embeddings, player_ids, player_names = load_embeddings()

    if player_id not in player_ids:
        raise ValueError(f"Player {player_id} not found in embedding store")

    idx = player_ids.index(player_id)
    query_vec = embeddings[idx]

    # Cosine similarity against all players
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    normed        = embeddings / norms
    query_normed  = query_vec / (np.linalg.norm(query_vec) or 1)
    similarities  = normed @ query_normed

    # Exclude self
    similarities[idx] = -1

    top_indices = np.argsort(similarities)[::-1][:top_n]

    return [
        {
            "player_id":   player_ids[i],
            "player_name": player_names[i],
            "similarity":  round(float(similarities[i]), 4),
        }
        for i in top_indices
    ]


def get_player_stats(player_id: int) -> dict | None:
    """Fetch per-90 stats for a single player."""
    from sqlalchemy import text
    query = text("SELECT * FROM player_per90_features WHERE player_id = :pid LIMIT 1")
    with get_connection() as conn:
        df = pd.read_sql(query, conn, params={"pid": player_id})

    if df.empty:
        return None
    row = df.iloc[0].to_dict()
    row.pop("embedding", None)   # don't expose raw vector
    return row
