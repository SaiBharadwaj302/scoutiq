"""
streamlit_app/similarity.py

Standalone cosine-similarity search over pre-loaded player embeddings —
a copy of the pure-numpy logic from models/similarity/query.py's
find_similar(), without that module's Postgres-backed load_embeddings().
"""
import numpy as np


def find_similar(player_id: int, top_n: int,
                  embeddings: np.ndarray, player_ids: list[int], player_names: list[str]) -> list[dict]:
    """
    Find the top-N most similar players to a given player_id by cosine
    similarity over their embedding vectors.
    """
    if player_id not in player_ids:
        raise ValueError(f"Player {player_id} not found in embedding store")

    idx = player_ids.index(player_id)
    query_vec = embeddings[idx]

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    normed       = embeddings / norms
    query_normed = query_vec / (np.linalg.norm(query_vec) or 1)
    similarities = normed @ query_normed

    similarities[idx] = -1  # exclude self
    top_indices = np.argsort(similarities)[::-1][:top_n]

    return [
        {
            "player_id":   player_ids[i],
            "player_name": player_names[i],
            "similarity":  round(float(similarities[i]), 4),
        }
        for i in top_indices
    ]
