"""
models/similarity/build_embeddings.py

Builds player embeddings for the similarity engine:
1. Aggregate per-90 stats per player-season from shot/pass features
2. Normalize with StandardScaler
3. Reduce to 32 dimensions with PCA
4. Store embedding vectors in PostgreSQL
5. Log to MLflow

Why PCA over neural autoencoder?
  ~2,000 unique players. PCA is optimal for linear variance at this scale,
  interpretable, and provably correct. An autoencoder would overfit.

Run directly:
    python -m models.similarity.build_embeddings
"""
import os
import warnings
warnings.filterwarnings("ignore")

import mlflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from loguru import logger
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sqlalchemy import text
from dotenv import load_dotenv

from db.store import get_connection, get_engine, upsert_dataframe

load_dotenv()

MLFLOW_URI      = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
EXPERIMENT_NAME = "scoutiq_similarity"
N_COMPONENTS    = 32   # PCA dimensions
MIN_MINUTES     = 200  # minimum minutes played to include player


# ── Per-90 Aggregation ────────────────────────────────────────────────────────

def aggregate_player_stats() -> pd.DataFrame:
    """
    Aggregate shot + pass features into per-90 player stats.
    Joins shot_features and pass_features by player_id and match_id.
    """
    logger.info("Aggregating player stats from feature store...")

    shot_query = """
        SELECT
            player_id,
            player_name,
            COUNT(*)                                    AS shots,
            SUM(CASE WHEN is_goal THEN 1 ELSE 0 END)   AS goals,
            SUM(CASE WHEN outcome_name = 'Saved'
                     THEN 1 ELSE 0 END)                AS shots_on_target,
            AVG(distance_to_goal)                       AS avg_shot_distance,
            AVG(angle_to_goal)                          AS avg_shot_angle,
            COUNT(DISTINCT match_id)                    AS matches_from_shots
        FROM shot_features
        WHERE player_id IS NOT NULL
        GROUP BY player_id, player_name
    """

    pass_query = """
        SELECT
            player_id,
            COUNT(*)                                        AS passes,
            SUM(CASE WHEN is_complete THEN 1 ELSE 0 END)   AS passes_complete,
            SUM(CASE WHEN is_switch THEN 1 ELSE 0 END)     AS switches,
            SUM(CASE WHEN is_through_ball THEN 1 ELSE 0 END) AS through_balls,
            SUM(CASE WHEN is_cross THEN 1 ELSE 0 END)      AS crosses,
            SUM(CASE WHEN goal_assist THEN 1 ELSE 0 END)   AS goal_assists,
            SUM(CASE WHEN shot_assist THEN 1 ELSE 0 END)   AS shot_assists,
            SUM(CASE WHEN under_pressure THEN 1 ELSE 0 END) AS passes_under_pressure,
            SUM(CASE WHEN end_x > start_x
                      AND SQRT(POWER(120 - end_x, 2) + POWER(40 - end_y, 2))
                        < 0.9 * SQRT(POWER(120 - start_x, 2) + POWER(40 - start_y, 2))
                     THEN 1 ELSE 0 END)                    AS progressive_passes,
            SUM(CASE WHEN end_x >= 102
                      AND end_y >= 18
                      AND end_y <= 62
                     THEN 1 ELSE 0 END)                    AS passes_into_box,
            AVG(length)                                     AS avg_pass_length,
            COUNT(DISTINCT match_id)                        AS matches_from_passes
        FROM pass_features
        WHERE player_id IS NOT NULL
        GROUP BY player_id
    """

    with get_connection() as conn:
        shots_df  = pd.read_sql(shot_query,  conn)
        passes_df = pd.read_sql(pass_query,  conn)

    logger.info(f"Shot aggregates: {len(shots_df)} players")
    logger.info(f"Pass aggregates: {len(passes_df)} players")

    # Merge on player_id
    df = shots_df.merge(passes_df, on="player_id", how="outer")

    # Estimate total matches (take max of shot/pass match counts)
    df["matches_played"] = df[["matches_from_shots", "matches_from_passes"]].max(axis=1)
    # Preserve string columns before filling numeric NaNs with 0
    _player_names = df.get("player_name")
    df = df.fillna(0)
    if _player_names is not None:
        df["player_name"] = _player_names

    # Rough minutes estimate: assume 85 mins average per match
    df["minutes_played"] = df["matches_played"] * 85

    # Filter minimum playing time
    df = df[df["minutes_played"] >= MIN_MINUTES].copy()
    logger.info(f"Players with {MIN_MINUTES}+ minutes: {len(df)}")

    return df


def compute_per90(df: pd.DataFrame) -> pd.DataFrame:
    """Convert raw counts to per-90-minute rates."""
    per90 = df.copy()
    mins  = df["minutes_played"].clip(lower=1)
    factor = 90 / mins

    count_cols = [
        "shots", "goals", "shots_on_target",
        "switches", "through_balls", "crosses",
        "goal_assists", "shot_assists",
        "passes_under_pressure", "progressive_passes",
        "passes_into_box", "passes", "passes_complete",
    ]

    for col in count_cols:
        if col in df.columns:
            per90[f"{col}_p90"] = df[col] * factor

    # Rates (not per90)
    per90["pass_completion_pct"] = np.where(
        df["passes"] > 0, df["passes_complete"] / df["passes"], 0
    )
    per90["shot_conversion_pct"] = np.where(
        df["shots"] > 0, df["goals"] / df["shots"], 0
    )
    per90["shots_on_target_pct"] = np.where(
        df["shots"] > 0, df["shots_on_target"] / df["shots"], 0
    )

    return per90


# ── Feature Matrix for PCA ────────────────────────────────────────────────────

EMBEDDING_FEATURES = [
    "shots_p90",
    "goals_p90",
    "shots_on_target_p90",
    "avg_shot_distance",
    "avg_shot_angle",
    "passes_p90",
    "pass_completion_pct",
    "shot_conversion_pct",
    "shots_on_target_pct",
    "switches_p90",
    "through_balls_p90",
    "crosses_p90",
    "goal_assists_p90",
    "shot_assists_p90",
    "passes_under_pressure_p90",
    "progressive_passes_p90",
    "passes_into_box_p90",
    "avg_pass_length",
]


def build_feature_matrix(per90_df: pd.DataFrame) -> pd.DataFrame:
    """Extract and clean the feature matrix for PCA."""
    available = [f for f in EMBEDDING_FEATURES if f in per90_df.columns]
    X = per90_df[available].copy()
    X = X.fillna(0)
    # Clip extreme outliers (> 99th percentile)
    for col in X.columns:
        cap = X[col].quantile(0.99)
        X[col] = X[col].clip(upper=cap)
    return X


# ── PCA Embedding ─────────────────────────────────────────────────────────────

def build_pca_embeddings(X: pd.DataFrame) -> tuple[np.ndarray, PCA, StandardScaler]:
    """
    Normalize → PCA → return embeddings + fitted transformers.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_components = min(N_COMPONENTS, X.shape[1], X.shape[0] - 1)
    pca = PCA(n_components=n_components, random_state=42)
    embeddings = pca.fit_transform(X_scaled)

    explained = pca.explained_variance_ratio_.cumsum()[-1]
    logger.info(f"PCA: {n_components} components explain "
                f"{explained*100:.1f}% of variance")

    return embeddings, pca, scaler


def plot_variance_explained(pca: PCA, run_id: str) -> str:
    """Plot cumulative explained variance curve."""
    cumvar = pca.explained_variance_ratio_.cumsum()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(cumvar)+1), cumvar * 100, "b-o", markersize=4)
    ax.axhline(y=80, color="r", linestyle="--", label="80% threshold")
    ax.axhline(y=90, color="g", linestyle="--", label="90% threshold")
    ax.set_xlabel("Number of PCA Components")
    ax.set_ylabel("Cumulative Explained Variance (%)")
    ax.set_title("Player Embedding — PCA Variance Explained")
    ax.legend()
    ax.grid(True, alpha=0.3)
    path = f"pca_variance_{run_id[:8]}.png"
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return path


# ── Store Embeddings ──────────────────────────────────────────────────────────

def store_embeddings(
    per90_df: pd.DataFrame,
    embeddings: np.ndarray,
) -> None:
    """Write player embeddings to PostgreSQL player_per90_features table."""
    rows = []
    for i, (_, row) in enumerate(per90_df.iterrows()):
        rows.append({
            "player_id":               int(row["player_id"]),
            "competition_id":          0,   # aggregated across all competitions
            "season_id":               0,
            "player_name":             row.get("player_name", ""),
            "team_name":               "",
            "position_group":          "Unknown",
            "minutes_played":          float(row["minutes_played"]),
            "matches_played":          int(row["matches_played"]),
            "goals_per90":             float(row.get("goals_p90", 0)),
            "assists_per90":           float(row.get("goal_assists_p90", 0)),
            "shots_per90":             float(row.get("shots_p90", 0)),
            "shots_on_target_per90":   float(row.get("shots_on_target_p90", 0)),
            "xg_per90":                0.0,
            "xg_overperformance":      0.0,
            "passes_per90":            float(row.get("passes_p90", 0)),
            "pass_completion_pct":     float(row.get("pass_completion_pct", 0)),
            "progressive_passes_per90": float(row.get("progressive_passes_p90", 0)),
            "key_passes_per90":        float(row.get("shot_assists_p90", 0)),
            "switches_per90":          float(row.get("switches_p90", 0)),
            "pressures_per90":         0.0,
            "pressure_success_pct":    0.0,
            "tackles_per90":           0.0,
            "interceptions_per90":     0.0,
            "blocks_per90":            0.0,
            "carries_per90":           0.0,
            "progressive_carries_per90": 0.0,
            "carries_into_final_third_per90": 0.0,
            "dribbles_completed_per90": 0.0,
            "embedding":               embeddings[i].tolist(),
        })

    df = pd.DataFrame(rows)
    upsert_dataframe(
        df, "player_per90_features",
        conflict_columns=["player_id", "competition_id", "season_id"]
    )
    logger.info(f"Stored {len(rows)} player embeddings in PostgreSQL")


# ── Similarity Query ──────────────────────────────────────────────────────────

def find_similar_players(
    player_id: int,
    embeddings: np.ndarray,
    player_ids: list[int],
    player_names: list[str],
    top_n: int = 5,
) -> pd.DataFrame:
    """Find top-N most similar players using cosine similarity."""
    if player_id not in player_ids:
        raise ValueError(f"Player {player_id} not found in embeddings")

    idx = player_ids.index(player_id)
    query_vec = embeddings[idx]

    # Cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    normed = embeddings / norms
    query_normed = query_vec / (np.linalg.norm(query_vec) or 1)
    similarities = normed @ query_normed

    # Exclude the query player itself
    similarities[idx] = -1

    top_indices = np.argsort(similarities)[::-1][:top_n]
    return pd.DataFrame({
        "player_id":   [player_ids[i] for i in top_indices],
        "player_name": [player_names[i] for i in top_indices],
        "similarity":  [round(float(similarities[i]), 4) for i in top_indices],
    })


# ── Main ──────────────────────────────────────────────────────────────────────

def build_similarity_model() -> dict:
    """Full pipeline: aggregate → per90 → PCA → store → validate."""
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    logger.info("═══ Player Similarity Model Starting ═══")

    # Aggregate stats
    raw_df   = aggregate_player_stats()
    per90_df = compute_per90(raw_df)
    X        = build_feature_matrix(per90_df)

    logger.info(f"Feature matrix: {X.shape[0]} players × {X.shape[1]} features")

    with mlflow.start_run(run_name="player_similarity_pca_v1") as run:
        run_id = run.info.run_id

        mlflow.log_params({
            "algorithm":        "pca_cosine_similarity",
            "n_components":     N_COMPONENTS,
            "n_players":        len(X),
            "n_features":       X.shape[1],
            "min_minutes":      MIN_MINUTES,
            "features":         ",".join(X.columns.tolist()),
        })

        # Build embeddings
        embeddings, pca, scaler = build_pca_embeddings(X)

        explained = float(pca.explained_variance_ratio_.cumsum()[-1])
        mlflow.log_metrics({
            "explained_variance_ratio": explained,
            "n_players_embedded":       len(embeddings),
            "n_components_used":        pca.n_components_,
        })

        # Plot variance
        var_path = plot_variance_explained(pca, run_id)
        mlflow.log_artifact(var_path)
        os.remove(var_path)

        # Store in PostgreSQL
        store_embeddings(per90_df, embeddings)

        # Validate — find similar players for a known player
        player_ids   = per90_df["player_id"].astype(int).tolist()
        player_names = per90_df["player_name"].fillna("Unknown").tolist()

        # Pick the player with most shots as validation target
        top_player_id = int(raw_df.loc[raw_df["shots"].idxmax(), "player_id"])
        top_player_name = raw_df.loc[raw_df["shots"].idxmax(), "player_name"]

        similar = find_similar_players(
            top_player_id, embeddings, player_ids, player_names, top_n=5
        )
        logger.info(f"\nSimilar players to {top_player_name}:")
        for _, row in similar.iterrows():
            logger.info(f"  {str(row['player_name']):<30s}  similarity={row['similarity']:.4f}")

        mlflow.log_metric("validation_top1_similarity", float(similar.iloc[0]["similarity"]))

    logger.info("═══ Similarity Model Complete ═══")
    return {
        "n_players":   len(embeddings),
        "n_components": pca.n_components_,
        "variance_explained": explained,
    }


if __name__ == "__main__":
    result = build_similarity_model()
    print(f"\n{'='*50}")
    print(f"Players embedded:    {result['n_players']}")
    print(f"PCA components:      {result['n_components']}")
    print(f"Variance explained:  {result['variance_explained']*100:.1f}%")
    print(f"{'='*50}")
