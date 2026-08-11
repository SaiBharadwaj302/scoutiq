"""
models/similarity/train_deep.py

Trains the deep metric learning similarity model:
1. Aggregate per-90 stats per player from shot/pass features (features.py)
2. Normalize with StandardScaler
3. Train PlayerEmbeddingNet with TripletMarginLoss (anchor/positive share a
   position_group, negative comes from a different one)
4. Extract 32-dim L2-normalized embeddings for every player
5. Upsert embeddings into PostgreSQL player_per90_features
6. Log the run + model to MLflow

Run directly:
    python -m models.similarity.train_deep
"""
import os
import warnings

import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
from loguru import logger
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from dotenv import load_dotenv

from db.store import upsert_dataframe
from models.similarity.dataset import PlayerTripletDataset
from models.similarity.features import (
    EMBEDDING_FEATURES,
    aggregate_player_stats,
    build_feature_matrix,
    compute_per90,
)
from models.similarity.network import PlayerEmbeddingNet
from models.similarity.query import find_similar

warnings.filterwarnings("ignore")
load_dotenv()

MLFLOW_URI      = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
EXPERIMENT_NAME = "scoutiq_similarity"
MODEL_NAME      = "scoutiq_similarity_net"

EMBEDDING_DIM = 32
BATCH_SIZE    = 64
EPOCHS        = 50
LR            = 0.001
MARGIN        = 1.0


def train_embedding_net(
    X_scaled: pd.DataFrame, positions: dict[int, str]
) -> tuple[PlayerEmbeddingNet, list[float]]:
    """Train PlayerEmbeddingNet with triplet margin loss."""
    dataset = PlayerTripletDataset(X_scaled, positions)
    loader = DataLoader(dataset, batch_size=min(BATCH_SIZE, len(dataset)), shuffle=True)

    model = PlayerEmbeddingNet(input_dim=X_scaled.shape[1], embedding_dim=EMBEDDING_DIM)
    criterion = torch.nn.TripletMarginLoss(margin=MARGIN)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    epoch_losses = []
    model.train()
    for epoch in range(1, EPOCHS + 1):
        running_loss = 0.0
        for anchor, positive, negative in loader:
            optimizer.zero_grad()
            anchor_out = model(anchor)
            pos_out = model(positive)
            neg_out = model(negative)
            loss = criterion(anchor_out, pos_out, neg_out)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * anchor.size(0)

        epoch_loss = running_loss / len(dataset)
        epoch_losses.append(epoch_loss)
        if epoch == 1 or epoch % 10 == 0 or epoch == EPOCHS:
            logger.info(f"Epoch {epoch:>3d}/{EPOCHS}  triplet_loss={epoch_loss:.4f}")

    return model, epoch_losses


def extract_embeddings(model: PlayerEmbeddingNet, X_scaled: pd.DataFrame) -> np.ndarray:
    """Run all players through the trained network to get final embeddings."""
    model.eval()
    with torch.no_grad():
        all_features = torch.tensor(X_scaled.values, dtype=torch.float32)
        embeddings = model(all_features).numpy()
    return embeddings


def store_embeddings(per90_df: pd.DataFrame, embeddings: np.ndarray) -> None:
    """Write player embeddings to PostgreSQL player_per90_features table."""
    rows = []
    for i, (_, row) in enumerate(per90_df.iterrows()):
        rows.append({
            "player_id":               int(row["player_id"]),
            "competition_id":          0,   # aggregated across all competitions
            "season_id":               0,
            "player_name":             row.get("player_name", ""),
            "team_name":               row.get("team_name", ""),
            "position_group":          row.get("position_group", "Unknown"),
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


def train_similarity_model() -> dict:
    """Full pipeline: aggregate -> per90 -> triplet train -> store -> validate."""
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    logger.info("=== Deep Metric Learning Similarity Model Starting ===")

    raw_df = aggregate_player_stats()
    per90_df = compute_per90(raw_df)
    X = build_feature_matrix(per90_df)

    logger.info(f"Feature matrix: {X.shape[0]} players x {X.shape[1]} features")

    positions = dict(zip(
        per90_df["player_id"].astype(int), per90_df["position_group"]
    ))
    positions = {pid: positions.get(int(pid), "Unknown") for pid in X.index}

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X), index=X.index, columns=X.columns
    )

    with mlflow.start_run(run_name="player_similarity_deep_metric_v1"):
        mlflow.log_params({
            "algorithm":      "triplet_margin_loss_cosine_similarity",
            "embedding_dim":  EMBEDDING_DIM,
            "n_players":      len(X),
            "n_features":     X.shape[1],
            "batch_size":     BATCH_SIZE,
            "epochs":         EPOCHS,
            "lr":             LR,
            "margin":         MARGIN,
            "features":       ",".join(EMBEDDING_FEATURES),
        })

        model, epoch_losses = train_embedding_net(X_scaled, positions)

        for epoch, loss in enumerate(epoch_losses, start=1):
            mlflow.log_metric("triplet_loss", loss, step=epoch)

        embeddings = extract_embeddings(model, X_scaled)

        # Store in PostgreSQL
        store_embeddings(per90_df, embeddings)

        # Validate — find similar players for the player with most shots
        player_ids = X.index.tolist()
        player_names = per90_df.set_index(
            per90_df["player_id"].astype(int)
        )["player_name"].reindex(player_ids).fillna("Unknown").tolist()

        top_player_id = int(raw_df.loc[raw_df["shots"].idxmax(), "player_id"])
        top_player_name = raw_df.loc[raw_df["shots"].idxmax(), "player_name"]

        similar = find_similar(
            top_player_id, top_n=5,
            embeddings=embeddings, player_ids=player_ids, player_names=player_names,
        )
        logger.info(f"\nSimilar players to {top_player_name}:")
        for r in similar:
            logger.info(f"  {str(r['player_name']):<30s}  similarity={r['similarity']:.4f}")

        mlflow.log_metric("validation_top1_similarity", float(similar[0]["similarity"]))

        sample_input = X_scaled.iloc[:3]
        mlflow.pytorch.log_model(
            model,
            artifact_path="scoutiq_similarity_net",
            registered_model_name=MODEL_NAME,
            input_example=sample_input.values.astype("float32"),
        )
        logger.info(f"Model registered as '{MODEL_NAME}'")

    logger.info("=== Deep Metric Learning Similarity Model Complete ===")
    return {
        "n_players":   len(embeddings),
        "embedding_dim": EMBEDDING_DIM,
        "final_loss":  epoch_losses[-1],
    }


if __name__ == "__main__":
    result = train_similarity_model()
    print(f"\n{'='*50}")
    print(f"Players embedded:    {result['n_players']}")
    print(f"Embedding dim:       {result['embedding_dim']}")
    print(f"Final triplet loss:  {result['final_loss']:.4f}")
    print(f"{'='*50}")
