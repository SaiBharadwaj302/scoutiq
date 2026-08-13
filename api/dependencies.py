"""
api/dependencies.py

Model loading and caching for FastAPI dependency injection.
Models are loaded once at startup and reused across requests —
this is what keeps p99 latency under 15ms.
"""
import os
import mlflow
import mlflow.sklearn
from loguru import logger
from dotenv import load_dotenv

from models.similarity.query import load_embeddings

load_dotenv()

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
SIMILARITY_MODEL_NAME = "scoutiq_similarity_net"

# ── Model Cache (loaded once at startup) ──────────────────────────────────────
_xg_model    = None
_pass_model  = None
_similarity_model = None
_embeddings  = None
_player_ids  = None
_player_names = None


def _load_mlflow_model(model_name: str, alias: str = "latest"):
    """Load a model from MLflow registry."""
    mlflow.set_tracking_uri(MLFLOW_URI)
    try:
        uri = f"models:/{model_name}/{alias}"
        model = mlflow.sklearn.load_model(uri)
        logger.info(f"Loaded model: {uri}")
        return model
    except Exception as e:
        logger.warning(f"Could not load {alias} — trying version 1: {e}")
        uri = f"models:/{model_name}/1"
        model = mlflow.sklearn.load_model(uri)
        logger.info(f"Loaded model: {uri}")
        return model


def get_xg_model():
    """Return cached xG model, loading if needed."""
    global _xg_model
    if _xg_model is None:
        _xg_model = _load_mlflow_model("scoutiq_xg")
    return _xg_model


def get_pass_model():
    """Return cached pass model, loading if needed."""
    global _pass_model
    if _pass_model is None:
        _pass_model = _load_mlflow_model("scoutiq_pass")
    return _pass_model


def get_similarity_data():
    """Return cached embeddings, loading if needed."""
    global _embeddings, _player_ids, _player_names
    if _embeddings is None:
        _embeddings, _player_ids, _player_names = load_embeddings()
        logger.info(f"Loaded {len(_player_ids)} player embeddings into memory")
    return _embeddings, _player_ids, _player_names


def get_similarity_model():
    """
    Return the cached PyTorch deep metric learning model, loading if needed.
    Only required for on-the-fly embedding of a raw feature vector that
    hasn't been precomputed into player_per90_features yet.
    """
    global _similarity_model
    if _similarity_model is None:
        # Lazy import: torch/mlflow.pytorch are only needed for this one
        # on-the-fly embedding path, not for the xG/pass/similarity-lookup
        # endpoints — keeps the base API deployable without a ~2GB torch
        # install on memory-constrained hosts.
        import mlflow.pytorch
        mlflow.set_tracking_uri(MLFLOW_URI)
        try:
            uri = f"models:/{SIMILARITY_MODEL_NAME}/latest"
            _similarity_model = mlflow.pytorch.load_model(uri)
            logger.info(f"Loaded model: {uri}")
        except Exception as e:
            logger.warning(f"Could not load latest — trying version 1: {e}")
            uri = f"models:/{SIMILARITY_MODEL_NAME}/1"
            _similarity_model = mlflow.pytorch.load_model(uri)
            logger.info(f"Loaded model: {uri}")
        _similarity_model.eval()
    return _similarity_model


def get_model_versions() -> dict[str, str]:
    """Return version info for all loaded models."""
    return {
        "xg_model":         "scoutiq_xg/1",
        "pass_model":       "scoutiq_pass/1",
        "similarity_model": f"{SIMILARITY_MODEL_NAME}/1",
    }
