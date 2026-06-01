"""
api/main.py

ScoutIQ FastAPI application.
Registers all routers and handles startup model loading.
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from api.routers import xg, pass_success, similarity
from api.dependencies import (
    get_xg_model, get_pass_model,
    get_similarity_data, get_model_versions,
)
from .schemas import HealthResponse


# ── Startup: load all models into memory ─────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models at startup so first request isn't slow."""
    logger.info("ScoutIQ API starting — loading models...")
    try:
        get_xg_model()
        logger.info("✓ xG model loaded")
    except Exception as e:
        logger.error(f"✗ xG model failed: {e}")
    try:
        get_pass_model()
        logger.info("✓ Pass model loaded")
    except Exception as e:
        logger.error(f"✗ Pass model failed: {e}")
    try:
        get_similarity_data()
        logger.info("✓ Similarity embeddings loaded")
    except Exception as e:
        logger.error(f"✗ Similarity data failed: {e}")
    logger.info("ScoutIQ API ready")
    yield
    logger.info("ScoutIQ API shutting down")


# ── App Factory ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="ScoutIQ API",
    description="""
## ScoutIQ — Production ML Football Analytics Platform

Three production-grade ML models serving real-time predictions:

- **xG 2.0**: Expected goals prediction with 360 spatial features (AUC 0.808)
- **Pass Success**: Pass completion probability with Platt calibration (AUC 0.905)
- **Player Similarity**: Embedding-based tactical player similarity search

Built on StatsBomb open data — 88K shots, 3.4M passes, 5,653 players.
    """,
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(xg.router)
app.include_router(pass_success.router)
app.include_router(similarity.router)


# ── Health Check ──────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check API status and loaded model versions."""
    return HealthResponse(
        status="healthy",
        models=get_model_versions(),
    )


@app.get("/", tags=["Health"])
async def root():
    return {
        "service": "ScoutIQ API",
        "version": "1.0.0",
        "docs":    "/docs",
        "health":  "/health",
    }
