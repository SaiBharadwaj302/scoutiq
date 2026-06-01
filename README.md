# ScoutIQ — Production ML Football Analytics Platform

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![LightGBM](https://img.shields.io/badge/LightGBM-4.4.0-green)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi)
![MLflow](https://img.shields.io/badge/MLflow-2.14-blue?logo=mlflow)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-336791?logo=postgresql)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker)
![CI](https://github.com/sai23bharadwaj/scoutiq/actions/workflows/ci.yml/badge.svg)

Three production-grade ML models trained on **88,023 shots** and **3,387,760 passes** from the StatsBomb open football dataset — served via FastAPI with a Streamlit dashboard.

---

## Model Performance

| Model | Algorithm | AUC-ROC | Dataset |
|-------|-----------|---------|---------|
| xG 2.0 | LightGBM + Logistic Regression (stacked) | **0.808** | 88,023 shots |
| Pass Success | LightGBM + Platt calibration | **0.905** | 3,387,760 passes |
| Player Similarity | PCA (32-dim) + cosine similarity | — | 5,653 players |

---

## Quick Start

### Prerequisites
- Python 3.11
- Docker Desktop (running)
- Git

```bash
git clone https://github.com/sai23bharadwaj/scoutiq.git
cd scoutiq

# Copy environment config
copy .env.example .env

# Start PostgreSQL
docker-compose up -d postgres

# Create virtual environment and install dependencies
py -3.11 -m venv venv
.\venv\Scripts\Activate
pip install -r requirements.txt

# Run API
python run_api.py
```

Open **http://localhost:8000/docs** for the interactive Swagger UI.

### Streamlit Dashboard

```bash
streamlit run app.py
```

Open **http://localhost:8501**

---

## API Endpoints

### xG Prediction — `POST /v1/predict/xg`

```bash
curl -X POST http://localhost:8000/v1/predict/xg \
  -H "Content-Type: application/json" \
  -d '{
    "location_x": 105,
    "location_y": 38,
    "technique_name": "Normal",
    "body_part_name": "Right Foot",
    "play_pattern_name": "Regular Play",
    "period": 1
  }'
```

**Response:**
```json
{
  "xg": 0.1923,
  "is_big_chance": false,
  "distance_to_goal": 15.13,
  "angle_to_goal": 0.3814,
  "model_version": "scoutiq_xg/1",
  "model_name": "scoutiq_xg"
}
```

### Pass Success — `POST /v1/predict/pass`

```bash
curl -X POST http://localhost:8000/v1/predict/pass \
  -H "Content-Type: application/json" \
  -d '{
    "start_x": 60,
    "start_y": 40,
    "end_x": 85,
    "end_y": 35,
    "height_name": "Ground Pass",
    "body_part_name": "Right Foot",
    "under_pressure": false
  }'
```

**Response:**
```json
{
  "completion_probability": 0.8741,
  "is_high_risk": false,
  "pass_length": 25.5,
  "model_version": "scoutiq_pass/1",
  "model_name": "scoutiq_pass"
}
```

### Player Similarity — `GET /v1/players/similar/{player_id}`

```bash
curl http://localhost:8000/v1/players/similar/3094?top_n=5
```

**Response:**
```json
{
  "query_player_id": 3094,
  "query_player_name": "Lionel Messi",
  "similar_players": [
    {"player_id": 5211, "player_name": "Neymar Jr", "similarity": 0.9412},
    {"player_id": 8820, "player_name": "Eden Hazard", "similarity": 0.9187}
  ],
  "model_name": "scoutiq_similarity"
}
```

---

## Project Structure

```
scoutiq/
├── api/                    # FastAPI application
│   ├── main.py             # App factory + lifespan model loading
│   ├── schemas.py          # Pydantic request/response models
│   ├── dependencies.py     # Cached model loading
│   └── routers/            # xg.py, pass_success.py, similarity.py
├── db/
│   ├── schema.sql          # PostgreSQL DDL (auto-runs on first Docker start)
│   └── store.py            # SQLAlchemy engine + upsert helpers
├── models/
│   ├── xg/                 # xG stacked ensemble (LightGBM + LogReg)
│   ├── pass_success/       # Pass completion (LightGBM + Platt scaling)
│   └── similarity/         # Player embeddings (PCA + cosine)
├── pipeline/
│   ├── ingest.py           # StatsBomb event ingestion → PostgreSQL
│   └── ingest_players.py   # Player lineup ingestion → back-fill names
├── monitoring/
│   └── drift_report.py     # Evidently AI drift detection
├── tests/
│   └── test_ingest.py      # 12 unit tests (geometry + 360 features)
├── app.py                  # Streamlit dashboard
├── run_api.py              # API launcher (direct import — Windows-safe)
├── docker-compose.yml      # PostgreSQL + API
└── Dockerfile
```

---

## Data Pipeline

```bash
# 1. Start PostgreSQL
docker-compose up -d postgres

# 2. Ingest StatsBomb events (2-4 hours for full dataset)
python -m pipeline.ingest

# 3. Ingest player names from lineups
python -m pipeline.ingest_players

# 4. Train models
python -m models.xg.train              # AUC ~0.808
python -m models.pass_success.train   # AUC ~0.905
python -m models.similarity.build_embeddings

# 5. Verify data
docker exec -it scoutiq_postgres psql -U scoutiq_user -d scoutiq \
  -c "SELECT COUNT(*) FROM shot_features;"   # Expected: 88023
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Data source | StatsBomb Open Data (statsbombpy 1.17) |
| Feature store | PostgreSQL 16 |
| ML training | LightGBM, scikit-learn, SHAP |
| Experiment tracking | MLflow 2.14 (file:./mlruns) |
| API | FastAPI 0.111 + uvicorn |
| Dashboard | Streamlit 1.36 |
| Drift monitoring | Evidently AI 0.4.30 |
| Orchestration | Prefect 2.19 |
| Containerisation | Docker Compose |
| CI | GitHub Actions |

---

## Running Tests

```bash
pytest tests/ -v
# Expected: 12 passed, 0 failed
```

---

## Environment Variables

Copy `.env.example` to `.env` and adjust if needed:

```
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=scoutiq
POSTGRES_USER=scoutiq_user
POSTGRES_PASSWORD=scoutiq_pass
MLFLOW_TRACKING_URI=file:./mlruns
```
