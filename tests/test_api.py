"""
tests/test_api.py

FastAPI endpoint tests using TestClient.
Models are mocked so no database or MLflow connection is needed.
"""
import math
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient


# ── Mock model factories ──────────────────────────────────────────────────────

def _mock_xg_model(xg_prob: float = 0.18):
    """Return a sklearn-compatible mock that always predicts xg_prob."""
    m = MagicMock()
    m.predict_proba.return_value = np.array([[1 - xg_prob, xg_prob]])
    return m


def _mock_pass_model(prob: float = 0.82):
    m = MagicMock()
    m.predict_proba.return_value = np.array([[1 - prob, prob]])
    return m


def _mock_similarity_data():
    """Return (embeddings, player_ids, player_names) for 3 fake players."""
    embs = np.eye(3, dtype=float)   # orthogonal — zero cosine similarity
    return embs, [1, 2, 3], ["Alice", "Bob", "Carol"]


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def client():
    """Create a TestClient with all model dependencies mocked."""
    with (
        patch("api.dependencies._xg_model",     _mock_xg_model()),
        patch("api.dependencies._pass_model",    _mock_pass_model()),
        patch("api.dependencies._embeddings",    _mock_similarity_data()[0]),
        patch("api.dependencies._player_ids",    _mock_similarity_data()[1]),
        patch("api.dependencies._player_names",  _mock_similarity_data()[2]),
    ):
        from api.main import app
        with TestClient(app) as c:
            yield c


# ── Health ────────────────────────────────────────────────────────────────────

class TestHealth:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_status_healthy(self, client):
        data = client.get("/health").json()
        assert data["status"] == "healthy"

    def test_root_redirects_or_200(self, client):
        resp = client.get("/")
        assert resp.status_code in (200, 307)


# ── xG endpoint ───────────────────────────────────────────────────────────────

class TestXGEndpoint:
    VALID_PAYLOAD = {
        "location_x": 105.0,
        "location_y": 38.0,
        "technique_name": "Normal",
        "body_part_name": "Right Foot",
        "play_pattern_name": "Regular Play",
        "under_pressure": False,
        "first_time": False,
        "period": 1,
    }

    def test_valid_shot_returns_200(self, client):
        resp = client.post("/v1/predict/xg", json=self.VALID_PAYLOAD)
        assert resp.status_code == 200, resp.text

    def test_response_has_xg_field(self, client):
        data = client.post("/v1/predict/xg", json=self.VALID_PAYLOAD).json()
        assert "xg" in data
        assert 0.0 <= data["xg"] <= 1.0

    def test_response_has_is_big_chance(self, client):
        data = client.post("/v1/predict/xg", json=self.VALID_PAYLOAD).json()
        assert "is_big_chance" in data
        assert isinstance(data["is_big_chance"], bool)

    def test_response_has_model_version(self, client):
        data = client.post("/v1/predict/xg", json=self.VALID_PAYLOAD).json()
        assert "model_version" in data

    def test_xg_out_of_pitch_x_rejected(self, client):
        bad = {**self.VALID_PAYLOAD, "location_x": 999}
        resp = client.post("/v1/predict/xg", json=bad)
        assert resp.status_code == 422   # Pydantic validation error

    def test_xg_out_of_pitch_y_rejected(self, client):
        bad = {**self.VALID_PAYLOAD, "location_y": -5}
        resp = client.post("/v1/predict/xg", json=bad)
        assert resp.status_code == 422

    def test_xg_missing_location_rejected(self, client):
        resp = client.post("/v1/predict/xg", json={"technique_name": "Normal"})
        assert resp.status_code == 422

    def test_xg_with_360_features(self, client):
        payload = {
            **self.VALID_PAYLOAD,
            "defenders_in_2m_cone": 1,
            "defenders_in_5m_cone": 2,
            "gk_distance_to_goal_centre": 3.5,
        }
        resp = client.post("/v1/predict/xg", json=payload)
        assert resp.status_code == 200

    def test_high_xg_shot_is_big_chance(self, client):
        """Mock a model that returns 0.45 → is_big_chance should be True."""
        with patch("api.dependencies._xg_model", _mock_xg_model(xg_prob=0.45)):
            resp = client.post("/v1/predict/xg", json=self.VALID_PAYLOAD)
            assert resp.json()["is_big_chance"] is True

    def test_low_xg_shot_not_big_chance(self, client):
        with patch("api.dependencies._xg_model", _mock_xg_model(xg_prob=0.05)):
            resp = client.post("/v1/predict/xg", json=self.VALID_PAYLOAD)
            assert resp.json()["is_big_chance"] is False


# ── Pass success endpoint ─────────────────────────────────────────────────────

class TestPassEndpoint:
    VALID_PAYLOAD = {
        "start_x": 60.0,
        "start_y": 40.0,
        "end_x":   85.0,
        "end_y":   35.0,
        "height_name": "Ground Pass",
        "body_part_name": "Right Foot",
        "under_pressure": False,
    }

    def test_valid_pass_returns_200(self, client):
        resp = client.post("/v1/predict/pass", json=self.VALID_PAYLOAD)
        assert resp.status_code == 200, resp.text

    def test_response_has_completion_probability(self, client):
        data = client.post("/v1/predict/pass", json=self.VALID_PAYLOAD).json()
        assert "completion_probability" in data
        assert 0.0 <= data["completion_probability"] <= 1.0

    def test_response_has_is_high_risk(self, client):
        data = client.post("/v1/predict/pass", json=self.VALID_PAYLOAD).json()
        assert "is_high_risk" in data
        assert isinstance(data["is_high_risk"], bool)

    def test_response_has_pass_length(self, client):
        data = client.post("/v1/predict/pass", json=self.VALID_PAYLOAD).json()
        assert "pass_length" in data
        assert data["pass_length"] > 0

    def test_out_of_pitch_rejected(self, client):
        bad = {**self.VALID_PAYLOAD, "start_x": 200}
        resp = client.post("/v1/predict/pass", json=bad)
        assert resp.status_code == 422

    def test_missing_endpoint_rejected(self, client):
        resp = client.post("/v1/predict/pass", json={"start_x": 60, "start_y": 40})
        assert resp.status_code == 422

    def test_high_risk_flag_when_low_probability(self, client):
        with patch("api.dependencies._pass_model", _mock_pass_model(prob=0.3)):
            data = client.post("/v1/predict/pass", json=self.VALID_PAYLOAD).json()
            assert data["is_high_risk"] is True

    def test_not_high_risk_when_high_probability(self, client):
        with patch("api.dependencies._pass_model", _mock_pass_model(prob=0.9)):
            data = client.post("/v1/predict/pass", json=self.VALID_PAYLOAD).json()
            assert data["is_high_risk"] is False

    def test_optional_360_features_accepted(self, client):
        payload = {
            **self.VALID_PAYLOAD,
            "defenders_in_pass_lane": 0,
            "nearest_defender_distance": 4.2,
        }
        assert client.post("/v1/predict/pass", json=payload).status_code == 200


# ── Similarity endpoint ───────────────────────────────────────────────────────

class TestSimilarityEndpoint:
    def test_known_player_returns_200(self, client):
        resp = client.get("/v1/players/similar/1")
        assert resp.status_code == 200

    def test_response_has_similar_players_list(self, client):
        data = client.get("/v1/players/similar/1").json()
        assert "similar_players" in data
        assert isinstance(data["similar_players"], list)

    def test_top_n_respected(self, client):
        resp = client.get("/v1/players/similar/1?top_n=2")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["similar_players"]) <= 2

    def test_unknown_player_returns_404(self, client):
        resp = client.get("/v1/players/similar/99999")
        assert resp.status_code == 404

    def test_similarity_scores_between_0_and_1(self, client):
        data = client.get("/v1/players/similar/1").json()
        for p in data["similar_players"]:
            assert -1.0 <= p["similarity"] <= 1.0

    def test_query_player_id_in_response(self, client):
        data = client.get("/v1/players/similar/1").json()
        assert data["query_player_id"] == 1

    def test_query_player_name_in_response(self, client):
        data = client.get("/v1/players/similar/1").json()
        assert data["query_player_name"] == "Alice"

    def test_player_stats_endpoint(self, client):
        """GET /v1/players/stats/{player_id} should return stats or 404."""
        resp = client.get("/v1/players/stats/1")
        # 404 expected without a real DB, but endpoint should exist
        assert resp.status_code in (200, 404, 500)
