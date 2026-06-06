"""
tests/test_api.py — API endpoint tests using FastAPI's TestClient.
"""

import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.api.main import app


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

def test_health_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["global_model_loaded"] is True
    assert data["calibration_loaded"] is True
    assert data["model_version"] != "unknown"


# ---------------------------------------------------------------------------
# Cold-start — zero history
# ---------------------------------------------------------------------------

def test_cold_start_no_sessions(client):
    payload = {
        "station_id": "test-zero-history",
        "site": "unknown",
        "lat": 34.2020,
        "lng": -118.1720,
        "num_ports": 2,
        "sessions": []
    }
    response = client.post("/forecast/cold-start", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["station_id"] == "test-zero-history"
    assert data["fine_tuned"] is False
    assert len(data["forecast"]) == 168
    for row in data["forecast"]:
        assert row["lower_80"] >= 0
        assert row["lower_90"] >= 0
        assert row["upper_80"] >= row["lower_80"]
        assert row["upper_90"] >= row["lower_90"]


def test_cold_start_no_sessions_field(client):
    """Sessions field omitted entirely — should behave identically to empty list."""
    payload = {
        "station_id": "test-omit-sessions",
        "site": "caltech",
        "lat": 34.1377,
        "lng": -118.1253,
        "num_ports": 4,
    }
    response = client.post("/forecast/cold-start", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["fine_tuned"] is False
    assert len(data["forecast"]) == 168


# ---------------------------------------------------------------------------
# Cold-start — with sparse history (fine-tuning path)
# ---------------------------------------------------------------------------

def test_cold_start_with_sessions(client):
    payload = {
        "station_id": "test-with-history",
        "site": "caltech",
        "lat": 34.1377,
        "lng": -118.1253,
        "num_ports": 2,
        "sessions": [
            {"start_time": "2024-03-04T09:00:00Z", "end_time": "2024-03-04T11:00:00Z", "energy_kwh": 15.0},
            {"start_time": "2024-03-04T13:00:00Z", "end_time": "2024-03-04T15:00:00Z", "energy_kwh": 12.0},
            {"start_time": "2024-03-05T08:00:00Z", "end_time": "2024-03-05T10:00:00Z", "energy_kwh": 18.0},
        ]
    }
    response = client.post("/forecast/cold-start", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["fine_tuned"] is True
    assert len(data["forecast"]) == 168
    predicted_values = [row["predicted"] for row in data["forecast"]]
    assert max(predicted_values) > 0


# ---------------------------------------------------------------------------
# Cold-start — schema validation
# ---------------------------------------------------------------------------

def test_cold_start_missing_required_fields(client):
    payload = {"station_id": "bad-request", "num_ports": 2}
    response = client.post("/forecast/cold-start", json=payload)
    assert response.status_code == 422


def test_cold_start_invalid_lat(client):
    payload = {
        "station_id": "bad-lat",
        "lat": 999.0,
        "lng": -118.1720,
        "num_ports": 2,
    }
    response = client.post("/forecast/cold-start", json=payload)
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Known station forecast
# ---------------------------------------------------------------------------

def test_station_forecast_known(client):
    response = client.get("/forecast/stations/caltech_2-39-123-23")
    assert response.status_code == 200
    data = response.json()
    assert data["station_id"] == "caltech_2-39-123-23"
    assert len(data["forecast"]) == 168


def test_station_forecast_unknown(client):
    response = client.get("/forecast/stations/does-not-exist")
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# Site evaluate
# ---------------------------------------------------------------------------

def test_site_evaluate(client):
    payload = {
        "lat": 34.2020,
        "lng": -118.1720,
        "num_ports": 4,
        "location_type": "workplace"
    }
    response = client.post("/site/evaluate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_weekly_sessions"] > 0
    assert data["demand_tier"] in ["Low", "Moderate", "High", "Very High"]
    assert data["recommendation"] in ["Strong candidate", "Moderate candidate", "Weak candidate"]
    assert len(data["similar_stations"]) == 3
    assert data["confidence_interval"]["high"] > data["confidence_interval"]["low"]


def test_site_evaluate_invalid_coords(client):
    payload = {
        "lat": 999.0,
        "lng": -118.1720,
        "num_ports": 4,
        "location_type": "workplace"
    }
    response = client.post("/site/evaluate", json=payload)
    assert response.status_code == 422