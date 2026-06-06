"""
main.py — FastAPI application for the EV Cold-Start Forecasting system.

Four endpoints:
    POST /forecast/cold-start          — 168-hour forecast for a new station
    GET  /forecast/stations/{id}       — 168-hour forecast for a known station
    POST /site/evaluate                — site scoring for a candidate location
    GET  /health                       — model status and version

All ML logic lives in predictor.py. This file handles only routing and errors.
"""

from __future__ import annotations
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api import predictor
from src.api.schemas import (
    ColdStartRequest, ColdStartResponse,
    SiteEvaluateRequest, SiteEvaluateResponse,
    HealthResponse,
)


# ---------------------------------------------------------------------------
# Lifespan — load all artifacts once before serving any request
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    predictor.startup()
    yield


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="EV Cold-Start Forecasting API",
    description=(
        "Predicts hourly EV charging demand for newly installed stations "
        "with zero usage history. Supports transfer learning fine-tuning, "
        "conformal prediction intervals, and pre-installation site scoring."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Allow React frontend (localhost:5173 is Vite's default dev port)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    """Returns model load status and version. Use this to verify the API is ready."""
    return predictor.get_health()


@app.post("/forecast/cold-start", response_model=ColdStartResponse, tags=["Forecast"])
def cold_start(request: ColdStartRequest):
    """
    Generate a one-week (168-hour) forecast for a newly installed station.

    - If `sessions` is omitted or empty: uses the global model directly (pure cold start).
    - If `sessions` is provided: fine-tunes the global model on the sparse history
      before forecasting. More sessions → better calibration.

    All forecasts include 80% and 90% conformal prediction intervals.
    """
    try:
        return predictor.run_cold_start(request)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forecast failed: {str(e)}")


@app.get("/forecast/stations/{station_id}", response_model=ColdStartResponse, tags=["Forecast"])
def station_forecast(station_id: str):
    """
    Generate a one-week forecast for a known training station.

    Uses the station's actual parquet history as context. Useful for
    validating model performance against stations with known demand patterns.
    """
    try:
        return predictor.run_station_forecast(station_id)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forecast failed: {str(e)}")


@app.post("/site/evaluate", response_model=SiteEvaluateResponse, tags=["Site Selection"])
def site_evaluate(request: SiteEvaluateRequest):
    """
    Score a candidate location for EV charging demand potential.

    Does not require any installed station or session history — purely
    location-based scoring using OpenStreetMap POI features and cosine
    similarity against 107 training stations.

    Returns a business-readable demand assessment with conformal intervals
    and the top-3 most similar reference stations.
    """
    try:
        return predictor.run_site_evaluate(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Site evaluation failed: {str(e)}")