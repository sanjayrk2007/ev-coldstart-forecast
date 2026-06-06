"""
predictor.py — ML bridge for the FastAPI layer.

Loads all model artifacts once at startup and exposes four clean functions
to main.py. No ML logic should live in main.py — only here.

Startup artifacts:
    - models/global_model.pkl        (LightGBM booster)
    - models/calibration.json        (conformal quantiles via load_calibration)
    - data/cache/station_profiles.json (107 training station profiles)
"""

from __future__ import annotations

import json
import pickle
import sys
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup — ensure src/ is importable regardless of working directory
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.uncertainty import load_calibration, predict_with_intervals
from src.models.transfer import fine_tune, predict, FEATURE_COLS, SITE_MAP
from src.api.schemas import (
    ColdStartRequest, ColdStartResponse,
    SiteEvaluateRequest, SiteEvaluateResponse,
    HourlyForecast, SimilarStation, ConfidenceInterval,
    HealthResponse,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MODEL_PATH      = PROJECT_ROOT / "models" / "global_model.pkl"
CALIB_PATH      = PROJECT_ROOT / "models" / "calibration.json"
PROFILES_PATH   = PROJECT_ROOT / "data" / "cache" / "station_profiles.json"
PROCESSED_DIR   = PROJECT_ROOT / "data" / "processed"

# ---------------------------------------------------------------------------
# Demand tier thresholds (derived from training data distribution)
# Min=0.6, p25=29.1, Median=34.2, p75=43.4, Max=65.8
# ---------------------------------------------------------------------------
TIER_LOW      = 20.0   # below this → Low
TIER_MODERATE = 35.0   # below this → Moderate
TIER_HIGH     = 50.0   # below this → High
                        # above      → Very High

# ---------------------------------------------------------------------------
# Module-level state — populated once at startup
# ---------------------------------------------------------------------------
_global_booster = None
_station_profiles: dict = {}
_model_version: str = "unknown"
_calibration_loaded: bool = False
_global_model_loaded: bool = False


def startup() -> None:
    """
    Load all artifacts into memory. Call this once from main.py lifespan.
    Raises on any missing artifact so the server fails fast rather than
    serving broken predictions.
    """
    global _global_booster, _station_profiles
    global _model_version, _calibration_loaded, _global_model_loaded

    # 1. Global LightGBM model
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Global model not found: {MODEL_PATH}")
    with open(MODEL_PATH, "rb") as f:
        _global_booster = pickle.load(f)
    _model_version = MODEL_PATH.stat().st_mtime.__str__()[:10]
    _global_model_loaded = True
    print(f"[predictor] Global model loaded from {MODEL_PATH}")

    # 2. Conformal calibration quantiles
    if not CALIB_PATH.exists():
        raise FileNotFoundError(f"Calibration file not found: {CALIB_PATH}")
    load_calibration(str(CALIB_PATH))
    _calibration_loaded = True
    print(f"[predictor] Calibration quantiles loaded from {CALIB_PATH}")

    # 3. Station profiles
    if not PROFILES_PATH.exists():
        raise FileNotFoundError(f"Station profiles not found: {PROFILES_PATH}")
    with open(PROFILES_PATH) as f:
        _station_profiles = json.load(f)
    print(f"[predictor] {len(_station_profiles)} station profiles loaded")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _demand_tier(weekly_sessions: float) -> str:
    if weekly_sessions < TIER_LOW:
        return "Low"
    elif weekly_sessions < TIER_MODERATE:
        return "Moderate"
    elif weekly_sessions < TIER_HIGH:
        return "High"
    else:
        return "Very High"


def _recommendation(tier: str, interval_width: float) -> str:
    """
    Combine demand tier and interval width into a operator recommendation.
    Wide intervals (high uncertainty) downgrade the recommendation by one level.
    """
    wide = interval_width > 40.0  # sessions/week — adjust if needed
    if tier == "Very High":
        return "Strong candidate" if not wide else "Moderate candidate"
    elif tier == "High":
        return "Moderate candidate" if not wide else "Moderate candidate"
    elif tier == "Moderate":
        return "Moderate candidate" if not wide else "Weak candidate"
    else:
        return "Weak candidate"


def _roi_signal(weekly: float, low: float, high: float, tier: str) -> str:
    return (
        f"This location is projected to see ~{weekly:.0f} charging sessions per week "
        f"({tier} demand). Based on calibrated uncertainty analysis, the realistic range "
        f"is {low:.0f}–{high:.0f} sessions/week. "
        f"At an average of 30 minutes per session, that translates to roughly "
        f"{weekly * 0.5:.0f} hours of active charger use per week."
    )


def _sessions_to_hourly_df(
    sessions: list,
    site_encoded: int = 2,
) -> pd.DataFrame:
    """
    Convert a list of SessionRecord objects into an hourly demand DataFrame
    with all 11 FEATURE_COLS populated.

    Each session contributes +1 to each hour it overlaps with. Lag and
    rolling features are computed from the resulting hourly series.
    """
    if not sessions:
        return pd.DataFrame(columns=FEATURE_COLS + ["timestamp", "sessions"])

    # Build hourly index spanning all sessions
    min_dt = min(s.start_time for s in sessions).replace(
        minute=0, second=0, microsecond=0, tzinfo=None)
    max_dt = max(s.end_time for s in sessions).replace(
        minute=0, second=0, microsecond=0, tzinfo=None)

    idx = pd.date_range(min_dt, max_dt, freq="h")
    demand = pd.Series(0.0, index=idx)

    for s in sessions:
        start = s.start_time.replace(tzinfo=None).replace(minute=0, second=0, microsecond=0)
        end   = s.end_time.replace(tzinfo=None).replace(minute=0, second=0, microsecond=0)
        hours = pd.date_range(start, end, freq="h")
        for h in hours:
            if h in demand.index:
                demand[h] += 1.0

    df = pd.DataFrame({"timestamp": idx, "sessions": demand.values})
    df["hour"]         = df["timestamp"].dt.hour
    df["day_of_week"]  = df["timestamp"].dt.dayofweek
    df["month"]        = df["timestamp"].dt.month
    df["is_weekend"]   = (df["day_of_week"] >= 5).astype(int)
    df["is_holiday"]   = 0  # conservative default; holidays library not called at inference
    df["lag_1h"]       = df["sessions"].shift(1).fillna(0)
    df["lag_24h"]      = df["sessions"].shift(24).fillna(0)
    df["lag_168h"]     = df["sessions"].shift(168).fillna(0)
    df["rolling_24h_mean"] = df["sessions"].shift(1).rolling(24,  min_periods=1).mean().fillna(0)
    df["rolling_7d_mean"]  = df["sessions"].shift(1).rolling(168, min_periods=1).mean().fillna(0)
    df["site_encoded"] = site_encoded

    return df


def _build_forecast_df(base_time: Optional[datetime] = None, site_encoded: int = 2) -> pd.DataFrame:
    """
    Build a 168-row feature DataFrame for a one-week ahead forecast
    starting from base_time (defaults to next Monday 00:00 UTC).
    Lag and rolling features are zeroed — appropriate for cold start.
    """
    if base_time is None:
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        days_ahead = (7 - now.weekday()) % 7 or 7
        base_time = (now + timedelta(days=days_ahead)).replace(
            hour=0, minute=0, second=0, microsecond=0)

    idx = pd.date_range(base_time, periods=168, freq="h")
    df = pd.DataFrame({"timestamp": idx})
    df["hour"]             = df["timestamp"].dt.hour
    df["day_of_week"]      = df["timestamp"].dt.dayofweek
    df["month"]            = df["timestamp"].dt.month
    df["is_weekend"]       = (df["day_of_week"] >= 5).astype(int)
    df["is_holiday"]       = 0
    df["lag_1h"]           = 0.0
    df["lag_24h"]          = 0.0
    df["lag_168h"]         = 0.0
    df["rolling_24h_mean"] = 0.0
    df["rolling_7d_mean"]  = 0.0
    df["site_encoded"]     = site_encoded
    return df


def _preds_to_forecast(preds_df: pd.DataFrame, timestamps: pd.Series) -> list[HourlyForecast]:
    """Convert predict_with_intervals() output DataFrame to list of HourlyForecast."""
    return [
        HourlyForecast(
            timestamp=pd.Timestamp(ts).to_pydatetime().replace(tzinfo=timezone.utc),
            predicted=float(preds_df["point_estimate"].iloc[i]),
            lower_80=float(preds_df["lower_80"].iloc[i]),
            upper_80=float(preds_df["upper_80"].iloc[i]),
            lower_90=float(preds_df["lower_90"].iloc[i]),
            upper_90=float(preds_df["upper_90"].iloc[i]),
        )
        for i, ts in enumerate(timestamps)
    ]

# ---------------------------------------------------------------------------
# Public API — called by main.py
# ---------------------------------------------------------------------------

def get_health() -> HealthResponse:
    return HealthResponse(
        status="ok" if (_global_model_loaded and _calibration_loaded) else "degraded",
        model_version=_model_version,
        calibration_loaded=_calibration_loaded,
        global_model_loaded=_global_model_loaded,
    )


def run_cold_start(request: ColdStartRequest) -> ColdStartResponse:
    """
    Generate a 168-hour forecast for a new station.

    If sessions are provided:
        1. Convert sessions → hourly demand DataFrame
        2. Fine-tune global booster on that DataFrame
        3. Build 168-row forecast DataFrame
        4. Run predict_with_intervals() on fine-tuned booster
    If no sessions:
        1. Build 168-row forecast DataFrame (zero lags)
        2. Run predict_with_intervals() on global booster directly
    """
    site_encoded = SITE_MAP.get(request.site, 2)  # unknown sites → office001 encoding
    fine_tuned = False

    forecast_df = _build_forecast_df(site_encoded=site_encoded)

    if request.sessions and len(request.sessions) > 0:
        train_df = _sessions_to_hourly_df(request.sessions, site_encoded=site_encoded)
        if len(train_df) >= 2:
            booster = fine_tune(_global_booster, train_df)
            fine_tuned = True
        else:
            booster = _global_booster
    else:
        booster = _global_booster

    raw_preds = predict(booster, forecast_df)
    preds_df = predict_with_intervals(raw_preds, method="conditional")

    forecast = _preds_to_forecast(preds_df, forecast_df["timestamp"])
    return ColdStartResponse(
        station_id=request.station_id,
        model_version=_model_version,
        fine_tuned=fine_tuned,
        forecast=forecast,
    )


def run_station_forecast(station_id: str) -> ColdStartResponse:
    """
    Generate a 168-hour forecast for a known training station
    using its actual parquet history as context.
    """
    # Find station in profiles to get its site
    if station_id not in _station_profiles:
        raise KeyError(f"Station '{station_id}' not found in training profiles.")

    profile = _station_profiles[station_id]
    site = profile.get("site", "caltech")
    site_encoded = SITE_MAP.get(site, 0)

    parquet_path = PROCESSED_DIR / f"{station_id}.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet file not found for station: {station_id}")

    df = pd.read_parquet(parquet_path)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["site_encoded"] = site_encoded
    df[FEATURE_COLS[:-1]] = df[FEATURE_COLS[:-1]].fillna(0)

    # Use last 168 rows as the forecast window
    forecast_df = df.tail(168).copy().reset_index(drop=True)

    raw_preds = predict(_global_booster, forecast_df)
    preds_df = predict_with_intervals(raw_preds, method="conditional")
    forecast = _preds_to_forecast(preds_df, forecast_df["timestamp"])
    

    return ColdStartResponse(
        station_id=station_id,
        model_version=_model_version,
        fine_tuned=False,
        forecast=forecast,
    )


def run_site_evaluate(request: SiteEvaluateRequest) -> SiteEvaluateResponse:
    """
    Score a candidate location for EV charging demand potential.

    Calls score_candidate() from global_model.py (which re-loads the pkl
    internally — acceptable since this is a heavier analytical endpoint,
    not a low-latency prediction endpoint).

    Builds the business-readable response from the raw scoring output.
    """
    # Import here to avoid circular imports at module load time
    from src.models.global_model import score_candidate

    raw = score_candidate(
        lat=request.lat,
        lng=request.lng,
        num_ports=request.num_ports,
        location_type=request.location_type,
        model_path=str(MODEL_PATH),
        profiles_path=str(PROFILES_PATH),
        cache_path=str(PROJECT_ROOT / "data" / "cache" / "poi_cache.json"),
        site_encoded=2,
        top_k=3,
    )

    weekly_total: float = raw["weekly_total"]

    hourly_preds = raw["weekly_profile"]["predicted_sessions"].values
    intervals = predict_with_intervals(hourly_preds, method="conditional")
    weekly_low  = float(np.sum(intervals["lower_80"].values))
    weekly_high = float(np.sum(intervals["upper_80"].values))
    tier = _demand_tier(weekly_total)
    interval_width = weekly_high - weekly_low

    similar = [
        SimilarStation(
            station_id=s["station_id"],
            site=s["site"],
            weekly_mean_sessions=float(_station_profiles.get(
                s["station_id"], {}).get("weekly_mean_sessions", 0.0)),
            similarity_score=float(s["similarity"]),
        )
        for s in raw["similar_stations"]
    ]

    return SiteEvaluateResponse(
        predicted_weekly_sessions=round(weekly_total, 1),
        confidence_interval=ConfidenceInterval(
            low=round(weekly_low, 1),
            high=round(weekly_high, 1),
        ),
        demand_tier=tier,
        roi_signal=_roi_signal(weekly_total, weekly_low, weekly_high, tier),
        recommendation=_recommendation(tier, interval_width),
        similar_stations=similar,
        model_version=_model_version,
    )