from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------

class SessionRecord(BaseModel):
    """A single observed charging session at the new station."""
    start_time: datetime = Field(..., description="Session start (ISO 8601, UTC)")
    end_time: datetime   = Field(..., description="Session end (ISO 8601, UTC)")
    energy_kwh: float    = Field(..., ge=0, description="Energy delivered in kWh")


class HourlyForecast(BaseModel):
    """Point forecast + conformal intervals for one hour."""
    timestamp:  datetime = Field(..., description="Hour start (ISO 8601, UTC)")
    predicted:  float    = Field(..., description="Predicted sessions this hour")
    lower_80:   float    = Field(..., description="80% interval lower bound")
    upper_80:   float    = Field(..., description="80% interval upper bound")
    lower_90:   float    = Field(..., description="90% interval lower bound")
    upper_90:   float    = Field(..., description="90% interval upper bound")


class SimilarStation(BaseModel):
    """A training station similar to the candidate, used as a reference point."""
    station_id:           str   = Field(..., description="Station identifier")
    site:                 str   = Field(..., description="Site name (caltech or jpl)")
    weekly_mean_sessions: float = Field(..., description="Actual mean weekly sessions")
    similarity_score:     float = Field(..., description="Cosine similarity to candidate (0–1)")


class ConfidenceInterval(BaseModel):
    """80% conformal prediction interval on a weekly demand estimate."""
    low:  float = Field(..., description="Lower bound (sessions/week)")
    high: float = Field(..., description="Upper bound (sessions/week)")


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class ColdStartRequest(BaseModel):
    """
    Request for a one-week hourly forecast at a newly installed station.
    Sparse session history is optional — if omitted, the global model
    is used directly without fine-tuning.
    """
    # Station identity
    station_id:   str   = Field(..., description="Caller-assigned station identifier")
    site:         str   = Field("unknown", description="Site type: 'office', 'campus', 'public', 'unknown'")
    lat:          float = Field(..., ge=-90,  le=90,  description="Latitude")
    lng:          float = Field(..., ge=-180, le=180, description="Longitude")
    num_ports:    int   = Field(2,   ge=1,            description="Number of charging ports")

    # Optional sparse history for fine-tuning
    sessions: Optional[list[SessionRecord]] = Field(
        default=None,
        description="Observed sessions since installation. Omit for zero-history cold start."
    )

    class Config:
        json_schema_extra = {
            "example": {
                "station_id": "my-station-001",
                "site": "office",
                "lat": 34.2020,
                "lng": -118.1720,
                "num_ports": 4,
                "sessions": [
                    {
                        "start_time": "2024-03-01T09:00:00Z",
                        "end_time":   "2024-03-01T11:30:00Z",
                        "energy_kwh": 18.5
                    }
                ]
            }
        }


class SiteEvaluateRequest(BaseModel):
    """
    Request to score a candidate location before any station is installed.
    No session history — purely location and site characteristics.
    """
    lat:           float = Field(..., ge=-90,  le=90,  description="Latitude")
    lng:           float = Field(..., ge=-180, le=180, description="Longitude")
    num_ports:     int   = Field(2,   ge=1,            description="Number of charging ports planned")
    location_type: str   = Field("office", description="Location type: 'office', 'campus', 'public', 'residential'")

    class Config:
        json_schema_extra = {
            "example": {
                "lat": 34.2020,
                "lng": -118.1720,
                "num_ports": 4,
                "location_type": "office"
            }
        }


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class ColdStartResponse(BaseModel):
    """One-week hourly forecast (168 hours) with conformal intervals."""
    station_id:    str                 = Field(..., description="Echo of request station_id")
    model_version: str                 = Field(..., description="Model artifact identifier")
    fine_tuned:    bool                = Field(..., description="True if sparse history was used for fine-tuning")
    forecast:      list[HourlyForecast] = Field(..., description="168 hourly predictions")


class SiteEvaluateResponse(BaseModel):
    """
    Site scoring result — designed to be readable by non-technical stakeholders.
    All numbers are weekly totals. Intervals are 80% conformal prediction intervals.
    """
    # Core numbers
    predicted_weekly_sessions: float             = Field(..., description="Predicted charging sessions per week")
    confidence_interval:       ConfidenceInterval = Field(..., description="80% interval on weekly sessions")

    # Business-readable signals
    demand_tier:    str = Field(..., description="Low / Moderate / High / Very High")
    roi_signal:     str = Field(..., description="Plain-English demand summary for operators")
    recommendation: str = Field(..., description="Strong candidate / Moderate candidate / Weak candidate")

    # Evidence
    similar_stations: list[SimilarStation] = Field(
        ..., description="Top-3 most similar training stations with actual demand history"
    )

    # Metadata
    model_version:  str = Field(..., description="Model artifact identifier")
    note:           str = Field(
        default="Model trained on US university/office campus data. Confidence intervals widen for non-US or retail locations.",
        description="Honest limitation notice"
    )


class HealthResponse(BaseModel):
    """API health and model status."""
    status:              str  = Field(..., description="'ok' or 'degraded'")
    model_version:       str  = Field(..., description="global_model.pkl artifact identifier")
    calibration_loaded:  bool = Field(..., description="True if conformal quantiles are loaded")
    global_model_loaded: bool = Field(..., description="True if LightGBM model is in memory")