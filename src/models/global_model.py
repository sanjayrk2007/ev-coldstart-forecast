import pandas as pd
import numpy as np
import lightgbm as lgb
import mlflow
import mlflow.lightgbm
from pathlib import Path
import pickle
import hashlib
from datetime import datetime

PROCESSED_DIR = Path(__file__).resolve().parents[2] / 'data' / 'processed'
MODEL_DIR     = Path(__file__).resolve().parents[2] / 'models'
MODEL_DIR.mkdir(exist_ok=True)

FEATURE_COLS = [
    'hour', 'day_of_week', 'month',
    'is_weekend', 'is_holiday',
    'lag_1h', 'lag_24h', 'lag_168h',
    'rolling_24h_mean', 'rolling_7d_mean',
    'site_encoded'          # station identity — coarse but generalises to new sites
]
TARGET_COL = 'sessions'

SITE_MAP = {
    'caltech': 0,
    'jpl': 1,
    'office001': 2,
    # Frontend dropdown mappings
    'office / workplace': 1,
    'office / corporate campus': 1,
    'university / campus': 0,
    'mixed-use commercial': 1,
    'public parking': 1,
    'public parking / municipal hub': 1,
    'highway / transit corridor': 1,
    'residential': 2,
}


def load_training_stations(sites=('caltech', 'jpl')):
    """
    Load all parquet files for the given sites, sort each station
    chronologically, fill NaNs, encode site, then concatenate.
    Crucially: sort within each station BEFORE concatenating so lag
    features never bleed across station boundaries.
    """
    frames = []
    for site in sites:
        files = sorted(PROCESSED_DIR.glob(f'{site}_*.parquet'))
        print(f"  {site}: {len(files)} stations")
        for f in files:
            df = pd.read_parquet(f)
            df = df.sort_values('timestamp').reset_index(drop=True)
            df['site_encoded'] = SITE_MAP[df['site'].iloc[0]]
            df[FEATURE_COLS[:-1]] = df[FEATURE_COLS[:-1]].fillna(0)
            frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    print(f"  Total rows: {len(combined):,}  |  Stations: {combined['station_id'].nunique()}")
    return combined


def train_global_model(df, params=None, num_boost_round=500):
    """
    Train a single LightGBM on all stations combined.
    Returns the trained booster object.
    """
    if params is None:
        params = {
            'objective':        'regression_l1',  # MAE loss — robust to zeros
            'learning_rate':    0.05,
            'num_leaves':       63,
            'min_child_samples': 20,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq':     5,
            'verbose':          -1,
        }

    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values

    dataset = lgb.Dataset(X, label=y, feature_name=FEATURE_COLS)

    print(f"  Training on {len(X):,} rows, {len(FEATURE_COLS)} features")
    booster = lgb.train(
        params,
        dataset,
        num_boost_round=num_boost_round,
    )
    return booster, params


def save_model(booster, path=None):
    if path is None:
        path = MODEL_DIR / 'global_model.pkl'
    with open(path, 'wb') as f:
        pickle.dump(booster, f)
    print(f"  Model saved to {path}")
    return path

def load_model(path=None):
    if path is None:
        path = MODEL_DIR / 'global_model.pkl'
    with open(path, 'rb') as f:
        return pickle.load(f)

# =============================================================================
# Site Selection Scoring — Phase 5.5
# Added in phase6_siteselection.ipynb
# =============================================================================

import json as _json
import numpy as _np
from pathlib import Path as _Path

_PROFILES_PATH = _Path("data/cache/station_profiles.json")

def _cosine_similarity(v1: list, v2: list) -> float:
    """
    Cosine similarity between two feature vectors.

    Why cosine over Euclidean: scale-invariant — captures the shape of
    the feature vector, not its magnitude. A candidate with 50 parking
    spots and a station with 10 parking spots still match on
    "parking-heavy location". For non-negative POI vectors this always
    returns a value in [0, 1].
    """
    a = _np.array(v1, dtype=float)
    b = _np.array(v2, dtype=float)
    norm_a = _np.linalg.norm(a)
    norm_b = _np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(_np.dot(a, b) / (norm_a * norm_b))

def find_similar_stations(
    candidate_vector: list,
    profiles_path: _Path = _PROFILES_PATH,
    top_k: int = 3,
) -> list:
    """
    Find the top-k most similar existing stations to a candidate location.

    Args:
        candidate_vector: 14-element feature vector from build_feature_vector()
        profiles_path:    path to station_profiles.json
        top_k:            number of similar stations to return (default 3)

    Returns:
        List of dicts sorted by similarity descending. Each dict contains:
        station_id, site, similarity, weight, hourly_stats,
        weekly_mean_sessions. Weights are normalized — sum to 1.0.
    """
    with open(profiles_path, "r") as f:
        profiles = _json.load(f)

    similarities = []
    for station_id, profile in profiles.items():
        sim = _cosine_similarity(candidate_vector, profile["feature_vector"])
        similarities.append({
            "station_id": station_id,
            "site": profile["site"],
            "similarity": sim,
            "hourly_stats": profile["hourly_stats"],
            "weekly_mean_sessions": profile["weekly_mean_sessions"],
        })

    similarities.sort(key=lambda x: x["similarity"], reverse=True)
    top = similarities[:top_k]

    total_sim = sum(s["similarity"] for s in top)
    if total_sim == 0:
        for s in top:
            s["weight"] = 1.0 / len(top)
    else:
        for s in top:
            s["weight"] = s["similarity"] / total_sim

    return top

def build_synthetic_profile(
    similar_stations: list,
    site_encoded: int = 0,
) -> "pd.DataFrame":
    """
    Build a 168-row synthetic feature DataFrame representing a typical week.

    Each row = one hour of a typical week (Mon 00:00 to Sun 23:00).
    Lag features are filled using hour-matched weighted averages from the
    top-k similar stations — preserves temporal structure (low lags at 3am,
    high lags at 9am on weekdays) rather than flattening to a global mean.

    Why hour-matched weighted average:
    Zero-fill biases predictions toward zero-demand hours (abandoned station).
    Global mean-fill flattens temporal structure.
    Hour-matched weighted average preserves the actual demand shape.

    Args:
        similar_stations: output of find_similar_stations()
        site_encoded:     integer site encoding (0=caltech, 1=jpl, 2=office001)
                          Use 0 for new candidates (closest training distribution)

    Returns:
        DataFrame with 168 rows and all features the global model expects.
    """
    import pandas as _pd
    import holidays as _holidays
    from datetime import date as _date

    us_holidays = _holidays.US()
    rows = []

    for hour_of_week in range(168):
        day_of_week = hour_of_week // 24
        hour = hour_of_week % 24
        month = datetime.now().month
        is_weekend = int(day_of_week >= 5)
        # June 3 2024 was a Monday — offset gives Mon through Sun
        ref_date = _date(2024, 6, 3 + day_of_week)
        is_holiday = int(ref_date in us_holidays)

        # Hour-matched weighted average of lag features across top-k stations
        lag_1h = lag_24h = lag_168h = rolling_24h = rolling_7d = 0.0

        for station in similar_stations:
            w = station["weight"]
            # hourly_stats keys are strings in JSON
            stats = station["hourly_stats"].get(str(hour_of_week), {})
            lag_1h      += w * stats.get("mean_lag_1h", 0.0)
            lag_24h     += w * stats.get("mean_lag_24h", 0.0)
            lag_168h    += w * stats.get("mean_lag_168h", 0.0)
            rolling_24h += w * stats.get("mean_rolling_24h", 0.0)
            rolling_7d  += w * stats.get("mean_rolling_7d", 0.0)

        rows.append({
            "hour": hour,
            "day_of_week": day_of_week,
            "month": month,
            "is_weekend": is_weekend,
            "is_holiday": is_holiday,
            "lag_1h": lag_1h,
            "lag_24h": lag_24h,
            "lag_168h": lag_168h,
            "rolling_24h_mean": rolling_24h,
            "rolling_7d_mean": rolling_7d,
            "site_encoded": site_encoded,
            "hour_of_week": hour_of_week,  # for output alignment only
        })

    return _pd.DataFrame(rows)

def score_candidate(
    lat: float,
    lng: float,
    num_ports: int,
    location_type: str = "workplace",
    model_path: str = "models/global_model.pkl",
    profiles_path: str = "data/cache/station_profiles.json",
    cache_path: str = "data/cache/poi_cache.json",
    site_encoded: int = 0,
    top_k: int = 3,
) -> dict:
    """
    Full site selection scoring pipeline for a candidate location.

    Dependency chain:
        lat/lng
            → POI features (hardcoded for Caltech/JPL, live query otherwise)
            → cosine similarity against all 107 station profiles
            → top-k similar stations with normalized weights
            → hour-matched weighted synthetic profile (168 rows)
            → global model inference → point predictions
            → weekly demand profile output

    Note on uncertainty:
        This function returns point predictions only. For calibrated
        conformal intervals, pass the weekly_profile through
        uncertainty.predict_with_intervals() after calling this function.

    Args:
        lat, lng:         Candidate location coordinates.
        num_ports:        Planned number of charging ports.
        location_type:    "workplace", "public", or "retail".
        model_path:       Path to saved global model pickle.
        profiles_path:    Path to station_profiles.json.
        cache_path:       Path to POI cache JSON.
        site_encoded:     Site encoding for synthetic profile (default 0).
        top_k:            Number of similar stations to use (default 3).

    Returns:
        dict with keys:
            weekly_profile:     DataFrame, 168 rows
            weekly_total:       float, predicted sessions over full week
            similar_stations:   list of top-k similarity dicts
            poi_features:       dict of 14 POI features for candidate
            candidate:          dict with lat, lng, num_ports, location_type
    """
    import pickle as _pickle
    import sys as _sys

    _src_data = str(_Path(model_path).parent.parent / "src" / "data")
    if _src_data not in _sys.path:
        _sys.path.insert(0, _src_data)
    from poi_features import get_poi_features, build_feature_vector

    # Step 1: POI features
    print("[score_candidate] Step 1/4: Fetching POI features...")
    poi_feats = get_poi_features(
        lat=lat, lng=lng,
        num_ports=num_ports,
        location_type=location_type,
        cache_path=_Path(cache_path),
        verbose=True,
    )
    candidate_vector = build_feature_vector(poi_feats)

    # Step 2: Station similarity
    print("[score_candidate] Step 2/4: Computing station similarity...")
    similar = find_similar_stations(
        candidate_vector=candidate_vector,
        profiles_path=_Path(profiles_path),
        top_k=top_k,
    )
    for s in similar:
        print(f"  {s['station_id']} ({s['site']}): "
              f"similarity={s['similarity']:.4f}, weight={s['weight']:.4f}")

    # Step 3: Synthetic weekly profile
    print("[score_candidate] Step 3/4: Building synthetic weekly profile...")
    profile_df = build_synthetic_profile(
        similar_stations=similar,
        site_encoded=site_encoded,
    )

    # Step 4: Global model inference
    print("[score_candidate] Step 4/4: Running global model inference...")
    with open(model_path, "rb") as f:
        model = _pickle.load(f)

    feature_cols = [
        "hour", "day_of_week", "month", "is_weekend", "is_holiday",
        "lag_1h", "lag_24h", "lag_168h", "rolling_24h_mean",
        "rolling_7d_mean", "site_encoded",
    ]
    predictions = _np.clip(model.predict(profile_df[feature_cols]), 0, None)

    # ------------------------------------------------------------------
    # Location-specific demand scaling
    # ------------------------------------------------------------------
    # The global model was trained on 107 stations at just 2 sites (Caltech,
    # JPL), so its predictions capture temporal patterns but not location-
    # specific demand magnitude.  Two adjustments differentiate candidates:
    #
    # 1. num_ports: more ports → more capacity → proportionally more sessions.
    #    Baseline is 2 ports (the training median).
    #
    # 2. Coordinate + location-type hash: a deterministic scalar in [0.7, 1.3]
    #    that encodes local demand potential not captured by the POI similarity
    #    (which collapses to 2 groups because training profiles share only 2
    #    unique feature vectors).
    # ------------------------------------------------------------------
    # sqrt scaling: diminishing returns per port (heuristic, not empirically validated)
    port_factor = (max(num_ports, 1) ** 0.5) / (2.0 ** 0.5)

    location_demand_baseline = {"workplace": 1.0, "public": 1.15, "retail": 1.25}
    type_factor = location_demand_baseline.get(location_type, 1.0)

    # NOTE: No coordinate-based multiplier is applied here.
    # Only 2 training sites exist (Caltech, JPL), so the POI similarity
    # collapses to 2 groups. Adding a hash-based ±30% multiplier would
    # introduce arbitrary noise that dominates real signal — any
    # differentiation must come from better POI features or more
    # training sites, not from coordinate hashing.

    predictions = predictions * port_factor * type_factor

    import pandas as _pd
    weekly_profile = profile_df[["hour_of_week", "hour", "day_of_week"]].copy()
    weekly_profile["predicted_sessions"] = predictions

    return {
        "weekly_profile": weekly_profile,
        "weekly_total": float(predictions.sum()),
        "similar_stations": similar,
        "poi_features": poi_feats,
        "candidate": {
            "lat": lat, "lng": lng,
            "num_ports": num_ports,
            "location_type": location_type,
        },
    }