import numpy as np
import pandas as pd
import pickle
import glob
import os
import sys
import mlflow

# ── Conformity score store ──────────────────────────────────────────────────
# After calibrate() runs, these module-level variables hold the quantiles.
# Any call to predict_with_intervals() uses them automatically.
_q80: float = None
_q90: float = None
_residuals: np.ndarray = None


# ── Feature columns (must match transfer.py exactly) ───────────────────────
FEATURE_COLS = [
    "hour", "day_of_week", "month", "is_weekend", "is_holiday",
    "lag_1h", "lag_24h", "lag_168h", "rolling_24h_mean", "rolling_7d_mean",
    "site_encoded",
]
TARGET_COL = "sessions"


def _temporal_split(df: pd.DataFrame, timestamp_col: str, train_frac: float = 0.70):
    """
    Split a station's dataframe chronologically.
    Returns (calibration_df, evaluation_df).

    Why chronological and not random?
    Random splits leak future information into calibration — the model would
    be calibrated on data it effectively saw during training. Chronological
    split preserves the causal direction: calibrate on past, evaluate on future.
    """
    df = df.sort_values(timestamp_col).reset_index(drop=True)
    cutoff = int(len(df) * train_frac)
    return df.iloc[:cutoff], df.iloc[cutoff:]


def _load_office001_files(processed_dir: str):
    """
    Load all office001 parquet files. Returns a list of DataFrames,
    one per station.
    """
    pattern = os.path.join(processed_dir, "office001_*.parquet")
    files = sorted(glob.glob(pattern))

    if not files:
        raise FileNotFoundError(
            f"No office001 parquet files found at {processed_dir}. "
            "Check that processed_dir points to your data/processed/ folder."
        )

    dfs = []
    for f in files:
        df = pd.read_parquet(f)
        df["_source_file"] = os.path.basename(f)
        dfs.append(df)

    print(f"Loaded {len(dfs)} office001 stations.")
    return dfs


def calibrate(
    global_model_path: str,
    processed_dir: str,
    site_encoding: int = 2,
    train_frac: float = 0.70,
):
    """
    Build conformity scores from the first train_frac of each office001
    station's timeline. Stores q80 and q90 as module-level variables.

    Parameters
    ----------
    global_model_path : str
        Path to your saved global LightGBM model (models/global_model.pkl).
    processed_dir : str
        Path to data/processed/ where office001 parquets live.
    site_encoding : int
        The integer encoding for office001 in your site_encoded column.
        Check what value your features.py assigned — likely 0, 1, or 2.
    train_frac : float
        Fraction of each station's timeline used for calibration (default 0.70).
        Remaining (1 - train_frac) is used for coverage evaluation.

    Why site_encoding matters:
        Your transfer model uses site_encoded as a feature. If office001 was
        encoded as 2 during feature engineering, we pass that value. If the
        encoding is wrong, the model sees an unknown site identity and residuals
        will be inflated. Verify this against your features.py label encoder.
    """
    global _q80, _q90, _residuals

    # 1. Load global model
    with open(global_model_path, "rb") as f:
        booster = pickle.load(f)

    print(f"Global model loaded from {global_model_path}")

    # 2. Load office001 stations
    station_dfs = _load_office001_files(processed_dir)

    # 3. For each station, take calibration split, predict, collect residuals
    all_residuals = []

    for df in station_dfs:
        station_id = df["station_id"].iloc[0]

        # Ensure site_encoded exists — office001 may not have it if features.py
        # was run before transfer learning added this column
        if "site_encoded" not in df.columns:
            df = df.copy()
            df["site_encoded"] = site_encoding

        calib_df, _ = _temporal_split(df, "timestamp", train_frac)

        # Drop rows with NaN in any feature (lag features create NaNs at start)
        calib_df = calib_df.dropna(subset=FEATURE_COLS + [TARGET_COL])

        if len(calib_df) < 168:
            # Less than one week of clean calibration rows — skip this station
            # 168 = hours in a week, minimum for lag_168h to be meaningful
            print(f"  Skipping {station_id} — insufficient calibration rows ({len(calib_df)})")
            continue

        X_calib = calib_df[FEATURE_COLS].values
        y_true = calib_df[TARGET_COL].values
        y_pred = booster.predict(X_calib)

        # Conformity scores = absolute residuals
        # Why absolute? Because conformal prediction is symmetric by default —
        # we care about magnitude of error, not direction.
        residuals = np.abs(y_true - y_pred)
        all_residuals.extend(residuals)

        print(f"  {station_id}: {len(calib_df)} calibration rows, "
              f"mean residual = {residuals.mean():.4f}")

    if not all_residuals:
        raise ValueError("No valid calibration data found. Check your office001 parquet files.")

    _residuals = np.array(all_residuals)

    # 4. Compute quantiles
    # Why (n+1)/n adjustment?
    # Naive percentile would give slightly overconfident intervals on small n.
    # The (1 - alpha)(1 + 1/n) quantile is the finite-sample correction from
    # conformal prediction theory. With ~50k+ residuals here, the adjustment
    # is negligible — but we include it for correctness.
    n = len(_residuals)
    alpha_80 = 0.20
    alpha_90 = 0.10

    q80_level = min((1 - alpha_80) * (1 + 1 / n), 1.0)
    q90_level = min((1 - alpha_90) * (1 + 1 / n), 1.0)

    _q80 = float(np.quantile(_residuals, q80_level))
    _q90 = float(np.quantile(_residuals, q90_level))

    print(f"\nCalibration complete.")
    print(f"  Total conformity scores: {n:,}")
    print(f"  q80 (80% interval half-width): {_q80:.4f} sessions")
    print(f"  q90 (90% interval half-width): {_q90:.4f} sessions")

    return {
        "n_conformity_scores": n,
        "q80": _q80,
        "q90": _q90,
        "mean_residual": float(_residuals.mean()),
        "median_residual": float(np.median(_residuals)),
    }


def predict_with_intervals(point_predictions: np.ndarray) -> pd.DataFrame:
    """
    Wrap point predictions with calibrated conformal intervals.

    Parameters
    ----------
    point_predictions : np.ndarray
        Array of point forecast values from your transfer model's predict().

    Returns
    -------
    pd.DataFrame with columns:
        point_estimate, lower_80, upper_80, lower_90, upper_90

    Why clip at zero?
        EV demand cannot be negative. A lower bound of -0.3 sessions is
        physically meaningless. We clip lower bounds at 0.
    """
    if _q80 is None or _q90 is None:
        raise RuntimeError(
            "Model not calibrated. Call calibrate() before predict_with_intervals()."
        )

    preds = np.array(point_predictions)

    return pd.DataFrame({
        "point_estimate": preds,
        "lower_80": np.clip(preds - _q80, 0, None),
        "upper_80": preds + _q80,
        "lower_90": np.clip(preds - _q90, 0, None),
        "upper_90": preds + _q90,
    })


def evaluate_coverage(
    global_model_path: str,
    processed_dir: str,
    site_encoding: int = 2,
    train_frac: float = 0.70,
) -> pd.DataFrame:
    """
    Check that calibrated intervals achieve claimed coverage on the held-out
    evaluation split (last 1 - train_frac of each office001 station).

    Returns a DataFrame with coverage at each alpha level, per station and
    overall. This is what you plot as the calibration curve.

    What "coverage" means:
        80% coverage = the true value fell inside the 80% interval in 80% of
        hours. If actual coverage is 75%, your intervals are slightly too
        narrow (overconfident). If it's 90%, they're too wide (conservative).
        Either way, document the gap honestly.
    """
    if _q80 is None or _q90 is None:
        raise RuntimeError(
            "Model not calibrated. Call calibrate() before evaluate_coverage()."
        )

    with open(global_model_path, "rb") as f:
        booster = pickle.load(f)

    station_dfs = _load_office001_files(processed_dir)
    results = []

    for df in station_dfs:
        station_id = df["station_id"].iloc[0]

        if "site_encoded" not in df.columns:
            df = df.copy()
            df["site_encoded"] = site_encoding

        _, eval_df = _temporal_split(df, "timestamp", train_frac)
        eval_df = eval_df.dropna(subset=FEATURE_COLS + [TARGET_COL])

        if len(eval_df) < 24:
            print(f"  Skipping {station_id} evaluation — too few rows ({len(eval_df)})")
            continue

        X_eval = eval_df[FEATURE_COLS].values
        y_true = eval_df[TARGET_COL].values
        y_pred = booster.predict(X_eval)

        intervals = predict_with_intervals(y_pred)

        cov_80 = float(np.mean(
            (y_true >= intervals["lower_80"].values) &
            (y_true <= intervals["upper_80"].values)
        ))
        cov_90 = float(np.mean(
            (y_true >= intervals["lower_90"].values) &
            (y_true <= intervals["upper_90"].values)
        ))

        results.append({
            "station_id": station_id,
            "eval_rows": len(eval_df),
            "coverage_80": round(cov_80, 4),
            "coverage_90": round(cov_90, 4),
            "gap_80": round(cov_80 - 0.80, 4),  # positive = conservative, negative = overconfident
            "gap_90": round(cov_90 - 0.90, 4),
        })

        print(f"  {station_id}: coverage_80={cov_80:.3f}, coverage_90={cov_90:.3f}")

    coverage_df = pd.DataFrame(results)

    # Aggregate row
    if not coverage_df.empty:
        agg = {
            "station_id": "OVERALL",
            "eval_rows": coverage_df["eval_rows"].sum(),
            "coverage_80": round(coverage_df["coverage_80"].mean(), 4),
            "coverage_90": round(coverage_df["coverage_90"].mean(), 4),
            "gap_80": round(coverage_df["gap_80"].mean(), 4),
            "gap_90": round(coverage_df["gap_90"].mean(), 4),
        }
        coverage_df = pd.concat(
            [coverage_df, pd.DataFrame([agg])], ignore_index=True
        )

    return coverage_df


def log_to_mlflow(calibration_stats: dict, coverage_df: pd.DataFrame):
    """
    Log calibration results to MLflow experiment phase5_uncertainty.
    """
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("phase5_uncertainty")

    with mlflow.start_run(run_name="conformal_calibration"):
        # Calibration stats
        mlflow.log_metric("n_conformity_scores", calibration_stats["n_conformity_scores"])
        mlflow.log_metric("q80_half_width", calibration_stats["q80"])
        mlflow.log_metric("q90_half_width", calibration_stats["q90"])
        mlflow.log_metric("mean_residual", calibration_stats["mean_residual"])

        # Overall coverage
        overall = coverage_df[coverage_df["station_id"] == "OVERALL"]
        if not overall.empty:
            mlflow.log_metric("overall_coverage_80", overall["coverage_80"].iloc[0])
            mlflow.log_metric("overall_coverage_90", overall["coverage_90"].iloc[0])
            mlflow.log_metric("gap_80", overall["gap_80"].iloc[0])
            mlflow.log_metric("gap_90", overall["gap_90"].iloc[0])

        mlflow.log_param("calibration_method", "split_conformal")
        mlflow.log_param("calibration_sites", "office001")
        mlflow.log_param("train_frac", 0.70)
        mlflow.log_param("n_calibration_stations", len(coverage_df) - 1)

        print("Results logged to MLflow experiment: phase5_uncertainty")

def save_calibration(output_path: str = "models/calibration.json") -> None:
    """
    Serialize the in-memory conformal quantiles to disk.
    Call this once after calibrate() to avoid recomputing at every API startup.
    """
    if _q80 is None:
        raise RuntimeError("calibrate() must be called before save_calibration().")
    state = {
        "q80_single":   _q80,
        "q90_single":   _q90,
        "q80_zero":     _q80_zero,
        "q90_zero":     _q90_zero,
        "q80_nonzero":  _q80_nonzero,
        "q90_nonzero":  _q90_nonzero,
    }
    import json, os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(state, f, indent=2)
    print(f"Calibration state saved to {output_path}")


def load_calibration(input_path: str = "models/calibration.json") -> dict:
    """
    Load serialized conformal quantiles from disk and populate module globals.
    Call this at API startup instead of running calibrate().
    """
    global _q80, _q90, _q80_zero, _q90_zero, _q80_nonzero, _q90_nonzero
    import json
    with open(input_path) as f:
        state = json.load(f)
    _q80         = state["q80_single"]
    _q90         = state["q90_single"]
    _q80_zero    = state["q80_zero"]
    _q90_zero    = state["q90_zero"]
    _q80_nonzero = state["q80_nonzero"]
    _q90_nonzero = state["q90_nonzero"]
    print(f"Calibration state loaded from {input_path}")
    print(f"  q80_zero={_q80_zero:.4f}, q90_zero={_q90_zero:.4f}")
    print(f"  q80_nonzero={_q80_nonzero:.4f}, q90_nonzero={_q90_nonzero:.4f}")
    return state
