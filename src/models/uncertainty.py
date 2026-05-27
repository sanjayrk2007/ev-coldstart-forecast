# src/models/uncertainty.py

import numpy as np
import pandas as pd
import pickle
import glob
import os
import mlflow

# ── Quantile store ──────────────────────────────────────────────────────────
# Single-quantile approach (baseline comparison)
_q80: float = None
_q90: float = None
_residuals: np.ndarray = None

# Conditional quantiles (zero vs non-zero regime)
_q80_zero: float = None
_q90_zero: float = None
_q80_nonzero: float = None
_q90_nonzero: float = None

# Threshold that separates zero-regime from non-zero-regime predictions
NONZERO_THRESHOLD = 0.05


# ── Feature columns (must match global_model.py exactly) ───────────────────
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
    Always chronological — never random. Shuffling a time series
    leaks future information into calibration.
    """
    df = df.sort_values(timestamp_col).reset_index(drop=True)
    cutoff = int(len(df) * train_frac)
    return df.iloc[:cutoff], df.iloc[cutoff:]


def _load_office001_files(processed_dir: str):
    """
    Load all office001 parquet files.
    Returns a list of DataFrames, one per station.
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


def _compute_quantile(residuals: np.ndarray, alpha: float) -> float:
    """
    Compute the finite-sample corrected quantile from a residual array.

    Why the (1 + 1/n) correction?
    Naive percentile gives slightly overconfident intervals on small n.
    This correction comes from conformal prediction theory and ensures
    the coverage guarantee holds exactly. With 70k+ residuals the
    difference is negligible, but we include it for correctness.
    """
    n = len(residuals)
    level = min((1 - alpha) * (1 + 1 / n), 1.0)
    return float(np.quantile(residuals, level))


def calibrate(
    global_model_path: str,
    processed_dir: str,
    site_encoding: int = 2,
    train_frac: float = 0.70,
):
    """
    Build conformity scores from the first train_frac of each office001
    station's timeline.

    Computes TWO sets of quantiles:
    - Single quantile: one q80/q90 across all residuals (baseline approach)
    - Conditional quantile: separate q80/q90 for zero-regime vs non-zero-regime

    Why two sets?
    The single quantile is dominated by near-zero residuals from dead hours
    (90% of all hours have zero demand). This makes q80 essentially 0,
    producing intervals that are too tight for active hours. The conditional
    approach uses the right error distribution for each regime.

    Parameters
    ----------
    global_model_path : str
        Path to models/global_model.pkl
    processed_dir : str
        Path to data/processed/
    site_encoding : int
        Integer encoding for office001 in site_encoded column (default 2).
    train_frac : float
        Fraction of each station's timeline used for calibration (default 0.70).
    """
    global _q80, _q90, _residuals
    global _q80_zero, _q90_zero, _q80_nonzero, _q90_nonzero

    # 1. Load global model
    with open(global_model_path, "rb") as f:
        booster = pickle.load(f)
    print(f"Global model loaded from {global_model_path}")

    # 2. Load office001 stations
    station_dfs = _load_office001_files(processed_dir)

    # 3. Collect residuals across all calibration splits
    all_residuals = []
    all_predictions = []

    for df in station_dfs:
        station_id = df["station_id"].iloc[0]

        if "site_encoded" not in df.columns:
            df = df.copy()
            df["site_encoded"] = site_encoding

        calib_df, _ = _temporal_split(df, "timestamp", train_frac)
        calib_df = calib_df.dropna(subset=FEATURE_COLS + [TARGET_COL])

        if len(calib_df) < 168:
            print(f"  Skipping {station_id} — insufficient rows ({len(calib_df)})")
            continue

        X_calib = calib_df[FEATURE_COLS].values
        y_true = calib_df[TARGET_COL].values
        y_pred = booster.predict(X_calib)

        residuals = np.abs(y_true - y_pred)
        all_residuals.extend(residuals)
        all_predictions.extend(y_pred)

        print(f"  {station_id}: {len(calib_df)} calibration rows, "
              f"mean residual = {residuals.mean():.4f}")

    if not all_residuals:
        raise ValueError("No valid calibration data found.")

    _residuals = np.array(all_residuals)
    all_predictions = np.array(all_predictions)

    # ── Single quantile (baseline) ──────────────────────────────────────────
    _q80 = _compute_quantile(_residuals, alpha=0.20)
    _q90 = _compute_quantile(_residuals, alpha=0.10)

    # ── Conditional quantiles ───────────────────────────────────────────────
    # Split residuals by prediction regime
    zero_mask = all_predictions < NONZERO_THRESHOLD
    nonzero_mask = ~zero_mask

    residuals_zero = _residuals[zero_mask]
    residuals_nonzero = _residuals[nonzero_mask]

    n_zero = zero_mask.sum()
    n_nonzero = nonzero_mask.sum()

    print(f"\n  Zero-regime hours:    {n_zero:,} ({100*n_zero/len(_residuals):.1f}%)")
    print(f"  Non-zero-regime hours: {n_nonzero:,} ({100*n_nonzero/len(_residuals):.1f}%)")

    if len(residuals_zero) > 0:
        _q80_zero = _compute_quantile(residuals_zero, alpha=0.20)
        _q90_zero = _compute_quantile(residuals_zero, alpha=0.10)
    else:
        _q80_zero = 0.0
        _q90_zero = 0.0

    if len(residuals_nonzero) > 0:
        _q80_nonzero = _compute_quantile(residuals_nonzero, alpha=0.20)
        _q90_nonzero = _compute_quantile(residuals_nonzero, alpha=0.10)
    else:
        # Fallback: if somehow no non-zero hours, use global quantile
        _q80_nonzero = _q80
        _q90_nonzero = _q90

    print(f"\nCalibration complete.")
    print(f"  Total conformity scores:     {len(_residuals):,}")
    print(f"\n  Single quantile approach:")
    print(f"    q80: {_q80:.4f} sessions")
    print(f"    q90: {_q90:.4f} sessions")
    print(f"\n  Conditional quantile approach:")
    print(f"    q80_zero:    {_q80_zero:.4f} sessions")
    print(f"    q90_zero:    {_q90_zero:.4f} sessions")
    print(f"    q80_nonzero: {_q80_nonzero:.4f} sessions")
    print(f"    q90_nonzero: {_q90_nonzero:.4f} sessions")

    return {
        "n_conformity_scores": len(_residuals),
        "n_zero_regime": int(n_zero),
        "n_nonzero_regime": int(n_nonzero),
        "q80_single": _q80,
        "q90_single": _q90,
        "q80_zero": _q80_zero,
        "q90_zero": _q90_zero,
        "q80_nonzero": _q80_nonzero,
        "q90_nonzero": _q90_nonzero,
        "mean_residual": float(_residuals.mean()),
        "median_residual": float(np.median(_residuals)),
    }


def predict_with_intervals(
    point_predictions: np.ndarray,
    method: str = "conditional",
) -> pd.DataFrame:
    """
    Wrap point predictions with calibrated conformal intervals.

    Parameters
    ----------
    point_predictions : np.ndarray
        Array of point forecast values from your transfer model.
    method : str
        "conditional" — use separate quantiles for zero/non-zero regime (recommended)
        "single"      — use one global quantile (baseline comparison)

    Returns
    -------
    pd.DataFrame with columns:
        point_estimate, lower_80, upper_80, lower_90, upper_90

    Why clip lower bounds at zero?
        EV demand cannot be negative. A lower bound of -0.3 sessions is
        physically meaningless. We clip at zero.
    """
    if _q80 is None:
        raise RuntimeError(
            "Model not calibrated. Call calibrate() before predict_with_intervals()."
        )

    preds = np.array(point_predictions)

    if method == "single":
        q80_arr = np.full_like(preds, _q80)
        q90_arr = np.full_like(preds, _q90)

    elif method == "conditional":
        if _q80_nonzero is None:
            raise RuntimeError(
                "Conditional quantiles not available. "
                "Run calibrate() first."
            )
        # Assign quantile based on which regime each prediction falls into
        zero_mask = preds < NONZERO_THRESHOLD
        q80_arr = np.where(zero_mask, _q80_zero, _q80_nonzero)
        q90_arr = np.where(zero_mask, _q90_zero, _q90_nonzero)

    else:
        raise ValueError(f"Unknown method '{method}'. Use 'single' or 'conditional'.")

    return pd.DataFrame({
        "point_estimate": preds,
        "lower_80": np.clip(preds - q80_arr, 0, None),
        "upper_80": preds + q80_arr,
        "lower_90": np.clip(preds - q90_arr, 0, None),
        "upper_90": preds + q90_arr,
    })


def evaluate_coverage(
    global_model_path: str,
    processed_dir: str,
    site_encoding: int = 2,
    train_frac: float = 0.70,
    method: str = "conditional",
) -> pd.DataFrame:
    """
    Check that calibrated intervals achieve claimed coverage on the held-out
    evaluation split (last 1 - train_frac of each office001 station).

    Returns a DataFrame with per-station and overall coverage.

    Parameters
    ----------
    method : str
        "conditional" or "single" — which quantile approach to evaluate.
    """
    if _q80 is None:
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
            print(f"  Skipping {station_id} — too few eval rows ({len(eval_df)})")
            continue

        X_eval = eval_df[FEATURE_COLS].values
        y_true = eval_df[TARGET_COL].values
        y_pred = booster.predict(X_eval)

        intervals = predict_with_intervals(y_pred, method=method)

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
            "gap_80": round(cov_80 - 0.80, 4),
            "gap_90": round(cov_90 - 0.90, 4),
        })

        print(f"  {station_id}: coverage_80={cov_80:.3f}, coverage_90={cov_90:.3f}")

    coverage_df = pd.DataFrame(results)

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


def log_to_mlflow(calibration_stats: dict, coverage_single: pd.DataFrame, coverage_conditional: pd.DataFrame):
    """
    Log calibration results and both coverage approaches to MLflow.
    """
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("phase5_uncertainty")

    with mlflow.start_run(run_name="conformal_calibration_conditional"):

        # Calibration stats
        mlflow.log_metric("n_conformity_scores", calibration_stats["n_conformity_scores"])
        mlflow.log_metric("n_zero_regime", calibration_stats["n_zero_regime"])
        mlflow.log_metric("n_nonzero_regime", calibration_stats["n_nonzero_regime"])
        mlflow.log_metric("q80_single", calibration_stats["q80_single"])
        mlflow.log_metric("q90_single", calibration_stats["q90_single"])
        mlflow.log_metric("q80_zero", calibration_stats["q80_zero"])
        mlflow.log_metric("q90_zero", calibration_stats["q90_zero"])
        mlflow.log_metric("q80_nonzero", calibration_stats["q80_nonzero"])
        mlflow.log_metric("q90_nonzero", calibration_stats["q90_nonzero"])
        mlflow.log_metric("mean_residual", calibration_stats["mean_residual"])

        # Single quantile overall coverage
        overall_single = coverage_single[coverage_single["station_id"] == "OVERALL"]
        if not overall_single.empty:
            mlflow.log_metric("single_coverage_80", overall_single["coverage_80"].iloc[0])
            mlflow.log_metric("single_coverage_90", overall_single["coverage_90"].iloc[0])

        # Conditional overall coverage
        overall_cond = coverage_conditional[coverage_conditional["station_id"] == "OVERALL"]
        if not overall_cond.empty:
            mlflow.log_metric("conditional_coverage_80", overall_cond["coverage_80"].iloc[0])
            mlflow.log_metric("conditional_coverage_90", overall_cond["coverage_90"].iloc[0])

        mlflow.log_param("calibration_method", "split_conformal_conditional")
        mlflow.log_param("calibration_sites", "office001")
        mlflow.log_param("train_frac", 0.70)
        mlflow.log_param("nonzero_threshold", NONZERO_THRESHOLD)
        mlflow.log_param("n_calibration_stations", 8)

        print("Results logged to MLflow experiment: phase5_uncertainty")