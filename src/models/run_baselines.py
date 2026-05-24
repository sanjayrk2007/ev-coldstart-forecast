import os
import sys
import glob
import warnings
import pandas as pd
import numpy as np
import mlflow
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mlflow
mlflow.set_tracking_uri("http://localhost:5000")

from src.models.baseline import SeasonalNaive, ARIMABaseline, LightGBMBaseline
from src.evaluation.metrics import evaluate, temporal_split

PROCESSED_DIR = Path("data/processed")
TRAIN_WEEKS_LIST = [1, 2, 3]
EVAL_WEEKS = 4
MIN_HISTORY_WEEKS = 26
HELD_OUT_SITE = "office001"
TIME_COL = "timestamp"
TARGET_COL = "sessions"
MLFLOW_EXPERIMENT = "phase2_baselines"


def load_stations():
    files = glob.glob(str(PROCESSED_DIR / "*.parquet"))
    stations = {}
    for f in files:
        station_id = Path(f).stem
        if HELD_OUT_SITE in station_id.lower():
            continue
        df = pd.read_parquet(f)
        df[TIME_COL] = pd.to_datetime(df[TIME_COL]).dt.tz_localize(None)
        df = df.sort_values(TIME_COL).reset_index(drop=True)
        stations[station_id] = df
    print(f"Loaded {len(stations)} stations")
    return stations


def has_enough_history(df, min_weeks):
    span = df[TIME_COL].max() - df[TIME_COL].min()
    return span.days >= (min_weeks * 7)


def run_baselines():
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    stations = load_stations()
    qualified = {sid: df for sid, df in stations.items() if has_enough_history(df, MIN_HISTORY_WEEKS)}
    print(f"{len(qualified)} stations qualify\n")

    models = {
        "seasonal_naive": SeasonalNaive,
        "arima": ARIMABaseline,
        "lgbm_vanilla": LightGBMBaseline,
    }

    all_results = []
    total = len(qualified) * len(TRAIN_WEEKS_LIST) * len(models)
    count = 0

    for station_id, df in qualified.items():
        for train_weeks in TRAIN_WEEKS_LIST:
            train_df, eval_df = temporal_split(df, train_weeks, EVAL_WEEKS, TIME_COL)
            if len(train_df) < 24 or len(eval_df) < 24:
                continue
            y_true = eval_df[TARGET_COL].values

            for model_name, ModelClass in models.items():
                count += 1
                print(f"[{count}/{total}] {station_id} | {train_weeks}w | {model_name}")
                try:
                    model = ModelClass()
                    model.fit(train_df)
                    y_pred = np.clip(model.predict(eval_df), 0, None)
                    min_len = min(len(y_true), len(y_pred))
                    metrics = evaluate(y_true[:min_len], y_pred[:min_len])

                    with mlflow.start_run(run_name=f"{model_name}_{station_id}_{train_weeks}w"):
                        mlflow.set_tags({"station_id": station_id, "model": model_name, "phase": "phase2"})
                        mlflow.log_params({"train_weeks": train_weeks, "eval_weeks": EVAL_WEEKS})
                        mlflow.log_metrics(metrics)

                    all_results.append({"station_id": station_id, "model": model_name, "train_weeks": train_weeks, **metrics})
                except Exception as e:
                    print(f"  FAILED: {e}")
                    continue

    if not all_results:
        print("No results.")
        return

    results_df = pd.DataFrame(all_results)
    print("\n" + "="*60)
    print("PHASE 2 RESULTS")
    print("="*60)
    summary = results_df.groupby(["model", "train_weeks"])[["mae", "rmse", "mape"]].mean().round(4)
    print(summary.to_string())
    results_df.to_csv("data/processed/baseline_results.csv", index=False)
    print("\nSaved to data/processed/baseline_results.csv")


if __name__ == "__main__":
    run_baselines()