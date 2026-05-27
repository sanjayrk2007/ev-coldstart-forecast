import os
import sys
import glob
import warnings
import pandas as pd
import numpy as np
from pathlib import Path

# Prevent Python from buffering console output
os.environ["PYTHONUNBUFFERED"] = "1"
warnings.filterwarnings("ignore")

# ==============================================================================
# ROBUST PATH RESOLUTION (Fixes silent PowerShell failures)
# ==============================================================================
try:
    # Get absolute path of this file to prevent relative resolution bugs
    SCRIPT_PATH = Path(os.path.abspath(__file__))
    # Safely move up 2 directories to project root
    PROJECT_ROOT = SCRIPT_PATH.parents[2]
    sys.path.insert(0, str(PROJECT_ROOT))
except IndexError:
    # Fallback if structure varies slightly
    PROJECT_ROOT = Path(os.getcwd())
    sys.path.insert(0, str(PROJECT_ROOT))

# ==============================================================================
# IMPORTS & CONFIGURATION
# ==============================================================================
try:
    import mlflow
    mlflow.set_tracking_uri("http://localhost:5000")
except Exception as e:
    print(f"⚠️ MLflow initialization warning (Will continue anyway): {e}")

from src.models.baseline import SeasonalNaive, ARIMABaseline, LightGBMBaseline
from src.evaluation.metrics import evaluate, temporal_split

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TRAIN_WEEKS_LIST = [1, 2, 3]
EVAL_WEEKS = 4
MIN_HISTORY_WEEKS = 26
HELD_OUT_SITE = "office001"
TIME_COL = "timestamp"
TARGET_COL = "sessions"
MLFLOW_EXPERIMENT = "phase2_baselines"


def load_stations():
    # Ensure directory exists before reading
    if not PROCESSED_DIR.exists():
        print(f"❌ ERROR: Processed directory not found at: {PROCESSED_DIR.resolve()}")
        return {}
        
    search_path = str(PROCESSED_DIR / "*.parquet")
    files = glob.glob(search_path)
    stations = {}
    
    for f in files:
        station_id = Path(f).stem
        if HELD_OUT_SITE in station_id.lower():
            continue
        try:
            df = pd.read_parquet(f)
            df[TIME_COL] = pd.to_datetime(df[TIME_COL]).dt.tz_localize(None)
            df = df.sort_values(TIME_COL).reset_index(drop=True)
            stations[station_id] = df
        except Exception as e:
            print(f"⚠️ Failed to read file {f}: {e}")
            
    print(f"Loaded {len(stations)} stations")
    return stations


def has_enough_history(df, min_weeks):
    if df.empty or TIME_COL not in df.columns:
        return False
    span = df[TIME_COL].max() - df[TIME_COL].min()
    return span.days >= (min_weeks * 7)


def run_baselines():
    print("🚀 Initializing baseline run...")
    
    # Try setting MLflow experiment safely
    mlflow_enabled = True
    try:
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
    except Exception as e:
        print(f"⚠️ Could not connect to MLflow server at localhost:5000. Logging to MLflow will be skipped. Error: {e}")
        mlflow_enabled = False

    stations = load_stations()
    if not stations:
        print("❌ Script terminated: No stations loaded. Check your data paths.")
        return

    qualified = {sid: df for sid, df in stations.items() if has_enough_history(df, MIN_HISTORY_WEEKS)}
    print(f"{len(qualified)} stations qualify\n")

    if not qualified:
        print("❌ Script terminated: No stations met the minimum history requirement.")
        return

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
            # Safely unpack split chunks
            try:
                train_df, eval_df = temporal_split(df, train_weeks, EVAL_WEEKS, TIME_COL)
            except Exception as e:
                print(f"  FAILED splitting data for {station_id}: {e}")
                continue

            if len(train_df) < 24 or len(eval_df) < 24:
                continue
            y_true = eval_df[TARGET_COL].values

            for model_name, ModelClass in models.items():
                count += 1
                print(f"[{count}/{total}] {station_id} | {train_weeks}w | {model_name}")
                sys.stdout.flush()  # Force immediate printing to PowerShell console
                
                try:
                    model = ModelClass()
                    model.fit(train_df)
                    y_pred = np.clip(model.predict(eval_df), 0, None)
                    min_len = min(len(y_true), len(y_pred))
                    metrics = evaluate(y_true[:min_len], y_pred[:min_len])

                    if mlflow_enabled:
                        try:
                            with mlflow.start_run(run_name=f"{model_name}_{station_id}_{train_weeks}w"):
                                mlflow.set_tags({"station_id": station_id, "model": model_name, "phase": "phase2"})
                                mlflow.log_params({"train_weeks": train_weeks, "eval_weeks": EVAL_WEEKS})
                                mlflow.log_metrics(metrics)
                        except Exception as e:
                            print(f"  ⚠️ MLflow tracking failed for this run step: {e}")

                    all_results.append({"station_id": station_id, "model": model_name, "train_weeks": train_weeks, **metrics})
                except Exception as e:
                    print(f"  ❌ FAILED running {model_name}: {e}")
                    continue

    if not all_results:
        print("\nNo evaluation results generated.")
        return

    # Process and log summary table
    results_df = pd.DataFrame(all_results)
    print("\n" + "="*60)
    print("PHASE 2 RESULTS")
    print("="*60)
    
    # Wrap in try-except in case metrics columns are entirely empty/NaN
    try:
        summary = results_df.groupby(["model", "train_weeks"])[["mae", "rmse", "mape"]].mean().round(4)
        print(summary.to_string())
    except Exception as e:
        print(f"Could not group summary statistics: {e}")

    # Save outputs safely
    try:
        output_file = PROCESSED_DIR / "baseline_results.csv"
        results_df.to_csv(output_file, index=False)
        print(f"\nSaved results table to: {output_file.resolve()}")
    except Exception as e:
        print(f"❌ Could not write results to CSV file: {e}")


if __name__ == "__main__":
    run_baselines()