import pandas as pd
import numpy as np
import lightgbm as lgb
import mlflow
import mlflow.lightgbm
from pathlib import Path
import pickle

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

SITE_MAP = {'caltech': 0, 'jpl': 1, 'office001': 2}


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