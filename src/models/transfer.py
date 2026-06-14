import pandas as pd
import numpy as np
import lightgbm as lgb
import pickle
from pathlib import Path

PROCESSED_DIR = Path(__file__).resolve().parents[2] / 'data' / 'processed'

FEATURE_COLS = [
    'hour', 'day_of_week', 'month',
    'is_weekend', 'is_holiday',
    'lag_1h', 'lag_24h', 'lag_168h',
    'rolling_24h_mean', 'rolling_7d_mean',
    'site_encoded'
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


def load_station(station_id, site):
    """
    Load a single station's parquet file, sort chronologically,
    fill NaNs, encode site. Returns a clean dataframe.
    """
    path = PROCESSED_DIR / f'{station_id}.parquet'
    df = pd.read_parquet(path)
    df = df.sort_values('timestamp').reset_index(drop=True)
    df['site_encoded'] = SITE_MAP[site]
    df[FEATURE_COLS[:-1]] = df[FEATURE_COLS[:-1]].fillna(0)
    return df


def temporal_split(df, train_weeks):
    """
    Split a station's data chronologically.
    First train_weeks of data → fine-tuning set.
    Everything after → evaluation set.
    Never shuffles.
    """
    start = df['timestamp'].min()
    cutoff = start + pd.Timedelta(weeks=train_weeks)
    train = df[df['timestamp'] < cutoff].copy()
    test  = df[df['timestamp'] >= cutoff].copy()
    return train, test


def fine_tune(global_booster, train_df, num_boost_round=100, learning_rate=0.01):
    """
    Add new trees on top of the global model using sparse station data.
    global_booster: trained lgb.Booster from global_model.py
    train_df:       sparse fine-tuning data for the new station
    Returns a new booster with global trees + station-specific correction trees.
    """
    if len(train_df) == 0:
        raise ValueError("Fine-tuning dataframe is empty — check train_weeks vs station history length")

    X = train_df[FEATURE_COLS].values
    y = train_df[TARGET_COL].values

    ft_params = {
        'objective':         'regression_l1',
        'learning_rate':     learning_rate,   # conservative — sparse data overfits fast
        'num_leaves':        31,              # shallower than global — less capacity to overfit
        'min_child_samples': 5,              # relaxed — fine-tuning set is small
        'feature_fraction':  0.8,
        'bagging_fraction':  0.8,
        'bagging_freq':      5,
        'verbose':           -1,
    }

    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    if len(X_val) < 10:
        # Not enough data for validation split, train without early stopping
        dataset = lgb.Dataset(X_train, label=y_train)
        fine_tuned = lgb.train(
            ft_params, dataset, num_boost_round=num_boost_round,
            init_model=global_booster,
        )
    else:
        dataset = lgb.Dataset(X_train, label=y_train)
        val_dataset = lgb.Dataset(X_val, label=y_val, reference=dataset)
        callbacks = [lgb.early_stopping(stopping_rounds=10, verbose=False),
                     lgb.log_evaluation(period=-1)]
        fine_tuned = lgb.train(
            ft_params, dataset, num_boost_round=num_boost_round,
            valid_sets=[val_dataset],
            init_model=global_booster,
            callbacks=callbacks,
        )
    return fine_tuned


def predict(booster, df):
    """
    Run inference on a dataframe. Returns numpy array of predictions,
    clipped to non-negative (sessions cannot be negative).
    """
    X = df[FEATURE_COLS].values
    preds = booster.predict(X)
    return np.clip(preds, 0, None)