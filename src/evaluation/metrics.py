import numpy as np
import pandas as pd


def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true, y_pred, epsilon=1e-8):
    return float(np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100)


def evaluate(y_true, y_pred):
    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mape": mape(y_true, y_pred),
    }


def temporal_split(df, train_weeks, eval_weeks=4, time_col="timestamp"):
    df = df.sort_values(time_col).reset_index(drop=True)
    train_cutoff = df[time_col].min() + pd.Timedelta(weeks=train_weeks)
    eval_cutoff = train_cutoff + pd.Timedelta(weeks=eval_weeks)
    train_df = df[df[time_col] < train_cutoff].copy()
    eval_df = df[(df[time_col] >= train_cutoff) & (df[time_col] < eval_cutoff)].copy()
    return train_df, eval_df