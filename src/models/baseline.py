import numpy as np
import pandas as pd
import lightgbm as lgb
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA

FEATURE_COLS = ["hour","day_of_week","is_weekend","is_holiday","rolling_24h_mean","rolling_7d_mean","lag_1h","lag_24h","lag_168h"]
TARGET_COL = "sessions"
TIME_COL = "timestamp"


class SeasonalNaive:
    def __init__(self):
        self.history = None

    def fit(self, train_df):
        self.history = train_df.set_index(TIME_COL)[TARGET_COL].sort_index()
        return self

    def predict(self, eval_df):
        fallback = float(self.history.mean())
        predictions = []
        for dt in eval_df[TIME_COL]:
            lookback = dt - pd.Timedelta(hours=168)
            if lookback in self.history.index:
                predictions.append(float(self.history[lookback]))
            else:
                predictions.append(fallback)
        return np.array(predictions)


class ARIMABaseline:
    def __init__(self, seasonal_period=24):
        self.seasonal_period = seasonal_period
        self.model = None

    def fit(self, train_df):
        sf_df = pd.DataFrame({
            "unique_id": "station",
            "ds": train_df[TIME_COL].values,
            "y": train_df[TARGET_COL].values.astype(float),
        })
        self.model = StatsForecast(
            models=[AutoARIMA(season_length=self.seasonal_period)],
            freq="h",
            n_jobs=1,
        )
        self.model.fit(sf_df)
        return self

    def predict(self, eval_df):
        horizon = len(eval_df)
        forecast = self.model.predict(h=horizon)
        return forecast["AutoARIMA"].values


class LightGBMBaseline:
    def __init__(self):
        self.model = None
        self.n_estimators = 200
        self.params = {
            "objective": "regression_l1",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_child_samples": 5,
            "verbosity": -1,
        }

    def fit(self, train_df):
        clean = train_df[FEATURE_COLS + [TARGET_COL]].copy()
        clean[FEATURE_COLS] = clean[FEATURE_COLS].fillna(0)
        clean = clean.dropna(subset=[TARGET_COL])
        X = clean[FEATURE_COLS].values
        y = clean[TARGET_COL].values
        train_data = lgb.Dataset(X, label=y, feature_name=FEATURE_COLS)
        self.model = lgb.train(self.params, train_data, num_boost_round=self.n_estimators)
        return self
    def predict(self, eval_df):
        X = eval_df[FEATURE_COLS].fillna(0).values
        return self.model.predict(X)