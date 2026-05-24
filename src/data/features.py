"""
src/data/features.py

Adds time-based and lag features to hourly demand time series.
These features are what LightGBM actually learns from —
the model has no concept of time without them.
"""

from pathlib import Path
import pandas as pd
import holidays

PROCESSED_DIR = Path("data/processed")
US_HOLIDAYS = holidays.US()


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds calendar-based features extracted from the timestamp.
    These tell the model WHERE in time each row sits.
    """
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek  # 0=Monday, 6=Sunday
    df["month"] = df["timestamp"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_holiday"] = df["timestamp"].dt.date.apply(
        lambda d: 1 if d in US_HOLIDAYS else 0
    )
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds lag features — past session counts at specific time offsets.
    These tell the model WHAT HAPPENED BEFORE each row.

    Critical for cold-start: once a new station has even 1 week of data,
    lag_168h becomes available and transfer learning kicks in strongly.
    """
    df = df.sort_values("timestamp")

    df["lag_1h"] = df["sessions"].shift(1)
    df["lag_24h"] = df["sessions"].shift(24)
    df["lag_168h"] = df["sessions"].shift(168)  # one week back

    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds rolling mean features — smoothed averages over past windows.
    These capture the recent trend and weekly baseline.
    min_periods=1 means we calculate even if the full window isn't available yet.
    """
    df = df.sort_values("timestamp")

    df["rolling_24h_mean"] = (
        df["sessions"]
        .shift(1)  # shift so we don't include current hour in its own mean
        .rolling(window=24, min_periods=1)
        .mean()
    )

    df["rolling_7d_mean"] = (
        df["sessions"]
        .shift(1)
        .rolling(window=168, min_periods=1)
        .mean()
    )

    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies all feature engineering steps in order.
    Input: hourly demand dataframe (output of preprocessor)
    Output: same dataframe with added feature columns
    """
    df = add_time_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    return df


def process_all_stations():
    """
    Loads each processed station parquet, adds features, saves back.
    """
    station_files = list(PROCESSED_DIR.glob("*.parquet"))
    print(f"Adding features to {len(station_files)} station files...")

    for i, filepath in enumerate(station_files):
        df = pd.read_parquet(filepath)
        df = build_features(df)
        df.to_parquet(filepath, index=False)

        if i % 20 == 0:
            print(f"  [{i+1}/{len(station_files)}] {filepath.stem}")

    print(f"\nDone. Features added to all {len(station_files)} station files.")


if __name__ == "__main__":
    process_all_stations()

    # Show what one station looks like after feature engineering
    sample = pd.read_parquet(next(PROCESSED_DIR.glob("*.parquet")))
    print(f"\nSample station shape: {sample.shape}")
    print(f"Columns: {list(sample.columns)}")
    print(sample.head(3).to_string())