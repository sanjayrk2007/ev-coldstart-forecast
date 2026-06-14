"""
src/data/preprocessor.py

Converts raw session records into hourly demand time series.
One output parquet file per station saved to data/processed/.

Pipeline:
    1. Expand each session into hourly rows
    2. Group by station + hour → count active sessions
    3. Resample to fill missing hours with 0
    4. Save one parquet per station
"""

from pathlib import Path
import pandas as pd
from src.data.loader import load_raw_sessions

PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def expand_session_to_hours(row: pd.Series) -> pd.DataFrame:
    """
    Takes one session row and returns a dataframe with one row per hour
    the session was active.

    Example:
        session_start : 2019-03-15 08:32
        session_end   : 2019-03-15 11:15
        → rows for hours: 08:00, 09:00, 10:00, 11:00
    """
    # Floor to nearest hour — partial hour counts as occupied
    start_hour = row["session_start"].floor("h")
    # Ceil then subtract 1h: 09:00-11:00 covers hours 9,10 only
    end_hour = row["session_end"].ceil("h") - pd.Timedelta(hours=1)
    if end_hour < start_hour:
        end_hour = start_hour

    # Generate every hour between start and end inclusive
    hours = pd.date_range(start=start_hour, end=end_hour, freq="h")

    return pd.DataFrame({
        "station_id": row["station_id"],
        "timestamp": hours,
        "site": row["site"],
    })


def sessions_to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts session-level dataframe to hourly demand time series.
    Returns one row per station per hour with session count.
    """
    print("Expanding sessions to hourly rows...")

    # Apply expansion to every session — this is the core transformation
    # Each session becomes multiple rows, one per hour it spans
    expanded = pd.concat(
        [expand_session_to_hours(row) for _, row in df.iterrows()],
        ignore_index=True
    )

    print(f"  Sessions: {len(df):,} → Hourly rows: {len(expanded):,}")

    # Group by station + hour and count how many sessions were active
    # This handles overlapping sessions at the same station
    hourly = (
        expanded
        .groupby(["station_id", "timestamp", "site"])
        .size()
        .reset_index(name="sessions")
    )

    print(f"  Unique station-hour combinations: {len(hourly):,}")
    return hourly


def resample_station(df: pd.DataFrame) -> pd.DataFrame:
    """
    For a single station's hourly data, fills in missing hours with 0.
    Ensures a continuous unbroken time series.
    """
    df = df.set_index("timestamp").sort_index()

    # Create a complete hourly range from first to last session
    full_range = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq="h",
        tz=df.index.tz  # preserve timezone
    )

    # Reindex to full range — missing hours become NaN, then fill with 0
    df = df.reindex(full_range)
    df["sessions"] = df["sessions"].fillna(0).astype(int)

    # Forward fill site and station_id for the newly created rows
    df["site"] = df["site"].ffill()
    df["station_id"] = df["station_id"].ffill()

    df.index.name = "timestamp"
    return df.reset_index()


def preprocess_and_save():
    """
    Full pipeline: load raw sessions → expand → group → resample → save.
    """
    # Step 1 — Load raw sessions from all three sites
    df = load_raw_sessions()

    # Step 2 — Expand and group into hourly demand
    hourly = sessions_to_hourly(df)

    # Step 3 — For each station, resample to fill gaps, save to parquet
    stations = hourly["station_id"].unique()
    print(f"\nSaving {len(stations)} station files to {PROCESSED_DIR}...")

    for i, station_id in enumerate(stations):
        station_df = hourly[hourly["station_id"] == station_id].copy()
        station_df = resample_station(station_df)

        # Save — use station_id as filename, replace special chars with _
        safe_name = station_id.replace("/", "_").replace(" ", "_")
        out_path = PROCESSED_DIR / f"{safe_name}.parquet"
        station_df.to_parquet(out_path, index=False)

        if i % 20 == 0:
            print(f"  [{i+1}/{len(stations)}] Saved {safe_name}")

    print(f"\nDone. {len(stations)} station files saved to {PROCESSED_DIR}")


if __name__ == "__main__":
    preprocess_and_save()