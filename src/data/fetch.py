"""
src/data/fetch.py

Downloads session data from ACN-Data API for all three sites
and saves them as parquet files in data/raw/.
"""

import os
from datetime import datetime
from pathlib import Path

import pandas as pd
from acnportal.acndata import DataClient
from dotenv import load_dotenv

# Load token from .env
load_dotenv()
API_TOKEN = os.getenv("ACN_API_TOKEN")

if not API_TOKEN:
    raise ValueError("ACN_API_TOKEN not found in .env file")

# Output directory
RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Sites to fetch
SITES = ["caltech", "jpl", "office001"]

# Date range
START = datetime(2018, 5, 1)
END = datetime(2020, 12, 31)


def fetch_site(site: str, client: DataClient) -> pd.DataFrame:
    """Fetch all sessions for a site using get_sessions_by_time generator."""
    print(f"\n>>> Fetching {site}...")

    sessions = []
    try:
        # get_sessions_by_time returns a generator — iterate through it
        generator = client.get_sessions_by_time(site, start=START, end=END)
        for i, session in enumerate(generator):
            sessions.append(session)
            if i % 500 == 0 and i > 0:
                print(f"  ...{i} sessions fetched so far")
    except Exception as e:
        print(f"  Error fetching {site}: {e}")
        return pd.DataFrame()

    if not sessions:
        print(f"  WARNING: No sessions returned for {site}")
        return pd.DataFrame()

    print(f"  Total sessions fetched: {len(sessions)}")
    df = pd.json_normalize(sessions)
    df["site"] = site
    return df


def clean_sessions(df: pd.DataFrame, site: str) -> pd.DataFrame:
    """Standardize columns across sites."""
    if df.empty:
        return df

    print(f"\n>>> Cleaning {site}...")
    print(f"  Raw columns: {list(df.columns)}")

    # Map ACN API field names to our standard names
    col_map = {
        "connectionTime": "session_start",
        "disconnectTime": "session_end",
        "kWhDelivered": "energy_kwh",
        "stationID": "station_id",
        "spaceID": "space_id",
        "userID": "user_id",
        "site": "site",
    }

    available = {k: v for k, v in col_map.items() if k in df.columns}
    df = df[list(available.keys())].rename(columns=available)

    # Parse timestamps
    for col in ["session_start", "session_end"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True)
            df[col] = df[col].dt.tz_convert("US/Pacific")

    # Drop rows missing critical fields
    df = df.dropna(subset=["session_start", "station_id"])

    if "site" not in df.columns:
        df["site"] = site

    print(f"  Clean shape: {df.shape}")
    print(f"  Date range: {df['session_start'].min()} to {df['session_start'].max()}")
    print(f"  Unique stations: {df['station_id'].nunique()}")

    return df


def main():
    print("=" * 50)
    print("ACN-Data Fetcher")
    print("=" * 50)

    client = DataClient(API_TOKEN)
    summary = []

    for site in SITES:
        raw_df = fetch_site(site, client)

        if raw_df.empty:
            print(f"  Skipping {site} — no data returned")
            continue

        clean_df = clean_sessions(raw_df, site)

        if clean_df.empty:
            continue

        out_path = RAW_DIR / f"{site}_sessions.parquet"
        clean_df.to_parquet(out_path, index=False)
        print(f"  Saved to {out_path}")

        summary.append({
            "site": site,
            "sessions": len(clean_df),
            "stations": clean_df["station_id"].nunique(),
            "start": clean_df["session_start"].min(),
            "end": clean_df["session_start"].max(),
        })

    print("\n" + "=" * 50)
    print("FETCH SUMMARY")
    print("=" * 50)
    for s in summary:
        print(f"\nSite     : {s['site']}")
        print(f"Sessions : {s['sessions']:,}")
        print(f"Stations : {s['stations']}")
        print(f"From     : {s['start']}")
        print(f"To       : {s['end']}")

    if not summary:
        print("No data fetched. Check your API token and connection.")
    else:
        print("\nDone. Files saved to data/raw/")


if __name__ == "__main__":
    main()