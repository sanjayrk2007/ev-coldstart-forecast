"""
src/data/fetch.py

Downloads session data from ACN-Data API for all three sites
and saves them as parquet files in data/raw/.
"""

import os
import time
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
SITES = ["caltech", "jpl", "office_01"]

# Date range — full available history
START = datetime(2018, 5, 1)
END = datetime(2020, 12, 31)


def fetch_site(site: str, client: DataClient) -> pd.DataFrame:
    """Fetch all sessions for a site and return as a DataFrame."""
    print(f"\n>>> Fetching {site}...")

    sessions = []
    page = 1

    while True:
        try:
            # acnportal paginates — keep fetching until empty
            data = client.get_sessions(
                site,
                start=START,
                end=END,
                page=page,
                page_size=1000,
            )

            if not data:
                break

            sessions.extend(data)
            print(f"  Page {page}: got {len(data)} sessions (total so far: {len(sessions)})")
            page += 1

            # Be polite to the API
            time.sleep(0.5)

        except Exception as e:
            print(f"  Error on page {page}: {e}")
            break

    if not sessions:
        print(f"  WARNING: No sessions returned for {site}")
        return pd.DataFrame()

    # Convert to DataFrame
    df = pd.json_normalize(sessions)
    df["site"] = site
    return df


def clean_sessions(df: pd.DataFrame, site: str) -> pd.DataFrame:
    """Standardize columns across sites."""
    if df.empty:
        return df

    print(f"\n>>> Cleaning {site}...")
    print(f"  Raw columns: {list(df.columns)}")

    # ACN API returns these key fields — map to standard names
    # Column names may vary slightly; we pick what's available
    col_map = {
        "connectionTime": "session_start",
        "disconnectTime": "session_end",
        "kWhDelivered": "energy_kwh",
        "stationID": "station_id",
        "spaceID": "space_id",
        "userID": "user_id",
        "site": "site",
    }

    # Only keep columns that exist
    available = {k: v for k, v in col_map.items() if k in df.columns}
    df = df[list(available.keys())].rename(columns=available)

    # Parse timestamps
    for col in ["session_start", "session_end"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True)
            # Convert to US/Pacific (ACN data timezone)
            df[col] = df[col].dt.tz_convert("US/Pacific")

    # Drop rows with missing critical fields
    df = df.dropna(subset=["session_start", "station_id"])

    # Add site column if not present
    if "site" not in df.columns:
        df["site"] = site

    print(f"  Clean shape: {df.shape}")
    print(f"  Date range: {df['session_start'].min()} → {df['session_start'].max()}")
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

        # Save to parquet
        out_path = RAW_DIR / f"{site}_sessions.parquet"
        clean_df.to_parquet(out_path, index=False)
        print(f"  Saved → {out_path}")

        summary.append({
            "site": site,
            "sessions": len(clean_df),
            "stations": clean_df["station_id"].nunique(),
            "start": clean_df["session_start"].min(),
            "end": clean_df["session_start"].max(),
        })

    # Print summary table
    print("\n" + "=" * 50)
    print("FETCH SUMMARY")
    print("=" * 50)
    for s in summary:
        print(f"\nSite: {s['site']}")
        print(f"  Sessions : {s['sessions']:,}")
        print(f"  Stations : {s['stations']}")
        print(f"  From     : {s['start']}")
        print(f"  To       : {s['end']}")

    print("\nDone. Files saved to data/raw/")


if __name__ == "__main__":
    main()
