"""
src/data/loader.py

Single responsibility: read raw parquet files from data/raw/
and return one combined dataframe with globally unique station IDs.
"""

from pathlib import Path
import pandas as pd

RAW_DIR = Path("data/raw")

REQUIRED_COLUMNS = [
    "station_id",
    "session_start", 
    "session_end",
    "energy_kwh",
    "site",
]

def load_raw_sessions() -> pd.DataFrame:
    """
    Scans data/raw/ for all parquet files, loads them,
    prefixes station_id with site name, combines into one dataframe.
    """
    parquet_files = list(RAW_DIR.glob("*.parquet"))
    
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {RAW_DIR}")
    
    frames = []
    
    for filepath in parquet_files:
        # Extract site name from filename e.g. caltech_sessions.parquet → caltech
        site_name = filepath.stem.replace("_sessions", "")
        
        df = pd.read_parquet(filepath)
        
        # Make station IDs globally unique
        df["station_id"] = site_name + "_" + df["station_id"].astype(str)
        
        frames.append(df)
        print(f"Loaded {filepath.name}: {len(df):,} sessions, {df['station_id'].nunique()} stations")
    
    combined = pd.concat(frames, ignore_index=True)
    
    # Verify all required columns exist
    missing = [c for c in REQUIRED_COLUMNS if c not in combined.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    print(f"\nTotal: {len(combined):,} sessions across {combined['station_id'].nunique()} stations")
    
    return combined[REQUIRED_COLUMNS]


if __name__ == "__main__":
    df = load_raw_sessions()
    print(df.head())
    print(df.dtypes)