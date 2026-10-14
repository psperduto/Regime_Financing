from pathlib import Path
import os

import pandas as pd
from dotenv import load_dotenv
from fredapi import Fred


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


def download_fred_series(
    series_ids,
    start_date="2000-01-01",
    end_date="2025-12-31",
):
    """Download FRED series and combine them into one DataFrame."""

    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        raise ValueError(
            "FRED_API_KEY not found. Check the .env file in the project root."
        )

    fred = Fred(api_key=api_key)
    data = {}

    for series_id in series_ids:
        print(f"Downloading {series_id}...")
        data[series_id] = fred.get_series(
            series_id,
            observation_start=start_date,
            observation_end=end_date,
        )

    return pd.DataFrame(data).sort_index()
