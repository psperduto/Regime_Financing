from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = PROJECT_ROOT / "data" / "raw" / "market_conditions_daily_2020_2025.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "derived"
OUTPUT_FILE = OUTPUT_DIR / "market_conditions_monthly_2020_2025.csv"


def main():
    if not RAW_FILE.exists():
        raise FileNotFoundError(f"Raw dataset not found: {RAW_FILE}")

    data = pd.read_csv(RAW_FILE, parse_dates=["date"], index_col="date")
    data = data.sort_index()

    expected_columns = ["DGS2", "DGS5", "DGS10", "VIXCLS", "DBAA"]
    missing_columns = set(expected_columns) - set(data.columns)

    if missing_columns:
        raise ValueError(f"Missing expected columns: {missing_columns}")

    # Monthly averages; missing daily values are ignored by the mean.
    monthly = data[expected_columns].resample("ME").mean()

    # Derived indicators.
    monthly["yield_curve_slope"] = monthly["DGS10"] - monthly["DGS2"]
    monthly["baa_credit_spread_proxy"] = monthly["DBAA"] - monthly["DGS10"]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    monthly.to_csv(OUTPUT_FILE, index_label="date")

    print(f"Saved: {OUTPUT_FILE}")
    print(f"Monthly shape: {monthly.shape}")
    print(f"Date range: {monthly.index.min().date()} to {monthly.index.max().date()}")
    print(f"Number of months: {len(monthly)}")

    print("\nMissing values by column:")
    print(monthly.isna().sum())

    print("\nSummary statistics:")
    print(monthly.describe().round(3))

    print("\nFirst five rows:")
    print(monthly.head())


if __name__ == "__main__":
    main()
