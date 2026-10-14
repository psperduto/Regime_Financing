from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FUNCTIONS_DIR = PROJECT_ROOT / "dslc_documentation" / "functions"
RAW_DIR = PROJECT_ROOT / "data" / "raw"

sys.path.insert(0, str(FUNCTIONS_DIR))

from fred_data import download_fred_series


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    series_ids = [
        "DGS2",
        "DGS5",
        "DGS10",
        "VIXCLS",
        "DBAA",
    ]

    data = download_fred_series(
        series_ids,
        start_date="2000-01-01",
        end_date="2025-12-31",
    )

    output_file = RAW_DIR / "market_conditions_daily_2000_2025.csv"
    data.to_csv(output_file, index_label="date")

    print(f"\nSaved: {output_file}")
    print(f"Shape: {data.shape}")
    print("\nMissing values:")
    print(data.isna().sum())
    print("\nLast five observations:")
    print(data.tail())


if __name__ == "__main__":
    main()

