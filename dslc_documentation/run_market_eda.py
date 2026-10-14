from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "data" / "derived" / "market_conditions_monthly_2020_2025.csv"
OUTPUT_DIR = PROJECT_ROOT / "results" / "eda"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(DATA_FILE, parse_dates=["date"], index_col="date")
    data = data.sort_index()

    print("DATA OVERVIEW")
    print("=" * 60)
    print("Shape:", data.shape)
    print("Date range:", data.index.min().date(), "to", data.index.max().date())
    print("\nMissing values:")
    print(data.isna().sum())

    print("\nSUMMARY STATISTICS")
    summary = data.describe().T
    print(summary.round(3))
    summary.to_csv(OUTPUT_DIR / "summary_statistics.csv")

    print("\nCORRELATIONS")
    correlations = data.corr()
    print(correlations.round(3))
    correlations.to_csv(OUTPUT_DIR / "correlation_matrix.csv")

    # Plot the original market variables and derived indicators.
    columns = [
        "DGS2", "DGS5", "DGS10", "VIXCLS",
        "DBAA", "yield_curve_slope", "baa_credit_spread_proxy",
    ]

    fig, axes = plt.subplots(
        nrows=len(columns),
        ncols=1,
        figsize=(12, 16),
        sharex=True,
    )

    for ax, column in zip(axes, columns):
        ax.plot(data.index, data[column])
        ax.set_ylabel(column)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Date")
    fig.suptitle("Monthly Market Conditions, 2020–2025", y=1.002)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "market_conditions_timeseries.png", dpi=160,
                bbox_inches="tight")
    plt.close(fig)

    # Correlation heatmap.
    fig, ax = plt.subplots(figsize=(9, 7))
    image = ax.imshow(correlations, vmin=-1, vmax=1, cmap="coolwarm")

    ax.set_xticks(range(len(correlations.columns)))
    ax.set_xticklabels(correlations.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(correlations.index)))
    ax.set_yticklabels(correlations.index)

    for i in range(len(correlations.index)):
        for j in range(len(correlations.columns)):
            ax.text(j, i, f"{correlations.iloc[i, j]:.2f}",
                    ha="center", va="center", fontsize=8)

    fig.colorbar(image, ax=ax, label="Pearson correlation")
    ax.set_title("Correlation Matrix: Monthly Market Variables")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "correlation_matrix.png", dpi=160,
                bbox_inches="tight")
    plt.close(fig)

    print("\nEDA outputs saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
