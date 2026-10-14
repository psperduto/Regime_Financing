from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "data" / "derived" / "market_conditions_monthly_2020_2025.csv"
OUTPUT_DIR = PROJECT_ROOT / "results" / "pca"

FEATURE_SETS = {
    "original": ["DGS2", "DGS5", "DGS10", "VIXCLS", "DBAA"],
    "interpretable": [
        "DGS2",
        "yield_curve_slope",
        "baa_credit_spread_proxy",
        "VIXCLS",
    ],
}


def run_pca(data, columns, label):
    X = data[columns].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA()
    scores = pca.fit_transform(X_scaled)

    explained = pd.DataFrame({
        "component": [f"PC{i + 1}" for i in range(len(columns))],
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative_variance": pca.explained_variance_ratio_.cumsum(),
    })

    loadings = pd.DataFrame(
        pca.components_.T,
        index=columns,
        columns=[f"PC{i + 1}" for i in range(len(columns))],
    )

    score_df = pd.DataFrame(
        scores,
        index=data.index,
        columns=[f"PC{i + 1}" for i in range(len(columns))],
    )
    score_df.index.name = "date"

    explained.to_csv(OUTPUT_DIR / f"{label}_explained_variance.csv", index=False)
    loadings.to_csv(OUTPUT_DIR / f"{label}_loadings.csv")
    score_df.to_csv(OUTPUT_DIR / f"{label}_scores.csv")

    print(f"\n{'=' * 55}")
    print(f"PCA RESULTS: {label.upper()}")
    print("=" * 55)
    print(explained.round(3).to_string(index=False))
    print("\nLoadings:")
    print(loadings.round(3))

    return explained


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(DATA_FILE, parse_dates=["date"], index_col="date")
    data = data.sort_index()

    for label, columns in FEATURE_SETS.items():
        run_pca(data, columns, label)

    fig, ax = plt.subplots(figsize=(9, 5))

    for label in FEATURE_SETS:
        explained = pd.read_csv(
            OUTPUT_DIR / f"{label}_explained_variance.csv"
        )
        ax.plot(
            range(1, len(explained) + 1),
            explained["cumulative_variance"],
            marker="o",
            label=label,
        )

    ax.axhline(0.80, linestyle="--", label="80% variance")
    ax.set_xlabel("Number of principal components")
    ax.set_ylabel("Cumulative explained variance")
    ax.set_xticks(range(1, 6))
    ax.set_ylim(0, 1.05)
    ax.set_title("PCA: Cumulative Explained Variance")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "cumulative_explained_variance.png",
        dpi=160,
    )
    plt.close(fig)

    print(f"\nPCA outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
