from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HMM_DIR = PROJECT_ROOT / "results" / "hmm"
OUTPUT_DIR = PROJECT_ROOT / "results" / "hmm"

STATE_FILE = HMM_DIR / "state_sequence_4_states.csv"
TRANSITION_FILE = HMM_DIR / "transition_matrix_4_states.csv"


def main():
    data = pd.read_csv(STATE_FILE, parse_dates=["date"], index_col="date")
    transition = pd.read_csv(TRANSITION_FILE, index_col=0)

    features = [
        "DGS2",
        "DGS5",
        "DGS10",
        "VIXCLS",
        "DBAA",
        "yield_curve_slope",
        "baa_credit_spread_proxy",
    ]

    print("\nSTATE OCCUPANCY")
    print("=" * 60)
    print(data["state"].value_counts().sort_index().to_string())

    print("\nAVERAGE MARKET CONDITIONS BY STATE")
    summary = data.groupby("state")[features].mean().round(3)
    print(summary.to_string())
    summary.to_csv(OUTPUT_DIR / "four_state_market_means.csv")

    print("\nMEDIAN MARKET CONDITIONS BY STATE")
    print(data.groupby("state")[features].median().round(3).to_string())

    print("\nTRANSITION MATRIX")
    print(transition.round(3).to_string())

    # Identify consecutive runs of the same state.
    state = data["state"]
    run_id = state.ne(state.shift()).cumsum()

    runs = data.groupby(run_id).agg(
        start_date=("state", lambda x: x.index.min()),
        end_date=("state", lambda x: x.index.max()),
        state=("state", "first"),
        months=("state", "size"),
    )

    print("\nCONSECUTIVE REGIME EPISODES")
    print(runs.to_string(index=False))
    runs.to_csv(OUTPUT_DIR / "four_state_regime_episodes.csv", index=False)

    # Plot the decoded state sequence through time.
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.step(data.index, data["state"], where="post")
    ax.set_yticks(sorted(data["state"].unique()))
    ax.set_ylabel("Inferred state")
    ax.set_xlabel("Date")
    ax.set_title("Four-State HMM: Inferred Market Regimes")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "four_state_regime_timeline.png", dpi=160)
    plt.close(fig)

    print(f"\nDiagnostic outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
