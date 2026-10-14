from pathlib import Path

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PCA_FILE = PROJECT_ROOT / "results" / "pca_extended" / "original_scores.csv"
MARKET_FILE = PROJECT_ROOT / "data" / "derived" / "market_conditions_monthly_2000_2025.csv"
OUTPUT_DIR = PROJECT_ROOT / "results" / "hmm_extended"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    scores = pd.read_csv(
        PCA_FILE, parse_dates=["date"], index_col="date"
    ).sort_index()

    market = pd.read_csv(
        MARKET_FILE, parse_dates=["date"], index_col="date"
    ).sort_index()

    # Align the datasets by date to prevent accidental row misalignment.
    combined = market.join(scores[["PC1", "PC2"]], how="inner")
    if combined.empty or combined[["PC1", "PC2"]].isna().any().any():
        raise ValueError("PCA scores are missing or could not be aligned.")

    X = combined[["PC1", "PC2"]].to_numpy(dtype=float)
    results = []

    for n_states in [2, 3, 4]:
        best_model = None
        best_loglik = -np.inf
        successful_fits = 0

        for seed in range(20):
            model = GaussianHMM(
                n_components=n_states,
                covariance_type="diag",
                n_iter=1000,
                tol=1e-4,
                min_covar=1e-3,
                random_state=seed,
            )

            try:
                model.fit(X)
                loglik = model.score(X)

                if np.isfinite(loglik):
                    successful_fits += 1
                    if loglik > best_loglik:
                        best_loglik = loglik
                        best_model = model
            except (ValueError, FloatingPointError) as exc:
                print(f"Skipped {n_states} states, seed {seed}: {exc}")

        if best_model is None:
            print(f"No valid fit for {n_states} states.")
            continue

        # Approximate parameter count for a diagonal Gaussian HMM.
        n_features = X.shape[1]
        n_params = (
            (n_states - 1)
            + n_states * (n_states - 1)
            + 2 * n_states * n_features
        )
        bic = -2 * best_loglik + n_params * np.log(len(X))

        # Viterbi path is a retrospective decoded path, not a real-time
        # filtered regime probability.
        states = best_model.predict(X)

        state_counts = (
            pd.Series(states, index=combined.index)
            .value_counts()
            .sort_index()
        )

        transition = pd.DataFrame(
            best_model.transmat_,
            index=[f"State_{i}" for i in range(n_states)],
            columns=[f"State_{i}" for i in range(n_states)],
        )
        transition.to_csv(
            OUTPUT_DIR / f"transition_matrix_{n_states}_states.csv"
        )

        state_data = combined.copy()
        state_data["state"] = states
        state_data.to_csv(
            OUTPUT_DIR / f"state_sequence_{n_states}_states.csv",
            index_label="date",
        )

        market_features = [
            "DGS2", "DGS5", "DGS10", "VIXCLS", "DBAA",
            "yield_curve_slope", "baa_credit_spread_proxy",
        ]
        summary = state_data.groupby("state")[market_features].mean()
        summary["months"] = state_data["state"].value_counts().sort_index()
        summary.to_csv(
            OUTPUT_DIR / f"state_summary_{n_states}_states.csv"
        )

        results.append({
            "n_states": n_states,
            "log_likelihood": best_loglik,
            "bic": bic,
            "converged": best_model.monitor_.converged,
            "iterations": best_model.monitor_.iter,
            "successful_fits": successful_fits,
            "min_state_months": int(state_counts.min()),
        })

        print(f"\n{'=' * 55}")
        print(f"{n_states}-STATE HMM | 2000-2025")
        print("=" * 55)
        print("Converged:", best_model.monitor_.converged)
        print("Iterations:", best_model.monitor_.iter)
        print("Successful fits:", successful_fits, "of 20")
        print("Log-likelihood:", round(best_loglik, 3))
        print("BIC:", round(bic, 3))
        print("\nMonths per state:")
        print(state_counts.to_string())
        print("\nTransition matrix:")
        print(transition.round(3).to_string())
        print("\nAverage market conditions:")
        print(summary.round(3).to_string())

    comparison = pd.DataFrame(results)
    comparison.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)

    print("\nMODEL COMPARISON")
    print(comparison.round(3).to_string(index=False))
    print(f"\nOutputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
