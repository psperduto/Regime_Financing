from pathlib import Path

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PCA_FILE = PROJECT_ROOT / "results" / "pca" / "original_scores.csv"
MARKET_FILE = PROJECT_ROOT / "data" / "derived" / "market_conditions_monthly_2020_2025.csv"
OUTPUT_DIR = PROJECT_ROOT / "results" / "hmm"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    scores = pd.read_csv(PCA_FILE, parse_dates=["date"], index_col="date")
    market = pd.read_csv(
        MARKET_FILE, parse_dates=["date"], index_col="date"
    )

    # Use the first two principal components.
    X = scores[["PC1", "PC2"]].to_numpy(dtype=float)

    results = []
    fitted_models = {}

    for n_states in [2, 3, 4]:
        best_model = None
        best_loglik = -np.inf

        # Multiple starts reduce dependence on one random initialization.
        for seed in range(10):
            model = GaussianHMM(
                n_components=n_states,
                covariance_type="diag",
                n_iter=500,
                tol=1e-4,
                min_covar=1e-3,
                random_state=seed,
            )

            try:
                model.fit(X)
                loglik = model.score(X)

                if np.isfinite(loglik) and loglik > best_loglik:
                    best_loglik = loglik
                    best_model = model
            except (ValueError, FloatingPointError) as exc:
                print(f"Skipped {n_states} states, seed {seed}: {exc}")

        if best_model is None:
            print(f"No valid fit for {n_states} states.")
            continue

        # Approximate parameter count for a diagonal Gaussian HMM:
        # initial probabilities, transition probabilities, means, variances.
        n_features = X.shape[1]
        n_params = (
            (n_states - 1)
            + n_states * (n_states - 1)
            + n_states * n_features
            + n_states * n_features
        )
        bic = -2 * best_loglik + n_params * np.log(len(X))

        states = best_model.predict(X)
        occupancy = pd.Series(states).value_counts().sort_index()

        results.append({
            "n_states": n_states,
            "log_likelihood": best_loglik,
            "bic": bic,
            "converged": best_model.monitor_.converged,
            "iterations": best_model.monitor_.iter,
            "min_state_months": int(occupancy.min()),
        })

        fitted_models[n_states] = (best_model, states)

        transition = pd.DataFrame(
            best_model.transmat_,
            index=[f"State_{i}" for i in range(n_states)],
            columns=[f"State_{i}" for i in range(n_states)],
        )
        transition.to_csv(OUTPUT_DIR / f"transition_matrix_{n_states}_states.csv")

        state_data = market.copy()
        state_data["state"] = states
        state_data.to_csv(OUTPUT_DIR / f"state_sequence_{n_states}_states.csv")

        state_summary = state_data.groupby("state").mean(numeric_only=True)
        state_summary["months"] = state_data["state"].value_counts().sort_index()
        state_summary.to_csv(OUTPUT_DIR / f"state_summary_{n_states}_states.csv")

        print(f"\n{n_states}-state HMM")
        print("Converged:", best_model.monitor_.converged)
        print("Iterations:", best_model.monitor_.iter)
        print("Log-likelihood:", round(best_loglik, 3))
        print("BIC:", round(bic, 3))
        print("Months per state:")
        print(occupancy.to_string())
        print("Transition matrix:")
        print(transition.round(3).to_string())

    comparison = pd.DataFrame(results)
    comparison.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)

    print("\nMODEL COMPARISON")
    print(comparison.round(3).to_string(index=False))
    print(f"\nHMM outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
