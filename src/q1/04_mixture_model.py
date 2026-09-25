#!/usr/bin/env python3
"""Q1 Step 4: Fit 2-component Gaussian mixture on Q_i."""
import json

import numpy as np
import pandas as pd

from config import A1_WITH_QI, Q1_MODELS, Q1_TABLES, SEED

np.random.seed(SEED)


def fit_gmm_2comp(x, n_iter=200, tol=1e-6):
    """Simple 2-component 1-D GMM via EM."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)

    # Init: split at median
    mu = np.array([np.quantile(x, 0.25), np.quantile(x, 0.75)])
    sd = np.array([x.std(), x.std()]) + 1e-6
    pi = np.array([0.5, 0.5])

    for _ in range(n_iter):
        # E-step
        p = np.zeros((n, 2))
        for k in range(2):
            p[:, k] = pi[k] * np.exp(-0.5 * ((x - mu[k]) / sd[k]) ** 2) / (sd[k] * np.sqrt(2 * np.pi))
        p_sum = p.sum(axis=1, keepdims=True) + 1e-300
        g = p / p_sum

        # M-step
        Nk = g.sum(axis=0)
        mu_new = (g * x[:, None]).sum(axis=0) / Nk
        sd_new = np.sqrt((g * (x[:, None] - mu_new) ** 2).sum(axis=0) / Nk) + 1e-6
        pi_new = Nk / n

        if np.allclose(mu_new, mu, atol=tol) and np.allclose(pi_new, pi, atol=tol):
            mu, sd, pi = mu_new, sd_new, pi_new
            break
        mu, sd, pi = mu_new, sd_new, pi_new

    # Order by mean
    order = np.argsort(mu)
    return mu[order], sd[order], pi[order], g[:, order]


def main():
    print("=" * 72)
    print("Q1 MIXTURE MODEL (2-component Gaussian on Q_i)")
    print("=" * 72)

    df = pd.read_parquet(A1_WITH_QI)
    x = df["Q_i"].dropna().to_numpy()
    print(f"A1 Q_i rows: {len(x)}")
    print(f"Q_i stats: mean={x.mean():.4f}  std={x.std():.4f}")

    mu, sd, pi, resp = fit_gmm_2comp(x)

    print(f"\nComponent 1 (low):  mu={mu[0]:+.4f}  sd={sd[0]:.4f}  weight={pi[0]:.4f}")
    print(f"Component 2 (high): mu={mu[1]:+.4f}  sd={sd[1]:.4f}  weight={pi[1]:.4f}")
    print(f"Separation: {mu[1] - mu[0]:.4f}  ({abs(mu[1] - mu[0]) / np.sqrt((sd[0]**2 + sd[1]**2)/2):.3f} pooled SDs)")

    # Save
    Q1_MODELS.mkdir(parents=True, exist_ok=True)
    with open(Q1_MODELS / "quality_mixture.json", "w") as f:
        json.dump({
            "components": [
                {"mean": float(mu[k]), "std": float(sd[k]), "weight": float(pi[k])}
                for k in range(2)
            ],
            "separation": float(mu[1] - mu[0]),
            "n_samples": len(x),
        }, f, indent=2)

    # Per-domain mixture weights
    domain_assign = df.copy()
    domain_assign = domain_assign[domain_assign["Q_i"].notna()].reset_index(drop=True)
    domain_assign["p_high"] = resp[:, 1]
    dom_stats = (domain_assign.groupby("_source_domain")
                 .agg(n=("Q_i", "size"),
                      mean_Q=("Q_i", "mean"),
                      mean_p_high=("p_high", "mean"))
                 .reset_index())
    dom_stats.to_csv(Q1_TABLES / "mixture_domain_weights.csv", index=False)

    print(f"\nSaved: {Q1_MODELS}/quality_mixture.json")
    print("\nPer-domain fraction in high-quality component:")
    print(dom_stats.to_string(index=False))
    print("\n[DONE] 04_mixture_model.py")


if __name__ == "__main__":
    main()