#!/usr/bin/env python3
"""
Q2 Step 1: B1 classical N-D scaling law + trajectory-cluster bootstrap.

Fits  L = E + A*N^(-alpha) + B*D^(-beta)  on B1 (pythia_training_log_existing).
Bootstrap resamples the 8 N-trajectories (clusters) with replacement.
"""
import json

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from config import B1_SCALING, Q2_B1_FIT, Q2_B1_BOOTSTRAP, Q2_TABLES, SEED

np.random.seed(SEED)


def loss_law(X, E, A, alpha, B, beta):
    N, D = X
    return E + A * np.power(N, -alpha) + B * np.power(D, -beta)


def main():
    print("=" * 72)
    print("Q2 STEP 1: B1 classical scaling law fit")
    print("=" * 72)

    df = pd.read_csv(B1_SCALING)
    print(f"B1 file: {B1_SCALING.name}")
    print(f"Raw rows: {len(df)}")
    print(f"Unique N levels: {df['N_params_B'].nunique()}")
    print(f"Unique D levels: {df['D_tokens_B'].nunique()}")

    df = df.dropna(subset=["N_params_B", "D_tokens_B", "val_loss"]).copy()
    df = df[df["val_loss"] > 0].copy()
    print(f"Clean rows: {len(df)}")

    N = df["N_params_B"].to_numpy(dtype=float)
    D = df["D_tokens_B"].to_numpy(dtype=float)
    L = df["val_loss"].to_numpy(dtype=float)

    # ---- Point fit ----
    p0 = [1.69, 0.35, 0.34, 1.24, 0.28]
    popt, _ = curve_fit(loss_law, (N, D), L, p0=p0, maxfev=200000)
    E, A, alpha, B, beta = popt

    pred = loss_law((N, D), *popt)
    resid = L - pred
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((L - L.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot
    mae = float(np.mean(np.abs(resid)))
    rmse = float(np.sqrt(np.mean(resid ** 2)))

    print("\nPoint estimates:")
    print(f"  E     = {E:.6f}")
    print(f"  A     = {A:.6f}")
    print(f"  alpha = {alpha:.6f}")
    print(f"  B     = {B:.6f}")
    print(f"  beta  = {beta:.6f}")
    print(f"\nFit statistics:")
    print(f"  R^2  = {r2:.8f}")
    print(f"  MAE  = {mae:.7f}")
    print(f"  RMSE = {rmse:.7f}")

    # ---- Trajectory-cluster bootstrap ----
    n_levels = sorted(df["N_params_B"].unique())
    print(f"\nCluster bootstrap by N-trajectory ({len(n_levels)} clusters)")
    groups = [df[df["N_params_B"] == n] for n in n_levels]

    n_boot = 2000
    rng = np.random.default_rng(SEED)
    boot = np.full((n_boot, 5), np.nan)

    for b in range(n_boot):
        idx = rng.integers(0, len(groups), size=len(groups))
        sample = pd.concat([groups[i] for i in idx], ignore_index=True)
        Nb = sample["N_params_B"].to_numpy(dtype=float)
        Db = sample["D_tokens_B"].to_numpy(dtype=float)
        Lb = sample["val_loss"].to_numpy(dtype=float)
        try:
            pb, _ = curve_fit(loss_law, (Nb, Db), Lb, p0=popt,
                              maxfev=100000)
            boot[b] = pb
        except Exception:
            pass

    valid = ~np.isnan(boot[:, 0])
    boot = boot[valid]
    print(f"  Successful draws: {len(boot)} / {n_boot}")

    names = ["E", "A", "alpha", "B", "beta"]
    print(f"\n{'Param':8s} {'Estimate':>12s} {'2.5%':>12s} {'97.5%':>12s}")
    print("-" * 50)
    params = {}
    for i, nm in enumerate(names):
        lo, hi = np.percentile(boot[:, i], [2.5, 97.5])
        print(f"{nm:8s} {popt[i]:12.6f} {lo:12.6f} {hi:12.6f}")
        params[nm] = {
            "estimate": float(popt[i]),
            "ci_low": float(lo),
            "ci_high": float(hi),
        }

    # ---- Save ----
    Q2_B1_FIT.parent.mkdir(parents=True, exist_ok=True)
    with open(Q2_B1_FIT, "w", encoding="utf-8") as f:
        json.dump({
            "formula": "L = E + A*N^(-alpha) + B*D^(-beta)",
            "parameters": params,
            "fit_stats": {
                "r2": r2, "mae": mae, "rmse": rmse,
                "n_rows": int(len(df)),
                "n_clusters": int(len(n_levels)),
            },
        }, f, indent=2)
    np.savez(Q2_B1_BOOTSTRAP, boot=boot, names=np.array(names))
    print(f"\nSaved: {Q2_B1_FIT}")
    print(f"Saved: {Q2_B1_BOOTSTRAP}")

    Q2_TABLES.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([
        {"param": nm,
         "estimate": params[nm]["estimate"],
         "ci_low": params[nm]["ci_low"],
         "ci_high": params[nm]["ci_high"]}
        for nm in names
    ]).to_csv(Q2_TABLES / "b1_parameters.csv", index=False)
    print(f"Saved: {Q2_TABLES / 'b1_parameters.csv'}")

    print("\n[DONE] 01_scaling_baseline.py")


if __name__ == "__main__":
    main()