#!/usr/bin/env python3
"""
Q2 Step 4: Cross-family / cross-scale validation.

Required by problem statement:
  - B2 (Cerebras) or B3 (Pythia trajectories) validation
  - B4 (scaling_baseline) + B5 (published_scaling_data) transfer
  - B9/B10 (100B+ estimated) extrapolation check

All evaluations use frozen B1 parameters (NO refit).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import spearmanr

from config import (
    Q2_B1_FIT, B1_SCALING, B2_CEREBRAS, B3_TRAJ_DIR,
    B4_BASELINE, B5_PUBLISHED, B9_LARGE_META, B10_LARGE_LOSS,
    Q2_MODELS, Q2_TABLES, SEED,
)

np.random.seed(SEED)


def loss_law(X, E, A, alpha, B, beta):
    N, D = X
    return E + A * np.power(N, -alpha) + B * np.power(D, -beta)


def load_b1():
    with open(Q2_B1_FIT, encoding="utf-8") as f:
        cfg = json.load(f)
    p = cfg["parameters"]
    return (p["E"]["estimate"], p["A"]["estimate"],
            p["alpha"]["estimate"], p["B"]["estimate"],
            p["beta"]["estimate"])


def eval_predictions(N, D, L, b1):
    """Apply B1 law (frozen), compute R², MAE, RMSE, Spearman."""
    E, A, alpha, B, beta = b1
    pred = loss_law((N, D), E, A, alpha, B, beta)
    resid = L - pred
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((L - L.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    mae = float(np.mean(np.abs(resid)))
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    rho = float(spearmanr(L, pred).correlation)
    return r2, mae, rmse, rho


# ==================================================================
# B3: Pythia interpolated trajectories
# ==================================================================
def validate_b3(b1):
    print("\n" + "=" * 72)
    print("B3: Pythia interpolated trajectory validation")
    print("=" * 72)

    traj_dir = Path(B3_TRAJ_DIR)
    if not traj_dir.exists():
        print(f"[SKIP] {traj_dir} not found")
        return []

    rows = []
    files = sorted(traj_dir.glob("*.csv"))
    print(f"Found {len(files)} trajectory files")
    print(f"\n{'N (B)':>10} {'R²':>10} {'MAE':>10} {'RMSE':>10} {'Spearman':>10}")
    print("-" * 56)

    for fp in files:
        df = pd.read_csv(fp)
        N = df["N_params_B"].to_numpy(dtype=float)
        D = df["D_tokens_B"].to_numpy(dtype=float)
        L = df["val_loss"].to_numpy(dtype=float)
        if len(N) < 3:
            continue
        r2, mae, rmse, rho = eval_predictions(N, D, L, b1)
        N_level = float(N[0])
        rows.append({
            "N_B": N_level, "n_points": len(N),
            "r2": r2, "mae": mae, "rmse": rmse, "spearman": rho,
        })
        print(f"{N_level:>10.6f} {r2:>10.6f} {mae:>10.6f} "
              f"{rmse:>10.6f} {rho:>10.6f}")

    return rows


# ==================================================================
# B2: Cerebras same-form structural check
# ==================================================================
def validate_b2(b1):
    print("\n" + "=" * 72)
    print("B2: Cerebras same-form structural check")
    print("=" * 72)

    if not Path(B2_CEREBRAS).exists():
        print(f"[SKIP] not found")
        return None

    df = pd.read_csv(B2_CEREBRAS)
    df = df.dropna(subset=["N_params_B", "D_tokens_B", "val_loss"])
    df = df[df["val_loss"] > 0]

    N = df["N_params_B"].to_numpy(dtype=float)
    D = df["D_tokens_B"].to_numpy(dtype=float)
    L = df["val_loss"].to_numpy(dtype=float)
    print(f"Cerebras rows: {len(df)}")

    # Independent same-form fit
    p0 = [1.67, 0.36, 0.34, 8.5, 0.28]
    try:
        popt, _ = curve_fit(loss_law, (N, D), L, p0=p0, maxfev=200000)
    except Exception as e:
        print(f"[FAIL] {e}")
        return None

    E2, A2, a2, B2_v, b2 = popt
    pred2 = loss_law((N, D), *popt)
    ss_res = float(np.sum((L - pred2) ** 2))
    ss_tot = float(np.sum((L - L.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    print(f"\nCerebras independent fit:")
    print(f"  E     = {E2:.6f}   (B1: {b1[0]:.6f})")
    print(f"  A     = {A2:.6f}   (B1: {b1[1]:.6f})")
    print(f"  alpha = {a2:.6f}   (B1: {b1[2]:.6f})")
    print(f"  B     = {B2_v:.6f}   (B1: {b1[3]:.6f})")
    print(f"  beta  = {b2:.6f}   (B1: {b1[4]:.6f})")
    print(f"  R²    = {r2:.6f}")
    print(f"  (report: E=1.6696, A=0.3603, alpha=0.3358, B=8.525, beta=0.2774, R²=0.9995)")

    return {"E": E2, "A": A2, "alpha": a2, "B": B2_v, "beta": b2, "r2": r2}


# ==================================================================
# B4/B5: Direct transfer on published data
# ==================================================================
def validate_transfer(path, label, b1):
    print(f"\n--- {label} ---")
    if not Path(path).exists():
        print(f"[SKIP] not found")
        return None

    df = pd.read_csv(path)
    df = df.dropna(subset=["N_params_B", "D_tokens_B", "val_loss"])
    df = df[df["val_loss"] > 0]
    N = df["N_params_B"].to_numpy(dtype=float)
    D = df["D_tokens_B"].to_numpy(dtype=float)
    L = df["val_loss"].to_numpy(dtype=float)
    print(f"Rows: {len(df)}")

    r2, mae, rmse, rho = eval_predictions(N, D, L, b1)
    print(f"  Direct B1 transfer: R²={r2:.4f}  MAE={mae:.4f}  "
          f"RMSE={rmse:.4f}  ρ={rho:.4f}")
    return {"n": len(df), "r2": r2, "mae": mae, "rmse": rmse, "rho": rho}


# ==================================================================
# B9/B10: 100B+ extrapolation consistency check
# ==================================================================
def validate_large(b1):
    print("\n" + "=" * 72)
    print("B9/B10: 100B+ extrapolation consistency check")
    print("=" * 72)

    if not (Path(B9_LARGE_META).exists() and Path(B10_LARGE_LOSS).exists()):
        print("[SKIP] B9/B10 not found")
        return None

    meta = pd.read_csv(B9_LARGE_META)
    loss = pd.read_csv(B10_LARGE_LOSS)
    print(f"B9 rows: {len(meta)}  B10 rows: {len(loss)}")

    # Try common column names
    def find(df, cands):
        for c in cands:
            if c in df.columns:
                return c
        return None

    meta_N = find(meta, ["N_params_B", "Parameters", "params_B", "N"])
    meta_D = find(meta, ["D_tokens_B", "Training tokens", "D", "tokens"])
    loss_N = find(loss, ["N_params_B", "N", "params_B"])
    loss_D = find(loss, ["D_tokens_B", "D", "tokens"])
    loss_L = find(loss, ["val_loss", "loss", "Loss", "val_loss_est"])

    print(f"meta cols: N={meta_N}  D={meta_D}")
    print(f"loss cols: N={loss_N}  D={loss_D}  L={loss_L}")

    if not all([loss_N, loss_D, loss_L]):
        print("[WARN] could not identify columns; skipping")
        print(f"meta columns: {list(meta.columns)}")
        print(f"loss columns: {list(loss.columns)}")
        return None

    df = loss.dropna(subset=[loss_N, loss_D, loss_L]).copy()
    N = df[loss_N].to_numpy(dtype=float)
    D = df[loss_D].to_numpy(dtype=float)
    L = df[loss_L].to_numpy(dtype=float)
    print(f"Usable rows: {len(df)}")

    if len(df) < 5:
        print("[SKIP] too few rows")
        return None

    r2, mae, rmse, rho = eval_predictions(N, D, L, b1)
    print(f"\n  B1 law on B10: R²={r2:.6f}  MAE={mae:.6f}  "
          f"RMSE={rmse:.6f}  ρ={rho:.6f}")
    print(f"  (report: R²=0.9999924, MAE=0.000614, ρ=0.999994)")
    print(f"  NOTE: B10 is estimated (not direct measurement)")

    return {"n": len(df), "r2": r2, "mae": mae, "rmse": rmse, "rho": rho}


# ==================================================================
# MAIN
# ==================================================================
def main():
    print("=" * 72)
    print("Q2 STEP 4: Cross-family / cross-scale validation")
    print("=" * 72)

    b1 = load_b1()
    print(f"B1 frozen: E={b1[0]:.6f}  A={b1[1]:.6f}  alpha={b1[2]:.6f}")
    print(f"           B={b1[3]:.6f}  beta={b1[4]:.6f}")

    b3_rows = validate_b3(b1)
    b2_res = validate_b2(b1)
    b4_res = validate_transfer(B4_BASELINE, "B4: scaling_baseline cross-family", b1)
    b5_res = validate_transfer(B5_PUBLISHED, "B5: published_scaling_data", b1)
    b9_b10 = validate_large(b1)

    # ==================================================================
    # Save
    # ==================================================================
    payload = {
        "b1_frozen": {"E": b1[0], "A": b1[1], "alpha": b1[2],
                       "B": b1[3], "beta": b1[4]},
        "b3_trajectories": b3_rows,
        "b2_cerebras": b2_res,
        "b4_transfer": b4_res,
        "b5_transfer": b5_res,
        "b9_b10_large": b9_b10,
    }
    Q2_MODELS.mkdir(parents=True, exist_ok=True)
    with open(Q2_MODELS / "validation.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved: {Q2_MODELS / 'validation.json'}")

    # CSV tables
    Q2_TABLES.mkdir(parents=True, exist_ok=True)
    if b3_rows:
        pd.DataFrame(b3_rows).to_csv(
            Q2_TABLES / "validation_b3_trajectories.csv", index=False)
    summary = [
        {"dataset": "B2_cerebras", **(b2_res or {})},
        {"dataset": "B4_baseline", **(b4_res or {})},
        {"dataset": "B5_published", **(b5_res or {})},
        {"dataset": "B9_B10_large", **(b9_b10 or {})},
    ]
    pd.DataFrame(summary).to_csv(
        Q2_TABLES / "validation_summary.csv", index=False)
    print(f"Saved: {Q2_TABLES / 'validation_summary.csv'}")

    print("\n[DONE] 04_validation.py")


if __name__ == "__main__":
    main()