#!/usr/bin/env python3
"""
Q2 Step 3: Scale attenuation omega(N).

Reads Q1's locked mixture model, refits on A4-A5 (deterministic given
locked family/lambda), computes L-on-g transfer slopes across scales,
fits Hill curve.

Q1 owns the mixture model itself (see src/q1/06_mixture_model.py).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    Q2_TRAIN_MIX, Q2_TRAIN_LOSS, Q2_TEST_MIX, Q2_TEST_LOSS,
    Q2_60M_MIX, Q2_60M_LOSS, Q2_1B_MIX, Q2_1B_LOSS,
    Q2_10B_MIX, Q2_10B_LOSS, Q2_70B_MIX, Q2_70B_LOSS,
    Q1_MODELS, Q2_MODELS, Q2_TABLES, SEED,
)

np.random.seed(SEED)

EPS = 5e-4
MIX_PREFIX = "train_the_pile_"
LOSS_PREFIX = "metric/the_pile_"
LOSS_SUFFIX = "_val_loss"


def _mix_cols(df):
    return [c for c in df.columns if c.startswith(MIX_PREFIX)]


def _loss_cols(df):
    return [c for c in df.columns if c.startswith(LOSS_PREFIX)]


def load_split(mix_path, loss_path):
    mix = pd.read_csv(mix_path)
    loss = pd.read_csv(loss_path)
    return (mix[_mix_cols(mix)].to_numpy(dtype=float),
            loss[_loss_cols(loss)].to_numpy(dtype=float))


def clr_transform(P, eps=EPS):
    Pt = (P + eps) / (P + eps).sum(axis=1, keepdims=True)
    log_Pt = np.log(Pt)
    return log_Pt - log_Pt.mean(axis=1, keepdims=True)


def quad_expand(X):
    n, d = X.shape
    terms = [X]
    for i in range(d):
        for j in range(i, d):
            terms.append((X[:, i] * X[:, j]).reshape(-1, 1))
    return np.column_stack(terms)


FAMILIES = {
    "raw_linear": lambda P: P.copy(),
    "raw_quadratic": quad_expand,
    "clr_linear": clr_transform,
    "clr_quadratic": lambda P: quad_expand(clr_transform(P)),
}


def ridge_fit(X, Y, lam):
    xm = X.mean(axis=0)
    ym = Y.mean(axis=0)
    Xc = X - xm
    A = Xc.T @ Xc + lam * np.eye(X.shape[1])
    W = np.linalg.solve(A, Xc.T @ (Y - ym))
    return W, xm, ym


def ridge_predict(X, W, xm, ym):
    return (X - xm) @ W + ym


def transfer_slopes(P_pred, L_true):
    """L-on-g regression slope per domain: cov(g, L) / var(g)."""
    slopes = []
    for j in range(L_true.shape[1]):
        g = P_pred[:, j]
        y = L_true[:, j]
        gc = g - g.mean()
        yc = y - y.mean()
        denom = float(np.dot(gc, gc))
        if denom <= 0:
            continue
        slopes.append(float(np.dot(gc, yc) / denom))
    return np.array(slopes)


def hill_curve(N, w_inf, Nc, eta, N0=0.001):
    return w_inf + (1 - w_inf) * (1 + (N0 / Nc) ** eta) / (1 + (N / Nc) ** eta)


def fit_hill(N_vals, w_vals):
    from scipy.optimize import curve_fit
    p0 = [0.19, 0.28, 0.85]
    popt, _ = curve_fit(hill_curve, N_vals, w_vals, p0=p0, maxfev=20000)
    return popt


def main():
    print("=" * 72)
    print("Q2 STEP 3: Scale attenuation omega(N)")
    print("=" * 72)

    # Read Q1's locked mixture model
    with open(Q1_MODELS / "mixture_model.json") as f:
        q1_model = json.load(f)
    fam = q1_model["locked_family"]
    lam = q1_model["locked_lambda"]
    print(f"Q1 locked model: {fam}  lambda={lam}")
    print(f"Aitchison center macro loss: {q1_model['base_macro_loss']:.6f}")

    # Refit deterministic on A4-A5
    P_tr, L_tr = load_split(Q2_TRAIN_MIX, Q2_TRAIN_LOSS)
    W, xm, ym = ridge_fit(FAMILIES[fam](P_tr), L_tr, lam)
    print(f"Refitted on A4-A5: P shape {P_tr.shape}")

    # Cross-scale attenuation
    print("\n" + "=" * 72)
    print("Cross-scale attenuation")
    print("=" * 72)

    scale_specs = [
        ("1M",   0.001, Q2_TEST_MIX, Q2_TEST_LOSS, "real"),
        ("60M",  0.06,  Q2_60M_MIX,  Q2_60M_LOSS,  "real"),
        ("1B",   1.0,   Q2_1B_MIX,   Q2_1B_LOSS,   "real"),
        ("10B",  10.0,  Q2_10B_MIX,  Q2_10B_LOSS,  "estimated"),
        ("70B",  70.0,  Q2_70B_MIX,  Q2_70B_LOSS,  "estimated"),
    ]

    N_vals, w_vals = [], []
    print(f"\n{'Scale':>6} {'N (B)':>8} {'median':>10} {'q25':>10} {'q75':>10}")
    print("-" * 50)
    ref = None
    for label, N_B, mix_path, loss_path, nature in scale_specs:
        if not (Path(mix_path).exists() and Path(loss_path).exists()):
            print(f"{label:>6} {N_B:>8.3f}   [missing]")
            continue
        Ps, Ls = load_split(mix_path, loss_path)
        pred_s = ridge_predict(FAMILIES[fam](Ps), W, xm, ym)
        slopes = transfer_slopes(pred_s, Ls)
        med_abs = float(np.median(np.abs(slopes)))
        q25 = float(np.percentile(np.abs(slopes), 25))
        q75 = float(np.percentile(np.abs(slopes), 75))

        if label == "1M":
            ref = med_abs
            ref_q25, ref_q75 = q25, q75

        rel = med_abs / ref
        print(f"{label:>6} {N_B:>8.3f} {rel:>10.3f} "
              f"{q25/ref_q25:>10.3f} {q75/ref_q75:>10.3f}")
        N_vals.append(N_B)
        w_vals.append(rel)

    print("\nFitting Hill attenuation curve omega(N):")
    w_inf, Nc, eta = fit_hill(np.array(N_vals), np.array(w_vals))
    print(f"  omega_inf  = {w_inf:.6f}")
    print(f"  N_c (B)    = {Nc:.6f}")
    print(f"  eta        = {eta:.6f}")

    # Save
    Q2_MODELS.mkdir(parents=True, exist_ok=True)
    with open(Q2_MODELS / "attenuation.json", "w", encoding="utf-8") as f:
        json.dump({
            "form": "omega(N) = w_inf + (1-w_inf)*(1+(N0/Nc)^eta)/(1+(N/Nc)^eta)",
            "N0": 0.001,
            "omega_inf": float(w_inf),
            "N_c": float(Nc),
            "eta": float(eta),
            "scale_points": [
                {"label": s, "N_B": N, "omega_rel": w}
                for s, N, w in zip([x[0] for x in scale_specs], N_vals, w_vals)
            ],
        }, f, indent=2)
    print(f"\nSaved: {Q2_MODELS / 'attenuation.json'}")

    print("\n[DONE] 03_mixture_attenuation.py")


if __name__ == "__main__":
    main()