#!/usr/bin/env python3
"""
Compare 3 bridge variants on Q3 losses:
  A. Global Ridge (current, all 43 rows, w=5 for high)
  B. Local Ridge (only rows within Q3 loss range ±0.1)
  C. High-only Ridge (only the 7 high-comparability rows)

Reports R2 (LOOCV within subset) + predictions for Q3 budgets.
Decide which (if any) is a better secondary predictor.
"""
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import LeaveOneOut

from config import C6_BRIDGE, Q3_POINT_SOLUTIONS


def family_from_name(name):
    s = str(name).split("/")[-1].lower()
    tok = s.split("-")[0]
    for pref in ["pythia", "llama", "qwen", "gemma", "mistral", "phi",
                 "olmo", "yi", "falcon", "gpt", "deepseek"]:
        if tok.startswith(pref):
            return pref
    return tok


def fit_and_eval(X, y, weights, name, q3_losses):
    """Fit ridge; report LOOCV + in-sample + Q3 predictions."""
    n = len(y)
    if n < 3:
        print(f"\n[{name}] n={n} — too few rows for LOOCV.")
        return None

    # LOOCV (leave-one-out is honest for small n)
    loo = LeaveOneOut()
    preds = np.full(n, np.nan)
    for tr, te in loo.split(X):
        m = Ridge(alpha=1.0)
        m.fit(X[tr], y[tr], sample_weight=weights[tr])
        preds[te] = m.predict(X[te])
    r2_loo = float(r2_score(y, preds))
    mae_loo = float(mean_absolute_error(y, preds))

    # In-sample
    m = Ridge(alpha=1.0)
    m.fit(X, y, sample_weight=weights)
    p_in = m.predict(X)
    r2_in = float(r2_score(y, p_in))
    mae_in = float(mean_absolute_error(y, p_in))

    # Q3 predictions
    q3_pred = m.predict(q3_losses.reshape(-1, 1))

    print(f"\n[{name}] n={n}")
    print(f"  slope = {m.coef_[0]:+.4f}, intercept = {m.intercept_:+.4f}")
    print(f"  In-sample:  R2={r2_in:+.4f}, MAE={mae_in:.4f}")
    print(f"  LOOCV:      R2={r2_loo:+.4f}, MAE={mae_loo:.4f}")
    print(f"  Q3 preds:   " + ", ".join(f"{v:.2f}" for v in q3_pred))

    return {
        "n": n,
        "slope": float(m.coef_[0]),
        "intercept": float(m.intercept_),
        "r2_in": r2_in,
        "mae_in": mae_in,
        "r2_loo": r2_loo,
        "mae_loo": mae_loo,
        "q3_preds": q3_pred.tolist(),
    }


def main():
    print("=" * 72)
    print("BRIDGE VARIANT COMPARISON")
    print("=" * 72)

    with open(Q3_POINT_SOLUTIONS, encoding="utf-8") as f:
        q3 = json.load(f)
    q3_losses = np.array([s["loss"] for s in q3["main_exponential"]])
    q3_lo, q3_hi = q3_losses.min(), q3_losses.max()
    print(f"\nQ3 losses: {q3_losses}")
    print(f"Q3 range:  [{q3_lo:.4f}, {q3_hi:.4f}]")

    c6 = pd.read_csv(C6_BRIDGE)
    c6 = c6.dropna(subset=["Val_Loss", "LB_Average"]).copy()
    c6["family"] = c6["Model"].apply(family_from_name)
    hm = c6["Loss_Comparability"].astype(str).str.contains(
        "High", case=False, na=False)
    c6["weight"] = np.where(hm, 5.0, 1.0)

    results = {}

    # === A. Global (current) ===
    X_all = c6["Val_Loss"].to_numpy(dtype=float).reshape(-1, 1)
    y_all = c6["LB_Average"].to_numpy(dtype=float)
    w_all = c6["weight"].to_numpy(dtype=float)
    results["A_global"] = fit_and_eval(
        X_all, y_all, w_all, "A. Global (all 43, w=5)", q3_losses)

    # === B. Local (within Q3 range ± 0.1) ===
    pad = 0.10
    mask_local = ((c6["Val_Loss"] >= q3_lo - pad) &
                  (c6["Val_Loss"] <= q3_hi + pad))
    sub = c6[mask_local]
    print(f"\nLocal subset: loss in [{q3_lo-pad:.2f}, {q3_hi+pad:.2f}] "
          f"→ n={len(sub)}")
    results["B_local"] = fit_and_eval(
        sub["Val_Loss"].to_numpy(dtype=float).reshape(-1, 1),
        sub["LB_Average"].to_numpy(dtype=float),
        sub["weight"].to_numpy(dtype=float),
        "B. Local (within Q3 range)", q3_losses)

    # === C. High-only (7 rows) ===
    high = c6[hm]
    print(f"\nHigh-only subset: n={len(high)}")
    results["C_high_only"] = fit_and_eval(
        high["Val_Loss"].to_numpy(dtype=float).reshape(-1, 1),
        high["LB_Average"].to_numpy(dtype=float),
        np.ones(len(high)),
        "C. High-only (7 pythia)", q3_losses)

    # === Summary table ===
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"\n{'Variant':22s} {'n':>4s} {'slope':>10s} {'intercept':>11s} "
          f"{'R2_LOO':>9s} {'MAE_LOO':>9s}")
    print("-" * 72)
    for k in ["A_global", "B_local", "C_high_only"]:
        r = results.get(k)
        if r:
            print(f"{k:22s} {r['n']:>4d} {r['slope']:>+10.3f} "
                  f"{r['intercept']:>+11.3f} {r['r2_loo']:>+9.4f} "
                  f"{r['mae_loo']:>9.4f}")

    print(f"\n{'Variant':22s}  Q3 predictions (budget 1e19 / 1e22 / 1e24)")
    print("-" * 72)
    for k in ["A_global", "B_local", "C_high_only"]:
        r = results.get(k)
        if r:
            preds_str = "  ".join(f"{v:6.2f}" for v in r["q3_preds"])
            print(f"{k:22s}  {preds_str}")

    # Save
    from config import Q4_MODELS
    Q4_MODELS.mkdir(parents=True, exist_ok=True)
    with open(Q4_MODELS / "bridge_variants_comparison.json", "w",
              encoding="utf-8") as f:
        json.dump({
            "q3_losses": q3_losses.tolist(),
            "variants": results,
            "note": ("Variant A is used as PRIMARY secondary bridge. "
                     "B and C reported for sensitivity."),
        }, f, indent=2)
    print(f"\nSaved: {Q4_MODELS / 'bridge_variants_comparison.json'}")


if __name__ == "__main__":
    main()