#!/usr/bin/env python3
"""
Q4 Step 7: EXPLICIT Q3 → Q4 bridge.

Compliance (competition Page 4):
  "前一问的输出是后一问的输入"

Q3's optimal losses are mapped through C6 Loss→Benchmark bridge
to produce benchmark predictions. These are compared with Q4's
direct benchmark forecast.

Outputs:
  models/q4/q3_to_q4_bridge.json
  outputs/q4/tables/q3_to_q4_prediction.csv
"""
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import KFold, GroupKFold

from config import (
    C6_BRIDGE, Q3_POINT_SOLUTIONS,
    Q4_MODELS, Q4_TABLES, Q4_FORECAST, SEED,
)

np.random.seed(SEED)


def family_from_name(name):
    s = str(name).split("/")[-1].lower()
    tok = s.split("-")[0]
    for pref in ["pythia", "llama", "qwen", "gemma", "mistral", "phi",
                 "olmo", "yi", "falcon", "gpt", "deepseek"]:
        if tok.startswith(pref):
            return pref
    return tok


def run_validation_schemes(X, y, fams, weights):
    """Return 4-scheme validation summary for the Ridge bridge."""
    summary = {}

    # (a) In-sample
    m = Ridge(alpha=1.0)
    m.fit(X, y, sample_weight=weights)
    p = m.predict(X)
    summary["in_sample"] = {
        "r2": float(r2_score(y, p, sample_weight=weights)),
        "mae": float(mean_absolute_error(y, p, sample_weight=weights)),
        "n": int(len(y)),
    }

    # (b) Random 5-fold
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    pr = np.full(len(y), np.nan)
    for tr, te in kf.split(X):
        mm = Ridge(alpha=1.0)
        mm.fit(X[tr], y[tr], sample_weight=weights[tr])
        pr[te] = mm.predict(X[te])
    summary["random_5fold"] = {
        "r2": float(r2_score(y, pr)),
        "mae": float(mean_absolute_error(y, pr)),
        "n": int(len(y)),
    }

    # (c) GroupKFold by model family
    fams_u = np.unique(fams)
    n_splits = min(5, len(fams_u))
    if n_splits >= 2:
        gkf = GroupKFold(n_splits=n_splits)
        pg = np.full(len(y), np.nan)
        for tr, te in gkf.split(X, y, fams):
            mm = Ridge(alpha=1.0)
            mm.fit(X[tr], y[tr], sample_weight=weights[tr])
            pg[te] = mm.predict(X[te])
        summary["groupkfold_family"] = {
            "r2": float(r2_score(y, pg)),
            "mae": float(mean_absolute_error(y, pg)),
            "n": int(len(y)),
        }
    else:
        summary["groupkfold_family"] = {"r2": None, "mae": None, "n": int(len(y))}

    # (d) High-comparability only
    hm = weights > 1.0
    if int(hm.sum()) >= 3:
        Xh, yh = X[hm], y[hm]
        mh = Ridge(alpha=1.0)
        mh.fit(Xh, yh)
        ph = mh.predict(Xh)
        summary["high_weight_only"] = {
            "r2": float(r2_score(yh, ph)),
            "mae": float(mean_absolute_error(yh, ph)),
            "n": int(hm.sum()),
        }
    else:
        summary["high_weight_only"] = {"r2": None, "mae": None,
                                       "n": int(hm.sum())}

    summary["interpretation"] = (
        "Bridge is data-limited: C6 has n=43 total, only 7 high-comparability "
        "rows. R2 varies widely across validation schemes. Treated as SECONDARY, "
        "direction-only indicator. PRIMARY forecast = Q4 §5 direct "
        "compute-frontier extrapolation."
    )
    summary["primary_forecast"] = "Q4 §5 direct compute-frontier extrapolation"
    summary["secondary_role"] = "direction-only cross-check"
    return summary


def main():
    print("=" * 72)
    print("Q4 STEP 7: EXPLICIT Q3 → Q4 bridge")
    print("=" * 72)

    with open(Q3_POINT_SOLUTIONS, encoding="utf-8") as f:
        q3 = json.load(f)
    main_sols = q3["main_exponential"]
    print(f"\nQ3 main solutions loaded: {len(main_sols)} budgets")
    for s in main_sols:
        print(f"  Budget {s['budget']:.0e}: "
              f"Loss={s['loss']:.4f}, Q*={s['Q_star']:.3f}")

    c6 = pd.read_csv(C6_BRIDGE)
    loss_col = "Val_Loss"
    target_col = "LB_Average"
    model_col = "Model"
    compar_col = "Loss_Comparability"

    c6 = c6.dropna(subset=[loss_col, target_col]).copy()
    c6["family"] = c6[model_col].apply(family_from_name)
    if compar_col in c6.columns:
        hm = c6[compar_col].astype(str).str.contains("High", case=False, na=False)
        c6["weight"] = np.where(hm, 5.0, 1.0)
    else:
        c6["weight"] = 1.0

    X = c6[loss_col].to_numpy(dtype=float).reshape(-1, 1)
    y = c6[target_col].to_numpy(dtype=float)
    fams = c6["family"].to_numpy()
    weights = c6["weight"].to_numpy()

    ridge = Ridge(alpha=1.0)
    ridge.fit(X, y, sample_weight=weights)
    a_coef = float(ridge.intercept_)
    b_coef = float(ridge.coef_[0])
    print(f"\nBridge: LB = {a_coef:.4f} + {b_coef:.4f} * Val_Loss")

    pred_in = ridge.predict(X)
    r2_in = float(r2_score(y, pred_in, sample_weight=weights))
    mae_in = float(mean_absolute_error(y, pred_in, sample_weight=weights))
    print(f"  In-sample: R²={r2_in:.4f}, MAE={mae_in:.4f}")

    # ---- FIX #2 — Full 4-scheme validation ----
    print(f"\n--- Full validation (4 schemes) ---")
    validation = run_validation_schemes(X, y, fams, weights)
    for scheme in ["random_5fold", "groupkfold_family", "high_weight_only"]:
        s = validation[scheme]
        if s["r2"] is not None:
            print(f"  {scheme:22s}  R²={s['r2']:+.4f}  MAE={s['mae']:.4f}  "
                  f"n={s['n']}")

    q3_losses = np.array([s["loss"] for s in main_sols])
    q3_bench = ridge.predict(q3_losses.reshape(-1, 1))

    rng = np.random.default_rng(SEED)
    n_boot = 500
    fams_uniq = np.unique(fams)
    boot = np.full((n_boot, len(q3_losses)), np.nan)
    for b in range(n_boot):
        idx_f = rng.choice(fams_uniq, size=len(fams_uniq), replace=True)
        rows = []
        for ff in idx_f:
            r = np.where(fams == ff)[0]
            rows.extend(rng.choice(r, size=len(r), replace=True))
        rows = np.array(rows)
        try:
            m = Ridge(alpha=1.0)
            m.fit(X[rows], y[rows], sample_weight=weights[rows])
            boot[b] = m.predict(q3_losses.reshape(-1, 1))
        except Exception:
            pass
    valid = ~np.isnan(boot[:, 0])
    boot = boot[valid]
    print(f"\nBootstrap: {len(boot)} draws")

    print(f"\n{'Budget':>10} {'Q3 Loss':>10} {'Pred Bench':>12} "
          f"{'95% lo':>10} {'95% hi':>10}")
    print("-" * 60)
    rows = []
    for i, s in enumerate(main_sols):
        pred = float(q3_bench[i])
        lo, hi = (np.percentile(boot[:, i], [2.5, 97.5])
                  if len(boot) > 10 else (np.nan, np.nan))
        rows.append({
            "budget": s["budget"],
            "q3_loss": s["loss"],
            "q3_N_star": s["N_star"],
            "q3_D_star": s["D_star"],
            "q3_Q_star": s["Q_star"],
            "predicted_benchmark": pred,
            "ci95_low": float(lo),
            "ci95_high": float(hi),
        })
        print(f"{s['budget']:>10.0e} {s['loss']:>10.4f} {pred:>12.4f} "
              f"{lo:>10.4f} {hi:>10.4f}")

    with open(Q4_FORECAST, encoding="utf-8") as f:
        fc = json.load(f)
    print(f"\nQ4 direct forecast (for comparison):")
    print(f"  Base score: {fc['base_score']:.4f}")
    print(f"  gC: {fc['gC']:.6f}")
    for r in fc["forecasts"]:
        if "50%" in r["scenario"]:
            print(f"  50% slowdown S12={r['S12']:.2f}, S24={r['S24']:.2f}")

    Q4_MODELS.mkdir(parents=True, exist_ok=True)
    with open(Q4_MODELS / "q3_to_q4_bridge.json", "w", encoding="utf-8") as f:
        json.dump({
            "compliance": "explicit_Q3_loss_to_Q4_benchmark",
            "model_form": f"LB = {a_coef:.4f} + {b_coef:.4f} * Val_Loss",
            "intercept": a_coef,
            "slope": b_coef,
            "fit_stats": {
                "n_rows": int(len(c6)),
                "in_sample_r2": r2_in,
                "in_sample_mae": mae_in,
            },
            "validation_summary": validation,
            "q3_to_q4_predictions": rows,
            "q4_direct_forecast_reference": {
                "base_score": float(fc["base_score"]),
                "gC": float(fc["gC"]),
            },
        }, f, indent=2)
    print(f"\nSaved: {Q4_MODELS / 'q3_to_q4_bridge.json'}")

    Q4_TABLES.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(
        Q4_TABLES / "q3_to_q4_prediction.csv", index=False)
    print(f"Saved: {Q4_TABLES / 'q3_to_q4_prediction.csv'}")

    print("\n[DONE] 07_q3_to_q4_bridge.py")


if __name__ == "__main__":
    main()