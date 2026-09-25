#!/usr/bin/env python3
"""Q4 Step 4: Loss-to-Benchmark bridge (C6)."""
import json

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import KFold, GroupKFold

from config import C6_BRIDGE, Q4_MODELS, Q4_TABLES, Q4_BRIDGE, SEED

np.random.seed(SEED)


def family_bucket(name):
    s = str(name).split("/")[-1].lower()
    tok = s.split("-")[0]
    for pref in ["llama", "qwen", "gemma", "mistral", "phi", "pythia",
                 "olmo", "yi", "falcon", "gpt", "deepseek"]:
        if tok.startswith(pref):
            return pref
    return tok


def main():
    print("=" * 72)
    print("Q4 STEP 4: Loss-to-Benchmark bridge")
    print("=" * 72)

    df = pd.read_csv(C6_BRIDGE)
    print(f"C6 rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    loss_col = next((c for c in df.columns
                     if "loss" in c.lower() and "val" in c.lower()), None)
    if loss_col is None:
        loss_col = next(c for c in df.columns if "loss" in c.lower())

    bench_cols = [c for c in df.columns if c.startswith("LB_")]
    compar_col = next((c for c in df.columns if "comparab" in c.lower()), None)
    model_col = next((c for c in df.columns
                      if c.lower() in ("model", "model_name")), None)

    print(f"loss_col='{loss_col}'  n_bench={len(bench_cols)}")
    print(f"compar_col='{compar_col}'  model_col='{model_col}'")

    if compar_col:
        high_mask = df[compar_col].astype(str).str.contains("High", case=False, na=False)
        med_mask = df[compar_col].astype(str).str.contains("Medium", case=False, na=False)
        print(f"High rows: {int(high_mask.sum())}  Medium rows: {int(med_mask.sum())}")
    else:
        high_mask = pd.Series(False, index=df.index)

    target_col = next((c for c in bench_cols if "average" in c.lower()), bench_cols[0])
    print(f"Target: {target_col}")

    X_loss = df[loss_col].to_numpy(dtype=float)
    y = df[target_col].to_numpy(dtype=float)
    fams = df[model_col].apply(family_bucket).to_numpy() if model_col else np.arange(len(df))

    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    preds_random = np.full(len(y), np.nan)
    for tr, te in kf.split(X_loss):
        iso = IsotonicRegression(increasing=False, out_of_bounds="clip")
        iso.fit(X_loss[tr], y[tr])
        preds_random[te] = iso.predict(X_loss[te])
    r2_rand = float(r2_score(y, preds_random))
    mae_rand = float(mean_absolute_error(y, preds_random))
    print(f"\nRandom 5-fold:  R2={r2_rand:.4f}  MAE={mae_rand:.4f}")

    n_fams = len(np.unique(fams))
    n_splits = min(5, n_fams)
    if n_splits >= 2:
        gkf = GroupKFold(n_splits=n_splits)
        preds_group = np.full(len(y), np.nan)
        for tr, te in gkf.split(X_loss, y, fams):
            iso = IsotonicRegression(increasing=False, out_of_bounds="clip")
            iso.fit(X_loss[tr], y[tr])
            preds_group[te] = iso.predict(X_loss[te])
        r2_grp = float(r2_score(y, preds_group))
        mae_grp = float(mean_absolute_error(y, preds_group))
    else:
        r2_grp, mae_grp = float("nan"), float("nan")
    print(f"GroupKFold:     R2={r2_grp:.4f}  MAE={mae_grp:.4f}")
    print(f"  (report: Random MAE 7.51, Family MAE 9.11, R2 0.091)")

    w = np.where(high_mask, 5.0, 1.0)
    iso_w = IsotonicRegression(increasing=False, out_of_bounds="clip")
    iso_w.fit(X_loss, y, sample_weight=w)
    preds_w = iso_w.predict(X_loss)
    r2_w = float(r2_score(y, preds_w))
    mae_w = float(mean_absolute_error(y, preds_w))
    print(f"High-weight (w=5): R2={r2_w:.4f}  MAE={mae_w:.4f}")

    iso_full = IsotonicRegression(increasing=False, out_of_bounds="clip")
    iso_full.fit(X_loss, y)

    Q4_MODELS.mkdir(parents=True, exist_ok=True)
    with open(Q4_BRIDGE, "w", encoding="utf-8") as f:
        json.dump({
            "n_rows": int(len(df)),
            "target_col": target_col,
            "loss_col": loss_col,
            "cv_random": {"r2": r2_rand, "mae": mae_rand},
            "cv_groupkfold": {"r2": r2_grp, "mae": mae_grp},
            "high_wt": {"r2": r2_w, "mae": mae_w},
            "loss_grid": iso_full.X_thresholds_.tolist(),
            "pred_grid": iso_full.y_thresholds_.tolist(),
        }, f, indent=2)
    print(f"\nSaved: {Q4_BRIDGE}")

    Q4_TABLES.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{
        "n": len(df),
        "cv_random_r2": r2_rand, "cv_random_mae": mae_rand,
        "cv_groupkfold_r2": r2_grp, "cv_groupkfold_mae": mae_grp,
        "highwt_r2": r2_w, "highwt_mae": mae_w,
    }]).to_csv(Q4_TABLES / "bridge_validation.csv", index=False)
    print(f"Saved: {Q4_TABLES / 'bridge_validation.csv'}")
    print("\n[DONE] 04_loss_benchmark_bridge.py")


if __name__ == "__main__":
    main()