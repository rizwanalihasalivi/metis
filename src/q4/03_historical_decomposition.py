#!/usr/bin/env python3
"""
Q4 Step 3: Historical observational decomposition.

Primary: S = b0 + bC*log10(C) + bt*t + e
Secondary: add family fixed effects.
gC loaded dynamically from forecast.json (no hardcoding).
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    Q4_MODELS, Q4_TABLES, Q4_DECOMPOSITION, Q4_FORECAST,
    C2_LEADERBOARD, C4_EPOCH_MODELS, SEED,
)

np.random.seed(SEED)

T0 = pd.Timestamp("2020-01-01")


def load_gC():
    """Load current gC from forecast.json; fallback: recompute from C4."""
    try:
        with open(Q4_FORECAST, "r", encoding="utf-8") as f:
            gC = float(json.load(f)["gC"])
        print(f"  [OK] Loaded gC={gC:.6f} from forecast.json")
        return gC
    except Exception as e:
        print(f"  [WARN] forecast.json unavailable ({e}); computing gC from C4...")

    # Fallback: recompute from C4
    df = pd.read_csv(C4_EPOCH_MODELS, low_memory=False)
    compute_col = "Training compute (FLOP)"
    date_col = "Publication date"
    domain_col = "Domain"
    open_col = "Open model weights?"

    df = df.dropna(subset=[compute_col, date_col]).copy()
    df[compute_col] = pd.to_numeric(df[compute_col], errors="coerce")
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[compute_col, date_col])
    df = df[df[compute_col] > 0]
    if domain_col in df.columns:
        df = df[df[domain_col].astype(str).str.contains("Language", na=False)]
    if open_col in df.columns:
        v = df[open_col].astype(str).str.lower()
        df = df[v.str.contains("yes|open", na=False, regex=True)]

    df["_q"] = df[date_col].dt.to_period("Q")
    rows = []
    for q in sorted(df["_q"].unique()):
        ts = q.to_timestamp()
        w = df[(df[date_col] >= ts - pd.Timedelta(days=365)) &
               (df[date_col] <= ts)]
        if len(w) < 5:
            continue
        q95 = float(w[compute_col].quantile(0.95))
        rows.append({"t": (ts - T0).days / 365.25, "y": np.log10(q95)})
    r = pd.DataFrame(rows)
    X = np.column_stack([np.ones(len(r)), r["t"].to_numpy()])
    b = np.linalg.lstsq(X, r["y"].to_numpy(), rcond=None)[0]
    gC = float(b[1])
    print(f"  [OK] Computed gC={gC:.6f} from C4 fallback")
    return gC


def normalize_key(s):
    s = str(s).lower().strip()
    if "/" in s:
        s = s.split("/")[-1]
    s = re.sub(r"[/_\s]+", "-", s)
    s = re.sub(r"-+", "-", s)
    return s


def fuzzy_key(s):
    s = normalize_key(s)
    s = re.sub(r"-(v\d+(\.\d+)?|instruct|chat|base|hf|it|sft|dpo|rlhf|preview)$", "", s)
    return s


def family_from_name(name):
    s = str(name).split("/")[-1].lower()
    tok = s.split("-")[0]
    for pref in ["llama", "qwen", "gemma", "mistral", "phi", "pythia",
                 "olmo", "yi", "falcon", "gpt", "deepseek"]:
        if tok.startswith(pref):
            return pref
    return tok


def load_c4():
    print("Loading C4 (Epoch AI models)...")
    df = pd.read_csv(C4_EPOCH_MODELS, low_memory=False)
    print(f"  rows: {len(df)}")

    def find_col(cands):
        for c in cands:
            if c in df.columns:
                return c
        return None

    model_col = "Model"
    compute_col = find_col(["Training compute (FLOP)"])
    date_col = find_col(["Publication date"])
    params_col = find_col(["Parameters"])
    open_col = find_col(["Open model weights?"])
    domain_col = find_col(["Domain"])

    keep = [model_col, compute_col, date_col, params_col, open_col, domain_col]
    keep = [c for c in keep if c is not None]
    df = df[keep].copy()

    if domain_col and domain_col in df.columns:
        df = df[df[domain_col].astype(str).str.contains("Language", na=False)]

    df = df.dropna(subset=[model_col, compute_col, date_col]).copy()
    df[compute_col] = pd.to_numeric(df[compute_col], errors="coerce")
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[compute_col, date_col])
    df = df[df[compute_col] > 0]

    df["_key"] = df[model_col].apply(normalize_key)
    df["_fkey"] = df[model_col].apply(fuzzy_key)
    df["family"] = df[model_col].apply(family_from_name)
    df["t_years"] = (df[date_col] - T0).dt.days / 365.25
    df["log10_C"] = np.log10(df[compute_col])
    print(f"  C4 usable: {len(df)}")
    return df


def load_c2_panel():
    panel = pd.read_parquet(Q4_MODELS / "capability_panel.parquet")
    panel["_key"] = panel["model_name"].apply(normalize_key)
    panel["_fkey"] = panel["model_name"].apply(fuzzy_key)
    panel["family"] = panel["model_name"].apply(family_from_name)
    return panel


def merge_c2_c4(panel, c4):
    print("\nMerging C2 capability panel with C4 compute metadata...")
    panel_keys = set(panel["_key"])
    c4_keys = set(c4["_key"])
    common = panel_keys & c4_keys
    print(f"  common exact keys: {len(common)}")

    c4_keep = c4[["_key", "log10_C", "t_years", "family"]].rename(
        columns={"family": "family_c4"})
    c4_keep = c4_keep.drop_duplicates(subset=["_key"], keep="first")
    merged = panel.merge(c4_keep, on="_key", how="inner",
                         suffixes=("", "_c4"))
    print(f"  exact matches: {len(merged)}")

    print("  [FUZZY] computing alias sensitivity sample...")
    panel_fkeys = set(panel["_fkey"])
    c4_fkeys = set(c4["_fkey"])
    common_f = panel_fkeys & c4_fkeys
    print(f"  common fuzzy keys: {len(common_f)}")

    c4_fuzzy = c4[["_fkey", "log10_C", "t_years", "family"]].rename(
        columns={"family": "family_c4"})
    c4_fuzzy = c4_fuzzy.drop_duplicates(subset=["_fkey"], keep="first")
    merged_fuzzy = panel.merge(c4_fuzzy, on="_fkey", how="inner",
                                suffixes=("", "_c4"))
    print(f"  fuzzy sensitivity matches: {len(merged_fuzzy)}")
    print(f"  PRIMARY sample uses exact keys only: {len(merged)}")

    merged = merged.drop_duplicates(subset=["model_name"], keep="first").copy()
    print(f"  after dedup: {len(merged)}")

    if "t_years" not in merged.columns or merged["t_years"].isna().all():
        t0 = pd.Timestamp("2020-01-01")
        merged["t_years"] = (pd.to_datetime(merged["submission_date"]) - t0).dt.days / 365.25

    if "family_c4" in merged.columns:
        merged["family_used"] = merged["family_c4"].fillna(merged["family"])
    else:
        merged["family_used"] = merged["family"]

    merged = merged.dropna(subset=["S", "log10_C", "t_years"]).copy()
    merged = merged[merged["t_years"] >= 0]
    print(f"  final: {len(merged)}  unique families: {merged['family_used'].nunique()}")

    Q4_MODELS.mkdir(parents=True, exist_ok=True)
    merged_fuzzy.drop_duplicates(subset=["model_name"], keep="first").to_parquet(
        Q4_MODELS / "matched_panel_fuzzy.parquet", index=False)

    return merged


def cluster_ols(y, X, groups):
    XtX = X.T @ X
    XtX_inv = np.linalg.pinv(XtX)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    n, k = X.shape
    meat = np.zeros((k, k))
    for g in np.unique(groups):
        mask = groups == g
        Xg = X[mask]
        rg = resid[mask]
        s = Xg.T @ rg
        meat += np.outer(s, s)
    G = len(np.unique(groups))
    corr = (G / (G - 1)) * ((n - 1) / (n - k)) if G > 1 else 1.0
    cov = XtX_inv @ meat @ XtX_inv * corr
    return beta, np.sqrt(np.diag(cov))


def compute_r2(y, yhat):
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def main():
    print("=" * 72)
    print("Q4 STEP 3: Historical decomposition")
    print("=" * 72)

    gC = load_gC()

    c4 = load_c4()
    panel = load_c2_panel()
    print(f"\nC2 panel: {len(panel)} rows")

    merged = merge_c2_c4(panel, c4)
    if len(merged) < 10:
        print(f"[FATAL] too few matches ({len(merged)})")
        return

    y = merged["S"].to_numpy(dtype=float)
    X_base = np.column_stack([
        np.ones(len(merged)),
        merged["log10_C"].to_numpy(dtype=float),
        merged["t_years"].to_numpy(dtype=float),
    ])
    groups = merged["family_used"].to_numpy()

    beta, se = cluster_ols(y, X_base, groups)
    b0, bC, bt = beta
    s0, sC, st = se
    r2 = compute_r2(y, X_base @ beta)

    print(f"\nPrimary decomposition (n={len(merged)}, G={len(np.unique(groups))}):")
    print(f"  beta0  = {b0:+.4f}  (SE {s0:.4f})")
    print(f"  beta_C = {bC:+.4f}  (SE {sC:.4f})  95% CI [{bC-1.96*sC:+.3f}, {bC+1.96*sC:+.3f}]")
    print(f"  beta_t = {bt:+.4f}  (SE {st:.4f})  95% CI [{bt-1.96*st:+.3f}, {bt+1.96*st:+.3f}]")
    print(f"  R^2    = {r2:.4f}")

    # Family FE
    print(f"\n--- Secondary: with family fixed effects ---")
    fam_dummies = pd.get_dummies(merged["family_used"], prefix="fam",
                                  drop_first=True, dtype=float)
    X_fe = np.column_stack([X_base, fam_dummies.to_numpy()])
    beta_fe, se_fe = cluster_ols(y, X_fe, groups)
    bC_fe, bt_fe = beta_fe[1], beta_fe[2]
    sC_fe, st_fe = se_fe[1], se_fe[2]
    r2_fe = compute_r2(y, X_fe @ beta_fe)

    print(f"  beta_C = {bC_fe:+.4f}  (SE {sC_fe:.4f})")
    print(f"  beta_t = {bt_fe:+.4f}  (SE {st_fe:.4f})")
    print(f"  R^2    = {r2_fe:.4f}  (ΔR²={r2_fe - r2:+.4f})")

    # LOFO
    lofo_C, lofo_t = [], []
    for g in np.unique(groups):
        mask = groups != g
        if mask.sum() < 10:
            continue
        b_g, _ = cluster_ols(y[mask], X_base[mask], groups[mask])
        lofo_C.append(b_g[1])
        lofo_t.append(b_g[2])

    # CV
    from sklearn.model_selection import GroupKFold, KFold
    from sklearn.metrics import r2_score, mean_absolute_error

    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    preds_random = np.zeros(len(y))
    for tr, te in kf.split(X_base):
        b_k = np.linalg.lstsq(X_base[tr], y[tr], rcond=None)[0]
        preds_random[te] = X_base[te] @ b_k
    r2_rand = float(r2_score(y, preds_random))
    mae_rand = float(mean_absolute_error(y, preds_random))
    print(f"\n  Random 5-fold:  R2={r2_rand:.4f}  MAE={mae_rand:.4f}")

    n_groups = len(np.unique(groups))
    n_splits = min(5, n_groups)
    if n_splits >= 2:
        gkf = GroupKFold(n_splits=n_splits)
        preds_group = np.zeros(len(y))
        for tr, te in gkf.split(X_base, y, groups):
            b_k = np.linalg.lstsq(X_base[tr], y[tr], rcond=None)[0]
            preds_group[te] = X_base[te] @ b_k
        r2_grp = float(r2_score(y, preds_group))
        mae_grp = float(mean_absolute_error(y, preds_group))
    else:
        r2_grp, mae_grp = float("nan"), float("nan")
    print(f"  GroupKFold:     R2={r2_grp:.4f}  MAE={mae_grp:.4f}")

    # 2024 holdout
    sub = merged.copy()
    sub["_year"] = pd.to_datetime(sub["submission_date"], errors="coerce").dt.year
    train_mask = (sub["_year"] < 2024).to_numpy()
    test_mask = (sub["_year"] == 2024).to_numpy()
    n_tr, n_te = int(train_mask.sum()), int(test_mask.sum())
    print(f"\n  2024 holdout: train={n_tr}, test={n_te}")
    if n_tr >= 5 and n_te >= 3:
        b_h = np.linalg.lstsq(X_base[train_mask], y[train_mask], rcond=None)[0]
        pred_h = X_base[test_mask] @ b_h
        r2_h = float(r2_score(y[test_mask], pred_h))
        mae_h = float(mean_absolute_error(y[test_mask], pred_h))
        print(f"  R2={r2_h:.4f}  MAE={mae_h:.4f}")
    else:
        r2_h, mae_h = float("nan"), float("nan")
        print(f"  [SKIP] insufficient rows")

    # Bootstrap scale share — uses dynamic gC
    print(f"\nBootstrap scale share (gC={gC:.4f}):")
    n_boot = 500
    shares = []
    family_list = np.unique(groups)
    rng = np.random.default_rng(SEED)
    for _ in range(n_boot):
        idx_fams = rng.choice(family_list, size=len(family_list), replace=True)
        idx_rows = []
        for f in idx_fams:
            rows_f = np.where(groups == f)[0]
            idx_rows.extend(rng.choice(rows_f, size=len(rows_f), replace=True))
        idx_rows = np.array(idx_rows)
        try:
            b_b, _ = cluster_ols(y[idx_rows], X_base[idx_rows], groups[idx_rows])
            sC_b = b_b[1] * gC
            sT_b = b_b[2]
            share = sC_b / (sC_b + sT_b) if (sC_b + sT_b) != 0 else 0.0
            shares.append(share)
        except Exception:
            pass

    shares = np.array(shares)
    lo, hi = np.percentile(shares, [2.5, 97.5]) if len(shares) > 10 else (float("nan"), float("nan"))
    point_share = (bC * gC) / (bC * gC + bt)
    print(f"  point scale share = {point_share*100:.1f}%  (dynamic gC)")
    print(f"  95% CI = [{lo*100:.1f}%, {hi*100:.1f}%]")

    # Save
    Q4_MODELS.mkdir(parents=True, exist_ok=True)
    with open(Q4_DECOMPOSITION, "w", encoding="utf-8") as f:
        json.dump({
            "n_models": int(len(merged)),
            "n_families": int(len(np.unique(groups))),
            "gC_used": float(gC),
            "beta0": float(b0), "beta_C": float(bC), "beta_t": float(bt),
            "se_beta_C": float(sC), "se_beta_t": float(st),
            "r2": float(r2),
            "family_fe": {
                "beta_C": float(bC_fe), "beta_t": float(bt_fe),
                "se_beta_C": float(sC_fe), "se_beta_t": float(st_fe),
                "r2": float(r2_fe), "delta_r2": float(r2_fe - r2),
            },
            "lofo_beta_C_range": [float(min(lofo_C)), float(max(lofo_C))] if lofo_C else None,
            "lofo_beta_t_range": [float(min(lofo_t)), float(max(lofo_t))] if lofo_t else None,
            "cv_random": {"r2": r2_rand, "mae": mae_rand},
            "cv_groupkfold": {"r2": r2_grp, "mae": mae_grp},
            "cv_2024_holdout": {"r2": r2_h, "mae": mae_h},
            "scale_share_point": float(point_share),
            "scale_share_ci95": [float(lo), float(hi)],
        }, f, indent=2)
    print(f"\nSaved: {Q4_DECOMPOSITION}")

    Q4_TABLES.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([
        {"spec": "primary_no_FE", "beta_C": bC, "beta_t": bt, "r2": r2},
        {"spec": "secondary_family_FE", "beta_C": bC_fe, "beta_t": bt_fe, "r2": r2_fe},
    ]).to_csv(Q4_TABLES / "decomposition.csv", index=False)

    merged.to_parquet(Q4_MODELS / "matched_panel.parquet", index=False)
    print(f"\n[DONE] 03_historical_decomposition.py")


if __name__ == "__main__":
    main()