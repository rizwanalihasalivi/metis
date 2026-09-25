#!/usr/bin/env python3
"""Q4 Step 2: Capability metric from C2 + C8 validation (with repo dedup)."""
import re

import numpy as np
import pandas as pd

from config import (
    C2_LEADERBOARD, Q4_C8_MODEL_AGG, Q4_C8_PRIMARY,
    Q4_MODELS, Q4_TABLES, SEED,
)
from scipy.stats import spearmanr, pearsonr

np.random.seed(SEED)

BENCH_COLS_FULL = ["IFEval", "BBH", "MATH", "GPQA", "MUSR", "MMLU-PRO"]


def normalize_key(s):
    s = str(s).lower().strip()
    s = re.sub(r"[/_\s]+", "-", s)
    s = re.sub(r"-+", "-", s)
    return s


def find_bench_columns(df):
    cols = list(df.columns)
    found = {}
    for target in BENCH_COLS_FULL:
        pat = re.compile(rf"^{re.escape(target)}\b", re.IGNORECASE)
        for c in cols:
            if pat.match(c.strip()):
                found[target] = c
                break
    if "MATH" not in found:
        for c in cols:
            if c.strip().lower().startswith("math"):
                found["MATH"] = c
                break
    return found


def load_c2():
    print("=" * 72)
    print("Q4 STEP 2: Capability metric (C2)")
    print("=" * 72)
    df = pd.read_csv(C2_LEADERBOARD, encoding="utf-8")
    print(f"C2 rows: {len(df)}")

    col_map = find_bench_columns(df)
    print(f"Benchmark columns found: {col_map}")

    avg_col = next((c for c in df.columns if "average" in c.lower()), None)
    par_col = next((c for c in df.columns if "params" in c.lower()), None)
    date_col = next((c for c in df.columns
                     if "date" in c.lower() and "submission" in c.lower()), None)
    model_col = "Model"

    print(f"avg_col='{avg_col}'  par_col='{par_col}'  date_col='{date_col}'")

    bench_cols = [col_map[k] for k in BENCH_COLS_FULL if k in col_map]
    df["S_recomputed"] = df[bench_cols].mean(axis=1, skipna=False)

    max_diff = float((df["S_recomputed"] - df[avg_col]).abs().max())
    print(f"Recomputed vs supplied Average: max diff = {max_diff:.3e}")

    df = df.rename(columns={
        model_col: "model_name",
        par_col: "N_params_B",
        date_col: "submission_date",
        avg_col: "Average_reported",
    })
    df["S"] = df["S_recomputed"]
    return df, bench_cols, col_map


def validate_c8(c2_df):
    print("\n" + "=" * 72)
    print("C8 task-level validation (with repo dedup)")
    print("=" * 72)

    if not Q4_C8_MODEL_AGG.exists():
        print(f"  [SKIP] {Q4_C8_MODEL_AGG.name} missing")
        return None

    c8 = pd.read_csv(Q4_C8_MODEL_AGG)
    print(f"C8 aggregate rows: {len(c8)}")

    macro_col = next((c for c in c8.columns
                      if "macro" in c.lower() or "average" in c.lower()
                      or "score" in c.lower()), None)
    repo_col = next((c for c in c8.columns
                     if "repo" in c.lower() or "model" in c.lower()
                     or "dir" in c.lower()), None)
    print(f"macro_col='{macro_col}'  repo_col='{repo_col}'")

    if repo_col is None or macro_col is None:
        return None

    c2_df["_key"] = c2_df["model_name"].apply(normalize_key)
    c8["_key"] = c8[repo_col].apply(normalize_key)

    common = set(c2_df["_key"]) & set(c8["_key"])
    print(f"Common keys: {len(common)}")

    merged = c2_df[["_key", "S"]].merge(
        c8[["_key", macro_col]].rename(columns={macro_col: "C8_macro"}),
        on="_key", how="inner",
    )
    n_pre = len(merged)
    print(f"Matched rows (pre-dedup): {n_pre}")

    # Repository-level dedup: keep highest S per repo key
    merged = (merged.sort_values("S", ascending=False)
                    .drop_duplicates(subset=["_key"], keep="first")
                    .copy())
    n_post = len(merged)
    print(f"Matched rows (post-dedup): {n_post}")

    if n_post < 3:
        print("  [SKIP] too few matches after dedup")
        return None

    r_p = float(pearsonr(merged["S"], merged["C8_macro"])[0])
    r_s = float(spearmanr(merged["S"], merged["C8_macro"]).correlation)
    mae = float(np.mean(np.abs(merged["S"] - merged["C8_macro"])))
    print(f"\nC2 vs C8 validation (dedup):")
    print(f"  Pearson  = {r_p:.4f}")
    print(f"  Spearman = {r_s:.4f}")
    print(f"  MAE      = {mae:.4f}")

    return {"n_pre_dedup": n_pre, "n_post_dedup": n_post,
            "pearson": r_p, "spearman": r_s, "mae": mae}


def main():
    df, bench_cols, col_map = load_c2()
    print(f"\nS stats: mean={df['S'].mean():.4f}  std={df['S'].std():.4f}")

    val = validate_c8(df)

    Q4_MODELS.mkdir(parents=True, exist_ok=True)
    out = df[["model_name", "N_params_B", "submission_date",
              "S", "Average_reported"] + bench_cols].copy()
    out.to_parquet(Q4_MODELS / "capability_panel.parquet", index=False)
    print(f"\nSaved: {Q4_MODELS / 'capability_panel.parquet'}")

    Q4_TABLES.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{
        "metric": "recomputed_vs_reported_max_abs_diff",
        "value": float((df["S"] - df["Average_reported"]).abs().max()),
    }]).to_csv(Q4_TABLES / "capability_check.csv", index=False)

    if val:
        pd.DataFrame([val]).to_csv(Q4_TABLES / "c8_validation.csv", index=False)
        print(f"Saved: {Q4_TABLES / 'c8_validation.csv'}")

    print("\n[DONE] 02_capability_metric.py")


if __name__ == "__main__":
    main()