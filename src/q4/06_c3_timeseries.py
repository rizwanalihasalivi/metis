#!/usr/bin/env python3
"""
Q4 Step 6: Historical dynamics from C3 (timeseries).

Report Table 3: yearly descriptive frontier statistics.
C3 file: leaderboard_extended_timeseries.csv (mixed, 2019-2025).
"""
import json

import numpy as np
import pandas as pd

from config import C_DATA, Q4_MODELS, Q4_TABLES, SEED

np.random.seed(SEED)

C3_PATH = C_DATA / "leaderboard_extended_timeseries.csv"


def find_avg_col(df):
    """Robust average column lookup."""
    for c in df.columns:
        if c.strip().lower() == "average" or "average" in c.lower():
            return c
    # Fallback: compute mean of 6 benchmarks
    bench = []
    for tgt in ["IFEval", "BBH", "MATH", "GPQA", "MUSR", "MMLU"]:
        for c in df.columns:
            if c.strip().lower().startswith(tgt.lower()):
                bench.append(c)
                break
    if len(bench) >= 3:
        return bench
    return None


def find_year_col(df):
    """Find year column (either 'Year' or derived from date)."""
    for c in df.columns:
        if c.strip().lower() == "year":
            return c
    for c in df.columns:
        if "date" in c.lower():
            return c
    return None


def main():
    print("=" * 72)
    print("Q4 STEP 6: Historical dynamics (C3)")
    print("=" * 72)

    if not C3_PATH.exists():
        print(f"[FATAL] not found: {C3_PATH}")
        return

    df = pd.read_csv(C3_PATH, low_memory=False)
    print(f"C3 rows: {len(df)}")
    print(f"C3 columns: {list(df.columns)}")

    year_col = find_year_col(df)
    avg_col = find_avg_col(df)

    if year_col is None:
        print("[FATAL] no year/date column found")
        return
    if avg_col is None:
        print("[FATAL] no average column found")
        return

    print(f"\nyear_col = '{year_col}'")

    if isinstance(avg_col, list):
        print(f"avg_cols = {avg_col}")
        df["_avg"] = df[avg_col].mean(axis=1, skipna=False)
    else:
        print(f"avg_col = '{avg_col}'")
        df["_avg"] = df[avg_col]

    # Handle Year column directly OR parse date
    if year_col.strip().lower() == "year":
        df["_year"] = pd.to_numeric(df[year_col], errors="coerce")
    else:
        df["_date"] = pd.to_datetime(df[year_col], errors="coerce")
        df["_year"] = df["_date"].dt.year

    df = df.dropna(subset=["_year", "_avg"])
    df["_year"] = df["_year"].astype(int)
    print(f"Valid rows with year + avg: {len(df)}")
    print(f"Year range: {int(df['_year'].min())} - {int(df['_year'].max())}")

    # Yearly descriptive frontier stats
    print("\n" + "=" * 72)
    print("Yearly descriptive frontier statistics")
    print("=" * 72)

    rows = []
    print(f"\n{'Year':>6} {'n':>6} {'Median':>10} {'q95':>10} {'Max':>10}")
    print("-" * 50)
    for year in sorted(df["_year"].unique()):
        sub = df[df["_year"] == year]
        n = len(sub)
        if n == 0:
            continue
        med = float(sub["_avg"].median())
        q95 = float(sub["_avg"].quantile(0.95))
        mx = float(sub["_avg"].max())
        rows.append({
            "year": int(year), "n": int(n),
            "median": med, "q95": q95, "max": mx,
        })
        print(f"{year:>6d} {n:>6d} {med:>10.4f} {q95:>10.4f} {mx:>10.4f}")

    print("\n(report Table 3):")
    print("  2019   n=1    median=5.00    q95=5.00     max=5.00")
    print("  2020   n=15   median=0.00    q95=50.00    max=50.00")
    print("  2021   n=2    median=3.50    q95=4.85     max=5.00")
    print("  2022   n=7    median=12.00   q95=33.50    max=35.00")
    print("  2023   n=13   median=12.00   q95=29.00    max=35.00")
    print("  2024   n=26   median=7.32    q95=21.10    max=38.58")
    print("  2025   n=1902 median=23.48   q95=41.25    max=47.22")

    # Pre-2024 vs post-2024 shift
    print("\n" + "=" * 72)
    print("Pre-2024 vs 2024+ regime shift")
    print("=" * 72)
    pre = df[df["_year"] < 2024]
    post = df[df["_year"] >= 2024]
    print(f"Pre-2024: n={len(pre)}  median={pre['_avg'].median():.3f}  "
          f"max={pre['_avg'].max():.3f}")
    print(f"2024+:    n={len(post)}  median={post['_avg'].median():.3f}  "
          f"max={post['_avg'].max():.3f}")
    print("\nNote: pre-2024 rows are mixed papers/reports; not directly comparable")
    print("      to 2024-2025 leaderboard population. This non-monotone early")
    print("      pattern motivates NOT fitting a single causal time trend from 2019.")

    # Save
    Q4_MODELS.mkdir(parents=True, exist_ok=True)
    with open(Q4_MODELS / "c3_timeseries.json", "w", encoding="utf-8") as f:
        json.dump({
            "n_total": len(df),
            "year_range": [int(df["_year"].min()), int(df["_year"].max())],
            "yearly_stats": rows,
            "pre_2024": {
                "n": len(pre),
                "median": float(pre["_avg"].median()),
                "max": float(pre["_avg"].max()),
            },
            "post_2024": {
                "n": len(post),
                "median": float(post["_avg"].median()),
                "max": float(post["_avg"].max()),
            },
            "note": "Pre-2024 rows are mixed sources; not directly comparable.",
        }, f, indent=2)
    print(f"\nSaved: {Q4_MODELS / 'c3_timeseries.json'}")

    Q4_TABLES.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(Q4_TABLES / "c3_yearly_stats.csv", index=False)
    print(f"Saved: {Q4_TABLES / 'c3_yearly_stats.csv'}")

    print("\n[DONE] 06_c3_timeseries.py")


if __name__ == "__main__":
    main()