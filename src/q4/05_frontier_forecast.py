#!/usr/bin/env python3
"""
Q4 Step 5: Compute frontier trend + forecast.

Fixes (reviewer feedback):
  - #BENCH: BENCH_BASE derived from C2 frontier (historical max over a
    documented 2-year window of open-license models), not just the
    latest-90-day window (which can miss the frontier peak).
  - Loads bC, bt dynamically from decomposition.json.
  - Bridge-based forecast (Q3 loss → benchmark) added as SECONDARY
    alongside direct forecast.
"""
import json

import numpy as np
import pandas as pd

from config import (
    C4_EPOCH_MODELS, C2_LEADERBOARD,
    Q4_MODELS, Q4_TABLES, Q4_FORECAST, Q4_DECOMPOSITION,
    Q3_POINT_SOLUTIONS,
    SEED,
)

np.random.seed(SEED)

T0 = pd.Timestamp("2020-01-01")


def derive_bench_base():
    """
    Derive the benchmark-score baseline dynamically from C2.

    Rule (reviewer fix): frontier = max six-benchmark Average over the
    last 2 years of open-license models. This is the observed open-model
    frontier, not just the latest 90-day window (which can miss the peak).

    Falls back to 51.231 (documented historical value) if C2 loading fails.

    Returns: (float, str)
    """
    try:
        df = pd.read_csv(C2_LEADERBOARD, encoding="utf-8", low_memory=False)
        avg_col = next((c for c in df.columns if "average" in c.lower()), None)
        date_col = next((c for c in df.columns
                         if "date" in c.lower() and "submission" in c.lower()), None)
        license_col = next((c for c in df.columns
                            if "license" in c.lower()), None)

        if avg_col is None or date_col is None:
            return 51.231, "fallback: avg/date columns not found in C2"

        df["_d"] = pd.to_datetime(df[date_col], errors="coerce")
        df[avg_col] = pd.to_numeric(df[avg_col], errors="coerce")
        df = df.dropna(subset=["_d", avg_col])
        n_total = len(df)

        # Open-license filter (permissive proxy)
        open_licenses = {
            "apache-2.0", "mit", "bsd-3-clause-clear", "bsd-3-clause",
            "gpl-3.0", "afl-3.0", "osl-3.0", "wtfpl",
            "cc-by-4.0", "cc-by-sa-4.0", "openrail",
        }
        if license_col is not None:
            lv = df[license_col].astype(str).str.lower().str.strip()
            df_open = df[lv.isin(open_licenses)].copy()
            if len(df_open) >= 10:
                df = df_open

        # Frontier = max score over last 2 years of open-license models
        latest = df["_d"].max()
        cutoff = latest - pd.Timedelta(days=730)
        recent = df[df["_d"] >= cutoff].copy()
        if len(recent) == 0:
            recent = df  # fallback: use all

        idx_max = recent[avg_col].idxmax()
        val = float(recent.loc[idx_max, avg_col])
        top_model = str(recent.loc[idx_max].get("Model", "unknown"))
        top_date = recent.loc[idx_max]["_d"].date()

        desc = (
            f"derived from C2: n_total={n_total}, "
            f"2y_window_n={len(recent)}, "
            f"latest_date={latest.date()}, "
            f"top_model='{top_model}', "
            f"top_date={top_date}, "
            f"open_license_filter_applied={license_col is not None}"
        )
        return val, desc
    except Exception as e:
        return 51.231, f"fallback: {type(e).__name__}: {e}"


BENCH_BASE, BENCH_BASE_SOURCE = derive_bench_base()


def load_beta_from_decomposition():
    """Load (bC, bt) from decomposition.json; fallback to report values."""
    try:
        with open(Q4_DECOMPOSITION, "r", encoding="utf-8") as f:
            decomp = json.load(f)
        bC = float(decomp["beta_C"])
        bt = float(decomp["beta_t"])
        print(f"  Loaded from decomposition: bC={bC:.4f}, bt={bt:.4f}")
        return bC, bt
    except Exception as e:
        print(f"  [WARN] Could not load decomposition ({e}); using defaults")
        return 6.111, 3.294


def main():
    print("=" * 72)
    print("Q4 STEP 5: Frontier trend + forecast")
    print("=" * 72)
    print(f"Baseline source: {BENCH_BASE_SOURCE}")
    print(f"Baseline value:  {BENCH_BASE:.4f}")
    print()

    df = pd.read_csv(C4_EPOCH_MODELS, low_memory=False)
    print(f"C4 rows: {len(df)}")

    compute_col = "Training compute (FLOP)"
    date_col = "Publication date"
    open_col = "Open model weights?"
    domain_col = "Domain"

    df = df.dropna(subset=[compute_col, date_col]).copy()
    df[compute_col] = pd.to_numeric(df[compute_col], errors="coerce")
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[compute_col, date_col])
    df = df[df[compute_col] > 0]

    if domain_col in df.columns:
        df = df[df[domain_col].astype(str).str.contains("Language", na=False)]

    if open_col in df.columns:
        vals = df[open_col].astype(str).str.lower()
        open_mask = (vals.str.contains("yes", na=False)
                     | vals.str.contains("open weights", na=False)
                     | vals.str.contains("open", na=False))
        df_open = df[open_mask].copy()
        if len(df_open) >= 20:
            df = df_open
            print(f"Filtered to open models: {len(df)}")
        else:
            print(f"[WARN] open filter too aggressive ({len(df_open)}), keeping all")
    print(f"Language models with compute + date: {len(df)}")

    df["quarter"] = df[date_col].dt.to_period("Q")
    df["log10_C"] = np.log10(df[compute_col])

    quarters = sorted(df["quarter"].unique())
    rows = []
    for q in quarters:
        q_ts = q.to_timestamp()
        window = df[(df[date_col] >= q_ts - pd.Timedelta(days=365)) &
                    (df[date_col] <= q_ts)]
        if len(window) < 5:
            continue
        q95 = float(window[compute_col].quantile(0.95))
        rows.append({
            "quarter": str(q),
            "quarter_ts": q_ts,
            "t": (q_ts - T0).days / 365.25,
            "log10_q95": float(np.log10(q95)),
            "n": len(window),
        })
    roll = pd.DataFrame(rows)
    print(f"Rolling quarterly anchors: {len(roll)}")

    if len(roll) < 4:
        print("[FATAL] not enough rolling anchors; cannot fit trend")
        return

    t = roll["t"].to_numpy()
    y = roll["log10_q95"].to_numpy()
    X = np.column_stack([np.ones(len(t)), t])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    gC = float(beta[1])
    intercept = float(beta[0])

    yhat = X @ beta
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    print(f"\nTrend: log10(q95) = {intercept:.4f} + {gC:.4f} * t")
    print(f"  gC = {gC:.4f} log10(FLOP)/year")
    print(f"  R² = {r2:.4f}")

    # Newey-West HAC SE
    resid = y - yhat
    n = len(t)
    lag = 3
    S = np.zeros((2, 2))
    for i in range(n):
        xi = X[i]
        S += np.outer(xi, xi) * resid[i] ** 2
    for l in range(1, lag + 1):
        w = 1.0 - l / (lag + 1)
        for i in range(l, n):
            S += w * np.outer(X[i], X[i - l]) * resid[i] * resid[i - l]
            S += w * np.outer(X[i - l], X[i]) * resid[i] * resid[i - l]
    XtX_inv = np.linalg.pinv(X.T @ X)
    cov_hac = XtX_inv @ S @ XtX_inv * (n / (n - 2))
    se_hac = float(np.sqrt(cov_hac[1, 1]))
    lo_hac = gC - 1.96 * se_hac
    hi_hac = gC + 1.96 * se_hac
    print(f"  HAC(3) SE: {se_hac:.4f}  95% CI [{lo_hac:.4f}, {hi_hac:.4f}]")

    # Block bootstrap
    rng = np.random.default_rng(SEED)
    n_boot = 1000
    block = 4
    n_blocks = int(np.ceil(len(t) / block))
    gC_boot = []
    for _ in range(n_boot):
        idx = []
        for _ in range(n_blocks):
            start = rng.integers(0, len(t))
            idx.extend([(start + j) % len(t) for j in range(block)])
        idx = idx[:len(t)]
        b_b = np.linalg.lstsq(X[idx], y[idx], rcond=None)[0]
        gC_boot.append(b_b[1])
    gC_boot = np.array(gC_boot)
    lo_bb, hi_bb = np.percentile(gC_boot, [2.5, 97.5])
    print(f"  Block bootstrap 95% CI [{lo_bb:.4f}, {hi_bb:.4f}]")

    # Forecast (DIRECT — primary)
    print(f"\n--- Forecast scenarios (base = {BENCH_BASE:.4f}) ---")
    bC, bt = load_beta_from_decomposition()

    scenarios = [
        ("Reference (recent growth)", 1.0),
        ("50% slowdown", 0.5),
        ("75% slowdown", 0.25),
        ("Compute frontier frozen", 0.0),
    ]
    print(f"{'Scenario':30s} {'12m':>8s} {'24m':>8s}")
    print("-" * 50)
    fc_rows = []
    for name, s in scenarios:
        S12 = BENCH_BASE + (bC * gC * s + bt) * 1
        S24 = BENCH_BASE + (bC * gC * s + bt) * 2
        print(f"{name:30s} {S12:>8.2f} {S24:>8.2f}")
        fc_rows.append({"scenario": name, "s": s, "S12": S12, "S24": S24})

    # ==============================================================
    # Q3 → Q4 BRIDGE-BASED FORECAST (secondary, data-limited)
    # ==============================================================
    print(f"\n--- Bridge-based forecast (Q3 loss → benchmark) ---")
    bridge_forecast = None
    try:
        with open(Q3_POINT_SOLUTIONS, encoding="utf-8") as f:
            q3 = json.load(f)
        with open(Q4_MODELS / "q3_to_q4_bridge.json", encoding="utf-8") as f:
            bridge = json.load(f)

        slope = float(bridge["slope"])
        intercept_b = float(bridge["intercept"])
        q3_ref_loss = float(q3["main_exponential"][1]["loss"])  # 1e22

        bench_base_bridge = intercept_b + slope * q3_ref_loss
        print(f"  Q3 ref loss (1e22) = {q3_ref_loss:.4f}")
        print(f"  Bridge slope = {slope:.6f}, intercept = {intercept_b:.6f}")
        print(f"  Bridge base = {bench_base_bridge:.4f}")
        print(f"  NOTE: This base is far below observed frontier "
              f"({BENCH_BASE:.2f}); bridge is data-limited.")

        bridge_fc_rows = []
        print(f"\n{'Scenario':30s} {'12m':>8s} {'24m':>8s}  (bridge-based)")
        print("-" * 60)
        for name, s in scenarios:
            S12_b = bench_base_bridge + (bC * gC * s + bt) * 1
            S24_b = bench_base_bridge + (bC * gC * s + bt) * 2
            print(f"{name:30s} {S12_b:>8.2f} {S24_b:>8.2f}")
            bridge_fc_rows.append({"scenario": name, "s": s,
                                   "S12": S12_b, "S24": S24_b})
        bridge_forecast = {
            "base": float(bench_base_bridge),
            "source_q3_loss": float(q3_ref_loss),
            "slope": slope,
            "intercept": intercept_b,
            "note": "Secondary. C6 data-limited: only 43 rows, 7 high-quality.",
            "scenarios": bridge_fc_rows,
        }
    except Exception as e:
        import traceback
        print(f"  [SKIP] bridge-based forecast failed: {e}")
        traceback.print_exc()

    # Save
    Q4_MODELS.mkdir(parents=True, exist_ok=True)
    with open(Q4_FORECAST, "w", encoding="utf-8") as f:
        json.dump({
            "gC": gC,
            "intercept": intercept,
            "r2": r2,
            "hac_ci": [float(lo_hac), float(hi_hac)],
            "block_ci": [float(lo_bb), float(hi_bb)],
            "base_score": BENCH_BASE,
            "base_score_source": BENCH_BASE_SOURCE,
            "beta_C": bC,
            "beta_t": bt,
            "beta_source": "decomposition.json"
                if Q4_DECOMPOSITION.exists() else "default",
            "forecasts": fc_rows,
            "bridge_based_forecast": bridge_forecast,
        }, f, indent=2)
    print(f"\nSaved: {Q4_FORECAST}")

    Q4_TABLES.mkdir(parents=True, exist_ok=True)
    roll.to_csv(Q4_TABLES / "frontier_rolling.csv", index=False)
    pd.DataFrame(fc_rows).to_csv(Q4_TABLES / "forecast_scenarios.csv", index=False)
    print(f"Saved: {Q4_TABLES / 'frontier_rolling.csv'}")
    print(f"Saved: {Q4_TABLES / 'forecast_scenarios.csv'}")

    print("\n[DONE] 05_frontier_forecast.py")


if __name__ == "__main__":
    main()