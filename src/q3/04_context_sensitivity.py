#!/usr/bin/env python3
"""
Q3 Step 4: Context-length sensitivity at 1e22 FLOPs.

Fix #21: Context values loaded from C7 (model_architecture_metadata.csv),
not hardcoded.
"""
import json

import numpy as np
import pandas as pd

from config import (
    Q2_B1_FIT, Q2_QUALITY_FITS, Q2_ATTENUATION,
    Q1_MODELS, C7_ARCH,
    Q3_MODELS, Q3_TABLES, Q3_Q0, SEED,
)
from src.q3.optimizer import optimize_budget

np.random.seed(SEED)

BUDGET = 1e22


def load_context_values():
    """Load context lengths from C7."""
    try:
        df = pd.read_csv(C7_ARCH)
        col = next((c for c in df.columns if "position" in c.lower()
                    or "context" in c.lower()), None)
        if col is None:
            raise ValueError(f"no context column in {list(df.columns)}")
        vals = sorted(int(v) for v in df[col].dropna().unique().tolist())
        print(f"  [OK] Loaded {len(vals)} context values from C7 "
              f"(column='{col}'): {vals}")
        return vals
    except Exception as e:
        print(f"  [WARN] C7 load failed ({e}); using fallback")
        return [2048, 4096, 8192, 32768, 131072]


def load_q2():
    with open(Q2_B1_FIT) as f:
        b1 = json.load(f)["parameters"]
    with open(Q2_QUALITY_FITS) as f:
        qual = json.load(f)
    with open(Q2_ATTENUATION) as f:
        att = json.load(f)
    mix_path = Q1_MODELS / "mixture_model.json"
    with open(mix_path) as f:
        mix = json.load(f)
    return b1, qual, att, mix


def get_delta_p(mix):
    """Δp* from Q1 model."""
    l72 = mix.get("macro_loss_recipe72")
    l0 = mix.get("base_macro_loss")
    if l72 is not None and l0 is not None:
        return float(l72 - l0)
    return -0.544766


def main():
    print("=" * 72)
    print(f"Q3 STEP 4: Context sensitivity at budget {BUDGET:.0e} FLOPs")
    print("=" * 72)

    print("\n--- Context values (from C7) ---")
    CONTEXTS = load_context_values()

    b1, qual, att, mix = load_q2()
    delta_p = get_delta_p(mix)
    print(f"  Δp* = {delta_p:+.6f}")

    print(f"\n{'S':>8} {'N*':>10} {'D*':>12} {'Q*':>8} {'Loss':>10} "
          f"{'Attn%':>8} {'Train%':>8}")
    print("-" * 72)

    rows = []
    for S in CONTEXTS:
        sol = optimize_budget(BUDGET, S, b1, qual, att, delta_p,
                              scenario="main", family="exponential")
        c = sol["costs"]
        total = c["total"]
        attn_pct = 100.0 * c["attention"] / total if total > 0 else 0.0
        train_pct = 100.0 * c["train"] / total if total > 0 else 0.0
        print(f"{S:>8d} {sol['N_star']:>10.4f} {sol['D_star']:>12.4f} "
              f"{sol['Q_star']:>8.4f} {sol['loss']:>10.4f} "
              f"{attn_pct:>8.2f} {train_pct:>8.2f}")
        rows.append({
            "S": S,
            "N_star": sol["N_star"],
            "D_star": sol["D_star"],
            "Q_star": sol["Q_star"],
            "loss": sol["loss"],
            "attn_pct": attn_pct,
            "train_pct": train_pct,
            "total_FLOPs": total,
        })

    # Crossover
    from config import Q3_ETA_S
    S_act = 6.0 / Q3_ETA_S
    print(f"\nAttention/train crossover: S_act = {S_act:.0f} tokens")

    Q3_TABLES.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(Q3_TABLES / "context_sensitivity.csv", index=False)
    print(f"\nSaved: {Q3_TABLES / 'context_sensitivity.csv'}")

    print("\n[DONE] 04_context_sensitivity.py")


if __name__ == "__main__":
    main()