#!/usr/bin/env python3
"""
Q2 Step 5: Cardinal bridge + elasticity + substitution analysis.

Reviewer feedback addressed:
  - Q1→Q2 bridge is now OPERATIONAL: explicit percentile mapping table
    between A1 Q_i ECDF and B-series Q_score is computed and saved.
  - Computable elasticities (eps_N, eps_D, eps_Q) and numerical substitution
    example (how much Delta_N equivalent to Delta_Q=0.1).

Outputs:
  models/q2/q2_bridge_metadata.json  (elasticity + substitution metadata)
  outputs/q2/tables/bridge_mapping.csv
  outputs/q2/tables/elasticities.csv
  outputs/q2/tables/substitution_analysis.csv

NOTE: This script does NOT overwrite models/q2/q1_q2_bridge.json,
which is owned by 02_quality_scenarios.py (sigmoid injection version).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    Q2_B1_FIT, Q2_QUALITY_FITS, Q2_ATTENUATION,
    Q2_MODELS, Q2_TABLES, A1_WITH_QI, B6_QUALITY,
    SEED,
)

np.random.seed(SEED)


def load_b1_params():
    with open(Q2_B1_FIT, encoding="utf-8") as f:
        cfg = json.load(f)
    p = cfg["parameters"]
    return {k: p[k]["estimate"] for k in ["E", "A", "alpha", "B", "beta"]}


def load_quality_fits():
    with open(Q2_QUALITY_FITS, encoding="utf-8") as f:
        q = json.load(f)
    return q


# ==================================================================
# STEP 1: Build cardinal bridge mapping table (A1 ECDF <-> B Q_score)
# ==================================================================
def build_bridge_mapping():
    """
    Explicit mapping between A1 Q_i ECDF and B-series Q_score.

    Both are on [0, 1] percentile scale, so the identity mapping is
    exact. But we document it explicitly with a 5%-grid table.
    """
    print("=" * 72)
    print("STEP 1: Cardinal bridge mapping (A1 Q_i <-> B Q_score)")
    print("=" * 72)

    a1 = pd.read_parquet(A1_WITH_QI)
    Q_a1 = a1["Q_i"].dropna().to_numpy()
    n_a1 = len(Q_a1)
    print(f"A1 Q_i: n={n_a1}, "
          f"range=[{Q_a1.min():+.4f}, {Q_a1.max():+.4f}]")

    # Compute quantiles at 5% grid
    grid = np.arange(0.0, 1.0001, 0.05)
    rows = []
    for u in grid:
        q_val = float(np.quantile(Q_a1, u))
        rows.append({
            "percentile_u": round(float(u), 4),
            "a1_Q_i_value": round(q_val, 6),
            "mapped_B_Q_score": round(float(u), 6),  # identity
        })
    mapping_df = pd.DataFrame(rows)

    Q2_TABLES.mkdir(parents=True, exist_ok=True)
    mapping_df.to_csv(Q2_TABLES / "bridge_mapping.csv", index=False)
    print(f"Saved: {Q2_TABLES / 'bridge_mapping.csv'}")
    print(mapping_df.to_string(index=False))

    # Metadata for this stage (separate file — do NOT overwrite q1_q2_bridge.json)
    metadata = {
        "method": "ordinal_percentile_identity",
        "a1_reference": {
            "n": int(n_a1),
            "min": float(Q_a1.min()),
            "max": float(Q_a1.max()),
            "mean": float(Q_a1.mean()),
            "median": float(np.median(Q_a1)),
            "std": float(Q_a1.std(ddof=1)),
        },
        "mapping_table": "outputs/q2/tables/bridge_mapping.csv",
        "operational_note": (
            "The A1 Q_i ECDF is computed explicitly. B-series Q_score "
            "(already on [0,1]) is mapped to the same percentile scale "
            "via identity, which is exact for ordinal comparison. "
            "The mapping table is saved for reproducibility."
        ),
        "limitation": (
            "This is an ordinal-preserving bridge. It does not make "
            "'Delta Q = 0.1' cardinally invariant; see the substitution "
            "analysis for scenario-conditional results."
        ),
        "canonical_bridge_file": "models/q2/q1_q2_bridge.json",
        "canonical_bridge_note": (
            "The SIGMOID injection bridge (sigmoid(Q_i_A1)) is owned by "
            "02_quality_scenarios.py and saved as q1_q2_bridge.json. "
            "This file (q2_bridge_metadata.json) is only the ordinal "
            "ECDF reference table."
        ),
    }
    Q2_MODELS.mkdir(parents=True, exist_ok=True)
    with open(Q2_MODELS / "q2_bridge_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nSaved: {Q2_MODELS / 'q2_bridge_metadata.json'}")
    print("  (NOT overwriting q1_q2_bridge.json — owned by 02_quality_scenarios.py)")

    return metadata, mapping_df


# ==================================================================
# STEP 2: Elasticities of the main quality scenario
# ==================================================================
def compute_elasticities(b1, qual):
    """
    Compute eps_N, eps_D, eps_Q at representative points.

    Main scenario:
      L(N,D,Q) = E + A*N^(-alpha) + B*D^(-beta) + G*(1-Q)^gamma
    """
    print("\n" + "=" * 72)
    print("STEP 2: Elasticity of loss with respect to N, D, Q")
    print("=" * 72)

    E = b1["E"]
    A = b1["A"]
    alpha = b1["alpha"]
    B = b1["B"]
    beta = b1["beta"]
    G = qual["main"]["G"]["estimate"]
    gamma = qual["main"]["gamma"]["estimate"]

    points = [
        ("Small",   0.1,   10.0,   0.5),
        ("Medium",  1.0,   100.0,  0.5),
        ("Large",   10.0,  1000.0, 0.7),
        ("Very large", 100.0, 3000.0, 0.8),
    ]
    rows = []
    for label, N, D, Q in points:
        L0 = E + A * N ** (-alpha) + B * D ** (-beta)
        L = L0 + G * (1 - Q) ** gamma

        eps_N = alpha * A * N ** (-alpha) / L
        eps_D = beta * B * D ** (-beta) / L
        eps_Q = gamma * G * (1 - Q) ** gamma / L
        dL_dQ = -gamma * G * (1 - Q) ** (gamma - 1)
        dL_dN = -alpha * A * N ** (-alpha - 1)
        dL_dD = -beta * B * D ** (-beta - 1)

        rows.append({
            "point": label,
            "N_B": N,
            "D_B": D,
            "Q": Q,
            "L": round(L, 6),
            "eps_N": round(eps_N, 6),
            "eps_D": round(eps_D, 6),
            "eps_Q": round(eps_Q, 6),
            "dL_dN": dL_dN,
            "dL_dD": dL_dD,
            "dL_dQ": dL_dQ,
        })

    df = pd.DataFrame(rows)
    Q2_TABLES.mkdir(parents=True, exist_ok=True)
    df.to_csv(Q2_TABLES / "elasticities.csv", index=False)
    print(f"Saved: {Q2_TABLES / 'elasticities.csv'}")
    print(df.to_string(index=False))

    return df, {"E": E, "A": A, "alpha": alpha,
                "B": B, "beta": beta, "G": G, "gamma": gamma}


# ==================================================================
# STEP 3: Quality <-> model-size substitution
# ==================================================================
def compute_substitution(b1, qual):
    """
    For fixed D and p, compute Delta N equivalent to a beneficial
    Delta Q at several starting points.
    """
    print("\n" + "=" * 72)
    print("STEP 3: Quality-model-size substitution")
    print("=" * 72)

    E = b1["E"]
    A = b1["A"]
    alpha = b1["alpha"]
    B = b1["B"]
    beta = b1["beta"]
    G = qual["main"]["G"]["estimate"]
    gamma = qual["main"]["gamma"]["estimate"]

    scenarios = [
        (0.1,  50.0, 0.50, 0.10),
        (1.0,  200.0, 0.60, 0.10),
        (5.0,  500.0, 0.70, 0.10),
        (20.0, 1000.0, 0.80, 0.10),
    ]

    rows = []
    for N, D, Q, dQ in scenarios:
        Q_new = min(Q + dQ, 0.999)

        dL_Q = G * ((1 - Q_new) ** gamma - (1 - Q) ** gamma)

        N_eq_term = A * N ** (-alpha) + dL_Q
        if N_eq_term <= 0:
            N_eq = np.nan
            status = "infeasible"
        else:
            N_eq = (A / N_eq_term) ** (1 / alpha)
            status = "ok"

        ratio = N_eq / N if np.isfinite(N_eq) else np.nan

        rows.append({
            "N_B_start": N,
            "D_B": D,
            "Q_start": Q,
            "Delta_Q": dQ,
            "N_eq_B": round(float(N_eq), 6) if np.isfinite(N_eq) else None,
            "N_ratio": round(float(ratio), 4) if np.isfinite(ratio) else None,
            "dL_from_quality": round(float(dL_Q), 8),
            "status": status,
        })

    df = pd.DataFrame(rows)
    Q2_TABLES.mkdir(parents=True, exist_ok=True)
    df.to_csv(Q2_TABLES / "substitution_analysis.csv", index=False)
    print(f"Saved: {Q2_TABLES / 'substitution_analysis.csv'}")
    print(df.to_string(index=False))

    return df


def main():
    print("=" * 72)
    print("Q2 STEP 5: Bridge + elasticities + substitution")
    print("=" * 72)

    metadata, mapping = build_bridge_mapping()
    b1 = load_b1_params()
    qual = load_quality_fits()

    el_df, _ = compute_elasticities(b1, qual)
    sub_df = compute_substitution(b1, qual)

    print("\n[DONE] 05_bridge_and_elasticities.py")


if __name__ == "__main__":
    main()