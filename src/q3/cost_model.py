#!/usr/bin/env python3
"""
Q3 Step 1: Compute accounting model + quality cost functions.

Cost formulas (N, D in billions):
  C_train = 6e18 * N * D
  C_Q     = 1e9 * D * max(g(Q) - g(Q0), 0)
  C_attn  = eta * S * 1e18 * N * D
  Budget:  D * [1e18 * N * (6 + eta*S) + 1e9 * dg(Q)] <= C

Quality cost families (from competition):
  g_exp(Q)  = 1e7 * exp(6 Q)
  g_pow(Q)  = 5e9 * Q^4
  g_log(Q)  = 2e9 * ln(1 + 10 Q)
"""
import json

import numpy as np

from config import Q3_MODELS, Q3_Q0, Q3_ETA_S, SEED

np.random.seed(SEED)

C_TRAIN_COEF = 6e18
C_Q_COEF = 1e9
C_ATTN_COEF = 1e18


def g_exp(Q):
    return 1e7 * np.exp(6.0 * Q)


def g_pow(Q):
    return 5e9 * np.power(Q, 4.0)


def g_log(Q):
    return 2e9 * np.log1p(10.0 * Q)


G_FAMILIES = {
    "exponential": g_exp,
    "power": g_pow,
    "logarithmic": g_log,
}


def delta_g(Q, family="exponential", Q0=Q3_Q0):
    """max(g(Q) - g(Q0), 0)"""
    fn = G_FAMILIES[family]
    return np.maximum(fn(Q) - fn(Q0), 0.0)


def g_prime(Q, family="exponential"):
    if family == "exponential":
        return 1e7 * 6.0 * np.exp(6.0 * Q)
    if family == "power":
        return 5e9 * 4.0 * np.power(Q, 3.0)
    if family == "logarithmic":
        return 2e9 * 10.0 / (1.0 + 10.0 * Q)
    raise ValueError(family)


def attention_coef(S, eta=Q3_ETA_S):
    return C_ATTN_COEF * eta * S


def d_max(N, Q, C, S, family="exponential", Q0=Q3_Q0):
    """Max D allowed by budget constraint."""
    a_S = C_TRAIN_COEF + attention_coef(S)
    b = C_Q_COEF * delta_g(Q, family, Q0)
    denom = a_S * N + b
    return C / np.maximum(denom, 1e-30)


def cost_breakdown(N, D, Q, S, family="exponential", Q0=Q3_Q0):
    """Return dict of compute components and total."""
    C_train = C_TRAIN_COEF * N * D
    C_attn = attention_coef(S) * N * D
    C_Q = C_Q_COEF * D * delta_g(Q, family, Q0)
    return {
        "train": float(C_train),
        "attention": float(C_attn),
        "quality": float(C_Q),
        "total": float(C_train + C_attn + C_Q),
    }


def main():
    print("=" * 72)
    print("Q3 STEP 1: Cost model check")
    print("=" * 72)

    # Sanity check at Q0
    Q0 = Q3_Q0
    for fam in G_FAMILIES:
        v = delta_g(Q0, fam)
        print(f"  delta_g(Q0=0.5, {fam:12s}) = {v:.4e}  (expect 0)")

    print("\n--- Quality-cost scale at Q0 -> 1 ---")
    print(f"{'family':12s}  {'g_prime(0.5)':>14s}  {'g_prime(1)':>14s}  {'increment':>14s}")
    for fam in G_FAMILIES:
        gp0 = float(g_prime(Q0, fam))
        gp1 = float(g_prime(1.0, fam))
        inc = float(delta_g(1.0, fam))
        print(f"{fam:12s}  {gp0:14.4e}  {gp1:14.4e}  {inc:14.4e}")

    print("\n--- Attention/train crossover ---")
    S_act = C_TRAIN_COEF / (C_ATTN_COEF * Q3_ETA_S)
    print(f"  S_act = 6 / eta = {S_act:.1f} tokens (expect 30000)")

    # Save reference values
    Q3_MODELS.mkdir(parents=True, exist_ok=True)
    with open(Q3_MODELS / "cost_model.json", "w") as f:
        json.dump({
            "C_TRAIN_COEF": C_TRAIN_COEF,
            "C_Q_COEF": C_Q_COEF,
            "C_ATTN_COEF": C_ATTN_COEF,
            "eta_S": Q3_ETA_S,
            "S_act_attention_train_crossover": float(S_act),
            "Q0": Q0,
        }, f, indent=2)
    print(f"\nSaved: {Q3_MODELS / 'cost_model.json'}")
    print("\n[DONE] 01_cost_model.py")


if __name__ == "__main__":
    main()