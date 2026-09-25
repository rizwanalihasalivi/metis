#!/usr/bin/env python3
"""
Q3 Step 3: Bootstrap uncertainty propagation + re-optimization.

Fixes (reviewer feedback):
  - #Q3-1: Δp* now loaded dynamically from Q1 model (was hardcoded -0.544766).
  - Transition draws saved to CSV for reproducibility.
  - Bootstrap methodology explicitly labeled as parameter perturbation.
"""
import json

import numpy as np
import pandas as pd

from config import (
    Q2_B1_BOOTSTRAP, Q2_QUALITY_FITS, Q2_ATTENUATION,
    Q1_MODELS,
    Q3_MODELS, Q3_BOOTSTRAP, Q3_TABLES,
    Q3_S_DEFAULT, Q3_Q0, SEED,
)
from src.q3.cost_model import d_max, cost_breakdown
from src.q3.optimizer import optimize_budget, loss_main, loss_no_quality, loss_reverse

np.random.seed(SEED)

BUDGETS = [1e19, 1e22, 1e24]
N_BOOT = 250


def load_q2():
    boot = np.load(Q2_B1_BOOTSTRAP)
    boot_b1 = boot["boot"]
    with open(Q2_QUALITY_FITS) as f:
        qual = json.load(f)
    with open(Q2_ATTENUATION) as f:
        att = json.load(f)
    # Load Q1 mixture model for dynamic Δp*
    with open(Q1_MODELS / "mixture_model.json", encoding="utf-8") as f:
        mix = json.load(f)
    return boot_b1, qual, att, mix


def get_delta_p_base(mix):
    """Load Δp* from Q1 model. Fallback to -0.544766 if fields missing."""
    l72 = mix.get("macro_loss_recipe72")
    l0 = mix.get("base_macro_loss")
    if l72 is not None and l0 is not None:
        val = float(l72 - l0)
        print(f"  [OK] Δp* loaded from Q1 model: {val:+.6f}")
        return val
    print(f"  [WARN] Q1 model missing recipe-72 fields; fallback -0.544766")
    return -0.544766


def att_perturb(att, rng):
    """Perturb attenuation; preserve Q2 key names (N_c, not Nc)."""
    return {
        "omega_inf": float(np.clip(att["omega_inf"] + rng.normal(0, 0.01), 0.1, 0.5)),
        "N_c": float(np.clip(att["N_c"] * rng.uniform(0.85, 1.15), 0.15, 0.5)),
        "eta": float(np.clip(att["eta"] + rng.normal(0, 0.05), 0.5, 1.3)),
        "N0": att["N0"],
    }


def qual_perturb(qual, rng):
    def jit(est, lo, hi):
        std = (hi - lo) / 3.92
        return float(est + rng.normal(0, std))
    return {
        "main": {
            "G": {"estimate": jit(qual["main"]["G"]["estimate"],
                                  qual["main"]["G"]["ci_low"],
                                  qual["main"]["G"]["ci_high"])},
            "gamma": {"estimate": jit(qual["main"]["gamma"]["estimate"],
                                      qual["main"]["gamma"]["ci_low"],
                                      qual["main"]["gamma"]["ci_high"])},
        },
        "reverse": {
            "H": {"estimate": jit(qual["reverse"]["H"]["estimate"],
                                  qual["reverse"]["H"]["ci_low"],
                                  qual["reverse"]["H"]["ci_high"])},
            "delta": {"estimate": jit(qual["reverse"]["delta"]["estimate"],
                                      qual["reverse"]["delta"]["ci_low"],
                                      qual["reverse"]["delta"]["ci_high"])},
        },
    }


def b1_dict_from_row(row):
    return {
        "E": {"estimate": float(row[0])},
        "A": {"estimate": float(row[1])},
        "alpha": {"estimate": float(row[2])},
        "B": {"estimate": float(row[3])},
        "beta": {"estimate": float(row[4])},
    }


def classify_regime(Q_star, Q0=Q3_Q0, tol=1e-3):
    if abs(Q_star - Q0) < tol:
        return "baseline"
    if abs(Q_star - 1.0) < tol:
        return "saturated"
    return "interior"


def main():
    print("=" * 72)
    print("Q3 STEP 3: Bootstrap uncertainty + re-optimization")
    print("=" * 72)
    print("METHODOLOGY: Parameter perturbation bootstrap.")
    print("  Layers:")
    print("    1. B1 draw          (from Q2's trajectory-cluster bootstrap)")
    print("    2. Quality jitter   (G, gamma, H, delta)")
    print("    3. Attenuation jit  (omega_inf, N_c, eta)")
    print("    4. Mixture Δp* jit  (from Q1 model + small noise)")
    print("  NOTE: This is NOT a full joint upstream resample.")
    print("        Intervals are conditional re-optimization intervals")
    print("        under the current pipeline parameter uncertainty.")
    print()

    boot_b1, qual, att, mix = load_q2()
    n_b1 = len(boot_b1)
    print(f"B1 bootstrap draws available: {n_b1}")

    # Dynamic Δp* from Q1 model (reviewer fix #Q3-1)
    delta_p_base = get_delta_p_base(mix)
    print(f"  Using delta_p_base = {delta_p_base:+.6f}")

    rng = np.random.default_rng(SEED)

    records = []
    n_fail = 0

    for k in range(N_BOOT):
        b1_row = boot_b1[rng.integers(0, n_b1)]
        b1 = b1_dict_from_row(b1_row)
        q_draw = qual_perturb(qual, rng)
        a_draw = att_perturb(att, rng)
        delta_p = delta_p_base + rng.normal(0, 0.08)

        for C in BUDGETS:
            for scen in ["main", "reverse", "no_quality"]:
                try:
                    sol = optimize_budget(C, Q3_S_DEFAULT, b1, q_draw, a_draw,
                                          delta_p, scenario=scen,
                                          family="exponential")
                    records.append({
                        "draw": k,
                        "budget": C,
                        "scenario": scen,
                        "N_star": sol["N_star"],
                        "D_star": sol["D_star"],
                        "Q_star": sol["Q_star"],
                        "loss": sol["loss"],
                        "regime": classify_regime(sol["Q_star"]),
                    })
                except Exception as e:
                    n_fail += 1
                    if k < 3:
                        print(f"  [WARN] draw {k}, C={C:.0e}, scen={scen}: "
                              f"{type(e).__name__}: {e}")

        if (k + 1) % 50 == 0:
            print(f"  Completed {k+1}/{N_BOOT} draws ({len(records)} records, {n_fail} fails)")

    df = pd.DataFrame(records)
    print(f"\nTotal draws collected: {len(df)}")

    if len(df) == 0:
        print("[FATAL] No draws collected — check warnings above.")
        return

    # ==================================================================
    # Main scenario: quantile summary
    # ==================================================================
    print("\n" + "=" * 72)
    print("Main B6/B7 scenario — re-optimized 95% intervals")
    print("=" * 72)
    print(f"{'Budget':>10} {'N* median':>12} {'N* [95%]':>26} "
          f"{'D* median':>12} {'D* [95%]':>26} {'Q* median':>10}")
    print("-" * 105)

    for C in BUDGETS:
        sub = df[(df.budget == C) & (df.scenario == "main")]
        if len(sub) == 0:
            continue
        N_med = sub.N_star.median()
        N_lo, N_hi = sub.N_star.quantile([0.025, 0.975])
        D_med = sub.D_star.median()
        D_lo, D_hi = sub.D_star.quantile([0.025, 0.975])
        Q_med = sub.Q_star.median()
        print(f"{C:>10.0e} {N_med:>12.4f} [{N_lo:>11.4f}, {N_hi:>11.4f}] "
              f"{D_med:>12.2f} [{D_lo:>11.2f}, {D_hi:>11.2f}] {Q_med:>10.4f}")

    # ==================================================================
    # Regime probabilities
    # ==================================================================
    print("\n" + "=" * 72)
    print("Regime probabilities")
    print("=" * 72)
    print(f"{'Scenario':>12} {'Budget':>10} {'baseline':>10} "
          f"{'interior':>10} {'saturated':>10}")
    print("-" * 60)

    for scen in ["main", "reverse", "no_quality"]:
        for C in BUDGETS:
            sub = df[(df.budget == C) & (df.scenario == scen)]
            if len(sub) == 0:
                continue
            counts = sub.regime.value_counts(normalize=True)
            p_base = counts.get("baseline", 0.0)
            p_int = counts.get("interior", 0.0)
            p_sat = counts.get("saturated", 0.0)
            print(f"{scen:>12} {C:>10.0e} {p_base:>10.3f} "
                  f"{p_int:>10.3f} {p_sat:>10.3f}")

    # ==================================================================
    # Transition budgets (main scenario)
    # ==================================================================
    print("\n" + "=" * 72)
    print("Transition budgets (main scenario)")
    print("=" * 72)
    print("Sweeping budgets per draw...")

    transition_records = []
    sweep_budgets = np.logspace(18.5, 21.5, 40)
    leave_q0_draws = []
    reach_q1_draws = []

    for k in range(min(120, N_BOOT)):
        b1_row = boot_b1[rng.integers(0, n_b1)]
        b1 = b1_dict_from_row(b1_row)
        q_draw = qual_perturb(qual, rng)
        a_draw = att_perturb(att, rng)
        delta_p = delta_p_base + rng.normal(0, 0.08)

        leave_found = False
        reach_found = False
        for C in sweep_budgets:
            try:
                sol = optimize_budget(C, Q3_S_DEFAULT, b1, q_draw, a_draw,
                                      delta_p, scenario="main")
                Q_star = sol["Q_star"]
                if not leave_found and Q_star > Q3_Q0 + 1e-3:
                    leave_q0_draws.append(C)
                    transition_records.append({
                        "draw": k, "transition": "leave_Q0", "budget": float(C),
                    })
                    leave_found = True
                if not reach_found and Q_star > 1.0 - 1e-3:
                    reach_q1_draws.append(C)
                    transition_records.append({
                        "draw": k, "transition": "reach_Q1", "budget": float(C),
                    })
                    reach_found = True
                if leave_found and reach_found:
                    break
            except Exception:
                pass

    if leave_q0_draws:
        lo, hi = np.percentile(leave_q0_draws, [2.5, 97.5])
        print(f"\nLeave Q0 threshold:")
        print(f"  median = {np.median(leave_q0_draws):.3e}")
        print(f"  95% CI = [{lo:.3e}, {hi:.3e}]")

    if reach_q1_draws:
        lo, hi = np.percentile(reach_q1_draws, [2.5, 97.5])
        print(f"\nReach Q=1 threshold:")
        print(f"  median = {np.median(reach_q1_draws):.3e}")
        print(f"  95% CI = [{lo:.3e}, {hi:.3e}]")

    # ==================================================================
    # Save
    # ==================================================================
    Q3_MODELS.mkdir(parents=True, exist_ok=True)
    Q3_TABLES.mkdir(parents=True, exist_ok=True)

    df.to_csv(Q3_BOOTSTRAP, index=False)
    print(f"\nSaved: {Q3_BOOTSTRAP}")

    if transition_records:
        trans_df = pd.DataFrame(transition_records)
        trans_df.to_csv(Q3_TABLES / "transition_draws.csv", index=False)
        print(f"Saved: {Q3_TABLES / 'transition_draws.csv'} "
              f"({len(trans_df)} draws)")

    sweep_settings = {
        "methodology": "parameter_perturbation_bootstrap",
        "layers": ["B1_draw", "quality_jitter", "attenuation_jitter",
                   "mixture_delta_p_jitter"],
        "n_draws": min(120, N_BOOT),
        "sweep_log10_start": 18.5,
        "sweep_log10_end": 21.5,
        "sweep_n_points": 40,
        "delta_p_base_used": float(delta_p_base),
        "leave_Q0_thresholds": {
            "median": float(np.median(leave_q0_draws)) if leave_q0_draws else None,
            "ci95_low": float(np.percentile(leave_q0_draws, 2.5)) if leave_q0_draws else None,
            "ci95_high": float(np.percentile(leave_q0_draws, 97.5)) if leave_q0_draws else None,
        },
        "reach_Q1_thresholds": {
            "median": float(np.median(reach_q1_draws)) if reach_q1_draws else None,
            "ci95_low": float(np.percentile(reach_q1_draws, 2.5)) if reach_q1_draws else None,
            "ci95_high": float(np.percentile(reach_q1_draws, 97.5)) if reach_q1_draws else None,
        },
    }
    with open(Q3_MODELS / "transition_sweep_settings.json", "w", encoding="utf-8") as f:
        json.dump(sweep_settings, f, indent=2)
    print(f"Saved: {Q3_MODELS / 'transition_sweep_settings.json'}")

    summary = df.groupby(["scenario", "budget"]).agg(
        N_med=("N_star", "median"),
        N_lo=("N_star", lambda x: x.quantile(0.025)),
        N_hi=("N_star", lambda x: x.quantile(0.975)),
        D_med=("D_star", "median"),
        Q_med=("Q_star", "median"),
        loss_med=("loss", "median"),
    ).reset_index()
    summary.to_csv(Q3_TABLES / "bootstrap_summary.csv", index=False)
    print(f"Saved: {Q3_TABLES / 'bootstrap_summary.csv'}")

    print("\n[DONE] 03_uncertainty.py")


if __name__ == "__main__":
    main()