#!/usr/bin/env python3
"""Q3 Step 2: Conditional (N, D, Q) optimizer.

Fixes:
  - #15: Δp* computed from Q1 mixture model (loaded from Q1_MODELS).
  - #16: All 3 cost families optimized.
  - #20: All scenarios saved together.
"""
import json

import numpy as np
from scipy.optimize import minimize

from config import (
    Q2_B1_FIT, Q2_QUALITY_FITS, Q2_ATTENUATION,
    Q1_MODELS,  # <-- Q1's mixture_model.json lives here
    Q3_MODELS, Q3_POINT_SOLUTIONS, Q3_TABLES,
    Q3_S_DEFAULT, Q3_Q0, SEED,
)
from src.q3.cost_model import d_max, cost_breakdown, delta_g

np.random.seed(SEED)

BUDGETS = [1e19, 1e22, 1e24]
COST_FAMILIES = ["exponential", "power", "logarithmic"]


def load_q2():
    with open(Q2_B1_FIT) as f:
        b1 = json.load(f)["parameters"]
    with open(Q2_QUALITY_FITS) as f:
        qual = json.load(f)
    with open(Q2_ATTENUATION) as f:
        att = json.load(f)
    # Mixture model lives in Q1's folder
    mix_path = Q1_MODELS / "mixture_model.json"
    if not mix_path.exists():
        raise FileNotFoundError(
            f"Q1 mixture model missing: {mix_path}\n"
            f"Run: python -m src.q1.06_mixture_model first"
        )
    with open(mix_path) as f:
        mix = json.load(f)
    return b1, qual, att, mix


def macro_perturbation(mix):
    """
    Δ_p*(p* - p0) = L(p*_recipe72) - L(p0) from Q1 model.

    Loads from Q1's mixture_model.json:
      - macro_loss_recipe72 : nonlinear model eval at recipe 72
      - base_macro_loss     : nonlinear model eval at Aitchison center

    If Q1 did not save recipe-72 field, falls back to hard-coded
    documented value -0.544766 (from Q1/Q3 report).
    """
    l72 = mix.get("macro_loss_recipe72")
    l0 = mix.get("base_macro_loss")

    if l72 is not None and l0 is not None:
        val = float(l72 - l0)
        print(f"  [OK] Δp* from Q1 model:")
        print(f"        L(recipe72) = {l72:.6f}")
        print(f"        L(center)   = {l0:.6f}")
        print(f"        Δp* = {val:+.6f}")
        return val

    print(f"  [WARN] Q1 mixture_model.json missing recipe-72 fields.")
    print(f"         Using documented fallback = -0.544766")
    return -0.544766


def omega(N, att):
    w_inf = att["omega_inf"]
    Nc = att["N_c"]
    eta = att["eta"]
    N0 = att["N0"]
    return w_inf + (1 - w_inf) * (1 + (N0 / Nc) ** eta) / (1 + (N / Nc) ** eta)


def loss_main(N, D, Q, b1, qual, att, delta_p):
    L0 = (b1["E"]["estimate"]
          + b1["A"]["estimate"] * N ** (-b1["alpha"]["estimate"])
          + b1["B"]["estimate"] * D ** (-b1["beta"]["estimate"]))
    G = qual["main"]["G"]["estimate"]
    gamma = qual["main"]["gamma"]["estimate"]
    dQ = G * (1.0 - Q) ** gamma
    return L0 + dQ + omega(N, att) * delta_p


def loss_no_quality(N, D, Q, b1, qual, att, delta_p):
    L0 = (b1["E"]["estimate"]
          + b1["A"]["estimate"] * N ** (-b1["alpha"]["estimate"])
          + b1["B"]["estimate"] * D ** (-b1["beta"]["estimate"]))
    return L0 + omega(N, att) * delta_p


def loss_reverse(N, D, Q, b1, qual, att, delta_p):
    L0 = (b1["E"]["estimate"]
          + b1["A"]["estimate"] * N ** (-b1["alpha"]["estimate"])
          + b1["B"]["estimate"] * D ** (-b1["beta"]["estimate"]))
    H = qual["reverse"]["H"]["estimate"]
    delta = qual["reverse"]["delta"]["estimate"]
    dQ = -H * (1.0 - Q) ** delta
    return L0 + dQ + omega(N, att) * delta_p


LOSS_FUNCS = {
    "main": loss_main,
    "reverse": loss_reverse,
    "no_quality": loss_no_quality,
}


def optimize_budget(C, S, b1, qual, att, delta_p, scenario="main",
                    family="exponential", Q0=Q3_Q0):
    log_N_bounds = (np.log(0.001), np.log(10000.0))
    Q_bounds = (Q0, 1.0)
    loss_fn = LOSS_FUNCS[scenario]

    def obj(theta):
        log_N, Q = theta
        N = np.exp(log_N)
        D = d_max(N, Q, C, S, family, Q0)
        return loss_fn(N, D, Q, b1, qual, att, delta_p)

    best = None
    starts = [
        (np.log(0.1), 0.5),
        (np.log(1.0), 0.7),
        (np.log(10.0), 0.9),
        (np.log(50.0), 0.5),
        (np.log(1.0), 1.0),
    ]
    for s in starts:
        res = minimize(obj, s,
                       bounds=[log_N_bounds, Q_bounds],
                       method="L-BFGS-B",
                       options={"ftol": 1e-12, "gtol": 1e-10})
        if best is None or res.fun < best.fun:
            best = res

    N_star = float(np.exp(best.x[0]))
    Q_star = float(best.x[1])
    D_star = float(d_max(N_star, Q_star, C, S, family, Q0))
    L_star = float(best.fun)
    costs = cost_breakdown(N_star, D_star, Q_star, S, family, Q0)
    return {
        "budget": float(C),
        "S": int(S),
        "scenario": scenario,
        "family": family,
        "N_star": N_star,
        "D_star": D_star,
        "Q_star": Q_star,
        "loss": L_star,
        "costs": costs,
    }


def main():
    print("=" * 72)
    print("Q3 STEP 2: Point optimization")
    print("=" * 72)

    b1, qual, att, mix = load_q2()
    delta_p = macro_perturbation(mix)
    print(f"B1: E={b1['E']['estimate']:.6f} A={b1['A']['estimate']:.6f} "
          f"alpha={b1['alpha']['estimate']:.6f}")
    print(f"    B={b1['B']['estimate']:.6f} beta={b1['beta']['estimate']:.6f}")
    print(f"Main: G={qual['main']['G']['estimate']:.6f} "
          f"gamma={qual['main']['gamma']['estimate']:.6f}")
    print(f"Attenuation: w_inf={att['omega_inf']:.6f} Nc={att['N_c']:.6f} "
          f"eta={att['eta']:.6f}")
    print(f"Delta_p* = {delta_p:.6f}")

    # --- Main ---
    print("\n--- Main B6/B7 scenario, S=4096, exponential ---")
    print(f"{'Budget':>10} {'N*':>10} {'D*':>12} {'Q*':>8} {'Loss':>10}")
    print("-" * 56)
    main_solutions = []
    for C in BUDGETS:
        sol = optimize_budget(C, Q3_S_DEFAULT, b1, qual, att, delta_p,
                              scenario="main", family="exponential")
        main_solutions.append(sol)
        print(f"{C:>10.0e} {sol['N_star']:>10.4f} {sol['D_star']:>12.4f} "
              f"{sol['Q_star']:>8.4f} {sol['loss']:>10.4f}")

    # --- Reverse ---
    print("\n--- Reverse B8 scenario ---")
    print(f"{'Budget':>10} {'N*':>10} {'D*':>12} {'Q*':>8} {'Loss':>10}")
    print("-" * 56)
    reverse_solutions = []
    for C in BUDGETS:
        sol = optimize_budget(C, Q3_S_DEFAULT, b1, qual, att, delta_p,
                              scenario="reverse", family="exponential")
        reverse_solutions.append(sol)
        print(f"{C:>10.0e} {sol['N_star']:>10.4f} {sol['D_star']:>12.4f} "
              f"{sol['Q_star']:>8.4f} {sol['loss']:>10.4f}")

    # --- No-quality ---
    print("\n--- No-quality scenario ---")
    print(f"{'Budget':>10} {'N*':>10} {'D*':>12} {'Q*':>8} {'Loss':>10}")
    print("-" * 56)
    noqual_solutions = []
    for C in BUDGETS:
        sol = optimize_budget(C, Q3_S_DEFAULT, b1, qual, att, delta_p,
                              scenario="no_quality", family="exponential")
        noqual_solutions.append(sol)
        print(f"{C:>10.0e} {sol['N_star']:>10.4f} {sol['D_star']:>12.4f} "
              f"{sol['Q_star']:>8.4f} {sol['loss']:>10.4f}")

    # --- All cost families ---
    print("\n" + "=" * 72)
    print("All cost families (main scenario, S=4096)")
    print("=" * 72)
    all_families = {}
    for fam_name in COST_FAMILIES:
        print(f"\n--- {fam_name} ---")
        print(f"{'Budget':>10} {'N*':>10} {'D*':>12} {'Q*':>8} {'Loss':>10}")
        print("-" * 56)
        fam_sols = []
        for C in BUDGETS:
            try:
                sol = optimize_budget(C, Q3_S_DEFAULT, b1, qual, att,
                                       delta_p, scenario="main",
                                       family=fam_name)
                fam_sols.append(sol)
                print(f"{C:>10.0e} {sol['N_star']:>10.4f} "
                      f"{sol['D_star']:>12.4f} {sol['Q_star']:>8.4f} "
                      f"{sol['loss']:>10.4f}")
            except Exception as e:
                print(f"{C:>10.0e} [FAIL] {type(e).__name__}: {e}")
        all_families[fam_name] = fam_sols

    # --- Save ---
    Q3_MODELS.mkdir(parents=True, exist_ok=True)
    with open(Q3_POINT_SOLUTIONS, "w", encoding="utf-8") as f:
        json.dump({
            "delta_p_star": float(delta_p),
            "main_exponential": main_solutions,
            "reverse_exponential": reverse_solutions,
            "no_quality_exponential": noqual_solutions,
            "all_families": all_families,
        }, f, indent=2)
    print(f"\nSaved: {Q3_POINT_SOLUTIONS}")

    Q3_TABLES.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    pd.DataFrame(main_solutions).to_csv(
        Q3_TABLES / "point_solutions_main.csv", index=False)
    pd.DataFrame(reverse_solutions).to_csv(
        Q3_TABLES / "point_solutions_reverse.csv", index=False)
    pd.DataFrame(noqual_solutions).to_csv(
        Q3_TABLES / "point_solutions_no_quality.csv", index=False)

    fam_rows = []
    for fam_name, sols in all_families.items():
        for s in sols:
            fam_rows.append({
                "family": fam_name, "budget": s["budget"],
                "N_star": s["N_star"], "D_star": s["D_star"],
                "Q_star": s["Q_star"], "loss": s["loss"],
            })
    pd.DataFrame(fam_rows).to_csv(
        Q3_TABLES / "point_solutions_cost_families.csv", index=False)

    print(f"Saved: {Q3_TABLES / 'point_solutions_main.csv'}")
    print(f"Saved: {Q3_TABLES / 'point_solutions_reverse.csv'}")
    print(f"Saved: {Q3_TABLES / 'point_solutions_no_quality.csv'}")
    print(f"Saved: {Q3_TABLES / 'point_solutions_cost_families.csv'}")

    print("\n[DONE] 02_optimizer.py")


if __name__ == "__main__":
    main()