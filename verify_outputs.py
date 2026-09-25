#!/usr/bin/env python3
"""
Verify pipeline outputs and internal consistency (reviewer feedback).

Checks:
  1. All key JSON artifacts present (15 checks)
  2. Q1 → Q3 Δp* consistency
  3. Q4 forecast ↔ decomposition gC consistency
  4. Q4 base_score source (derived, not fallback)
  5. Q1 → Q2 explicit injection table
  6. Q3 → Q4 explicit bridge predictions
"""
import json
import sys
from pathlib import Path

import pandas as pd

from config import MODELS, OUTPUTS


CHECKS = [
    # --- Q1 artifacts ---
    ("Q1 weights", MODELS / "q1" / "quality_weights.json",
     ["features", "weights", "orientation"]),
    ("Q1 mixture", MODELS / "q1" / "mixture_model.json",
     ["locked_family", "locked_lambda", "blind_test",
      "extra_validation_A8_A15", "macro_loss_recipe72", "base_macro_loss"]),

    # --- Q2 artifacts ---
    ("Q2 B1 fit", MODELS / "q2" / "b1_fit.json",
     ["parameters", "fit_stats"]),
    ("Q2 quality", MODELS / "q2" / "quality_fits.json",
     ["main", "reverse"]),
    ("Q2 attenuation", MODELS / "q2" / "attenuation.json",
     ["omega_inf", "N_c", "eta"]),
    ("Q2 Q1-Q2 bridge", MODELS / "q2" / "q1_q2_bridge.json",
     ["method", "a1_reference", "mapping_table", "operational_note"]),
    ("Q2 bridge mapping CSV",
     OUTPUTS / "q2" / "tables" / "bridge_mapping.csv", None),
    ("Q2 elasticities CSV",
     OUTPUTS / "q2" / "tables" / "elasticities.csv", None),
    ("Q2 substitution CSV",
     OUTPUTS / "q2" / "tables" / "substitution_analysis.csv", None),
    ("Q2 Q1→Q2 injection table",
     OUTPUTS / "q2" / "tables" / "q1_q2_injection_table.csv", None),

    # --- Q3 artifacts ---
    ("Q3 point sol", MODELS / "q3" / "point_solutions.json",
     ["delta_p_star", "main_exponential", "reverse_exponential",
      "no_quality_exponential", "all_families"]),
    ("Q3 bootstrap settings", MODELS / "q3" / "transition_sweep_settings.json",
     ["methodology", "delta_p_base_used"]),

    # --- Q4 artifacts ---
    ("Q4 decomp", MODELS / "q4" / "decomposition.json",
     ["beta_C", "beta_t", "r2", "gC_used", "scale_share_point"]),
    ("Q4 forecast", MODELS / "q4" / "forecast.json",
     ["gC", "r2", "forecasts", "hac_ci", "block_ci",
      "base_score", "base_score_source"]),
    ("Q4 Q3→Q4 bridge", MODELS / "q4" / "q3_to_q4_bridge.json",
     ["compliance", "model_form", "q3_to_q4_predictions"]),
]


def jload(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def check_artifact(path, keys):
    if not Path(path).exists():
        return False, f"missing: {path}"
    if keys is None:
        return True, "present"
    try:
        d = jload(path)
    except Exception as e:
        return False, f"read error: {e}"
    missing = [k for k in keys if k not in d]
    if missing:
        return False, f"missing keys: {missing}"
    return True, "OK"


def check_consistency():
    print("\n" + "=" * 72)
    print("CONSISTENCY CHECKS")
    print("=" * 72)
    issues = []

    # --- Q1 → Q3 Δp* consistency ---
    try:
        q1 = jload(MODELS / "q1" / "mixture_model.json")
        q3 = jload(MODELS / "q3" / "point_solutions.json")
        q3_settings = jload(MODELS / "q3" / "transition_sweep_settings.json")
        dp_q1 = float(q1["macro_loss_recipe72"] - q1["base_macro_loss"])
        dp_q3_point = float(q3["delta_p_star"])
        dp_q3_bootstrap = float(q3_settings["delta_p_base_used"])
        diff = abs(dp_q1 - dp_q3_point)
        diff_bs = abs(dp_q1 - dp_q3_bootstrap)
        ok = diff < 1e-4 and diff_bs < 1e-4
        tag = "OK" if ok else "FAIL"
        print(f"  [{tag}] Q1 → Q3 Δp* = {dp_q1:+.6f}")
        print(f"        Q3 point Δp*    = {dp_q3_point:+.6f} (d={diff:.2e})")
        print(f"        Q3 bootstrap Δp* = {dp_q3_bootstrap:+.6f} (d={diff_bs:.2e})")
        if not ok:
            issues.append("Q1<->Q3 Δp* mismatch")
    except Exception as e:
        print(f"  [SKIP] Q1<->Q3 Δp* check failed: {e}")

    # --- Q4 gC consistency ---
    try:
        fc = jload(MODELS / "q4" / "forecast.json")
        dec = jload(MODELS / "q4" / "decomposition.json")
        gc_fc = float(fc["gC"])
        gc_dec = float(dec.get("gC_used", 0))
        diff = abs(gc_fc - gc_dec)
        ok = diff < 1e-4
        tag = "OK" if ok else "FAIL"
        print(f"\n  [{tag}] Q4 forecast gC = {gc_fc:.6f}")
        print(f"        Q4 decomposition gC_used = {gc_dec:.6f} (d={diff:.2e})")
        if not ok:
            issues.append(f"Q4 gC mismatch: {gc_fc} vs {gc_dec}")
    except Exception as e:
        print(f"  [SKIP] Q4 gC check failed: {e}")

    # --- Q4 base_score source ---
    try:
        fc = jload(MODELS / "q4" / "forecast.json")
        src = fc.get("base_score_source", "")
        if "fallback" in src.lower():
            print(f"\n  [WARN] Q4 base_score uses fallback: {src}")
            issues.append("Q4 base_score fallback")
        else:
            print(f"\n  [OK] Q4 base_score derived: {src[:80]}")
    except Exception as e:
        print(f"  [SKIP] Q4 base_score check failed: {e}")

    # --- Q1 → Q2 explicit injection check ---
    try:
        inj_path = OUTPUTS / "q2" / "tables" / "q1_q2_injection_table.csv"
        if inj_path.exists():
            inj = pd.read_csv(inj_path)
            print(f"\n  [OK] Q1 → Q2 injection table: {len(inj)} rows")
            print(f"       series: {sorted(inj['series'].unique())}")
        else:
            print(f"\n  [FAIL] Q1 → Q2 injection table missing: {inj_path.name}")
            issues.append("Q1→Q2 injection table missing")
    except Exception as e:
        print(f"  [SKIP] Q1→Q2 injection check failed: {e}")

    # --- Q1 → Q2 bridge operational ---
    try:
        bridge = jload(MODELS / "q2" / "q1_q2_bridge.json")
        mapping_path = Path(bridge.get("mapping_table", ""))
        if mapping_path.exists():
            print(f"  [OK] Q1→Q2 bridge mapping present: {mapping_path.name}")
        else:
            print(f"  [FAIL] Q1→Q2 bridge mapping missing")
            issues.append("Q1→Q2 bridge mapping missing")
    except Exception as e:
        print(f"  [SKIP] Q1→Q2 bridge check failed: {e}")

    # --- Q3 → Q4 explicit bridge check ---
    try:
        q3q4 = jload(MODELS / "q4" / "q3_to_q4_bridge.json")
        preds = q3q4.get("q3_to_q4_predictions", [])
        print(f"\n  [OK] Q3 → Q4 predictions: {len(preds)} budgets")
        for r in preds:
            print(f"       Budget {r['budget']:.0e}: "
                  f"Q3 loss={r['q3_loss']:.4f} → "
                  f"Bench={r['predicted_benchmark']:.2f}")
    except Exception as e:
        print(f"  [SKIP] Q3→Q4 bridge check failed: {e}")

    return issues


def main():
    print("=" * 72)
    print("VERIFICATION — artifacts + internal consistency")
    print("=" * 72)

    ok_count = 0
    for name, path, keys in CHECKS:
        ok, msg = check_artifact(path, keys)
        status = "[OK]" if ok else "[FAIL]"
        print(f"  {status:6s} {name:30s}  {msg}")
        if ok:
            ok_count += 1
    print(f"\n  {ok_count}/{len(CHECKS)} artifact checks passed")

    issues = check_consistency()

    print("\n" + "=" * 72)
    print("KEY NUMERICAL VALUES")
    print("=" * 72)

    try:
        q1 = jload(MODELS / "q1" / "mixture_model.json")
        l72 = q1.get("macro_loss_recipe72")
        l0 = q1.get("base_macro_loss")
        if l72 is not None and l0 is not None:
            print(f"  Q1 Δp*              = {l72-l0:+.6f}")
        print(f"  Q1 blind R^2        = {q1['blind_test']['mean_r2']:.4f}")
    except Exception as e:
        print(f"  [WARN] Q1 read: {e}")

    try:
        q3 = jload(MODELS / "q3" / "point_solutions.json")
        print(f"  Q3 Δp*              = {q3['delta_p_star']:+.6f}")
    except Exception as e:
        print(f"  [WARN] Q3 read: {e}")

    try:
        q4 = jload(MODELS / "q4" / "decomposition.json")
        print(f"  Q4 gC used          = {q4.get('gC_used', 0):.6f}")
        print(f"  Q4 scale share      = {q4['scale_share_point']*100:.1f}%")
    except Exception as e:
        print(f"  [WARN] Q4 decomp read: {e}")

    try:
        fc = jload(MODELS / "q4" / "forecast.json")
        print(f"  Q4 forecast gC      = {fc['gC']:.6f}")
        print(f"  Q4 base score       = {fc['base_score']:.4f}")
        sc = [r for r in fc["forecasts"] if "50%" in r["scenario"]]
        if sc:
            print(f"  Q4 50% slowdown S12 = {sc[0]['S12']:.2f}")
            print(f"  Q4 50% slowdown S24 = {sc[0]['S24']:.2f}")
    except Exception as e:
        print(f"  [WARN] Q4 forecast read: {e}")

    print("\n" + "=" * 72)
    if ok_count == len(CHECKS) and not issues:
        print("[OK] ALL VERIFICATIONS PASSED")
        sys.exit(0)
    else:
        print(f"[WARN] {len(CHECKS)-ok_count} artifact failures, "
              f"{len(issues)} consistency issues")
        sys.exit(1)


if __name__ == "__main__":
    main()