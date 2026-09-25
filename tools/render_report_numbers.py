#!/usr/bin/env python3
"""
Auto-generate REPORT_NUMBERS.md from all model JSONs.

Single source of truth: every number in the report comes from a JSON
produced by a pipeline script. Re-run after any pipeline change.

Usage:
  python -m tools.render_report_numbers
Output:
  REPORT_NUMBERS.md (repo root)
"""
import json
from pathlib import Path

from config import (
    Q2_QUALITY_FITS, Q3_POINT_SOLUTIONS,
    Q4_FORECAST, Q4_DECOMPOSITION, Q4_BRIDGE, Q4_MODELS,
)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "REPORT_NUMBERS.md"


def _load(p):
    if p is None or not Path(p).exists():
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def fmt(v, n=4):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{n}f}"
    return str(v)


def main():
    q2 = _load(Q2_QUALITY_FITS)
    q3 = _load(Q3_POINT_SOLUTIONS)
    q4f = _load(Q4_FORECAST)
    q4d = _load(Q4_DECOMPOSITION)
    q4b = _load(Q4_BRIDGE)
    q34b = _load(Q4_MODELS / "q3_to_q4_bridge.json")

    lines = []
    lines.append("# REPORT NUMBERS (auto-generated)\n")
    lines.append("> Source: pipeline JSONs. Refresh: "
                 "`python -m tools.render_report_numbers`\n")

    # ---- Q2 ----
    lines.append("\n## Q2 — Quality scenarios (Q1-injected)\n")
    if q2:
        m, r = q2["main"], q2["reverse"]
        lines.append(f"- **Main**: G = {fmt(m['G']['estimate'])} "
                     f"[{fmt(m['G']['ci_low'])}, {fmt(m['G']['ci_high'])}], "
                     f"gamma = {fmt(m['gamma']['estimate'])} "
                     f"[{fmt(m['gamma']['ci_low'])}, {fmt(m['gamma']['ci_high'])}]")
        lines.append(f"- **Main fit**: R² = {fmt(m['fit_r2'])}, "
                     f"MAE = {fmt(m['fit_mae'], 5)}, n = {m['n_rows']}")
        lines.append(f"- **Reverse**: H = {fmt(r['H']['estimate'])} "
                     f"[{fmt(r['H']['ci_low'])}, {fmt(r['H']['ci_high'])}], "
                     f"delta = {fmt(r['delta']['estimate'])} "
                     f"[{fmt(r['delta']['ci_low'])}, {fmt(r['delta']['ci_high'])}]")
        lines.append(f"- **Reverse fit**: R² = {fmt(r['fit_r2'])}, "
                     f"MAE = {fmt(r['fit_mae'], 5)}, n = {r['n_rows']}")
        dd = q2.get("direction_diagnostic", {})
        if dd:
            l1 = dd.get("L1_within_series", {})
            l2 = dd.get("L2_compression", {})
            l3 = dd.get("L3_cross_series", {})
            lines.append(f"- **L1 sign**: B6 ρ = {fmt(l1.get('B6_spearman_Q_loss'))} "
                         f"({'OK' if l1.get('B6_sign_ok') else 'FAIL'}), "
                         f"B8 ρ = {fmt(l1.get('B8_spearman_Q_loss'))} "
                         f"({'OK' if l1.get('B8_sign_ok') else 'FAIL'})")
            lines.append(f"- **L2 width**: B6 = {fmt(l2.get('B6_Q_sigmoid_width'), 3)}, "
                         f"B8 = {fmt(l2.get('B8_Q_sigmoid_width'), 3)}")
            lines.append(f"- **L3 cross**: n = {l3.get('n_shared')}, "
                         f"ρ = {fmt(l3.get('spearman_b6_b8'))}")

    # ---- Q3 ----
    lines.append("\n## Q3 — Point solutions\n")
    if q3:
        lines.append(f"- **δp\\*** = {fmt(q3.get('delta_p_star'))}")
        lines.append("\n### Main exponential scenario\n")
        lines.append("| Budget | N* | D* | Q* | Loss |")
        lines.append("|---|---|---|---|---|")
        for r in q3["main_exponential"]:
            lines.append(f"| {r['budget']:.0e} | {fmt(r['N_star'])} | "
                         f"{fmt(r['D_star'])} | {fmt(r['Q_star'])} | "
                         f"{fmt(r['loss'])} |")

    # ---- Q4 direct ----
    lines.append("\n## Q4 — Direct frontier forecast (PRIMARY)\n")
    if q4f:
        lines.append(f"- **BENCH_BASE** = {fmt(q4f['base_score'])}")
        lines.append(f"- **Source**: {q4f.get('base_score_source', '?')}")
        lines.append(f"- **gC** = {fmt(q4f['gC'])} log10(FLOP)/year")
        lines.append(f"- **R²** = {fmt(q4f['r2'])}")
        lines.append(f"- **HAC 95% CI**: "
                     f"[{fmt(q4f['hac_ci'][0])}, {fmt(q4f['hac_ci'][1])}]")
        lines.append(f"- **Block boot 95% CI**: "
                     f"[{fmt(q4f['block_ci'][0])}, {fmt(q4f['block_ci'][1])}]")
        lines.append(f"- **β_C** = {fmt(q4f['beta_C'])}, "
                     f"**β_t** = {fmt(q4f['beta_t'])}")
        lines.append("\n| Scenario | 12m | 24m |")
        lines.append("|---|---|---|")
        for r in q4f["forecasts"]:
            lines.append(f"| {r['scenario']} | {fmt(r['S12'], 2)} | "
                         f"{fmt(r['S24'], 2)} |")

    # ---- Q4 bridge forecast (secondary) ----
    bf = (q4f or {}).get("bridge_based_forecast")
    if bf:
        lines.append("\n## Q4 — Bridge-based forecast (SECONDARY)\n")
        lines.append(f"- **Bridge base** = {fmt(bf['base'])}")
        lines.append(f"- **Source Q3 loss** = {fmt(bf['source_q3_loss'])}")
        lines.append(f"- **NOTE**: {bf['note']}")
        lines.append("\n| Scenario | 12m | 24m |")
        lines.append("|---|---|---|")
        for r in bf["scenarios"]:
            lines.append(f"| {r['scenario']} | {fmt(r['S12'], 2)} | "
                         f"{fmt(r['S24'], 2)} |")

    # ---- Q4 decomposition ----
    lines.append("\n## Q4 — Decomposition\n")
    if q4d:
        for k in ["beta_C", "beta_t", "R2", "n_rows"]:
            if k in q4d:
                lines.append(f"- **{k}** = {fmt(q4d[k])}")

    # ---- Bridge validation (step 04) ----
    lines.append("\n## Q4 — Bridge validation, Isotonic (step 04)\n")
    if q4b:
        lines.append(f"- n_rows = {q4b.get('n_rows')}")
        for k in ["cv_random", "cv_groupkfold", "high_wt"]:
            s = q4b.get(k)
            if s:
                lines.append(f"- **{k}**: R² = {fmt(s.get('r2'))}, "
                             f"MAE = {fmt(s.get('mae'))}")

    # ---- Q3→Q4 bridge (step 07) ----
    lines.append("\n## Q4 — Q3→Q4 bridge, Ridge (step 07, SECONDARY)\n")
    if q34b:
        lines.append(f"- **{q34b.get('model_form')}**")
        fs = q34b.get("fit_stats", {})
        lines.append(f"- In-sample: R² = {fmt(fs.get('in_sample_r2'))}, "
                     f"MAE = {fmt(fs.get('in_sample_mae'))}")
        vs = q34b.get("validation_summary", {})
        for k in ["random_5fold", "groupkfold_family", "high_weight_only"]:
            s = vs.get(k)
            if s and s.get("r2") is not None:
                lines.append(f"- **{k}**: R² = {fmt(s['r2'])}, "
                             f"MAE = {fmt(s['mae'])}, n = {s['n']}")
        lines.append(f"\n> {vs.get('interpretation', '')}")
        lines.append("\n| Budget | Q3 Loss | Pred Bench | 95% lo | 95% hi |")
        lines.append("|---|---|---|---|---|")
        for r in q34b.get("q3_to_q4_predictions", []):
            lines.append(f"| {r['budget']:.0e} | {fmt(r['q3_loss'])} | "
                         f"{fmt(r['predicted_benchmark'], 2)} | "
                         f"{fmt(r['ci95_low'], 2)} | "
                         f"{fmt(r['ci95_high'], 2)} |")

    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] Wrote {OUT}")
    print(f"     {len(lines)} lines, {OUT.stat().st_size} bytes")


if __name__ == "__main__":
    main()