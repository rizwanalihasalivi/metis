# Known Limitations — Honest Disclosure

## Q1
- Human raw-text reliability panel NOT completed (honest gap).
- A1/A2/A3 overlaps are pipeline-consistency checks, not external validity.
- Quality score uses 11 of 22 indicators; rest used in conflict families.
- Conflict score is a separate diagnostic (not fused into Q_i).

## Q2
- Q1 → B-series bridge IS executed: B's Q_score → A1 ECDF percentile u
  → A1 Q_i value → sigmoid(Q_i_A1) ∈ (0, 1). See
  `outputs/q2/tables/q1_q2_injection_table.csv` (21 rows across B6/B7/B8).
- Sigmoid compression: Q_sigmoid width 0.38–0.41. This weakens
  cross-series ordering (L3 ρ(B6,B8) = -0.26) but within-series sign
  is correct (L1: B6 ρ = -0.32 NEG ✓, B8 ρ = +0.92 POS ✓).
- Primary direction evidence = within-series (L1), not cross-series (L3).
- Attenuation curve includes 10B/70B *estimated* points.
- B10 (100B+) is an estimated extrapolation consistency check.

## Q3
- Optimization conditional on fixed observed recipe p*.
- Cost-family comparisons conditional on supplied absolute scales.
- Bootstrap uncertainty = model/fit perturbation, not full joint
  upstream resampling.
- High-budget optima omit hardware/wall-clock/data caps.

## Q4 (Weakest — Multiple Issues)

### 1. Loss-to-Benchmark Bridge (n=43, 7 high-comparability)

Two parallel bridge models, both fitted on C6:

| Model | In-sample | Random 5-fold | GroupKFold | High-wt (n=7) |
|---|---|---|---|---|
| Isotonic (step 04) | R² = 0.25 | R² = 0.31 | R² = -0.08 | R² = 0.35 |
| Ridge    (step 07) | R² = 0.25 | R² = 0.10 | R² = -0.24 | R² = 0.05 |

**Variant sensitivity (Ridge, LOOCV):**

| Variant | n | Slope | LOOCV R² | MAE | Q3 preds (1e19/1e22/1e24) |
|---|---|---|---|---|---|
| Global (all rows, w=5)      | 43 | -14.41 | +0.073 | 7.34 | 4.48 / 15.05 / 17.71 |
| Local (within Q3 loss ± 0.1)| 40 | -16.01 | +0.089 | 7.52 | 3.74 / 15.49 / 18.44 |
| High-comparability only     |  7 |  -0.14 | -0.334 | 0.38 | 5.60 / 5.70 /  5.73 |

- Local retrain differs from global by ΔR² = +0.016 (noise-level; MAE
  actually worse). Not adopted.
- High-only variant fails: all 7 Pythia rows have similar benchmark
  (~5–6) across loss values, slope collapses, LOOCV R² negative. Not adopted.
- Cause of wide range: C6 has only 43 rows (7 high-comparability Pythia)
  across 10+ families. Cross-family generalization not identifiable.
- Diagnostic artifact: `models/q4/bridge_variants_comparison.json`.
- Both bridges retained as *direction-only* indicators, NOT primary forecast.
- PRIMARY forecast = §5 direct compute-frontier extrapolation (n=1779).

### 2. Historical Decomposition — Specification Sensitivity

| Spec | β_C | β_t | R² |
|---|---|---|---|
| Primary (no FE) | +4.68 | +4.13 | 0.50 |
| With family FE | +9.33 | **-1.95** | 0.85 |

β_t sign flips with family FE → attribution NOT robustly identified.
Both specs reported; primary used for continuity.

### 3. 2024 Out-of-Time Holdout
- 0 pre-2024 rows in matched sample → holdout skipped.

### 4. Forecast Space (two-track)
- **PRIMARY**: frontier forecast computed **directly in benchmark-score
  space**:
  `S(h) = 51.231 + [β_C·(s·g_C) + β_t]·h`.
  - g_C loaded dynamically from `forecast.json` (no hardcoding).
  - n=1779 C2 rows; out-of-sample R² = 0.34–0.43.
- **SECONDARY**: bridge-based forecast via Q3 loss → C6 bridge.
  - Bridge base = 15.05 vs frontier 51.23 (3.4× gap, data-limited).
  - Reported as directional cross-check ONLY, not headline forecast.

### 5. Baseline Origin
- 51.2313 = max six-benchmark Average over a documented 2-year window
  of open-license models (C2). Top model: dfurman/CalmeRys-78B-Orpo-v0.1,
  2024-09-24. n=4564 total, 1779 in 2-year window.

### 6. Forecast Uncertainty
- Reported scores are point scenarios, NOT full predictive intervals.

### 7. C8 Validation
- Statistics computed on repo-key de-duplicated matches.

## References
- Q4 PDF Sections 3, 4, 6, 9, 10, A.
- Bridge sensitivity diagnostic: `models/q4/bridge_variants_comparison.json`.