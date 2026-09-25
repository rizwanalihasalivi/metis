# REPORT NUMBERS (auto-generated)

> Source: pipeline JSONs. Refresh: `python -m tools.render_report_numbers`


## Q2 — Quality scenarios (Q1-injected)

- **Main**: G = 1.7262 [1.5009, 1.9606], gamma = 3.3510 [3.1797, 3.5504]
- **Main fit**: R² = 0.9465, MAE = 0.06245, n = 360
- **Reverse**: H = 6.6649 [6.4515, 6.8799], delta = 2.7257 [2.6802, 2.7710]
- **Reverse fit**: R² = 0.8661, MAE = 0.21121, n = 1704
- **L1 sign**: B6 ρ = -0.3239 (OK), B8 ρ = 0.9208 (OK)
- **L2 width**: B6 = 0.378, B8 = 0.406
- **L3 cross**: n = 160, ρ = -0.2590

## Q3 — Point solutions

- **δp\*** = -0.5309

### Main exponential scenario

| Budget | N* | D* | Q* | Loss |
|---|---|---|---|---|
| 1e+19 | 0.1475 | 7.5519 | 0.6582 | 2.7205 |
| 1e+22 | 1.1644 | 1052.5318 | 0.8619 | 1.9871 |
| 1e+24 | 27.4340 | 5262.2086 | 0.9591 | 1.8026 |

## Q4 — Direct frontier forecast (PRIMARY)

- **BENCH_BASE** = 51.2313
- **Source**: derived from C2: n_total=4564, 2y_window_n=1779, latest_date=2025-03-13, top_model='dfurman/CalmeRys-78B-Orpo-v0.1', top_date=2024-09-24, open_license_filter_applied=True
- **gC** = 0.5340 log10(FLOP)/year
- **R²** = 0.9507
- **HAC 95% CI**: [0.4550, 0.6131]
- **Block boot 95% CI**: [0.4651, 0.5942]
- **β_C** = 4.6796, **β_t** = 4.1279

| Scenario | 12m | 24m |
|---|---|---|
| Reference (recent growth) | 57.86 | 64.49 |
| 50% slowdown | 56.61 | 61.99 |
| 75% slowdown | 55.98 | 60.74 |
| Compute frontier frozen | 55.36 | 59.49 |

## Q4 — Bridge-based forecast (SECONDARY)

- **Bridge base** = 15.0476
- **Source Q3 loss** = 1.9871
- **NOTE**: Secondary. C6 data-limited: only 43 rows, 7 high-quality.

| Scenario | 12m | 24m |
|---|---|---|
| Reference (recent growth) | 21.67 | 28.30 |
| 50% slowdown | 20.43 | 25.80 |
| 75% slowdown | 19.80 | 24.55 |
| Compute frontier frozen | 19.18 | 23.30 |

## Q4 — Decomposition

- **beta_C** = 4.6796
- **beta_t** = 4.1279

## Q4 — Bridge validation, Isotonic (step 04)

- n_rows = 43
- **cv_random**: R² = 0.3133, MAE = 6.4708
- **cv_groupkfold**: R² = -0.0824, MAE = 8.1589
- **high_wt**: R² = 0.3499, MAE = 6.0792

## Q4 — Q3→Q4 bridge, Ridge (step 07, SECONDARY)

- **LB = 43.6853 + -14.4118 * Val_Loss**
- In-sample: R² = 0.2464, MAE = 6.3123
- **random_5fold**: R² = 0.1043, MAE = 7.2151, n = 43
- **groupkfold_family**: R² = -0.2429, MAE = 8.8232, n = 43
- **high_weight_only**: R² = 0.0461, MAE = 0.3197, n = 7

> Bridge is data-limited: C6 has n=43 total, only 7 high-comparability rows. R2 varies widely across validation schemes. Treated as SECONDARY, direction-only indicator. PRIMARY forecast = Q4 §5 direct compute-frontier extrapolation.

| Budget | Q3 Loss | Pred Bench | 95% lo | 95% hi |
|---|---|---|---|---|
| 1e+19 | 2.7205 | 4.48 | 0.79 | 14.63 |
| 1e+22 | 1.9871 | 15.05 | 9.33 | 24.05 |
| 1e+24 | 1.8026 | 17.71 | 10.80 | 27.40 |
