#!/usr/bin/env python3
"""
Q1 Step 6: 17-domain mixture-to-loss model (A4-A15).

Fixes (reviewer feedback):
  - #6: A8-A15 validation reports BOTH raw R² and centered R²
        (centered = post-hoc, uses eval mean).
  - Lambda policy documented (pre-registered).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from config import (
    Q2_TRAIN_MIX, Q2_TRAIN_LOSS, Q2_TEST_MIX, Q2_TEST_LOSS,
    Q2_60M_MIX, Q2_60M_LOSS, Q2_1B_MIX, Q2_1B_LOSS,
    Q2_10B_MIX, Q2_10B_LOSS, Q2_70B_MIX, Q2_70B_LOSS,
    Q1_MODELS, Q1_TABLES, A_DATA, SEED,
)

np.random.seed(SEED)

EPS = 5e-4
LAMBDAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
N_OUTER = 5
N_INNER = 5
LOCKED_LAMBDA = 10.0
BEST_RECIPE_INDEX = 71  # 0-based (= 1-based recipe 72)

MIX_PREFIX = "train_the_pile_"
LOSS_PREFIX = "metric/the_pile_"
LOSS_SUFFIX = "_val_loss"


def _mix_cols(df):
    return [c for c in df.columns if c.startswith(MIX_PREFIX)]


def _loss_cols(df):
    return [c for c in df.columns if c.startswith(LOSS_PREFIX)]


def load_split(mix_path, loss_path):
    mix = pd.read_csv(mix_path)
    loss = pd.read_csv(loss_path)
    mix_cols = _mix_cols(mix)
    loss_cols = _loss_cols(loss)
    P = mix[mix_cols].to_numpy(dtype=float)
    L = loss[loss_cols].to_numpy(dtype=float)
    mix_domains = [c.replace(MIX_PREFIX, "") for c in mix_cols]
    loss_domains = [c.replace(LOSS_PREFIX, "").replace(LOSS_SUFFIX, "")
                    for c in loss_cols]
    return P, L, mix_domains, loss_domains


def clr_transform(P, eps=EPS):
    Pt = (P + eps) / (P + eps).sum(axis=1, keepdims=True)
    log_Pt = np.log(Pt)
    return log_Pt - log_Pt.mean(axis=1, keepdims=True)


def quad_expand(X):
    n, d = X.shape
    terms = [X]
    for i in range(d):
        for j in range(i, d):
            terms.append((X[:, i] * X[:, j]).reshape(-1, 1))
    return np.column_stack(terms)


def feat_raw_linear(P): return P.copy()
def feat_raw_quad(P): return quad_expand(P)
def feat_clr_linear(P): return clr_transform(P)
def feat_clr_quad(P): return quad_expand(clr_transform(P))

FAMILIES = {
    "raw_linear": feat_raw_linear,
    "raw_quadratic": feat_raw_quad,
    "clr_linear": feat_clr_linear,
    "clr_quadratic": feat_clr_quad,
}


def ridge_fit(X, Y, lam):
    xm = X.mean(axis=0)
    ym = Y.mean(axis=0)
    Xc = X - xm
    Yc = Y - ym
    A = Xc.T @ Xc + lam * np.eye(X.shape[1])
    W = np.linalg.solve(A, Xc.T @ Yc)
    return W, xm, ym


def ridge_predict(X, W, xm, ym):
    return (X - xm) @ W + ym


def per_output_r2(y_true, y_pred):
    """Raw R² — uses eval mean as reference."""
    r2s = []
    for j in range(y_true.shape[1]):
        y = y_true[:, j]
        yp = y_pred[:, j]
        ss_res = float(np.sum((y - yp) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2s.append(1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0)
    return np.array(r2s)


def per_output_r2_centered(y_true, y_pred):
    """Centered R² — removes per-domain mean shift (cross-scale)."""
    r2s = []
    for j in range(y_true.shape[1]):
        y = y_true[:, j]
        yp = y_pred[:, j]
        yc = y - y.mean()
        ypc = yp - yp.mean()
        ss_res = float(np.sum((yc - ypc) ** 2))
        ss_tot = float(np.sum(yc ** 2))
        r2s.append(1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0)
    return np.array(r2s)


def macro_mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def nested_cv(P, L):
    rng = np.random.default_rng(SEED)
    n = len(P)
    idx = rng.permutation(n)
    folds = np.array_split(idx, N_OUTER)

    print(f"\nNested {N_OUTER}x{N_INNER} CV on {n} recipes (family selection only)")
    print(f"{'Fold':>5} {'Best family':>16} {'lambda':>8} "
          f"{'inner MAE':>10} {'outer MAE':>10} {'mean R2':>9} {'med R2':>9}")
    print("-" * 74)

    results = []
    for k_out in range(N_OUTER):
        test_idx = folds[k_out]
        train_idx = np.concatenate(
            [folds[i] for i in range(N_OUTER) if i != k_out])

        best = None
        for fam_name, fn in FAMILIES.items():
            X_all = fn(P[train_idx])
            for lam in LAMBDAS:
                inner_perm = rng.permutation(len(train_idx))
                inner_folds = np.array_split(inner_perm, N_INNER)
                maes = []
                for k_in in range(N_INNER):
                    it = inner_folds[k_in]
                    tr = np.concatenate(
                        [inner_folds[i] for i in range(N_INNER) if i != k_in])
                    W, xm, ym = ridge_fit(X_all[tr], L[train_idx][tr], lam)
                    pred = ridge_predict(X_all[it], W, xm, ym)
                    maes.append(np.mean(np.abs(pred - L[train_idx][it])))
                m = float(np.mean(maes))
                if best is None or m < best[0]:
                    best = (m, fam_name, lam)

        _, fam, lam = best
        X_tr = FAMILIES[fam](P[train_idx])
        X_te = FAMILIES[fam](P[test_idx])
        W, xm, ym = ridge_fit(X_tr, L[train_idx], lam)
        pred = ridge_predict(X_te, W, xm, ym)

        r2s = per_output_r2(L[test_idx], pred)
        row = {
            "fold": k_out, "family": fam, "lambda_fold_selection": lam,
            "inner_mae": best[0],
            "outer_mae": macro_mae(L[test_idx], pred),
            "mean_r2": float(r2s.mean()),
            "median_r2": float(np.median(r2s)),
        }
        results.append(row)
        print(f"{k_out:>5} {fam:>16} {lam:>8.2f} {best[0]:>10.4f} "
              f"{row['outer_mae']:>10.4f} {row['mean_r2']:>9.4f} "
              f"{row['median_r2']:>9.4f}")

    from collections import Counter
    modal_fam = Counter(r["family"] for r in results).most_common(1)[0][0]
    fold_lams = [r["lambda_fold_selection"] for r in results]
    print(f"\nModal family: {modal_fam}")
    print(f"Fold-selected lambdas: {fold_lams}")
    print(f"FINAL lambda locked at {LOCKED_LAMBDA} (pre-registered)")
    return results, modal_fam, LOCKED_LAMBDA


def locked_fit_and_blind(P_tr, L_tr, P_te, L_te, fam, lam):
    print(f"\nLocking: {fam}  lambda={lam}")
    X_tr = FAMILIES[fam](P_tr)
    X_te = FAMILIES[fam](P_te)
    W, xm, ym = ridge_fit(X_tr, L_tr, lam)
    pred_te = ridge_predict(X_te, W, xm, ym)

    r2s = per_output_r2(L_te, pred_te)
    maes = np.array([float(np.mean(np.abs(L_te[:, j] - pred_te[:, j])))
                     for j in range(L_te.shape[1])])
    rhos = np.array([float(spearmanr(L_te[:, j], pred_te[:, j]).correlation)
                     for j in range(L_te.shape[1])])

    print(f"\nBlind test A6-A7 (n={len(P_te)}):")
    print(f"  mean R2   = {r2s.mean():.4f}")
    print(f"  median R2 = {np.median(r2s):.4f}")
    print(f"  mean MAE  = {maes.mean():.4f}")
    print(f"  median rho = {np.median(rhos):.4f}")
    return W, xm, ym, r2s, maes, rhos


def validate_on_split(P_tr, L_tr, fam, lam, P_eval, L_eval, label):
    """
    Reviewer fix #6: report BOTH raw and centered R².

    - raw R² = absolute fit (uses eval-set mean as baseline; can be very negative
      when the eval set's absolute loss scale differs from the training scale)
    - centered R² = post-hoc; per-domain mean of predictions is removed before
      computing R² (removes the "domain baseline mismatch" component)
    - Spearman ρ = rank preservation (scale-independent)
    """
    X_tr = FAMILIES[fam](P_tr)
    X_ev = FAMILIES[fam](P_eval)
    W, xm, ym = ridge_fit(X_tr, L_tr, lam)
    pred = ridge_predict(X_ev, W, xm, ym)

    r2s_raw = per_output_r2(L_eval, pred)
    r2s_centered = per_output_r2_centered(L_eval, pred)
    maes = np.array([float(np.mean(np.abs(L_eval[:, j] - pred[:, j])))
                     for j in range(L_eval.shape[1])])
    rhos = np.array([float(spearmanr(L_eval[:, j], pred[:, j]).correlation)
                     for j in range(L_eval.shape[1])])
    return {
        "split": label,
        "n_eval": int(len(P_eval)),
        "mean_r2_raw": float(r2s_raw.mean()),
        "mean_r2_centered": float(np.mean(r2s_centered)),
        "median_rho": float(np.median(rhos)),
        "mean_mae": float(maes.mean()),
    }


def aitchison_center(P, eps=EPS):
    Pt = (P + eps) / (P + eps).sum(axis=1, keepdims=True)
    g = np.exp(np.log(Pt).mean(axis=0))
    return g / g.sum()


def macro_loss(P, W, xm, ym, fam):
    pred = ridge_predict(FAMILIES[fam](P), W, xm, ym)
    return pred.mean(axis=1)


def local_effects(p0, W, xm, ym, fam, mix_domains, delta_pp=0.01):
    base = macro_loss(p0.reshape(1, -1), W, xm, ym, fam)[0]
    effects = {}
    for j, dom in enumerate(mix_domains):
        p_new = p0.copy()
        p_new[j] += delta_pp
        p_new = p_new / p_new.sum()
        effects[dom] = float(
            macro_loss(p_new.reshape(1, -1), W, xm, ym, fam)[0] - base)
    return effects, float(base)


def pairwise_interactions(p0, W, xm, ym, fam, mix_domains):
    effects, base = local_effects(p0, W, xm, ym, fam, mix_domains)
    interactions = {}
    G = len(mix_domains)
    for j in range(G):
        for r in range(j + 1, G):
            p_new = p0.copy()
            p_new[j] += 0.01
            p_new[r] += 0.01
            p_new = p_new / p_new.sum()
            L_jr = macro_loss(p_new.reshape(1, -1), W, xm, ym, fam)[0]
            I_jr = float(L_jr - base - effects[mix_domains[j]]
                         - effects[mix_domains[r]])
            interactions[(mix_domains[j], mix_domains[r])] = I_jr
    return interactions


def main():
    print("=" * 72)
    print("Q1 STEP 6: 17-domain mixture-to-loss model")
    print("=" * 72)

    P_tr, L_tr, mix_dom, loss_dom = load_split(Q2_TRAIN_MIX, Q2_TRAIN_LOSS)
    print(f"A4-A5: P shape={P_tr.shape}, L shape={L_tr.shape}")
    print(f"  {len(mix_dom)} mixture domains, {len(loss_dom)} loss domains")

    P_te, L_te, _, _ = load_split(Q2_TEST_MIX, Q2_TEST_LOSS)
    print(f"A6-A7: P shape={P_te.shape}, L shape={L_te.shape}")

    cv_results, fam, lam = nested_cv(P_tr, L_tr)
    W, xm, ym, r2s, maes, rhos = locked_fit_and_blind(
        P_tr, L_tr, P_te, L_te, fam, lam)

    # ---- Additional validations (reviewer fix #6) ----
    print("\n" + "=" * 72)
    print("Additional validation on A8-A15 (reviewer fix #6)")
    print("=" * 72)
    print("(raw R² = absolute, uses eval-set mean;")
    print(" centered R² = post-hoc mean-shift removal;")
    print(" ρ = rank preservation — the scale-independent metric)")
    extra_val = []
    for label, mix_p, loss_p in [
        ("A8-A9 (60M)", Q2_60M_MIX, Q2_60M_LOSS),
        ("A10-A11 (1B)", Q2_1B_MIX, Q2_1B_LOSS),
        ("A12-A13 (10B est)", Q2_10B_MIX, Q2_10B_LOSS),
        ("A14-A15 (70B est)", Q2_70B_MIX, Q2_70B_LOSS),
    ]:
        if not (Path(mix_p).exists() and Path(loss_p).exists()):
            print(f"  [SKIP] {label}: file missing")
            continue
        P_ev, L_ev, _, _ = load_split(mix_p, loss_p)
        res = validate_on_split(P_tr, L_tr, fam, lam, P_ev, L_ev, label)
        extra_val.append(res)
        print(f"  {label:22s}  n={res['n_eval']:>4d}  "
              f"raw R²={res['mean_r2_raw']:+.4f}  "
              f"centered R²={res['mean_r2_centered']:+.4f}  "
              f"ρ={res['median_rho']:+.4f}  "
              f"MAE={res['mean_mae']:.4f}")

    print("\n" + "=" * 72)
    print("Local effects around Aitchison center")
    print("=" * 72)
    p0 = aitchison_center(P_tr)
    print("Center shares (top 5):")
    for i in np.argsort(-p0)[:5]:
        print(f"  {mix_dom[i]:20s}  {p0[i]*100:.2f}%")

    effects, base_loss = local_effects(p0, W, xm, ym, fam, mix_dom)
    print(f"\nMacro loss at center: {base_loss:.6f}")
    print("Top favorable (+1pp):")
    for dom, d in sorted(effects.items(), key=lambda x: x[1])[:5]:
        print(f"  {dom:20s}  {d:+.6f}")
    print("Top adverse (+1pp):")
    for dom, d in sorted(effects.items(), key=lambda x: -x[1])[:5]:
        print(f"  {dom:20s}  {d:+.6f}")

    print("\nTop favorable interactions:")
    inter = pairwise_interactions(p0, W, xm, ym, fam, mix_dom)
    for (a, b), v in sorted(inter.items(), key=lambda x: x[1])[:5]:
        print(f"  {a} + {b}: {v:+.6f}")
    print("\nTop adverse interactions:")
    for (a, b), v in sorted(inter.items(), key=lambda x: -x[1])[:5]:
        print(f"  {a} + {b}: {v:+.6f}")

    # Recipe-72 macro loss
    print("\n" + "=" * 72)
    print(f"Macro loss at observed recipe {BEST_RECIPE_INDEX} (for Q3 Δp*)")
    print("=" * 72)
    macro_loss_recipe72 = None
    try:
        mix_path = A_DATA / "regmix_tables" / "train_mixture_1m.csv"
        df_mix = pd.read_csv(mix_path)
        mix_cols_q = [c for c in df_mix.columns if c.startswith(MIX_PREFIX)]
        if BEST_RECIPE_INDEX < len(df_mix):
            p_star_72 = df_mix.iloc[BEST_RECIPE_INDEX][mix_cols_q].to_numpy(dtype=float)
            p_star_72 = p_star_72 / p_star_72.sum()
            macro_loss_recipe72 = float(
                macro_loss(p_star_72.reshape(1, -1), W, xm, ym, fam)[0]
            )
            delta_p_star = macro_loss_recipe72 - base_loss
            print(f"  L(p*_recipe72) = {macro_loss_recipe72:.6f}")
            print(f"  L(p0_center)   = {base_loss:.6f}")
            print(f"  Δp* = L(p*) - L(p0) = {delta_p_star:+.6f}")
    except Exception as e:
        print(f"  [WARN] could not compute recipe-72 macro loss: {type(e).__name__}: {e}")

    # Save
    Q1_MODELS.mkdir(parents=True, exist_ok=True)
    with open(Q1_MODELS / "mixture_model.json", "w", encoding="utf-8") as f:
        json.dump({
            "locked_family": fam, "locked_lambda": lam,
            "lambda_policy": "pre-registered_10.0_before_blind_test",
            "fold_selected_lambdas": [r["lambda_fold_selection"] for r in cv_results],
            "nested_cv": cv_results,
            "blind_test": {
                "mean_r2": float(r2s.mean()),
                "median_r2": float(np.median(r2s)),
                "mean_mae": float(maes.mean()),
                "median_rho": float(np.median(rhos)),
            },
            "extra_validation_A8_A15": extra_val,
            "aitchison_center": {d: float(p0[i]) for i, d in enumerate(mix_dom)},
            "local_effects": effects,
            "pairwise_interactions": {f"{a}|{b}": v for (a, b), v in inter.items()},
            "base_macro_loss": float(base_loss),
            "macro_loss_recipe72": (
                float(macro_loss_recipe72) if macro_loss_recipe72 is not None else None
            ),
            "recipe_index_used": BEST_RECIPE_INDEX,
        }, f, indent=2)
    print(f"\nSaved: {Q1_MODELS / 'mixture_model.json'}")

    Q1_TABLES.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(cv_results).to_csv(
        Q1_TABLES / "mixture_cv_results.csv", index=False)
    pd.DataFrame([
        {"domain": d, "center_share": float(p0[i]), "local_effect": effects[d]}
        for i, d in enumerate(mix_dom)
    ]).to_csv(Q1_TABLES / "mixture_local_effects.csv", index=False)
    if extra_val:
        pd.DataFrame(extra_val).to_csv(
            Q1_TABLES / "mixture_validation_A8_A15.csv", index=False)
        print(f"Saved: {Q1_TABLES / 'mixture_validation_A8_A15.csv'}")
    print(f"Saved: {Q1_TABLES / 'mixture_cv_results.csv'}")
    print(f"Saved: {Q1_TABLES / 'mixture_local_effects.csv'}")

    print("\n[DONE] 06_mixture_model.py")


if __name__ == "__main__":
    main()