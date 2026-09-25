#!/usr/bin/env python3
"""
Q1 Step 2: Build Q_i quality score + extensions.

Fixes addressed:
  - #2: Explicit direction (orientation) table for ALL 22 indicators.
  - #3: Document which 11 of 22 are used in the score vs excluded.
  - #5: Full A2/A3 domain-level Q summary (not just overlap validation).
  - #Direction-mismatch: Report which indicators have semantic direction
    different from anchor-implied sign (reviewer feedback).
"""
import json
import lzma

import numpy as np
import pandas as pd

from config import (
    A1_SCALARIZED, A1_WITH_QI, ARXIV_RAW, GITHUB_RAW,
    ARXIV_WITH_QI, GITHUB_WITH_QI,
    Q1_WEIGHTS, Q1_ECDF, Q1_NORM, Q1_MODELS, Q1_TABLES,
    SEED,
)
from src.common.metrics import rankdata, spearman, ecdf_rank, parse_list

np.random.seed(SEED)

ANCHORS = ["qurater_mean", "fineweb_edu_mean"]

V3_FEATURES = [
    "modernbert_professionalism_mean",
    "modernbert_readability_mean",
    "fluency_en_mean",
    "dsir_composite",
    "ad_en_mean",
    "rps_doc_word_count",
    "rps_doc_frac_no_alph_words",
    "rps_doc_frac_chars_top_2gram",
    "rps_doc_mean_word_length",
    "rps_doc_unigram_entropy",
    "rps_lines_uppercase_letter_fraction",
]

ALL_22_INDICATORS = [
    "modernbert_cleanliness_mean", "modernbert_readability_mean",
    "modernbert_reasoning_mean", "modernbert_professionalism_mean",
    "fluency_en_mean", "dsir_books", "dsir_wiki", "dsir_math",
    "qurater_mean", "fineweb_edu_mean", "ad_en_mean",
    "rps_doc_word_count", "rps_doc_num_sentences",
    "rps_doc_unigram_entropy", "rps_doc_frac_unique_words",
    "rps_doc_frac_no_alph_words", "rps_doc_frac_chars_top_2gram",
    "rps_doc_frac_chars_top_3gram",
    "rps_lines_uppercase_letter_fraction",
    "rps_lines_ending_with_terminal_punctution_mark",
    "rps_lines_numerical_chars_fraction", "rps_doc_mean_word_length",
]

SEMANTIC_DIRECTION = {
    "modernbert_cleanliness_mean": +1,
    "modernbert_readability_mean": +1,
    "modernbert_reasoning_mean": +1,
    "modernbert_professionalism_mean": +1,
    "fluency_en_mean": +1,
    "dsir_books": +1,
    "dsir_wiki": +1,
    "dsir_math": +1,
    "qurater_mean": +1,
    "fineweb_edu_mean": +1,
    "ad_en_mean": -1,
    "rps_doc_word_count": +1,
    "rps_doc_num_sentences": +1,
    "rps_doc_unigram_entropy": +1,
    "rps_doc_frac_unique_words": +1,
    "rps_doc_frac_no_alph_words": -1,
    "rps_doc_frac_chars_top_2gram": -1,
    "rps_doc_frac_chars_top_3gram": -1,
    "rps_lines_uppercase_letter_fraction": -1,
    "rps_lines_ending_with_terminal_punctution_mark": +1,
    "rps_lines_numerical_chars_fraction": -1,
    "rps_doc_mean_word_length": -1,
}

MANUAL_OVERRIDE = {
    "qurater_mean": +1,
    "fineweb_edu_mean": +1,
    "ad_en_mean": -1,
}

LAMBDA = 1.0

EXT_LIST_METRICS = [
    "modernbert_professionalism", "modernbert_readability",
    "fluency_en", "ad_en", "fineweb_edu", "qurater",
]
EXT_RAW_SCALARS = [
    "dsir_books", "dsir_math", "dsir_wiki",
    "rps_doc_word_count", "rps_doc_frac_no_alph_words",
    "rps_doc_frac_chars_top_2gram", "rps_doc_mean_word_length",
    "rps_doc_unigram_entropy",
    "rps_lines_uppercase_letter_fraction",
]


def load_a1():
    print("=" * 72)
    print("STEP 1: Load A1 scalarized features")
    print("=" * 72)
    df = pd.read_parquet(A1_SCALARIZED).copy()
    print(f"A1 rows: {len(df)}")
    return df


def build_dsir_composite_a1(df):
    df = df.copy()
    dsir_cols = ["dsir_books", "dsir_math", "dsir_wiki"]
    n = len(df)
    ranks = np.zeros((n, 3))
    for i, c in enumerate(dsir_cols):
        ranks[:, i] = rankdata(df[c].to_numpy()) / n
    df["dsir_composite"] = ranks.mean(axis=1)
    return df


def apply_qi(df, features, orient, weights_dict, ecdf, mu, sd, dsir_ecdfs,
             impute_neutral=True):
    df = df.copy()
    if "dsir_composite" not in df.columns:
        b = ecdf_rank(df["dsir_books"].fillna(np.median(dsir_ecdfs["dsir_books"])).to_numpy(),
                      dsir_ecdfs["dsir_books"])
        m = ecdf_rank(df["dsir_math"].fillna(np.median(dsir_ecdfs["dsir_math"])).to_numpy(),
                      dsir_ecdfs["dsir_math"])
        w = ecdf_rank(df["dsir_wiki"].fillna(np.median(dsir_ecdfs["dsir_wiki"])).to_numpy(),
                      dsir_ecdfs["dsir_wiki"])
        df["dsir_composite"] = (b + m + w) / 3.0

    R = pd.DataFrame(index=df.index)
    for f in features:
        vals = df[f].to_numpy(dtype=float)
        mask_nan = ~np.isfinite(vals)
        if f == "dsir_composite":
            r = ecdf_rank(vals, ecdf[f])
        else:
            r = np.empty(len(vals))
            if mask_nan.all():
                r[:] = 0.5
            else:
                r[~mask_nan] = ecdf_rank(vals[~mask_nan], ecdf[f])
                if impute_neutral:
                    r[mask_nan] = 0.5
                else:
                    r[mask_nan] = np.nan
        if orient[f] < 0:
            r = 1.0 - r
        R[f] = r

    Z = (R - pd.Series(mu)) / pd.Series(sd)
    w = np.array([weights_dict[f] for f in features], dtype=float)
    return Z[features].to_numpy() @ w


def build_direction_table(df):
    """Issue #2 + #3 + reviewer direction mismatch."""
    print("\n" + "=" * 72)
    print("STEP 1b: Direction table for all 22 indicators (issue #2, #3)")
    print("=" * 72)

    rows = []
    n = len(df)
    r_q = rankdata(df["qurater_mean"].to_numpy()) / n
    r_f = rankdata(df["fineweb_edu_mean"].to_numpy()) / n
    anchor = (r_q + r_f) / 2.0

    for f in ALL_22_INDICATORS:
        if f not in df.columns:
            continue
        semantic = SEMANTIC_DIRECTION.get(f, 0)
        rf = rankdata(df[f].fillna(df[f].median()).to_numpy()) / n
        corr = float(np.corrcoef(rf, anchor)[0, 1])
        anchor_sign = +1 if corr >= 0 else -1
        in_score = (f in V3_FEATURES) or (f == "dsir_composite")
        in_anchor = (f in ANCHORS)
        rows.append({
            "indicator": f,
            "semantic_direction": semantic,
            "anchor_correlation": corr,
            "anchor_implied_sign": anchor_sign,
            "used_in_score": in_score,
            "used_as_anchor": in_anchor,
            "agreement_semantic_vs_anchor": (semantic == anchor_sign)
                if semantic != 0 else None,
        })

    table = pd.DataFrame(rows)
    print(table.to_string(index=False))

    # Excluded indicators
    excluded = [f for f in ALL_22_INDICATORS
                if f not in V3_FEATURES and f not in ANCHORS]
    print(f"\nExcluded from score ({len(excluded)}):")
    for f in excluded:
        print(f"  - {f}")

    # ---- Direction mismatch audit (reviewer feedback) ----
    mismatches = table[
        table["agreement_semantic_vs_anchor"] == False
    ][["indicator", "semantic_direction", "anchor_implied_sign"]]
    if len(mismatches) > 0:
        print(f"\n  [NOTE] {len(mismatches)} indicators have semantic direction")
        print(f"         different from anchor-implied sign.")
        print(f"         The score uses anchor-implied sign (via orientation);")
        print(f"         semantic direction is for reference only.")
        for _, r in mismatches.iterrows():
            print(f"         {r['indicator']:50s} "
                  f"semantic={r['semantic_direction']:+d}, "
                  f"anchor={r['anchor_implied_sign']:+d}")
    else:
        print(f"\n  [OK] All indicators have semantic direction matching anchor sign.")

    Q1_TABLES.mkdir(parents=True, exist_ok=True)
    table.to_csv(Q1_TABLES / "q1_direction_table.csv", index=False)
    print(f"\nSaved: {Q1_TABLES / 'q1_direction_table.csv'}")
    return table, excluded


def train_weights(df):
    print("\n" + "=" * 72)
    print("STEP 2: Train Q_i weights (ridge regression)")
    print("=" * 72)

    all_cols = V3_FEATURES + ANCHORS
    df_v3 = df.dropna(subset=all_cols).copy()
    print(f"Complete cases: {len(df_v3)}")

    df_rank = pd.DataFrame(index=df_v3.index)
    for f in all_cols:
        df_rank[f] = rankdata(df_v3[f].to_numpy()) / len(df_v3)

    anchor = (df_rank["qurater_mean"] + df_rank["fineweb_edu_mean"]) / 2

    orient = {}
    print("\n--- Orientations (11 score features + 2 anchors) ---")
    for f in all_cols:
        if f in MANUAL_OVERRIDE:
            orient[f] = MANUAL_OVERRIDE[f]
            tag = " [OVERRIDE]"
        else:
            r = np.corrcoef(df_rank[f].to_numpy(), anchor.to_numpy())[0, 1]
            orient[f] = +1 if r >= 0 else -1
            tag = ""
        print(f"  {f:55s} {orient[f]:+d}{tag}")

    df_oriented = df_rank.copy()
    for f in all_cols:
        if orient[f] < 0:
            df_oriented[f] = 1.0 - df_oriented[f]

    Z = (df_oriented - df_oriented.mean()) / df_oriented.std()
    X = Z[V3_FEATURES].to_numpy()
    y = anchor.to_numpy() - anchor.to_numpy().mean()

    XtX = X.T @ X + LAMBDA * np.eye(X.shape[1])
    w = np.linalg.solve(XtX, X.T @ y)
    w_norm = w / np.abs(w).sum()

    print(f"\n{'Feature':55s} {'w_raw':>10s} {'w_norm':>10s}")
    print("-" * 78)
    for f, wr, wn in sorted(zip(V3_FEATURES, w, w_norm),
                             key=lambda x: -abs(x[1])):
        print(f"{f:55s} {wr:+10.4f} {wn:+10.4f}")

    a1_mu = df_oriented[V3_FEATURES].mean()
    a1_sd = df_oriented[V3_FEATURES].std(ddof=1).replace(0, 1.0)
    ecdf = {f: np.sort(df_v3[f].to_numpy(dtype=float)) for f in V3_FEATURES}
    dsir_ecdfs = {
        "dsir_books": np.sort(df["dsir_books"].dropna().to_numpy()),
        "dsir_math": np.sort(df["dsir_math"].dropna().to_numpy()),
        "dsir_wiki": np.sort(df["dsir_wiki"].dropna().to_numpy()),
    }
    weights_dict = dict(zip(V3_FEATURES, w_norm.tolist()))
    q_a1 = apply_qi(df, V3_FEATURES, orient, weights_dict, ecdf,
                    a1_mu, a1_sd, dsir_ecdfs, impute_neutral=True)
    df_v3_eval = df.copy()
    df_v3_eval["Q_i"] = q_a1

    print("\n--- Anchor alignment (A1) ---")
    for a in ANCHORS:
        r = spearman(df_v3_eval["Q_i"], df_v3_eval[a])
        print(f"  Q_i vs {a:20s} = {r:+.4f}")

    return {
        "features": V3_FEATURES,
        "orientation": orient,
        "weights": weights_dict,
        "weights_raw": dict(zip(V3_FEATURES, w.tolist())),
        "anchors": ANCHORS,
        "lambda": LAMBDA,
        "a1_mu": {f: float(a1_mu[f]) for f in V3_FEATURES},
        "a1_sd": {f: float(a1_sd[f]) for f in V3_FEATURES},
        "ecdf": ecdf, "dsir_ecdfs": dsir_ecdfs,
        "a1_scored": df_v3_eval,
    }


def save_artifacts(model):
    print("\n" + "=" * 72)
    print("STEP 3: Save model artifacts")
    print("=" * 72)
    Q1_MODELS.mkdir(parents=True, exist_ok=True)
    with open(Q1_WEIGHTS, "w", encoding="utf-8") as f:
        json.dump({
            "version": "v3_final",
            "features": model["features"],
            "anchors": model["anchors"],
            "weights": model["weights"],
            "weights_raw": model["weights_raw"],
            "orientation": model["orientation"],
            "lambda": model["lambda"],
        }, f, indent=2)
    print(f"Saved: {Q1_WEIGHTS}")
    np.savez(Q1_ECDF, **model["ecdf"])
    print(f"Saved: {Q1_ECDF}")
    with open(Q1_NORM, "w", encoding="utf-8") as f:
        json.dump({
            "version": "v3_final",
            "features": model["features"],
            "mean_oriented_ecdf_rank_A1": model["a1_mu"],
            "std_oriented_ecdf_rank_A1": model["a1_sd"],
        }, f, indent=2)
    print(f"Saved: {Q1_NORM}")


def stream_extension(path, domain, model):
    print(f"\n  Reading {path.name} ...")
    rows = []
    with lzma.open(path, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            rec = json.loads(line)
            row = {"id": rec.get("id")}
            for m in EXT_LIST_METRICS:
                row[m] = rec.get(m)
            for m in EXT_RAW_SCALARS:
                row[m] = rec.get(m)
            rows.append(row)
            if (i + 1) % 50000 == 0:
                print(f"    ... {i+1}")
    df = pd.DataFrame(rows)
    for col in EXT_LIST_METRICS:
        df[f"{col}_mean"] = (df[col].apply(parse_list)
                             .apply(lambda v: float(np.mean(v)) if v else np.nan))
    df["_source_domain"] = domain
    df["Q_i"] = apply_qi(
        df, model["features"], model["orientation"], model["weights"],
        model["ecdf"], model["a1_mu"], model["a1_sd"], model["dsir_ecdfs"],
        impute_neutral=True,
    )
    df["id_str"] = df["id"].astype(str)
    keep = ["id_str", "_source_domain", "Q_i"]
    print(f"  {domain}: {len(df)} rows scored")
    return df[keep].copy()


def full_extension_summary(ext_df, label):
    if len(ext_df) == 0:
        return
    s = ext_df["Q_i"]
    row = {
        "dataset": label,
        "domain": ext_df["_source_domain"].iloc[0],
        "n": int(len(ext_df)),
        "mean_Q": float(s.mean()),
        "std_Q": float(s.std(ddof=1)),
        "median_Q": float(s.median()),
        "q05": float(s.quantile(0.05)),
        "q25": float(s.quantile(0.25)),
        "q75": float(s.quantile(0.75)),
        "q95": float(s.quantile(0.95)),
        "min_Q": float(s.min()),
        "max_Q": float(s.max()),
    }
    return row


def main():
    df_a1 = load_a1()
    df_a1 = build_dsir_composite_a1(df_a1)

    direction_table, excluded = build_direction_table(df_a1)

    model = train_weights(df_a1)
    save_artifacts(model)

    a1_scored = model["a1_scored"].copy()
    a1_scored["id_str"] = a1_scored["id"].astype(str)
    a1_out = a1_scored[["id_str", "_source_domain", "Q_i"]].copy()
    a1_out.to_parquet(A1_WITH_QI, index=False)
    print(f"Saved: {A1_WITH_QI}")

    print("\n" + "=" * 72)
    print("STEP 4: Apply to extension data")
    print("=" * 72)

    ext_rows = []
    arx = None
    gh = None
    if ARXIV_RAW.exists():
        arx = stream_extension(ARXIV_RAW, "arxiv", model)
        arx.to_parquet(ARXIV_WITH_QI, index=False)
        ext_rows.append(full_extension_summary(arx, "A2_arxiv_full"))
    if GITHUB_RAW.exists():
        gh = stream_extension(GITHUB_RAW, "github", model)
        gh.to_parquet(GITHUB_WITH_QI, index=False)
        ext_rows.append(full_extension_summary(gh, "A3_github_full"))

    print("\n" + "=" * 72)
    print("STEP 4b: Full A2/A3 domain-level Q summary (issue #5)")
    print("=" * 72)

    a1_dom_rows = []
    for dom, g in a1_scored.groupby("_source_domain"):
        s = g["Q_i"]
        a1_dom_rows.append({
            "dataset": "A1_sample", "domain": dom, "n": int(len(g)),
            "mean_Q": float(s.mean()), "std_Q": float(s.std(ddof=1)),
            "median_Q": float(s.median()),
            "q05": float(s.quantile(0.05)), "q25": float(s.quantile(0.25)),
            "q75": float(s.quantile(0.75)), "q95": float(s.quantile(0.95)),
            "min_Q": float(s.min()), "max_Q": float(s.max()),
        })

    full_summary = pd.DataFrame(a1_dom_rows + [r for r in ext_rows if r])
    full_summary = full_summary.sort_values(["dataset", "domain"])
    Q1_TABLES.mkdir(parents=True, exist_ok=True)
    full_summary.to_csv(Q1_TABLES / "q1_domain_summary_A1_A2_A3.csv", index=False)
    print(full_summary.to_string(index=False))
    print(f"\nSaved: {Q1_TABLES / 'q1_domain_summary_A1_A2_A3.csv'}")

    print("\n" + "=" * 72)
    print("STEP 5: Overlap validation")
    print("=" * 72)
    if arx is not None:
        a1_sub = a1_scored[a1_scored["_source_domain"] == "arxiv"][["id_str", "Q_i"]]
        m = a1_sub.merge(arx, on="id_str", suffixes=("_a1", "_ext"))
        if len(m) > 0:
            r = spearman(m["Q_i_a1"], m["Q_i_ext"])
            d = np.abs(m["Q_i_a1"] - m["Q_i_ext"]).max()
            print(f"  A1 arxiv vs A2: n={len(m)}, Spearman={r:.6f}, max|dQ|={d:.3e}")
    if gh is not None:
        a1_sub = a1_scored[a1_scored["_source_domain"] == "github"][["id_str", "Q_i"]]
        m = a1_sub.merge(gh, on="id_str", suffixes=("_a1", "_ext"))
        if len(m) > 0:
            r = spearman(m["Q_i_a1"], m["Q_i_ext"])
            d = np.abs(m["Q_i_a1"] - m["Q_i_ext"]).max()
            print(f"  A1 github vs A3: n={len(m)}, Spearman={r:.6f}, max|dQ|={d:.3e}")

    print("\n[DONE] 02_build_quality_score.py")


if __name__ == "__main__":
    main()