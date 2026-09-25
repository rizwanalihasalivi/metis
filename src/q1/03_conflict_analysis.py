#!/usr/bin/env python3
"""Q1 Step 3: 7-family conflict analysis + resolution + confidence-shrunk Q_i."""
import json

import numpy as np
import pandas as pd

from config import (
    A1_SCALARIZED, A1_WITH_QI,
    Q1_CONFLICT, Q1_TABLES, Q1_FIGURES,
    CONFLICT_PRIMARY,
)
from src.common.metrics import rankdata, spearman, ecdf_rank, pairwise_mean_abs_diff

ALL22 = [
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

FAMILIES = {
    "semantic_model": ["modernbert_cleanliness_mean", "modernbert_readability_mean",
                       "modernbert_reasoning_mean", "modernbert_professionalism_mean",
                       "fluency_en_mean"],
    "retrieval_dsir": ["dsir_books", "dsir_wiki", "dsir_math"],
    "quality_anchors": ["qurater_mean", "fineweb_edu_mean"],
    "advertising": ["ad_en_mean"],
    "document_scale": ["rps_doc_word_count", "rps_doc_num_sentences"],
    "lexical_quality": ["rps_doc_unigram_entropy", "rps_doc_frac_unique_words",
                        "rps_doc_mean_word_length"],
    "surface_noise": ["rps_doc_frac_no_alph_words", "rps_doc_frac_chars_top_2gram",
                      "rps_doc_frac_chars_top_3gram", "rps_lines_uppercase_letter_fraction",
                      "rps_lines_ending_with_terminal_punctution_mark",
                      "rps_lines_numerical_chars_fraction"],
}


def main():
    print("=" * 78)
    print("Q1 CONFLICT ANALYSIS (7 families)")
    print("=" * 78)

    df = pd.read_parquet(A1_SCALARIZED).copy()
    df = df.dropna(subset=ALL22).copy()
    df["id_str"] = df["id"].astype(str)
    print(f"A1 rows with all 22 indicators: {len(df):,}")

    # ECDFs from A1
    ECDF = {f: np.sort(df[f].to_numpy(dtype=float)) for f in ALL22}

    # Orientation vs anchor
    n = len(df)
    r_q = rankdata(df["qurater_mean"].to_numpy()) / n
    r_f = rankdata(df["fineweb_edu_mean"].to_numpy()) / n
    anchor = (r_q + r_f) / 2.0

    orient = {}
    for f in ALL22:
        if f in ("qurater_mean", "fineweb_edu_mean"):
            orient[f] = +1
            continue
        rf = rankdata(df[f].to_numpy()) / n
        orient[f] = +1 if np.corrcoef(rf, anchor)[0, 1] >= 0 else -1
    orient["ad_en_mean"] = -1

    # Oriented ranks
    R = pd.DataFrame(index=df.index)
    for f in ALL22:
        r = ecdf_rank(df[f].to_numpy(), ECDF[f])
        if orient[f] < 0:
            r = 1.0 - r
        R[f] = r

    # Family scores
    S = pd.DataFrame(index=df.index)
    for fam, members in FAMILIES.items():
        S[fam] = R[members].mean(axis=1)

    # Conflict C_i = mean pairwise |S_i - S_j|
    C = pairwise_mean_abs_diff(S.to_numpy())
    df["conflict_C"] = C

    # Thresholds
    tau85 = float(np.quantile(C, 0.85))
    tau90 = float(np.quantile(C, 0.90))
    tau95 = float(np.quantile(C, 0.95))
    tau = tau90
    print(f"\nConflict thresholds: q85={tau85:.6f}  q90={tau90:.6f}  q95={tau95:.6f}")

    # Resolution: 2^(-excess)
    denom = max(1e-12, 1.0 - tau)
    excess = np.clip((C - tau) / denom, 0.0, 1.0)
    df["conflict_confidence"] = 2.0 ** (-excess)
    df["high_conflict"] = C > tau

    # Save
    Q1_CONFLICT.mkdir(parents=True, exist_ok=True)

    # Per-domain summary
    rows = []
    for dom, g in df.groupby("_source_domain"):
        rows.append({
            "domain": dom, "n": len(g),
            "mean_conflict_C": g["conflict_C"].mean(),
            "high_conflict_rate": g["high_conflict"].mean(),
            "mean_confidence": g["conflict_confidence"].mean(),
        })
    rows.append({
        "domain": "ALL", "n": len(df),
        "mean_conflict_C": df["conflict_C"].mean(),
        "high_conflict_rate": df["high_conflict"].mean(),
        "mean_confidence": df["conflict_confidence"].mean(),
    })
    summary = pd.DataFrame(rows)
    summary.to_csv(Q1_CONFLICT / "conflict_summary.csv", index=False)

    # Threshold sensitivity
    sens = []
    for q, t in [("q85", tau85), ("q90", tau90), ("q95", tau95)]:
        sens.append({"quantile": q, "tau": t,
                     "high_conflict_rate": float((C > t).mean())})
    pd.DataFrame(sens).to_csv(Q1_CONFLICT / "threshold_sensitivity.csv", index=False)

    # Save scored slim
    df[["id_str", "_source_domain", "conflict_C",
        "high_conflict", "conflict_confidence"]].to_parquet(
        Q1_CONFLICT / "A1_conflict_scored.parquet", index=False
    )

    # Orientation json
    with open(Q1_CONFLICT / "orientation.json", "w") as f:
        json.dump({"orientation": orient, "families": FAMILIES,
                   "thresholds": {"q85": tau85, "q90": tau90, "q95": tau95}},
                  f, indent=2)

    # ============================================================
    # Confidence-shrunk Q_i (Review issue #4)
    # ============================================================
    print("\n--- Confidence-shrunk Q_i (issue #4) ---")
    try:
        if A1_WITH_QI.exists():
            q1 = pd.read_parquet(A1_WITH_QI)
            # A1_WITH_QI has id_str, not id
            df_merge = df.merge(q1, on="id_str", how="inner", suffixes=("", "_q"))
            if "Q_i" in df_merge.columns:
                df_merge["Q_i_resolved"] = (
                    df_merge["Q_i"] * df_merge["conflict_confidence"]
                )
                print(f"  Merged rows: {len(df_merge)}")
                print(f"  Mean Q_i          = {df_merge['Q_i'].mean():+.4f}")
                print(f"  Mean Q_i_resolved = {df_merge['Q_i_resolved'].mean():+.4f}")
                corr = df_merge[["Q_i", "Q_i_resolved"]].corr().iloc[0, 1]
                print(f"  Corr(Q_i, Q_i_resolved) = {corr:.4f}")
                df_merge[["id_str", "_source_domain", "Q_i", "conflict_C",
                          "conflict_confidence", "Q_i_resolved"]].to_parquet(
                    Q1_CONFLICT / "A1_with_conflict_resolved.parquet", index=False
                )
                print(f"  Saved: {Q1_CONFLICT}/A1_with_conflict_resolved.parquet")
        else:
            print(f"  [WARN] {A1_WITH_QI} not found — run 02_build_quality_score.py first")
    except Exception as e:
        print(f"  [WARN] could not compute resolved Q_i: {type(e).__name__}: {e}")

    print(f"\nSaved: {Q1_CONFLICT}/conflict_summary.csv")
    print(summary.to_string(index=False))
    print("\n[DONE] 03_conflict_analysis.py")


if __name__ == "__main__":
    main()