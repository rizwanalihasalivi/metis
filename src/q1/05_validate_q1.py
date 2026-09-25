#!/usr/bin/env python3
"""Q1 Step 5: Validation — correlations, slimming, summaries."""
import numpy as np
import pandas as pd

from config import (
    A1_SCALARIZED, A1_WITH_QI, ARXIV_WITH_QI, GITHUB_WITH_QI,
    Q1_TABLES, Q1_DIR,
)
from src.common.metrics import spearman


METRICS = [
    "modernbert_cleanliness_mean", "modernbert_reasoning_mean",
    "modernbert_professionalism_mean", "modernbert_readability_mean",
    "fluency_en_mean", "dsir_books", "dsir_math", "dsir_wiki",
    "ad_en_mean", "rps_doc_num_sentences", "rps_doc_word_count",
    "rps_doc_frac_no_alph_words", "rps_doc_frac_chars_top_2gram",
    "rps_doc_frac_chars_top_3gram", "rps_doc_frac_unique_words",
    "rps_doc_mean_word_length", "rps_doc_unigram_entropy",
    "rps_lines_uppercase_letter_fraction",
    "rps_lines_numerical_chars_fraction",
    "rps_lines_ending_with_terminal_punctution_mark",
    "fineweb_edu_mean", "qurater_mean",
]


def main():
    print("=" * 72)
    print("Q1 VALIDATION")
    print("=" * 72)

    # ------------------------------------------------------------------
    # Load A1 scalarized (has raw metrics)
    # ------------------------------------------------------------------
    df = pd.read_parquet(A1_SCALARIZED)
    df_metrics = df[METRICS].dropna()
    print(f"A1 rows: {len(df_metrics)}")

    # ------------------------------------------------------------------
    # Correlation matrices (Pearson + Spearman)
    # ------------------------------------------------------------------
    pearson = df_metrics.corr(method="pearson")
    spearman_m = df_metrics.corr(method="spearman")
    pearson.to_csv(Q1_TABLES / "A1_corr_pearson.csv")
    spearman_m.to_csv(Q1_TABLES / "A1_corr_spearman.csv")
    print("Saved: A1_corr_pearson.csv, A1_corr_spearman.csv")

    # ------------------------------------------------------------------
    # Load A1 with Q_i and merge anchors on id_str
    # ------------------------------------------------------------------
    q1 = pd.read_parquet(A1_WITH_QI)

    df_anchors = df[["id", "qurater_mean", "fineweb_edu_mean"]].copy()
    df_anchors["id_str"] = df_anchors["id"].astype(str)

    merged = q1.merge(
        df_anchors[["id_str", "qurater_mean", "fineweb_edu_mean"]],
        on="id_str", how="inner",
    )
    print(f"\nMerged for anchor check: {len(merged)} rows")

    print("\n--- Q_i vs anchors ---")
    for a in ["qurater_mean", "fineweb_edu_mean"]:
        r = spearman(merged["Q_i"], merged[a])
        print(f"  Q_i vs {a:20s} = {r:+.4f}")

    # ------------------------------------------------------------------
    # Domain summary
    # ------------------------------------------------------------------
    print("\n--- Q_d by domain ---")
    qd = (
        q1.groupby("_source_domain")["Q_i"]
        .agg(["mean", "std", "count", "median"])
        .sort_values("mean", ascending=False)
    )
    qd.to_csv(Q1_TABLES / "quality_domain_summary.csv")
    print(qd.to_string())

    # ------------------------------------------------------------------
    # Cross-dataset overlap validation
    # ------------------------------------------------------------------
    print("\n--- Cross-dataset validation ---")
    for ext_path, domain in [(ARXIV_WITH_QI, "arxiv"),
                             (GITHUB_WITH_QI, "github")]:
        try:
            ext = pd.read_parquet(ext_path)

            a1_sub = q1[q1["_source_domain"] == domain][["id_str", "Q_i"]].copy()
            ext_sub = ext[["id_str", "Q_i"]].rename(columns={"Q_i": "Q_i_ext"})

            m = a1_sub.merge(ext_sub, on="id_str")
            if len(m) > 0:
                r = spearman(m["Q_i"], m["Q_i_ext"])
                d = float(np.abs(m["Q_i"] - m["Q_i_ext"]).max())
                print(f"  A1 {domain} vs full: n={len(m)}  "
                      f"Spearman={r:.6f}  max|dQ|={d:.3e}")
            else:
                print(f"  A1 {domain}: no overlap")
        except FileNotFoundError:
            print(f"  [SKIP] {ext_path.name} missing")

    # ------------------------------------------------------------------
    # Slim outputs
    # ------------------------------------------------------------------
    print("\n--- Slim outputs ---")
    slim = q1[["id_str", "_source_domain", "Q_i"]].copy()
    slim.to_parquet(Q1_DIR / "A1_with_Qi_slim.parquet", index=False)
    print("  Saved: A1_with_Qi_slim.parquet")

    for ext_path in [ARXIV_WITH_QI, GITHUB_WITH_QI]:
        try:
            ext = pd.read_parquet(ext_path)
            out = ext_path.with_name(ext_path.stem + "_slim.parquet")
            ext[["id_str", "_source_domain", "Q_i"]].to_parquet(out, index=False)
            print(f"  Saved: {out.name}")
        except FileNotFoundError:
            pass

    print("\n[DONE] 05_validate_q1.py")


if __name__ == "__main__":
    main()