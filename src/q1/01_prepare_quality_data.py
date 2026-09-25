#!/usr/bin/env python3
"""
Q1 Step 1: Prepare quality data.

Merges logic from:
  - check_keys.py        (verify raw inputs)
  - extract_A1.py        (extract 22 metrics)
  - scalarize_A1.py      (list -> mean/std/min/max/nchunks)

Inputs:
  data/raw/A_data_value/slimpajama_quality_signal_sample.jsonl.xz

Outputs:
  data/processed/q1/A1_quality_features.parquet
  data/processed/q1/A1_quality_scalarized.parquet
"""
import json
import lzma
from collections import Counter

import numpy as np
import pandas as pd

from config import A_DATA, SLIMPAJAMA_SAMPLE, A1_FEATURES, A1_SCALARIZED
from src.common.metrics import parse_list


RAW_FILES = [
    "slimpajama_quality_signal_sample.jsonl.xz",
    "regmix_domain_sample.jsonl.xz",
]

METRICS = [
    "modernbert_cleanliness", "modernbert_reasoning",
    "modernbert_professionalism", "modernbert_readability",
    "fluency_en", "dsir_books", "dsir_math", "dsir_wiki",
    "ad_en", "rps_doc_num_sentences", "rps_doc_word_count",
    "rps_doc_frac_no_alph_words", "rps_doc_frac_chars_top_2gram",
    "rps_doc_frac_chars_top_3gram", "rps_doc_frac_unique_words",
    "rps_doc_mean_word_length", "rps_doc_unigram_entropy",
    "rps_lines_uppercase_letter_fraction",
    "rps_lines_numerical_chars_fraction",
    "rps_lines_ending_with_terminal_punctution_mark",
    "fineweb_edu", "qurater",
]

LIST_METRICS = [
    "modernbert_cleanliness", "modernbert_reasoning",
    "modernbert_professionalism", "modernbert_readability",
    "fluency_en", "ad_en", "fineweb_edu", "qurater",
]


def verify_inputs():
    print("=" * 72)
    print("STEP 1: Verify raw inputs")
    print("=" * 72)
    for name in RAW_FILES:
        path = A_DATA / name
        status = "[OK]" if path.exists() else "[MISSING]"
        print(f"  {status} {path.name}")
    if not SLIMPAJAMA_SAMPLE.exists():
        raise FileNotFoundError(f"Missing: {SLIMPAJAMA_SAMPLE}")
    print()


def extract_features():
    print("=" * 72)
    print("STEP 2: Extract quality features")
    print("=" * 72)

    rows = []
    domains = Counter()

    with lzma.open(SLIMPAJAMA_SAMPLE, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            rec = json.loads(line)
            row = {
                "id": rec.get("id"),
                "_source_domain": rec.get("_source_domain"),
                "_source_path": rec.get("_source_path"),
                "sub_path": rec.get("sub_path"),
            }
            for m in METRICS:
                row[m] = rec.get(m)
            rows.append(row)
            domains[rec.get("_source_domain")] += 1
            if (i + 1) % 5000 == 0:
                print(f"  Processed {i+1} records...")

    df = pd.DataFrame(rows)
    print(f"\nTotal records: {len(df)}")
    print(f"Domains: {dict(domains)}")

    A1_FEATURES.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(A1_FEATURES, index=False)
    print(f"Saved: {A1_FEATURES}")
    return df


def scalarize(df):
    print("\n" + "=" * 72)
    print("STEP 3: Scalarize list metrics")
    print("=" * 72)

    df = df.copy()
    for col in LIST_METRICS:
        print(f"  {col}...")
        parsed = df[col].apply(parse_list)
        df[f"{col}_mean"] = parsed.apply(lambda v: float(np.mean(v)) if v else np.nan)
        df[f"{col}_std"] = parsed.apply(lambda v: float(np.std(v)) if len(v) > 1 else 0.0)
        df[f"{col}_min"] = parsed.apply(lambda v: float(np.min(v)) if v else np.nan)
        df[f"{col}_max"] = parsed.apply(lambda v: float(np.max(v)) if v else np.nan)
        df[f"{col}_nchunks"] = parsed.apply(len)
        df = df.drop(columns=[col])

    df.to_parquet(A1_SCALARIZED, index=False)
    print(f"\nSaved: {A1_SCALARIZED}")
    print(f"Rows: {len(df)}, Cols: {len(df.columns)}")
    return df


def main():
    verify_inputs()
    df_raw = extract_features()
    scalarize(df_raw)
    print("\n[DONE] 01_prepare_quality_data.py")


if __name__ == "__main__":
    main()