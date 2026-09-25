#!/usr/bin/env python3
"""
Q4 / C8 detailed_results extractor
==================================

Put this script in:
    real_attachments/C_efficiency_evolution/

Expected folder:
    real_attachments/C_efficiency_evolution/detailed_results/

It scans every model subfolder, chooses the latest parseable JSON, skips broken
JSON files, extracts per-task metrics, and creates small CSV files that can be
uploaded back to ChatGPT.

Outputs
-------
1) C8_task_metrics_long.csv
   One row per model/task/metric.

2) C8_task_primary_scores.csv
   One primary task-level score per model/task, with benchmark family labels.

3) C8_model_family_aggregate.csv
   Per-model aggregation over the six benchmark families:
   IFEval, BBH, MATH, GPQA, MUSR, MMLU-PRO.

4) C8_extraction_audit.csv
   One row per model directory: total JSON count, parseable count, chosen JSON.

5) C8_broken_json.csv
   Broken/truncated JSON files and their errors.

6) C8_extraction_report.txt
   Human-readable summary.

Important:
- The script NEVER silently replaces broken JSON.
- It uses the latest parseable result in each model directory.
- Raw task metrics are preserved in long format.
- For task-level aggregation, a transparent primary-metric rule is used:
    BBH / GPQA / MUSR -> acc_norm when available
    MATH -> exact_match when available
    MMLU-PRO -> acc when available
    IFEval -> mean of available strict/loose prompt/instance accuracies
  with sensible fallbacks.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DETAILS = ROOT / "detailed_results"

if not DETAILS.exists():
    # Allow running from another directory.
    alt = Path("detailed_results")
    if alt.exists():
        DETAILS = alt.resolve()
    else:
        raise FileNotFoundError(
            "Could not find detailed_results/. "
            "Put this script in C_efficiency_evolution/ and run it there."
        )


DATE_PATTERNS = [
    re.compile(r"(20\d{2})[-_](\d{2})[-_](\d{2})[T_\-](\d{2})[-_:](\d{2})[-_:](\d{2})"),
    re.compile(r"(20\d{2})[-_](\d{2})[-_](\d{2})"),
]


def timestamp_from_name(path: Path) -> float:
    """Prefer embedded timestamp; fall back to file modification time."""
    s = path.name
    for pat in DATE_PATTERNS:
        m = pat.search(s)
        if m:
            try:
                parts = [int(x) for x in m.groups()]
                if len(parts) == 6:
                    dt = datetime(*parts)
                else:
                    dt = datetime(parts[0], parts[1], parts[2])
                return dt.timestamp()
            except Exception:
                pass
    try:
        return path.stat().st_mtime
    except Exception:
        return 0.0


def parse_json(path: Path):
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def find_results_dict(obj):
    """
    Find a dictionary that looks like a benchmark task->metrics mapping.
    Handles common Open LLM Leaderboard layouts:
      {"results": {...}}
      {"all": {...}}
      nested wrappers.
    """
    candidates = []

    def walk(x, depth=0):
        if depth > 8:
            return
        if isinstance(x, dict):
            # strong candidates
            for key in ("results", "all"):
                v = x.get(key)
                if isinstance(v, dict):
                    score = sum(
                        1 for k, vv in v.items()
                        if isinstance(vv, dict) and (
                            "leaderboard" in str(k).lower()
                            or any(
                                isinstance(mk, str) and
                                any(z in mk.lower() for z in ("acc", "exact_match", "score"))
                                for mk in vv.keys()
                            )
                        )
                    )
                    if score:
                        candidates.append((score, depth, v))

            # the object itself may be task->metrics
            score = sum(
                1 for k, vv in x.items()
                if isinstance(vv, dict) and (
                    "leaderboard" in str(k).lower()
                    or any(
                        isinstance(mk, str) and
                        any(z in mk.lower() for z in ("acc", "exact_match", "score"))
                        for mk in vv.keys()
                    )
                )
            )
            if score >= 3:
                candidates.append((score, depth, x))

            for v in x.values():
                if isinstance(v, (dict, list)):
                    walk(v, depth + 1)

        elif isinstance(x, list):
            for v in x:
                if isinstance(v, (dict, list)):
                    walk(v, depth + 1)

    walk(obj)

    if not candidates:
        return None

    # Prefer more task entries, then shallower depth.
    candidates.sort(key=lambda t: (t[0], -t[1]), reverse=True)
    return candidates[0][2]


def is_numeric(v):
    return isinstance(v, (int, float, np.integer, np.floating)) and np.isfinite(v)


def family_from_task(task: str):
    t = task.lower().replace("-", "_")

    # avoid double assigning group aliases
    if "ifeval" in t:
        return "IFEval"
    if "mmlu_pro" in t or "mmlu-pro" in t:
        return "MMLU-PRO"
    if "gpqa" in t:
        return "GPQA"
    if "musr" in t:
        return "MUSR"
    if "math" in t and ("hard" in t or "leaderboard_math" in t):
        return "MATH"
    if "bbh" in t:
        return "BBH"
    return None


def clean_metric_name(s: str):
    return str(s).strip().lower().replace(" ", "_")


def primary_metric(task: str, metrics: dict):
    """
    Return (metric_name, value) for one task.
    Excludes stderr and metadata.
    """
    num = {}
    for k, v in metrics.items():
        kk = clean_metric_name(k)
        if (
            is_numeric(v)
            and "stderr" not in kk
            and kk not in {"n_samples", "samples", "num_samples"}
        ):
            num[str(k)] = float(v)

    if not num:
        return None, None

    fam = family_from_task(task)
    lowered = {clean_metric_name(k): (k, v) for k, v in num.items()}

    def get_exact(*names):
        for name in names:
            if name in lowered:
                return lowered[name]
        return None

    if fam == "IFEval":
        # The aggregate IFEval task commonly exposes 4 meaningful accuracies.
        vals = []
        names = []
        for key, (orig, val) in lowered.items():
            if any(
                z in key
                for z in (
                    "prompt_level_strict_acc",
                    "prompt_level_loose_acc",
                    "inst_level_strict_acc",
                    "inst_level_loose_acc",
                )
            ):
                vals.append(val)
                names.append(orig)
        if vals:
            return "mean(" + ";".join(sorted(names)) + ")", float(np.mean(vals))

    priority = {
        "BBH": [
            "acc_norm,none", "acc_norm",
            "acc,none", "acc",
        ],
        "GPQA": [
            "acc_norm,none", "acc_norm",
            "acc,none", "acc",
        ],
        "MUSR": [
            "acc_norm,none", "acc_norm",
            "acc,none", "acc",
        ],
        "MATH": [
            "exact_match,none", "exact_match",
            "acc_norm,none", "acc_norm",
            "acc,none", "acc",
        ],
        "MMLU-PRO": [
            "acc,none", "acc",
            "acc_norm,none", "acc_norm",
        ],
    }

    for pref in priority.get(fam, []):
        ans = get_exact(pref)
        if ans is not None:
            return ans

    # Generic fallback.
    generic = [
        "acc_norm,none", "acc_norm",
        "acc,none", "acc",
        "exact_match,none", "exact_match",
        "score,none", "score",
    ]
    for pref in generic:
        ans = get_exact(pref)
        if ans is not None:
            return ans

    # Last resort: mean of all non-stderr numeric metrics.
    return "mean_numeric_nonstderr", float(np.mean(list(num.values())))


def normalize_score(v):
    """
    Convert probability-like [0,1] score to percentage scale.
    Preserve values that are already percentages.
    """
    if v is None or not np.isfinite(v):
        return np.nan
    if -0.01 <= v <= 1.01:
        return 100.0 * v
    return float(v)


broken_rows = []
audit_rows = []
long_rows = []
primary_rows = []

model_dirs = [p for p in DETAILS.iterdir() if p.is_dir()]
model_dirs.sort(key=lambda p: p.name.lower())

for ix, model_dir in enumerate(model_dirs, start=1):
    json_files = sorted(model_dir.rglob("*.json"))
    parseable = []

    for fp in json_files:
        obj, err = parse_json(fp)
        if err is None:
            parseable.append((fp, obj))
        else:
            broken_rows.append({
                "model_dir": model_dir.name,
                "json_path": str(fp.relative_to(DETAILS)),
                "error": err,
            })

    if parseable:
        parseable.sort(key=lambda t: timestamp_from_name(t[0]), reverse=True)
        chosen_path, chosen_obj = parseable[0]
        results = find_results_dict(chosen_obj)
    else:
        chosen_path, chosen_obj, results = None, None, None

    audit_rows.append({
        "model_dir": model_dir.name,
        "json_count": len(json_files),
        "parseable_json_count": len(parseable),
        "chosen_json": (
            str(chosen_path.relative_to(DETAILS))
            if chosen_path is not None else ""
        ),
        "results_mapping_found": results is not None,
    })

    if results is None:
        continue

    for task, metrics in results.items():
        if not isinstance(metrics, dict):
            continue

        fam = family_from_task(str(task))

        # Preserve all numeric non-stderr metrics in long format.
        for metric, value in metrics.items():
            m = clean_metric_name(metric)
            if (
                is_numeric(value)
                and "stderr" not in m
                and m not in {"n_samples", "samples", "num_samples"}
            ):
                long_rows.append({
                    "model_dir": model_dir.name,
                    "chosen_json": str(chosen_path.relative_to(DETAILS)),
                    "task": str(task),
                    "benchmark_family": fam,
                    "metric": str(metric),
                    "value_raw": float(value),
                    "value_percent": normalize_score(float(value)),
                })

        if fam is not None:
            metric, value = primary_metric(str(task), metrics)
            if value is not None:
                primary_rows.append({
                    "model_dir": model_dir.name,
                    "chosen_json": str(chosen_path.relative_to(DETAILS)),
                    "task": str(task),
                    "benchmark_family": fam,
                    "primary_metric": metric,
                    "score_raw": float(value),
                    "score_percent": normalize_score(float(value)),
                })

    if ix % 100 == 0:
        print(f"Processed {ix:,}/{len(model_dirs):,} model directories")


audit = pd.DataFrame(audit_rows)
broken = pd.DataFrame(broken_rows)
long_df = pd.DataFrame(long_rows)
primary = pd.DataFrame(primary_rows)

audit.to_csv(ROOT / "C8_extraction_audit.csv", index=False)
broken.to_csv(ROOT / "C8_broken_json.csv", index=False)
long_df.to_csv(ROOT / "C8_task_metrics_long.csv", index=False)
primary.to_csv(ROOT / "C8_task_primary_scores.csv", index=False)


# ------------------------------------------------------------
# Model x family aggregation
# ------------------------------------------------------------

# Prefer child/subtask entries over the broad aggregate group when several
# tasks exist within a family. This makes the analysis genuinely task-level.
def is_broad_group(task, fam):
    t = task.lower().replace("-", "_")
    exacts = {
        "IFEval": {"leaderboard_ifeval", "ifeval"},
        "BBH": {"leaderboard_bbh", "bbh"},
        "MATH": {"leaderboard_math", "leaderboard_math_hard", "math", "math_hard"},
        "GPQA": {"leaderboard_gpqa", "gpqa"},
        "MUSR": {"leaderboard_musr", "musr"},
        "MMLU-PRO": {"leaderboard_mmlu_pro", "mmlu_pro"},
    }
    return t in exacts.get(fam, set())


if len(primary):
    primary["is_broad_group"] = [
        is_broad_group(t, f)
        for t, f in zip(primary["task"], primary["benchmark_family"])
    ]

    family_rows = []

    for (model, fam), g in primary.groupby(
        ["model_dir", "benchmark_family"], dropna=False
    ):
        children = g[~g["is_broad_group"]]
        use = children if len(children) else g

        vals = use["score_percent"].dropna()

        if len(vals):
            family_rows.append({
                "model_dir": model,
                "benchmark_family": fam,
                "n_tasks": int(len(vals)),
                "mean_task_score": float(vals.mean()),
                "median_task_score": float(vals.median()),
                "min_task_score": float(vals.min()),
                "max_task_score": float(vals.max()),
                "aggregation_source": (
                    "child_tasks"
                    if len(children) else
                    "aggregate_task_fallback"
                ),
            })

    family_long = pd.DataFrame(family_rows)

    if len(family_long):
        wide = family_long.pivot(
            index="model_dir",
            columns="benchmark_family",
            values="mean_task_score",
        ).reset_index()

        nwide = family_long.pivot(
            index="model_dir",
            columns="benchmark_family",
            values="n_tasks",
        ).add_prefix("n_tasks_").reset_index()

        model_family = wide.merge(nwide, on="model_dir", how="left")

        six = ["IFEval", "BBH", "MATH", "GPQA", "MUSR", "MMLU-PRO"]
        present = [c for c in six if c in model_family.columns]

        model_family["n_complete_families"] = model_family[present].notna().sum(axis=1)
        model_family["six_family_macro_average"] = model_family[present].mean(
            axis=1, skipna=True
        )
    else:
        model_family = pd.DataFrame()
else:
    family_long = pd.DataFrame()
    model_family = pd.DataFrame()

family_long.to_csv(ROOT / "C8_family_aggregation_long.csv", index=False)
model_family.to_csv(ROOT / "C8_model_family_aggregate.csv", index=False)


# ------------------------------------------------------------
# Report
# ------------------------------------------------------------

n_dirs = len(model_dirs)
n_dirs_parseable = int((audit["parseable_json_count"] > 0).sum()) if len(audit) else 0
n_results = int(audit["results_mapping_found"].sum()) if len(audit) else 0
n_broken = len(broken)
n_long = len(long_df)
n_primary = len(primary)

if len(model_family):
    n_six = int((model_family["n_complete_families"] >= 6).sum())
else:
    n_six = 0

report = f"""C8 DETAILED-RESULTS EXTRACTION REPORT
=====================================

Detailed-results root:
{DETAILS}

Model directories found:             {n_dirs:,}
Directories with parseable JSON:     {n_dirs_parseable:,}
Directories with results mapping:    {n_results:,}
Broken/truncated JSON files skipped: {n_broken:,}

Raw numeric task-metric rows:         {n_long:,}
Primary task-score rows:              {n_primary:,}
Models with all 6 benchmark families: {n_six:,}

Processing rule:
1. Scan every immediate model directory under detailed_results/.
2. Parse every JSON recursively.
3. If a JSON is broken/truncated, record it and continue.
4. For each model, choose the latest parseable JSON.
5. Preserve every numeric non-stderr task metric in long format.
6. Build one transparent primary score per task.
7. Aggregate child tasks within each benchmark family when available;
   otherwise use the broad aggregate task as a fallback.

Expected reference from the competition data description:
- 1,863 model directories
- 1,958 JSON files
- 4 truncated JSON files
- 1,860 directories with at least one parseable JSON
- about 1,854 models with complete six-dimensional benchmark coverage

Do not worry if your exact local counts differ by a few records; upload this
report and the audit CSV so the discrepancy can be diagnosed explicitly.
"""

(ROOT / "C8_extraction_report.txt").write_text(report, encoding="utf-8")

print("\n" + report)
print("\nSaved outputs in:", ROOT)
