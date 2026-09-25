#!/usr/bin/env python3
"""Run entire Q1 pipeline."""
import subprocess
import sys
from config import ensure_dirs

STEPS = [
    "src.q1.01_prepare_quality_data",
    "src.q1.02_build_quality_score",
    "src.q1.03_conflict_analysis",
    "src.q1.04_mixture_model",
    "src.q1.05_validate_q1",
    "src.q1.06_mixture_model",
]


def main():
    ensure_dirs()
    print("=" * 72)
    print("RUNNING Q1 PIPELINE")
    print("=" * 72)
    for step in STEPS:
        print(f"\n>>> {step}")
        r = subprocess.run([sys.executable, "-m", step])
        if r.returncode != 0:
            print(f"\n[FAILED] {step}")
            sys.exit(1)
    print("\n[OK] Q1 pipeline complete")


if __name__ == "__main__":
    main()