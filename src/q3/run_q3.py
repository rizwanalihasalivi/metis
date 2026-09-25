#!/usr/bin/env python3
"""Run entire Q3 pipeline."""
import subprocess
import sys

STEPS = [
    "src.q3.01_cost_model",
    "src.q3.02_optimizer",
    "src.q3.03_uncertainty",
    "src.q3.04_context_sensitivity",
]


def main():
    print("=" * 72)
    print("RUNNING Q3 PIPELINE")
    print("=" * 72)
    for step in STEPS:
        print(f"\n>>> {step}")
        r = subprocess.run([sys.executable, "-m", step])
        if r.returncode != 0:
            print(f"\n[FAILED] {step}")
            sys.exit(1)
    print("\n[OK] Q3 pipeline complete")


if __name__ == "__main__":
    main()