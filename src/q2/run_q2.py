#!/usr/bin/env python3
"""Run entire Q2 pipeline."""
import subprocess
import sys

STEPS = [
    "src.q2.01_scaling_baseline",
    "src.q2.02_quality_scenarios",
    "src.q2.03_mixture_attenuation",
    "src.q2.04_validation",
    "src.q2.05_bridge_and_elasticities",
]


def main():
    print("=" * 72)
    print("RUNNING Q2 PIPELINE")
    print("=" * 72)
    for step in STEPS:
        print(f"\n>>> {step}")
        r = subprocess.run([sys.executable, "-m", step])
        if r.returncode != 0:
            print(f"\n[FAILED] {step}")
            sys.exit(1)
    print("\n[OK] Q2 pipeline complete")


if __name__ == "__main__":
    main()