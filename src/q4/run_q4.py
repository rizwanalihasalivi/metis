#!/usr/bin/env python3
"""Run entire Q4 pipeline."""
import subprocess
import sys
from pathlib import Path
from config import ensure_dirs, Q4_C8_MODEL_AGG

STEPS = [
    "src.q4.02_capability_metric",
    "src.q4.05_frontier_forecast",
    "src.q4.03_historical_decomposition",
    "src.q4.04_loss_benchmark_bridge",
    "src.q4.06_c3_timeseries",
    "src.q4.07_q3_to_q4_bridge",
]


def main():
    ensure_dirs()
    print("=" * 72)
    print("RUNNING Q4 PIPELINE")
    print("=" * 72)

    if not Path(Q4_C8_MODEL_AGG).exists():
        r = subprocess.run([sys.executable, "-m", "src.q4.01_extract_c8"])
        if r.returncode != 0:
            print("[WARN] C8 extractor failed")
    else:
        print(f"[SKIP] C8 processed present: {Q4_C8_MODEL_AGG.name}")

    for step in STEPS:
        print(f"\n>>> {step}")
        r = subprocess.run([sys.executable, "-m", step])
        if r.returncode != 0:
            print(f"\n[FAILED] {step}")
            sys.exit(1)
    print("\n[OK] Q4 pipeline complete")


if __name__ == "__main__":
    main()