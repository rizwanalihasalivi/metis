#!/usr/bin/env python3
"""Top-level orchestrator for all 4 questions + verification."""
import subprocess
import sys

from config import ensure_dirs

STEPS = [
    ("Q1", "src.q1.run_q1"),
    ("Q2", "src.q2.run_q2"),
    ("Q3", "src.q3.run_q3"),
    ("Q4", "src.q4.run_q4"),
]

VERIFY = "verify_outputs"


def main():
    ensure_dirs()
    print("=" * 72)
    print("PROBLEM F — FULL PIPELINE (Q1 → Q4)")
    print("=" * 72)

    for name, module in STEPS:
        print(f"\n{'='*72}\n>>> {name} — {module}\n{'='*72}")
        r = subprocess.run([sys.executable, "-m", module])
        if r.returncode != 0:
            print(f"\n[FAILED] {name}")
            sys.exit(1)

    # Reviewer fix: run verification as final step
    print(f"\n{'='*72}\n>>> VERIFICATION — {VERIFY}\n{'='*72}")
    r = subprocess.run([sys.executable, f"{VERIFY}.py"])
    if r.returncode != 0:
        print(f"\n[FAILED] Verification did not pass")
        sys.exit(1)

    print("\n" + "=" * 72)
    print("[OK] Full pipeline complete — all 4 questions + verification")
    print("=" * 72)


if __name__ == "__main__":
    main()