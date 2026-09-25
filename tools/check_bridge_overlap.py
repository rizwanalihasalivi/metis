#!/usr/bin/env python3
"""
Diagnostic: check overlap between Q3 optimal losses and C6 loss range.
Determines whether a local/subset bridge can legitimately improve fit.
"""
import json

import numpy as np
import pandas as pd

from config import C6_BRIDGE, Q3_POINT_SOLUTIONS


def family_from_name(name):
    s = str(name).split("/")[-1].lower()
    tok = s.split("-")[0]
    for pref in ["pythia", "llama", "qwen", "gemma", "mistral", "phi",
                 "olmo", "yi", "falcon", "gpt", "deepseek"]:
        if tok.startswith(pref):
            return pref
    return tok


def main():
    print("=" * 72)
    print("DIAGNOSTIC: Q3 loss range vs C6 loss range")
    print("=" * 72)

    # Q3 losses
    with open(Q3_POINT_SOLUTIONS, encoding="utf-8") as f:
        q3 = json.load(f)
    q3_losses = np.array([s["loss"] for s in q3["main_exponential"]])
    print(f"\nQ3 losses (main): {q3_losses}")
    print(f"Q3 range: [{q3_losses.min():.4f}, {q3_losses.max():.4f}]")

    # C6
    c6 = pd.read_csv(C6_BRIDGE)
    c6 = c6.dropna(subset=["Val_Loss", "LB_Average"]).copy()
    c6["family"] = c6["Model"].apply(family_from_name)
    hm = c6["Loss_Comparability"].astype(str).str.contains(
        "High", case=False, na=False)
    c6["weight"] = np.where(hm, 5.0, 1.0)

    print(f"\nC6 total rows: {len(c6)}")
    print(f"C6 loss range: [{c6['Val_Loss'].min():.4f}, "
          f"{c6['Val_Loss'].max():.4f}]")
    print(f"C6 bench range: [{c6['LB_Average'].min():.4f}, "
          f"{c6['LB_Average'].max():.4f}]")

    # Distribution
    print(f"\n--- C6 loss percentiles ---")
    for p in [0, 10, 25, 50, 75, 90, 100]:
        v = np.percentile(c6["Val_Loss"], p)
        print(f"  p{p:3d}: {v:.4f}")

    # High-comparability subset
    high = c6[hm].copy()
    print(f"\n--- High-comparability rows (n={len(high)}) ---")
    for _, r in high.iterrows():
        print(f"  {r['Model'][:40]:40s}  "
              f"loss={r['Val_Loss']:.4f}  bench={r['LB_Average']:.2f}  "
              f"family={r['family']}")

    if len(high) > 0:
        print(f"\nHigh-comparability loss range: "
              f"[{high['Val_Loss'].min():.4f}, "
              f"{high['Val_Loss'].max():.4f}]")

    # Overlap test
    q3_lo, q3_hi = q3_losses.min(), q3_losses.max()
    print(f"\n--- Overlap analysis ---")
    print(f"Q3 range: [{q3_lo:.4f}, {q3_hi:.4f}]")

    overlap_all = c6[(c6["Val_Loss"] >= q3_lo) & (c6["Val_Loss"] <= q3_hi)]
    print(f"All C6 rows within Q3 range: {len(overlap_all)} / {len(c6)}")

    overlap_high = high[(high["Val_Loss"] >= q3_lo) &
                        (high["Val_Loss"] <= q3_hi)]
    print(f"High-comparability within Q3 range: {len(overlap_high)} / {len(high)}")

    # Extend window
    pad = 0.3
    ext = c6[(c6["Val_Loss"] >= q3_lo - pad) & (c6["Val_Loss"] <= q3_hi + pad)]
    print(f"C6 within Q3 range ±{pad}: {len(ext)} / {len(c6)}")

    ext_high = high[(high["Val_Loss"] >= q3_lo - pad) &
                    (high["Val_Loss"] <= q3_hi + pad)]
    print(f"High-comparability within ±{pad}: {len(ext_high)} / {len(high)}")

    # Verdict
    print(f"\n--- VERDICT ---")
    if len(overlap_high) >= 3:
        print(f"✅ Substantial high-quality overlap ({len(overlap_high)} rows). "
              f"Local bridge retrain is meaningful.")
    elif len(ext_high) >= 3:
        print(f"⚠️  Overlap only with ±{pad} padding "
              f"({len(ext_high)} high rows). Local bridge possible "
              f"but with caveat.")
    else:
        print(f"❌ No substantial overlap. Local bridge NOT justified. "
              f"Keep global bridge only.")

    # Save
    out = pd.DataFrame({
        "q3_loss": q3_losses,
    })
    print(f"\nDone.")


if __name__ == "__main__":
    main()