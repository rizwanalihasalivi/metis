# Problem F: Resource Allocation for LLM Capability

## Overview

Complete solution to Problem F (2026 China Graduate Mathematical Modeling Contest).

Four connected sub-problems:

- **Q1**: Anchor-aligned quality proxy `Q_i` from 22 indicators, 7-family
  conflict, 2-component GMM, AND 17-domain mixture-to-loss model (A4-A15).
- **Q2**: Classical N-D scaling law (B1), 3 quality scenarios (B6/B7/B8),
  scale attenuation omega(N), cross-family validation (B2/B3/B4/B5/B9/B10).
- **Q3**: Conditional (N, D, Q) optimizer, budget regimes, bootstrap,
  context sensitivity.
- **Q4**: C8 capability validation, historical decomposition, C3 timeseries,
  Loss-Benchmark bridge, frontier forecast.

## Quick Start

```bash
pip install -r requirements.txt
python verify_outputs.py
python run_all.py
python -m src.q1.run_q1
python -m src.q2.run_q2
python -m src.q3.run_q3
python -m src.q4.run_q4
