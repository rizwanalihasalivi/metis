"""Shared metric helpers: rank, Spearman, ECDF, parsing, KS."""
from itertools import combinations
import numpy as np
import pandas as pd


def rankdata(x):
    """Average-rank of x (scipy-free replacement)."""
    return pd.Series(np.asarray(x, dtype=float)).rank(method="average").to_numpy()


def spearman(a, b):
    """Spearman correlation, NaN-safe."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    return float(np.corrcoef(rankdata(a[ok]), rankdata(b[ok]))[0, 1])


def ecdf_rank(values, reference_sorted):
    """Rank values against a pre-sorted reference ECDF."""
    values = np.asarray(values, dtype=float)
    return np.searchsorted(reference_sorted, values, side="right") / len(reference_sorted)


def ks_statistic(a, b):
    """Two-sample Kolmogorov-Smirnov statistic."""
    a = np.sort(np.asarray(a, dtype=float))
    b = np.sort(np.asarray(b, dtype=float))
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    x = np.sort(np.unique(np.concatenate([a, b])))
    cdf_a = np.searchsorted(a, x, side="right") / len(a)
    cdf_b = np.searchsorted(b, x, side="right") / len(b)
    return float(np.max(np.abs(cdf_a - cdf_b)))


def parse_list(x):
    """Parse list-like field from JSONL / CSV into list of floats."""
    if x is None:
        return []
    if isinstance(x, (list, tuple, np.ndarray)):
        try:
            return [float(v) for v in x]
        except Exception:
            return []
    try:
        if pd.isna(x):
            return []
    except Exception:
        pass
    s = str(x).strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    if not s:
        return []
    try:
        return [float(v) for v in s.split(",") if v.strip()]
    except Exception:
        return []


def pairwise_mean_abs_diff(vals):
    """For each row, mean of |v_i - v_j| over all pairs of columns."""
    vals = np.asarray(vals, dtype=float)
    g = vals.shape[1]
    diffs = [np.abs(vals[:, i] - vals[:, j]) for i, j in combinations(range(g), 2)]
    return np.vstack(diffs).mean(axis=0)