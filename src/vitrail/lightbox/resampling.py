from __future__ import annotations

from typing import Callable

import numpy as np
from scipy import stats


def bootstrap_ci(
    values: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = values.shape[0]
    draws = np.empty(resamples, dtype=np.float64)
    for i in range(resamples):
        idx = rng.integers(0, n, n)
        draws[i] = statistic(values[idx])
    point = statistic(values)
    lo = float(np.percentile(draws, 100 * alpha / 2))
    hi = float(np.percentile(draws, 100 * (1 - alpha / 2)))
    return point, lo, hi


def _midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    sorted_x = x[order]
    n = x.shape[0]
    ranks = np.zeros(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n, dtype=np.float64)
    out[order] = ranks
    return out


def delong_auc_variance(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    m, n = pos.shape[0], neg.shape[0]
    tx = _midrank(pos)
    ty = _midrank(neg)
    tz = _midrank(np.concatenate([pos, neg]))
    auc = (tz[:m].sum() - m * (m + 1) / 2) / (m * n)
    v01 = (tz[:m] - tx) / n
    v10 = 1.0 - (tz[m:] - ty) / m
    var = v01.var(ddof=1) / m + v10.var(ddof=1) / n
    return float(auc), float(var)


def delong_test(
    scores_a: np.ndarray, scores_b: np.ndarray, labels: np.ndarray
) -> tuple[float, float]:
    auc_a, var_a = delong_auc_variance(scores_a, labels)
    auc_b, var_b = delong_auc_variance(scores_b, labels)
    denom = np.sqrt(var_a + var_b)
    if denom == 0:
        return 0.0, 1.0
    z = (auc_a - auc_b) / denom
    p = 2.0 * stats.norm.sf(abs(z))
    return float(z), float(p)


def wilcoxon_signed_rank(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    diff = x - y
    diff = diff[diff != 0]
    n = diff.shape[0]
    if n == 0:
        return 0.0, 1.0
    ranks = _midrank(np.abs(diff))
    w_plus = ranks[diff > 0].sum()
    w_minus = ranks[diff < 0].sum()
    stat = min(w_plus, w_minus)
    mean = n * (n + 1) / 4.0
    _, counts = np.unique(np.abs(diff), return_counts=True)
    tie = (counts**3 - counts).sum()
    var = n * (n + 1) * (2 * n + 1) / 24.0 - tie / 48.0
    if var <= 0:
        return float(stat), 1.0
    z = (stat - mean + 0.5) / np.sqrt(var)
    p = 2.0 * stats.norm.cdf(z)
    return float(stat), float(min(p, 1.0))


def holm_bonferroni(pvalues: np.ndarray) -> np.ndarray:
    order = np.argsort(pvalues)
    m = pvalues.shape[0]
    adjusted = np.empty(m, dtype=np.float64)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * pvalues[idx]
        running = max(running, val)
        adjusted[idx] = min(running, 1.0)
    return adjusted
