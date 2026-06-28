from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

from vitrail.lightbox.resampling import (
    bootstrap_ci,
    delong_auc_variance,
    holm_bonferroni,
    wilcoxon_signed_rank,
)


def test_delong_auc_matches_sklearn():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, 200)
    scores = rng.random(200) + 0.4 * labels
    auc, var = delong_auc_variance(scores, labels)
    assert abs(auc - roc_auc_score(labels, scores)) < 1e-9
    assert var > 0


def test_wilcoxon_matches_scipy_approx():
    rng = np.random.default_rng(1)
    x = rng.normal(size=40)
    y = x + rng.normal(scale=0.5, size=40)
    stat, p = wilcoxon_signed_rank(x, y)
    ref = stats.wilcoxon(x, y, correction=True, method="approx")
    assert abs(stat - ref.statistic) < 1e-6
    assert abs(p - ref.pvalue) < 1e-3


def test_holm_is_monotone_and_bounded():
    p = np.array([0.01, 0.02, 0.03, 0.04])
    adj = holm_bonferroni(p)
    assert np.all(adj <= 1.0)
    order = np.argsort(p)
    assert np.all(np.diff(adj[order]) >= -1e-12)
    assert abs(adj[order][0] - 4 * 0.01) < 1e-12


def test_bootstrap_brackets_mean():
    rng = np.random.default_rng(2)
    values = rng.normal(loc=3.0, size=500)
    point, lo, hi = bootstrap_ci(values, lambda v: float(v.mean()), resamples=400, seed=5)
    assert lo < point < hi
    assert lo < 3.0 < hi
