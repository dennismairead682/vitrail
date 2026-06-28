from __future__ import annotations

import numpy as np


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, bins: int = 15) -> float:
    confidence = probs.max(axis=1)
    prediction = probs.argmax(axis=1)
    correct = (prediction == labels).astype(np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = labels.shape[0]
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (confidence > lo) & (confidence <= hi)
        if not sel.any():
            continue
        weight = sel.sum() / total
        ece += weight * abs(correct[sel].mean() - confidence[sel].mean())
    return float(ece)
