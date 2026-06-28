from __future__ import annotations

import numpy as np
from scipy import ndimage


def dice_per_class(pred: np.ndarray, target: np.ndarray, num_classes: int) -> np.ndarray:
    out = np.zeros(num_classes, dtype=np.float64)
    for c in range(num_classes):
        p = pred == c
        t = target == c
        denom = p.sum() + t.sum()
        if denom == 0:
            out[c] = np.nan
            continue
        out[c] = 2.0 * np.logical_and(p, t).sum() / denom
    return out


def mean_dice(pred: np.ndarray, target: np.ndarray, num_classes: int) -> float:
    vals = dice_per_class(pred, target, num_classes)
    return float(np.nanmean(vals))


def iou_per_class(pred: np.ndarray, target: np.ndarray, num_classes: int) -> np.ndarray:
    out = np.zeros(num_classes, dtype=np.float64)
    for c in range(num_classes):
        p = pred == c
        t = target == c
        union = np.logical_or(p, t).sum()
        if union == 0:
            out[c] = np.nan
            continue
        out[c] = np.logical_and(p, t).sum() / union
    return out


def mean_iou(pred: np.ndarray, target: np.ndarray, num_classes: int) -> float:
    return float(np.nanmean(iou_per_class(pred, target, num_classes)))


def _surface_distances(a: np.ndarray, b: np.ndarray, spacing: float) -> np.ndarray:
    if a.sum() == 0 or b.sum() == 0:
        return np.array([np.nan])
    border_a = a ^ ndimage.binary_erosion(a)
    dt_b = ndimage.distance_transform_edt(~b) * spacing
    return dt_b[border_a]


def assd(pred: np.ndarray, target: np.ndarray, spacing: float = 0.30) -> float:
    d_ab = _surface_distances(pred, target, spacing)
    d_ba = _surface_distances(target, pred, spacing)
    joined = np.concatenate([d_ab, d_ba])
    return float(np.nanmean(joined))


def hd95(pred: np.ndarray, target: np.ndarray, spacing: float = 0.30) -> float:
    d_ab = _surface_distances(pred, target, spacing)
    d_ba = _surface_distances(target, pred, spacing)
    joined = np.concatenate([d_ab, d_ba])
    finite = joined[np.isfinite(joined)]
    if finite.size == 0:
        return float("nan")
    return float(np.percentile(finite, 95))


def chamfer_l1(a: np.ndarray, b: np.ndarray) -> float:
    diff_ab = np.abs(a[:, None, :] - b[None, :, :]).sum(axis=2)
    forward = diff_ab.min(axis=1).mean()
    backward = diff_ab.min(axis=0).mean()
    return float(0.5 * (forward + backward))
