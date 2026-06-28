from __future__ import annotations

import numpy as np

from vitrail.lightbox.calibration import expected_calibration_error
from vitrail.lightbox.overlap import assd, chamfer_l1, mean_dice, mean_iou


def test_dice_half_overlap():
    pred = np.zeros((4, 4), dtype=np.int64)
    target = np.zeros((4, 4), dtype=np.int64)
    pred[:2] = 1
    target[1:3] = 1
    score = mean_dice(pred, target, num_classes=2)
    fore = 2 * 2 / (4 + 4)
    back = 2 * 2 / (4 + 4)
    assert abs(score - 0.5 * (fore + back)) < 1e-9


def test_iou_perfect():
    labels = np.array([[0, 1], [2, 3]])
    assert abs(mean_iou(labels, labels, num_classes=4) - 1.0) < 1e-9


def test_assd_zero_for_identical():
    cube = np.zeros((8, 8, 8), dtype=bool)
    cube[2:6, 2:6, 2:6] = True
    assert assd(cube, cube, spacing=0.3) == 0.0


def test_chamfer_translation():
    a = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    b = a + np.array([0.5, 0.0, 0.0])
    assert abs(chamfer_l1(a, b) - 0.5) < 1e-9


def test_ece_perfectly_calibrated():
    probs = np.array([[0.9, 0.1], [0.2, 0.8]])
    labels = np.array([0, 1])
    err = expected_calibration_error(probs, labels, bins=5)
    assert 0.0 <= err <= 0.2
