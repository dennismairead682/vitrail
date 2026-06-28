from __future__ import annotations

import torch

from vitrail.soldering.consistency import consistency_loss
from vitrail.soldering.reconstruction import block_mask, masked_recon_loss
from vitrail.soldering.segmentation import soft_dice_loss


def test_consistency_target_is_detached():
    paired = torch.randn(1, 4, 2, 2, 2, requires_grad=True)
    target = torch.randn(1, 4, 2, 2, 2, requires_grad=True)
    loss = consistency_loss(paired, target)
    loss.backward()
    assert paired.grad is not None
    assert target.grad is None


def test_block_mask_keeps_block_structure():
    gen = torch.Generator()
    gen.manual_seed(0)
    mask = block_mask((16, 16, 16), block=8, ratio=0.5, generator=gen)
    assert mask.shape == (16, 16, 16)
    corner = mask[:8, :8, :8]
    assert corner.all() or (~corner).all()


def test_masked_recon_scores_only_masked():
    pred = torch.zeros(1, 1, 4, 4, 4)
    target = torch.ones(1, 1, 4, 4, 4)
    mask = torch.zeros(4, 4, 4, dtype=torch.bool)
    mask[0, 0, 0] = True
    loss = masked_recon_loss(pred, target, mask.view(1, 1, 4, 4, 4))
    assert abs(loss.item() - 1.0) < 1e-9


def test_soft_dice_perfect_is_zero():
    target = torch.zeros(1, 4, 4, 4, dtype=torch.long)
    target[0, :2] = 1
    logits = torch.zeros(1, 2, 4, 4, 4)
    logits[0, 1, :2] = 20.0
    logits[0, 0, 2:] = 20.0
    assert soft_dice_loss(logits, target).item() < 1e-2
