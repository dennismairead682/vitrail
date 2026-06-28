from __future__ import annotations

import torch
import torch.nn.functional as F


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    classes = logits.shape[1]
    probs = F.softmax(logits, dim=1)
    onehot = F.one_hot(target.clamp(0, classes - 1), classes)
    axes = tuple(range(1, onehot.ndim - 1))
    onehot = onehot.permute(0, onehot.ndim - 1, *[a for a in axes]).float()
    dims = tuple(range(2, probs.ndim))
    inter = (probs * onehot).sum(dim=dims)
    denom = probs.sum(dim=dims) + onehot.sum(dim=dims)
    dice = (2.0 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()


def voxel_objective(logits: torch.Tensor, target: torch.Tensor, dice_weight: float) -> torch.Tensor:
    ce = F.cross_entropy(logits, target)
    dice = soft_dice_loss(logits, target)
    return (1.0 - dice_weight) * ce + dice_weight * dice


def vertex_objective(
    logits: torch.Tensor, target: torch.Tensor, dice_weight: float
) -> torch.Tensor:
    ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1))
    probs = F.softmax(logits, dim=-1)
    onehot = F.one_hot(target.clamp(0, logits.shape[-1] - 1), logits.shape[-1]).float()
    inter = (probs * onehot).sum(dim=(0, 1))
    denom = probs.sum(dim=(0, 1)) + onehot.sum(dim=(0, 1))
    dice = 1.0 - ((2.0 * inter + 1e-5) / (denom + 1e-5)).mean()
    return (1.0 - dice_weight) * ce + dice_weight * dice
