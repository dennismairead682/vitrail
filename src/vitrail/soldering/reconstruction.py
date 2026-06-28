from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class PatchPredictor(nn.Module):
    def __init__(self, in_ch: int, scale: int) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        ch = in_ch
        steps = int(round(_log2(scale)))
        for _ in range(steps):
            nxt = max(ch // 2, 16)
            blocks.append(nn.ConvTranspose3d(ch, nxt, 2, stride=2))
            blocks.append(nn.GroupNorm(8 if nxt % 8 == 0 else 1, nxt))
            blocks.append(nn.LeakyReLU(0.01, inplace=True))
            ch = nxt
        blocks.append(nn.Conv3d(ch, 1, 1))
        self.net = nn.Sequential(*blocks)

    def forward(self, dense: torch.Tensor) -> torch.Tensor:
        return self.net(dense)


def _log2(value: int) -> float:
    return float(int(value).bit_length() - 1)


def block_mask(
    shape: tuple[int, ...],
    block: int,
    ratio: float,
    generator: torch.Generator,
) -> torch.Tensor:
    dims = [s // block for s in shape]
    grid = torch.rand(dims, generator=generator) < ratio
    mask = grid.float()
    for axis, factor in enumerate((block, block, block)):
        mask = mask.repeat_interleave(factor, dim=axis)
    return mask[: shape[0], : shape[1], : shape[2]] > 0.5


def masked_recon_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    sel = mask.expand_as(pred)
    if sel.sum() == 0:
        return pred.sum() * 0.0
    return F.mse_loss(pred[sel], target[sel])
