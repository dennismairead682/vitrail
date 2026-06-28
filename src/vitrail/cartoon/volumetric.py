from __future__ import annotations

import torch
from torch import nn

from vitrail.atelier.schema import CartoonSpec


def _norm(channels: int) -> nn.Module:
    groups = 8 if channels % 8 == 0 else 1
    return nn.GroupNorm(groups, channels)


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.norm1 = _norm(out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False)
        self.norm2 = _norm(out_ch)
        self.act = nn.LeakyReLU(0.01, inplace=True)
        if stride != 1 or in_ch != out_ch:
            self.skip: nn.Module = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 1, stride=stride, bias=False), _norm(out_ch)
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        h = self.act(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return self.act(h + residual)


class VolumetricEncoder(nn.Module):
    def __init__(self, spec: CartoonSpec) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(1, spec.vol_stem, 3, padding=1, bias=False),
            _norm(spec.vol_stem),
            nn.LeakyReLU(0.01, inplace=True),
        )
        stages: list[nn.Module] = []
        prev = spec.vol_stem
        for i, width in enumerate(spec.vol_stages):
            stride = 1 if i == 0 else 2
            block: list[nn.Module] = [ResBlock(prev, width, stride)]
            for _ in range(spec.vol_blocks - 1):
                block.append(ResBlock(width, width, 1))
            stages.append(nn.Sequential(*block))
            prev = width
        self.stages = nn.ModuleList(stages)
        self.out_channels = spec.vol_stages[-1]

    def forward(self, volume: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        h = self.stem(volume)
        skips: list[torch.Tensor] = []
        for stage in self.stages:
            h = stage(h)
            skips.append(h)
        return h, skips
