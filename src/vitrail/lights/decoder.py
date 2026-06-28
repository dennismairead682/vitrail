from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from vitrail.cartoon.volumetric import ResBlock


class UpStep(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, 2, stride=2)
        self.fuse = ResBlock(out_ch + skip_ch, out_ch, 1)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        up = self.up(x)
        if up.shape[-3:] != skip.shape[-3:]:
            up = F.interpolate(up, size=skip.shape[-3:], mode="trilinear", align_corners=False)
        return self.fuse(torch.cat([up, skip], dim=1))


class VoxelDecoder(nn.Module):
    def __init__(self, stage_widths: tuple[int, ...], out_ch: int) -> None:
        super().__init__()
        widths = list(stage_widths)
        steps: list[UpStep] = []
        cur = widths[-1]
        for skip_w in reversed(widths[:-1]):
            steps.append(UpStep(cur, skip_w, skip_w))
            cur = skip_w
        self.steps = nn.ModuleList(steps)
        self.classifier = nn.Conv3d(cur, out_ch, 1)

    def forward(self, bottleneck: torch.Tensor, skips: list[torch.Tensor]) -> torch.Tensor:
        h = bottleneck
        encoder_skips = skips[:-1]
        for step, skip in zip(self.steps, reversed(encoder_skips)):
            h = step(h, skip)
        return self.classifier(h)
