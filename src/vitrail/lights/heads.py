from __future__ import annotations

from typing import TypedDict

import torch
import torch.nn.functional as F
from torch import nn

from vitrail.atelier.schema import CartoonSpec, GlassSpec
from vitrail.cutline.types import RegionBundle
from vitrail.lights.decoder import VoxelDecoder


class FusionOutput(TypedDict):
    recon: torch.Tensor
    implant: torch.Tensor


class CbctLight(nn.Module):
    def __init__(self, cartoon: CartoonSpec, glass: GlassSpec) -> None:
        super().__init__()
        self.decoder = VoxelDecoder(cartoon.vol_stages, glass.cbct_classes)

    def forward(
        self, dense: torch.Tensor, skips: list[torch.Tensor], full: tuple[int, int, int]
    ) -> torch.Tensor:
        logits = self.decoder(dense, skips)
        if logits.shape[-3:] != full:
            logits = F.interpolate(logits, size=full, mode="trilinear", align_corners=False)
        return logits


class IosLight(nn.Module):
    def __init__(self, cartoon: CartoonSpec, glass: GlassSpec) -> None:
        super().__init__()
        point_in = 3 + glass.surface_feats
        self.head = nn.Sequential(
            nn.Linear(point_in + 2 * cartoon.surf_width, cartoon.surf_width),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Linear(cartoon.surf_width, glass.fdi_classes),
        )
        self.width = cartoon.surf_width

    def forward(
        self,
        bundle: RegionBundle,
        glob: torch.Tensor,
        per_point: torch.Tensor,
    ) -> torch.Tensor:
        points = bundle["points"]
        feats = bundle["feats"]
        b, r, p, _ = points.shape
        glob_exp = glob.view(b, 1, 1, self.width).expand(b, r, p, self.width)
        node_in = torch.cat([points, feats, per_point, glob_exp], dim=-1)
        return self.head(node_in)


class FusionLight(nn.Module):
    def __init__(self, cartoon: CartoonSpec, glass: GlassSpec) -> None:
        super().__init__()
        self.recon = VoxelDecoder(cartoon.vol_stages, glass.cbct_classes)
        self.implant = VoxelDecoder(cartoon.vol_stages, 2)

    def forward(
        self, fused: torch.Tensor, skips: list[torch.Tensor], full: tuple[int, int, int]
    ) -> FusionOutput:
        recon = self.recon(fused, skips)
        implant = self.implant(fused, skips)
        if recon.shape[-3:] != full:
            recon = F.interpolate(recon, size=full, mode="trilinear", align_corners=False)
            implant = F.interpolate(implant, size=full, mode="trilinear", align_corners=False)
        return FusionOutput(recon=recon, implant=implant)
