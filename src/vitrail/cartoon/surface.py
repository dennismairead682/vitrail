from __future__ import annotations

import torch
from torch import nn

from vitrail.atelier.schema import CartoonSpec, GlassSpec
from vitrail.cutline.types import RegionBundle


class _SetEncoder(nn.Module):
    def __init__(self, in_dim: int, width: int, depth: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for _ in range(depth):
            layers.append(nn.Conv1d(prev, width, 1))
            layers.append(nn.GroupNorm(8 if width % 8 == 0 else 1, width))
            layers.append(nn.LeakyReLU(0.01, inplace=True))
            prev = width
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SurfaceEncoder(nn.Module):
    def __init__(self, cartoon: CartoonSpec, glass: GlassSpec) -> None:
        super().__init__()
        self.point_in = 3 + glass.surface_feats
        self.width = cartoon.surf_width
        self.local = _SetEncoder(self.point_in, cartoon.surf_width, cartoon.surf_depth)
        self.global_mlp = nn.Sequential(
            nn.Linear(cartoon.surf_width, cartoon.surf_width),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Linear(cartoon.surf_width, cartoon.surf_width),
        )

    def forward(self, bundle: RegionBundle) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        points = bundle["points"]
        feats = bundle["feats"]
        present = bundle["present"]
        b, r, p, _ = points.shape
        stacked = torch.cat([points, feats], dim=-1).reshape(b * r, p, self.point_in)
        encoded = self.local(stacked.transpose(1, 2))
        regional = encoded.amax(dim=2).reshape(b, r, self.width) * present.unsqueeze(-1)
        denom = present.sum(dim=1, keepdim=True).clamp_min(1.0)
        glob = self.global_mlp(regional.sum(dim=1) / denom)
        per_point = encoded.reshape(b, r, self.width, p).permute(0, 1, 3, 2)
        return regional, glob, per_point
