from __future__ import annotations

import math

import torch

from vitrail.atelier.schema import GlassSpec
from vitrail.cutline.meshes import tokenize_surface
from vitrail.cutline.types import PairedSample


def _signature(region: int, dim: int, generator: torch.Generator) -> torch.Tensor:
    local = torch.Generator()
    local.manual_seed(int(generator.initial_seed()) * 131 + region * 977)
    return torch.randn(dim, generator=local)


def _grid(size: tuple[int, int, int]) -> torch.Tensor:
    axes = [torch.linspace(-1.0, 1.0, n) for n in size]
    zz, yy, xx = torch.meshgrid(axes[0], axes[1], axes[2], indexing="ij")
    return torch.stack([zz, yy, xx], dim=0)


def synth_paired(
    spec: GlassSpec,
    size: tuple[int, int, int],
    n_teeth: int,
    seed: int,
    paired: bool = True,
    with_implant: bool = False,
) -> PairedSample:
    generator = torch.Generator()
    generator.manual_seed(seed)
    grid = _grid(size)
    volume = torch.randn(1, *size, generator=generator) * 0.05
    seg = torch.zeros(size, dtype=torch.long)
    fdi_map = torch.zeros(size, dtype=torch.long)
    implant = torch.zeros(size, dtype=torch.long)

    slots = spec.region_slots
    chosen = torch.randperm(slots, generator=generator)[:n_teeth] + 1
    sig_dim = 8

    verts: list[torch.Tensor] = []
    vfeat: list[torch.Tensor] = []
    vfdi: list[torch.Tensor] = []

    for region in chosen.tolist():
        sig = _signature(region, sig_dim, generator)
        centre = torch.tensor(
            [
                math.cos(region) * 0.6,
                math.sin(region * 1.7) * 0.6,
                math.sin(region) * 0.6,
            ]
        )
        radius = 0.18 + 0.05 * torch.sigmoid(sig[0]).item()
        dist = ((grid - centre.view(3, 1, 1, 1)) ** 2).sum(dim=0).sqrt()
        blob = torch.exp(-(dist**2) / (2 * radius**2))
        orient = (grid * sig[1:4].view(3, 1, 1, 1)).sum(dim=0)
        volume[0] += blob * (1.0 + 0.3 * torch.tanh(orient))
        mask = blob > 0.4
        seg[mask] = region
        fdi_map[mask] = region

        surf = _surface_points(
            centre, radius, sig, spec.points_per_region, spec.surface_feats, generator
        )
        verts.append(surf[0])
        vfeat.append(surf[1])
        vfdi.append(torch.full((surf[0].shape[0],), region, dtype=torch.long))

    if with_implant:
        centre = torch.tensor([0.0, 0.0, 0.0])
        dist = ((grid[:2] - centre[:2].view(2, 1, 1, 1)) ** 2).sum(dim=0).sqrt()
        cyl = (dist < 0.12) & (grid[0].abs() < 0.5)
        implant[cyl] = 1
        volume[0][cyl] += 0.8

    vertices = torch.cat(verts) if verts else torch.zeros(1, 3)
    feats = torch.cat(vfeat) if vfeat else torch.zeros(1, spec.surface_feats)
    fdi = torch.cat(vfdi) if vfdi else torch.zeros(1, dtype=torch.long)
    bundle = tokenize_surface(
        vertices, feats, fdi, spec.region_slots, spec.points_per_region, generator
    )

    return PairedSample(
        volume=volume,
        seg=seg,
        implant=implant,
        fdi_map=fdi_map,
        bundle=bundle,
        paired=torch.tensor(1.0 if paired else 0.0),
    )


def _surface_points(
    centre: torch.Tensor,
    radius: float,
    sig: torch.Tensor,
    count: int,
    feat_dim: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    directions = torch.randn(count, 3, generator=generator)
    directions = directions / directions.norm(dim=1, keepdim=True).clamp_min(1e-6)
    points = centre.view(1, 3) + directions * radius
    curvature = (directions * sig[1:4].view(1, 3)).sum(dim=1, keepdim=True)
    feats = torch.cat([directions, curvature], dim=1)
    if feat_dim > feats.shape[1]:
        pad = sig[: feat_dim - feats.shape[1]].view(1, -1).expand(count, -1)
        feats = torch.cat([feats, pad], dim=1)
    return points, feats[:, :feat_dim]
