from __future__ import annotations

import torch

from vitrail.cutline.types import RegionBundle


def tokenize_surface(
    vertices: torch.Tensor,
    feats: torch.Tensor,
    vertex_fdi: torch.Tensor,
    slots: int,
    points_per_region: int,
    generator: torch.Generator,
) -> RegionBundle:
    feat_dim = feats.shape[1]
    points = vertices.new_zeros(slots, points_per_region, 3)
    bagged = vertices.new_zeros(slots, points_per_region, feat_dim)
    present = vertices.new_zeros(slots)
    vertex_region = torch.zeros_like(vertex_fdi)
    for r in range(1, slots + 1):
        idx = torch.nonzero(vertex_fdi == r, as_tuple=False).flatten()
        if idx.numel() == 0:
            continue
        vertex_region[idx] = r
        present[r - 1] = 1.0
        pick = _resample(idx, points_per_region, generator)
        points[r - 1] = vertices[pick]
        bagged[r - 1] = feats[pick]
    return RegionBundle(
        points=points,
        feats=bagged,
        present=present,
        vertex_fdi=vertex_fdi,
        vertex_region=vertex_region,
    )


def _resample(idx: torch.Tensor, count: int, generator: torch.Generator) -> torch.Tensor:
    n = idx.numel()
    if n >= count:
        order = torch.randperm(n, generator=generator)[:count]
        return idx[order]
    reps = torch.randint(0, n, (count - n,), generator=generator)
    return torch.cat([idx, idx[reps]], dim=0)


def centroid_normalize(vertices: torch.Tensor) -> torch.Tensor:
    centre = vertices.mean(dim=0, keepdim=True)
    shifted = vertices - centre
    scale = shifted.abs().amax().clamp_min(1e-6)
    return shifted / scale
