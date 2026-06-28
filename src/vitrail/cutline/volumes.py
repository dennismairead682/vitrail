from __future__ import annotations

import torch
import torch.nn.functional as F


def window_normalize(volume: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    clamped = volume.clamp(lo, hi)
    return (clamped - lo) / (hi - lo) * 2.0 - 1.0


def random_patch(
    volume: torch.Tensor,
    patch: tuple[int, int, int],
    generator: torch.Generator,
) -> tuple[int, int, int]:
    spans = []
    for size, want in zip(volume.shape[-3:], patch):
        if size <= want:
            spans.append(0)
        else:
            spans.append(int(torch.randint(0, size - want + 1, (1,), generator=generator).item()))
    return spans[0], spans[1], spans[2]


def crop(
    tensor: torch.Tensor, origin: tuple[int, int, int], patch: tuple[int, int, int]
) -> torch.Tensor:
    z, y, x = origin
    dz, dy, dx = patch
    return tensor[..., z : z + dz, y : y + dy, x : x + dx]


def downsample_labels(fdi_map: torch.Tensor, size: tuple[int, ...]) -> torch.Tensor:
    grid = fdi_map.float().view(1, 1, *fdi_map.shape[-3:])
    pooled = F.interpolate(grid, size=size, mode="nearest")
    return pooled.view(*size).round().long()


def region_pool(
    feature: torch.Tensor, region_map: torch.Tensor, slots: int
) -> tuple[torch.Tensor, torch.Tensor]:
    b, c = feature.shape[:2]
    flat = feature.reshape(b, c, -1)
    labels = region_map.reshape(b, -1)
    out = feature.new_zeros(b, slots, c)
    present = feature.new_zeros(b, slots)
    for r in range(1, slots + 1):
        mask = (labels == r).unsqueeze(1).float()
        count = mask.sum(dim=2)
        pooled = (flat * mask).sum(dim=2) / count.clamp_min(1.0)
        out[:, r - 1] = pooled
        present[:, r - 1] = (count.squeeze(1) > 0).float()
    return out, present
