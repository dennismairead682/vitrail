from __future__ import annotations

import torch
import torch.nn.functional as F


def gather_valid(
    z_cbct: torch.Tensor, z_ios: torch.Tensor, present: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    flag = present.reshape(-1) > 0.5
    a = z_cbct.reshape(-1, z_cbct.shape[-1])[flag]
    b = z_ios.reshape(-1, z_ios.shape[-1])[flag]
    return a, b


def cross_modal_infonce(
    z_cbct: torch.Tensor,
    z_ios: torch.Tensor,
    present: torch.Tensor,
    temperature: torch.Tensor,
) -> torch.Tensor:
    anchor, positive = gather_valid(z_cbct, z_ios, present)
    if anchor.shape[0] < 2:
        return z_cbct.sum() * 0.0
    logits = anchor @ positive.t() / temperature
    target = torch.arange(anchor.shape[0], device=anchor.device)
    return F.cross_entropy(logits, target)
