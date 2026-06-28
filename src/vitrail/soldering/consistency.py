from __future__ import annotations

import torch
import torch.nn.functional as F


def consistency_loss(dense_paired: torch.Tensor, dense_cbct_only: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(dense_paired, dense_cbct_only.detach())
