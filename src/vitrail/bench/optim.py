from __future__ import annotations

import math
from collections.abc import Iterable

import torch
from torch.optim.lr_scheduler import LambdaLR

from vitrail.atelier.schema import RegimenSpec


def build_optimizer(
    params: Iterable[torch.nn.Parameter], regimen: RegimenSpec
) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        params,
        lr=regimen.lr,
        betas=regimen.betas,
        weight_decay=regimen.weight_decay,
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer, regimen: RegimenSpec, total_steps: int, warmup_steps: int
) -> LambdaLR:
    def curve(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(warmup_steps, 1)
        if regimen.scheduler == "constant":
            return 1.0
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return LambdaLR(optimizer, lr_lambda=curve)
