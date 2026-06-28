from __future__ import annotations

import torch
from torch import nn


class ShadowWeights:
    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for key, value in model.state_dict().items():
            stored = self.shadow[key]
            if value.dtype.is_floating_point:
                stored.mul_(self.decay).add_(value.detach(), alpha=1.0 - self.decay)
            else:
                stored.copy_(value)

    def copy_to(self, model: nn.Module) -> None:
        model.load_state_dict(self.shadow, strict=True)
