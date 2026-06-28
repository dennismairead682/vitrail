from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch


def stamp(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    seed: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "seed": seed,
    }
    if extra:
        payload.update(extra)
    return payload


def write_pane(payload: dict[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    scratch = target.with_suffix(target.suffix + ".inflight")
    torch.save(payload, scratch)
    os.replace(scratch, target)


def read_pane(path: str | Path, map_location: str = "cpu") -> dict[str, Any]:
    return torch.load(Path(path), map_location=map_location, weights_only=False)
