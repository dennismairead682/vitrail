from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


def maybe_init() -> tuple[int, int]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        rank = dist.get_rank()
        world = dist.get_world_size()
        if torch.cuda.is_available():
            torch.cuda.set_device(rank % torch.cuda.device_count())
        return rank, world
    return 0, 1


def wrap(model: torch.nn.Module, rank: int) -> torch.nn.Module:
    if dist.is_available() and dist.is_initialized():
        device_ids = [rank] if torch.cuda.is_available() else None
        return DistributedDataParallel(model, device_ids=device_ids)
    return model


def shutdown() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


@contextlib.contextmanager
def autocast(precision: str, enabled: bool) -> Iterator[None]:
    if not enabled or not torch.cuda.is_available():
        yield
        return
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    with torch.autocast(device_type="cuda", dtype=dtype):
        yield
