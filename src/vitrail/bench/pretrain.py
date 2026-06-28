from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from vitrail.atelier.journal import get_logger
from vitrail.atelier.provenance import stamp, write_pane
from vitrail.atelier.schema import Runtime, TopConfig
from vitrail.atelier.seeding import fork_generator, set_seed
from vitrail.bench.assembly import PretrainAtelier
from vitrail.bench.ema import ShadowWeights
from vitrail.bench.optim import build_optimizer, build_scheduler
from vitrail.cutline.pairing import GlazedCorpus
from vitrail.cutline.types import collate_paired

_LOG = get_logger("pretrain")


def run_pretrain(
    config: TopConfig,
    runtime: Runtime,
    corpus_len: int = 64,
    max_steps: int | None = None,
) -> list[dict[str, float]]:
    set_seed(config.regimen.seed)
    device = torch.device(runtime.device)
    model = PretrainAtelier(config.cartoon, config.glass, config.regimen).to(device)

    corpus = GlazedCorpus(
        config.glass, corpus_len, config.glass.patch, base_seed=config.regimen.seed
    )
    loader = DataLoader(
        corpus,
        batch_size=config.regimen.micro_batch,
        shuffle=True,
        collate_fn=collate_paired,
        num_workers=0,
        drop_last=True,
    )

    steps_per_epoch = max(len(loader) // config.regimen.grad_accum, 1)
    total_steps = steps_per_epoch * config.regimen.epochs
    warmup = steps_per_epoch * config.regimen.warmup_epochs
    optimizer = build_optimizer(model.parameters(), config.regimen)
    scheduler = build_scheduler(optimizer, config.regimen, total_steps, warmup)
    shadow = ShadowWeights(model, config.regimen.ema_decay)
    generator = fork_generator(config.regimen.seed + 1)

    history: list[dict[str, float]] = []
    opt_step = 0
    optimizer.zero_grad(set_to_none=True)
    accum = 0
    for epoch in range(config.regimen.epochs):
        for batch in loader:
            batch = _to_device(batch, device)
            total, terms = model(batch, epoch, generator)
            (total / config.regimen.grad_accum).backward()
            accum += 1
            if accum % config.regimen.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.regimen.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                shadow.update(model)
                opt_step += 1
                record = {
                    "step": float(opt_step),
                    "total": float(total.detach()),
                    "xm": float(terms.xm.detach()),
                    "rec": float(terms.rec.detach()),
                    "cons": float(terms.cons.detach()),
                }
                history.append(record)
                if opt_step % runtime.log_every == 0:
                    _LOG.info("step %d total=%.4f xm=%.4f", opt_step, record["total"], record["xm"])
                if max_steps is not None and opt_step >= max_steps:
                    _checkpoint(model, optimizer, epoch, config, runtime)
                    return history
    _checkpoint(model, optimizer, config.regimen.epochs, config, runtime)
    return history


def _to_device(batch: Any, device: torch.device) -> Any:
    if torch.is_tensor(batch):
        return batch.to(device)
    if isinstance(batch, dict):
        return {k: _to_device(v, device) for k, v in batch.items()}
    return batch


def _checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: TopConfig,
    runtime: Runtime,
) -> None:
    payload = stamp(model, optimizer, epoch, config.regimen.seed, {"tag": config.tag})
    target = Path(runtime.out_dir) / f"{config.tag}_encoder.pane"
    write_pane(payload, target)
    _LOG.info("wrote %s", target)
