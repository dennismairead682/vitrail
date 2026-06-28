from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from vitrail.atelier.journal import get_logger
from vitrail.atelier.provenance import read_pane, stamp, write_pane
from vitrail.atelier.schema import Runtime, TaskSpec, TopConfig
from vitrail.atelier.seeding import set_seed
from vitrail.cutline.pairing import GlazedCorpus
from vitrail.cutline.types import PairedSample, collate_paired
from vitrail.lights.heads import FusionOutput
from vitrail.lights.panel import Window
from vitrail.soldering.segmentation import vertex_objective, voxel_objective

_LOG = get_logger("finetune")


def _vertex_targets(batch: PairedSample, slots: int) -> torch.Tensor:
    present = batch["bundle"]["present"]
    b, r = present.shape
    p = batch["bundle"]["points"].shape[2]
    ids = torch.arange(1, r + 1, device=present.device).view(1, r, 1)
    target = ids.expand(b, r, p) * (present.unsqueeze(-1) > 0.5).long()
    return target


def _task_loss(
    window: Window, batch: PairedSample, task: str, dice_weight: float, slots: int
) -> torch.Tensor:
    if task == "seg_cbct":
        logits = window.seg_cbct(batch["volume"])
        return voxel_objective(logits, batch["seg"], dice_weight)
    if task == "seg_ios":
        logits = window.seg_ios(batch["bundle"])
        return vertex_objective(logits, _vertex_targets(batch, slots), dice_weight)
    out = window.fusion(batch["volume"], batch["bundle"])
    assert isinstance(out, dict)
    typed: FusionOutput = out
    recon = voxel_objective(typed["recon"], batch["seg"], dice_weight)
    implant = F.cross_entropy(typed["implant"], batch["implant"])
    return recon + implant


def run_finetune(
    config: TopConfig,
    runtime: Runtime,
    task: TaskSpec,
    pretrained: str | None = None,
    corpus_len: int = 48,
    max_steps: int | None = None,
) -> list[dict[str, float]]:
    set_seed(config.regimen.seed + 7)
    device = torch.device(runtime.device)
    window = Window(config.cartoon, config.glass, config.regimen).to(device)
    if pretrained is not None:
        state = read_pane(pretrained)["model"]
        absorbed = {k[len("encoder.") :]: v for k, v in state.items() if k.startswith("encoder.")}
        window.encoder.load_state_dict(absorbed, strict=False)

    corpus = GlazedCorpus(
        config.glass, corpus_len, config.glass.patch, base_seed=config.regimen.seed + 3
    )
    loader = DataLoader(
        corpus,
        batch_size=task.micro_batch,
        shuffle=True,
        collate_fn=collate_paired,
        num_workers=0,
        drop_last=True,
    )

    head_params = [p for name, p in window.named_parameters() if not name.startswith("encoder.")]
    encoder_params = [p for name, p in window.named_parameters() if name.startswith("encoder.")]
    optimizer = torch.optim.AdamW(
        [
            {"params": head_params, "lr": task.head_lr},
            {"params": encoder_params, "lr": task.encoder_lr},
        ],
        weight_decay=config.regimen.weight_decay,
    )

    history: list[dict[str, float]] = []
    best = float("inf")
    stale = 0
    opt_step = 0
    for epoch in range(task.epochs):
        for batch in loader:
            batch = _to_device(batch, device)
            loss = _task_loss(window, batch, task.name, task.dice_weight, config.glass.region_slots)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(window.parameters(), config.regimen.grad_clip)
            optimizer.step()
            opt_step += 1
            value = float(loss.detach())
            history.append({"step": float(opt_step), "loss": value})
            if value < best - 1e-4:
                best = value
                stale = 0
            else:
                stale += 1
            if max_steps is not None and opt_step >= max_steps:
                _save(window, optimizer, epoch, config, runtime, task.name)
                return history
        if stale >= task.early_stop_patience:
            _LOG.info("early stop at epoch %d", epoch)
            break
    _save(window, optimizer, task.epochs, config, runtime, task.name)
    return history


def _to_device(batch: Any, device: torch.device) -> Any:
    if torch.is_tensor(batch):
        return batch.to(device)
    if isinstance(batch, dict):
        return {k: _to_device(v, device) for k, v in batch.items()}
    return batch


def _save(
    window: Window,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: TopConfig,
    runtime: Runtime,
    task: str,
) -> None:
    payload = stamp(window, optimizer, epoch, config.regimen.seed, {"task": task})
    target = Path(runtime.out_dir) / f"{config.tag}_{task}.pane"
    write_pane(payload, target)
    _LOG.info("wrote %s", target)
