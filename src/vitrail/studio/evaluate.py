from __future__ import annotations

import numpy as np
import torch

from vitrail.atelier.schema import Runtime, TopConfig
from vitrail.cutline.pairing import GlazedCorpus
from vitrail.cutline.types import collate_paired, sample_to
from vitrail.lightbox.overlap import mean_dice
from vitrail.lights.panel import Window


def appraise(
    window: Window, config: TopConfig, runtime: Runtime, task: str, samples: int = 8
) -> dict[str, float]:
    device = torch.device(runtime.device)
    window.eval()
    corpus = GlazedCorpus(
        config.glass, samples, config.glass.patch, base_seed=config.regimen.seed + 99
    )
    batch = sample_to(collate_paired([corpus[i] for i in range(samples)]), device)

    with torch.no_grad():
        if task == "seg_cbct":
            pred = window.seg_cbct(batch["volume"]).argmax(dim=1).cpu().numpy()
            ref = batch["seg"].cpu().numpy()
            score = _frame_mean(pred, ref, config.glass.cbct_classes)
            return {"mean_dice": score}
        if task == "seg_ios":
            logits = window.seg_ios(batch["bundle"])
            present = batch["bundle"]["present"]
            b, r = present.shape
            p = batch["bundle"]["points"].shape[2]
            ids = torch.arange(1, r + 1, device=present.device).view(1, r, 1)
            target = (ids.expand(b, r, p) * (present.unsqueeze(-1) > 0.5).long()).cpu()
            acc = float((logits.argmax(dim=-1).cpu() == target).float().mean())
            return {"vertex_accuracy": acc}
        out = window.fusion(batch["volume"], batch["bundle"])
        assert isinstance(out, dict)
        pred = out["recon"].argmax(dim=1).cpu().numpy()
        ref = batch["seg"].cpu().numpy()
        return {"recon_dice": _frame_mean(pred, ref, config.glass.cbct_classes)}


def _frame_mean(pred: np.ndarray, ref: np.ndarray, classes: int) -> float:
    return float(np.mean([mean_dice(pred[i], ref[i], classes) for i in range(pred.shape[0])]))
