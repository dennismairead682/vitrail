from __future__ import annotations

import torch

from vitrail.cutline.synthetic import synth_paired
from vitrail.cutline.types import collate_paired
from vitrail.lights.panel import Window
from vitrail.soldering.segmentation import voxel_objective


def test_seg_cbct_overfits_tiny_batch(tiny_specs):
    cartoon, glass, regimen = tiny_specs
    torch.manual_seed(0)
    window = Window(cartoon, glass, regimen)
    batch = collate_paired([synth_paired(glass, (16, 16, 16), n_teeth=2, seed=4)])
    optimizer = torch.optim.AdamW(window.parameters(), lr=3e-3)

    first = None
    last = None
    for step in range(40):
        logits = window.seg_cbct(batch["volume"])
        loss = voxel_objective(logits, batch["seg"], dice_weight=0.5)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 0:
            first = loss.item()
        last = loss.item()
    assert first is not None and last is not None
    assert last < first * 0.7
