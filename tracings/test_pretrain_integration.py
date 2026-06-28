from __future__ import annotations

import math

import torch

from vitrail.atelier.schema import Runtime
from vitrail.bench.assembly import PretrainAtelier
from vitrail.bench.pretrain import run_pretrain
from vitrail.cutline.synthetic import synth_paired
from vitrail.cutline.types import collate_paired


def test_run_pretrain_writes_pane(smoke_config, tmp_path):
    runtime = Runtime(device="cpu", out_dir=tmp_path, log_every=100)
    history = run_pretrain(smoke_config, runtime, corpus_len=8, max_steps=3)
    assert len(history) == 3
    assert all(math.isfinite(rec["total"]) for rec in history)
    assert (tmp_path / "smoke_encoder.pane").exists()


def test_pretrain_total_drops_on_fixed_batch(tiny_specs):
    cartoon, glass, regimen = tiny_specs
    torch.manual_seed(0)
    model = PretrainAtelier(cartoon, glass, regimen)
    batch = collate_paired([synth_paired(glass, (16, 16, 16), n_teeth=3, seed=2) for _ in range(2)])
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
    generator = torch.Generator()
    generator.manual_seed(0)

    first = None
    last = None
    for step in range(30):
        total, _ = model(batch, epoch=0, generator=generator)
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        optimizer.step()
        if step == 0:
            first = total.item()
        last = total.item()
    assert first is not None and last is not None and last < first
