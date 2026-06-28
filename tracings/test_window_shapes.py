from __future__ import annotations

import torch

from vitrail.cutline.synthetic import synth_paired
from vitrail.cutline.types import collate_paired
from vitrail.lights.panel import Window


def _batch(glass):
    items = [synth_paired(glass, (16, 16, 16), n_teeth=3, seed=i) for i in range(2)]
    return collate_paired(items)


def test_seg_cbct_shape(tiny_specs):
    cartoon, glass, regimen = tiny_specs
    window = Window(cartoon, glass, regimen)
    batch = _batch(glass)
    out = window.seg_cbct(batch["volume"])
    assert out.shape == (2, glass.cbct_classes, 16, 16, 16)


def test_seg_ios_shape(tiny_specs):
    cartoon, glass, regimen = tiny_specs
    window = Window(cartoon, glass, regimen)
    batch = _batch(glass)
    out = window.seg_ios(batch["bundle"])
    assert out.shape[-1] == glass.fdi_classes
    assert out.shape[:3] == (2, glass.region_slots, glass.points_per_region)


def test_fusion_dual_branch_shapes(tiny_specs):
    cartoon, glass, regimen = tiny_specs
    window = Window(cartoon, glass, regimen)
    batch = _batch(glass)
    out = window.fusion(batch["volume"], batch["bundle"])
    assert out["recon"].shape == (2, glass.cbct_classes, 16, 16, 16)
    assert out["implant"].shape == (2, 2, 16, 16, 16)


def test_fusion_off_matches_dense(tiny_specs):
    cartoon, glass, regimen = tiny_specs
    no_bridge = cartoon.model_copy(update={"use_fusion": False})
    window = Window(no_bridge, glass, regimen).eval()
    batch = _batch(glass)
    with torch.no_grad():
        with_surface = window.fusion(batch["volume"], batch["bundle"], use_surface=True)
        without = window.fusion(batch["volume"], batch["bundle"], use_surface=False)
    assert torch.allclose(with_surface["recon"], without["recon"], atol=1e-6)
