from __future__ import annotations

from pathlib import Path

import pytest
import torch

from vitrail.atelier.schema import CartoonSpec, GlassSpec, RegimenSpec, TopConfig, load_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def smoke_config() -> TopConfig:
    return load_config(ROOT / "configs" / "panels" / "_smoke.yaml")


@pytest.fixture
def tiny_specs() -> tuple[CartoonSpec, GlassSpec, RegimenSpec]:
    glass = GlassSpec(
        patch=(16, 16, 16),
        cbct_classes=6,
        fdi_classes=5,
        region_slots=4,
        surface_feats=7,
        points_per_region=24,
        mask_ratio=0.5,
    )
    cartoon = CartoonSpec(
        vol_stem=8,
        vol_stages=(8, 16, 24),
        vol_blocks=1,
        surf_width=32,
        surf_depth=2,
        embed_dim=24,
        proj_dim=16,
        fusion_heads=2,
    )
    regimen = RegimenSpec(world_size=1, micro_batch=2, grad_accum=1, modality_dropout=0.0)
    return cartoon, glass, regimen


@pytest.fixture(autouse=True)
def _determinism() -> None:
    torch.manual_seed(0)
