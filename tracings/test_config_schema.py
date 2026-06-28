from __future__ import annotations

import pytest
from pydantic import ValidationError

from vitrail.atelier.schema import CartoonSpec, TopConfig, load_config


def test_main_effective_batch(tmp_path):
    cfg = load_config("configs/panels/main.yaml")
    assert cfg.regimen.effective_batch == 1024
    assert cfg.glass.patch == (128, 128, 128)


def test_overrides_apply(tmp_path):
    cfg = load_config(
        "configs/panels/main.yaml",
        overrides={"regimen.lr": 1e-6, "cartoon.use_fusion": False},
    )
    assert cfg.regimen.lr == 1e-6
    assert cfg.cartoon.use_fusion is False


def test_embed_dim_must_match_last_stage():
    with pytest.raises(ValidationError):
        TopConfig(cartoon=CartoonSpec(vol_stages=(32, 64), embed_dim=320))


def test_extra_keys_rejected(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("tag: x\nglass:\n  not_a_field: 3\n")
    with pytest.raises(ValidationError):
        load_config(bad)
