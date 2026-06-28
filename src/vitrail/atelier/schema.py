from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class GlassSpec(_Frozen):
    voxel_mm: float = 0.30
    patch: tuple[int, int, int] = (128, 128, 128)
    cbct_classes: int = 42
    fdi_classes: int = 33
    region_slots: int = 32
    surface_feats: int = 7
    points_per_region: int = 256
    mask_ratio: float = 0.6
    intensity_clip: tuple[float, float] = (-1000.0, 3000.0)


class CartoonSpec(_Frozen):
    vol_stem: int = 44
    vol_stages: tuple[int, ...] = (44, 88, 176, 288, 352)
    vol_blocks: int = 2
    surf_width: int = 320
    surf_depth: int = 3
    embed_dim: int = 352
    proj_dim: int = 256
    fusion_heads: int = 4
    fusion_dropout: float = 0.0
    use_fusion: bool = True


class Lambdas(_Frozen):
    xm: float = 1.0
    rec: float = 0.5
    cons: float = 0.3


class RegimenSpec(_Frozen):
    world_size: int = 8
    precision: Literal["fp32", "bf16", "fp16"] = "fp32"
    amp: bool = False
    micro_batch: int = 2
    grad_accum: int = 64
    epochs: int = 4000
    warmup_epochs: int = 200
    lr: float = 2.0e-7
    weight_decay: float = 0.05
    betas: tuple[float, float] = (0.9, 0.999)
    grad_clip: float = 1.0
    scheduler: Literal["cosine", "constant"] = "cosine"
    temperature_init: float = 0.07
    modality_dropout: float = 0.25
    dropout_onset: int = 400
    lambdas: Lambdas = Field(default_factory=Lambdas)
    ema_decay: float = 0.999
    seed: int = 20259

    @property
    def effective_batch(self) -> int:
        return self.micro_batch * self.grad_accum * self.world_size


class TaskSpec(_Frozen):
    name: Literal["seg_cbct", "seg_ios", "fusion", "implant"]
    head_lr: float = 1.0e-5
    encoder_lr: float = 2.0e-7
    epochs: int = 600
    micro_batch: int = 2
    grad_accum: int = 16
    dice_weight: float = 0.5
    early_stop_patience: int = 40


class TopConfig(_Frozen):
    tag: str = "main"
    glass: GlassSpec = Field(default_factory=GlassSpec)
    cartoon: CartoonSpec = Field(default_factory=CartoonSpec)
    regimen: RegimenSpec = Field(default_factory=RegimenSpec)
    task: TaskSpec | None = None

    @model_validator(mode="after")
    def _bridge_dims(self) -> TopConfig:
        if self.cartoon.embed_dim != self.cartoon.vol_stages[-1]:
            raise ValueError("embed_dim must equal the deepest volumetric stage width")
        return self


class Runtime(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VITRAIL_", extra="ignore")

    device: str = "cpu"
    workers: int = 4
    out_dir: Path = Path("runs")
    log_every: int = 20


def _coerce_tuples(node: object) -> object:
    if isinstance(node, list):
        return tuple(_coerce_tuples(x) for x in node)
    if isinstance(node, dict):
        return {k: _coerce_tuples(v) for k, v in node.items()}
    return node


def load_config(path: str | Path, overrides: dict[str, object] | None = None) -> TopConfig:
    raw = yaml.safe_load(Path(path).read_text())
    if raw is None:
        raw = {}
    payload = _coerce_tuples(raw)
    assert isinstance(payload, dict)
    if overrides:
        payload = _apply_overrides(dict(payload), overrides)
    return TopConfig.model_validate(payload)


def _apply_overrides(payload: dict[str, object], overrides: dict[str, object]) -> dict[str, object]:
    for dotted, value in overrides.items():
        cursor = payload
        parts = dotted.split(".")
        for key in parts[:-1]:
            nxt = cursor.get(key)
            if not isinstance(nxt, dict):
                nxt = {}
                cursor[key] = nxt
            cursor = nxt
        cursor[parts[-1]] = value
    return payload
