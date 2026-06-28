from __future__ import annotations

import torch
from torch import nn

from vitrail.atelier.schema import CartoonSpec, GlassSpec, RegimenSpec
from vitrail.cartoon.foundation import CoupledEncoder
from vitrail.cutline.types import RegionBundle
from vitrail.lights.heads import CbctLight, FusionLight, FusionOutput, IosLight


class Window(nn.Module):
    def __init__(self, cartoon: CartoonSpec, glass: GlassSpec, regimen: RegimenSpec) -> None:
        super().__init__()
        self.glass = glass
        self.encoder = CoupledEncoder(cartoon, glass, regimen)
        self.cbct_light = CbctLight(cartoon, glass)
        self.ios_light = IosLight(cartoon, glass)
        self.fusion_light = FusionLight(cartoon, glass)

    def seg_cbct(self, volume: torch.Tensor) -> torch.Tensor:
        dense, skips = self.encoder.encode_volume(volume)
        return self.cbct_light(dense, skips, tuple(volume.shape[-3:]))

    def seg_ios(self, bundle: RegionBundle) -> torch.Tensor:
        regional, glob, per_point = self.encoder.f_surf(bundle)
        return self.ios_light(bundle, glob, per_point)

    def fusion(
        self, volume: torch.Tensor, bundle: RegionBundle, use_surface: bool = True
    ) -> FusionOutput:
        dense, skips = self.encoder.encode_volume(volume)
        if use_surface and self.encoder.cartoon.use_fusion:
            regional, _, _ = self.encoder.f_surf(bundle)
            fused = self.encoder.fuse(dense, regional, bundle["present"])
        else:
            fused = dense
        return self.fusion_light(fused, skips, tuple(volume.shape[-3:]))

    def dispatch(
        self,
        task: str,
        volume: torch.Tensor | None = None,
        bundle: RegionBundle | None = None,
        use_surface: bool = True,
    ) -> torch.Tensor | FusionOutput:
        if task == "seg_cbct":
            assert volume is not None
            return self.seg_cbct(volume)
        if task == "seg_ios":
            assert bundle is not None
            return self.seg_ios(bundle)
        if task in ("fusion", "implant"):
            assert volume is not None and bundle is not None
            return self.fusion(volume, bundle, use_surface)
        raise ValueError(f"unknown task {task}")
