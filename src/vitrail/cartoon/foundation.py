from __future__ import annotations

import math
from typing import TypedDict

import torch
from torch import nn

from vitrail.atelier.schema import CartoonSpec, GlassSpec, RegimenSpec
from vitrail.cartoon.projection import ProjectionHead
from vitrail.cartoon.surface import SurfaceEncoder
from vitrail.cartoon.volumetric import VolumetricEncoder
from vitrail.cutline.types import RegionBundle
from vitrail.cutline.volumes import downsample_labels, region_pool
from vitrail.leading.crossattn import CameFusion


class CoupledOutput(TypedDict):
    dense: torch.Tensor
    fused: torch.Tensor
    skips: list[torch.Tensor]
    z_cbct: torch.Tensor
    z_ios: torch.Tensor
    pair_present: torch.Tensor


class CoupledEncoder(nn.Module):
    def __init__(self, cartoon: CartoonSpec, glass: GlassSpec, regimen: RegimenSpec) -> None:
        super().__init__()
        self.glass = glass
        self.cartoon = cartoon
        self.f_vol = VolumetricEncoder(cartoon)
        self.f_surf = SurfaceEncoder(cartoon, glass)
        self.fusion = CameFusion(
            self.f_vol.out_channels,
            cartoon.surf_width,
            cartoon.fusion_heads,
            cartoon.fusion_dropout,
        )
        self.proj_vol = ProjectionHead(self.f_vol.out_channels, cartoon.proj_dim)
        self.proj_surf = ProjectionHead(cartoon.surf_width, cartoon.proj_dim)
        self.log_temperature = nn.Parameter(torch.tensor(math.log(regimen.temperature_init)))

    def encode_volume(self, volume: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        return self.f_vol(volume)

    def encode_surface(self, bundle: RegionBundle) -> tuple[torch.Tensor, torch.Tensor]:
        regional, glob, _ = self.f_surf(bundle)
        return regional, glob

    def fuse(
        self, dense: torch.Tensor, regional: torch.Tensor, present: torch.Tensor
    ) -> torch.Tensor:
        return self.fusion(dense, regional, present)

    def couple(
        self,
        volume: torch.Tensor,
        fdi_map: torch.Tensor,
        bundle: RegionBundle,
        use_surface: bool,
    ) -> CoupledOutput:
        dense, skips = self.encode_volume(volume)
        slots = self.glass.region_slots
        small = torch.stack(
            [downsample_labels(fdi_map[i], dense.shape[-3:]) for i in range(fdi_map.shape[0])]
        )
        vol_regions, vol_present = region_pool(dense, small, slots)
        z_cbct = self.proj_vol(vol_regions)
        if use_surface:
            regional, _ = self.encode_surface(bundle)
            if self.cartoon.use_fusion:
                fused = self.fuse(dense, regional, bundle["present"])
            else:
                fused = dense
            z_ios = self.proj_surf(regional)
            pair_present = vol_present * bundle["present"]
        else:
            fused = dense
            z_ios = torch.zeros_like(z_cbct)
            pair_present = torch.zeros_like(vol_present)
        return CoupledOutput(
            dense=dense,
            fused=fused,
            skips=skips,
            z_cbct=z_cbct,
            z_ios=z_ios,
            pair_present=pair_present,
        )

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp().clamp(0.01, 0.5)
