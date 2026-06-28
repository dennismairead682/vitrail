from __future__ import annotations

import torch
from torch import nn

from vitrail.atelier.schema import CartoonSpec, GlassSpec, RegimenSpec
from vitrail.cartoon.foundation import CoupledEncoder
from vitrail.cutline.types import PairedSample, RegionBundle
from vitrail.soldering.composite import PretrainTerms, weighted_total
from vitrail.soldering.consistency import consistency_loss
from vitrail.soldering.infonce import cross_modal_infonce
from vitrail.soldering.reconstruction import PatchPredictor, block_mask, masked_recon_loss


def _stride(glass: GlassSpec, stages: int) -> int:
    return 2 ** (stages - 1)


class PretrainAtelier(nn.Module):
    def __init__(self, cartoon: CartoonSpec, glass: GlassSpec, regimen: RegimenSpec) -> None:
        super().__init__()
        self.glass = glass
        self.regimen = regimen
        self.encoder = CoupledEncoder(cartoon, glass, regimen)
        scale = _stride(glass, len(cartoon.vol_stages))
        self.predictor = PatchPredictor(self.encoder.f_vol.out_channels, scale)
        self.mask_block = 16

    def _drop_present(
        self, bundle: RegionBundle, paired: torch.Tensor, epoch: int, generator: torch.Generator
    ) -> RegionBundle:
        keep = torch.ones_like(paired)
        if epoch >= self.regimen.dropout_onset:
            sample = torch.rand(paired.shape, generator=generator)
            keep = (sample >= self.regimen.modality_dropout).float()
        gate = (paired * keep).view(-1, 1)
        masked = dict(bundle)
        masked["present"] = bundle["present"] * gate
        return RegionBundle(**masked)  # type: ignore[typeddict-item]

    def forward(
        self, batch: PairedSample, epoch: int, generator: torch.Generator
    ) -> tuple[torch.Tensor, PretrainTerms]:
        volume = batch["volume"]
        bundle = self._drop_present(batch["bundle"], batch["paired"], epoch, generator)
        out = self.encoder.couple(volume, batch["fdi_map"], bundle, use_surface=True)

        l_xm = cross_modal_infonce(
            out["z_cbct"], out["z_ios"], out["pair_present"], self.encoder.temperature
        )

        pred = self.predictor(out["dense"])
        mask = block_mask(
            tuple(volume.shape[-3:]), self.mask_block, self.glass.mask_ratio, generator
        ).to(volume.device)
        mask = mask.view(1, 1, *mask.shape).expand_as(volume)
        l_rec = masked_recon_loss(pred, volume, mask)

        l_cons = consistency_loss(out["fused"], out["dense"])

        terms = PretrainTerms(xm=l_xm, rec=l_rec, cons=l_cons)
        total = weighted_total(terms, self.regimen.lambdas)
        return total, terms
