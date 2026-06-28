from __future__ import annotations

from vitrail.soldering.composite import PretrainTerms, weighted_total
from vitrail.soldering.consistency import consistency_loss
from vitrail.soldering.infonce import cross_modal_infonce, gather_valid
from vitrail.soldering.reconstruction import (
    PatchPredictor,
    block_mask,
    masked_recon_loss,
)
from vitrail.soldering.segmentation import (
    soft_dice_loss,
    vertex_objective,
    voxel_objective,
)

__all__ = [
    "PretrainTerms",
    "weighted_total",
    "consistency_loss",
    "cross_modal_infonce",
    "gather_valid",
    "PatchPredictor",
    "block_mask",
    "masked_recon_loss",
    "soft_dice_loss",
    "vertex_objective",
    "voxel_objective",
]
