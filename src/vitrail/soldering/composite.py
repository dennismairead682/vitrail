from __future__ import annotations

from typing import NamedTuple

import torch

from vitrail.atelier.schema import Lambdas


class PretrainTerms(NamedTuple):
    xm: torch.Tensor
    rec: torch.Tensor
    cons: torch.Tensor


def weighted_total(terms: PretrainTerms, weights: Lambdas) -> torch.Tensor:
    return weights.xm * terms.xm + weights.rec * terms.rec + weights.cons * terms.cons
