from __future__ import annotations

from typing import TypedDict

import torch


class RegionBundle(TypedDict):
    points: torch.Tensor
    feats: torch.Tensor
    present: torch.Tensor
    vertex_fdi: torch.Tensor
    vertex_region: torch.Tensor


class VolumeSample(TypedDict):
    volume: torch.Tensor
    seg: torch.Tensor
    implant: torch.Tensor
    fdi_map: torch.Tensor


class PairedSample(TypedDict):
    volume: torch.Tensor
    seg: torch.Tensor
    implant: torch.Tensor
    fdi_map: torch.Tensor
    bundle: RegionBundle
    paired: torch.Tensor


def stack_bundles(items: list[RegionBundle]) -> RegionBundle:
    return RegionBundle(
        points=torch.stack([b["points"] for b in items]),
        feats=torch.stack([b["feats"] for b in items]),
        present=torch.stack([b["present"] for b in items]),
        vertex_fdi=torch.stack([b["vertex_fdi"] for b in items]),
        vertex_region=torch.stack([b["vertex_region"] for b in items]),
    )


def bundle_to(bundle: RegionBundle, device: torch.device) -> RegionBundle:
    return RegionBundle(
        points=bundle["points"].to(device),
        feats=bundle["feats"].to(device),
        present=bundle["present"].to(device),
        vertex_fdi=bundle["vertex_fdi"].to(device),
        vertex_region=bundle["vertex_region"].to(device),
    )


def sample_to(sample: PairedSample, device: torch.device) -> PairedSample:
    return PairedSample(
        volume=sample["volume"].to(device),
        seg=sample["seg"].to(device),
        implant=sample["implant"].to(device),
        fdi_map=sample["fdi_map"].to(device),
        bundle=bundle_to(sample["bundle"], device),
        paired=sample["paired"].to(device),
    )


def collate_paired(items: list[PairedSample]) -> PairedSample:
    return PairedSample(
        volume=torch.stack([s["volume"] for s in items]),
        seg=torch.stack([s["seg"] for s in items]),
        implant=torch.stack([s["implant"] for s in items]),
        fdi_map=torch.stack([s["fdi_map"] for s in items]),
        bundle=stack_bundles([s["bundle"] for s in items]),
        paired=torch.stack([s["paired"] for s in items]),
    )
