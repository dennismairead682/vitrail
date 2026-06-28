from __future__ import annotations

import inspect
from pathlib import Path

import torch
from torch import nn

from vitrail.cartoon.foundation import CoupledEncoder


class _DenseTrace(nn.Module):
    def __init__(self, encoder: CoupledEncoder) -> None:
        super().__init__()
        self.encoder = encoder

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        dense, _ = self.encoder.encode_volume(volume)
        return dense


def export_encoder(encoder: CoupledEncoder, patch: tuple[int, int, int], out: str | Path) -> Path:
    encoder.eval()
    trace = _DenseTrace(encoder).eval()
    dummy = torch.randn(1, 1, *patch)
    target = Path(out)
    target.parent.mkdir(parents=True, exist_ok=True)
    axes = {"volume": {0: "batch"}, "dense": {0: "batch"}}
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        torch.onnx.export(
            trace,
            (dummy,),
            str(target),
            input_names=["volume"],
            output_names=["dense"],
            dynamic_axes=axes,
            opset_version=17,
            dynamo=False,
        )
    else:
        torch.onnx.export(
            trace,
            (dummy,),
            str(target),
            input_names=["volume"],
            output_names=["dense"],
            dynamic_axes=axes,
            opset_version=17,
        )
    return target


def trace_reference(
    encoder: CoupledEncoder,
    patch: tuple[int, int, int],
    volume: torch.Tensor | None = None,
) -> torch.Tensor:
    trace = _DenseTrace(encoder).eval()
    if volume is None:
        volume = torch.randn(1, 1, *patch)
    with torch.no_grad():
        return trace(volume)
