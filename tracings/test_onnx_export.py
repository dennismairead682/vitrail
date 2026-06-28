from __future__ import annotations

import importlib.util

import numpy as np
import onnx
import torch

from vitrail.cartoon.foundation import CoupledEncoder
from vitrail.studio.onnx import export_encoder, trace_reference

_HAS_ORT = importlib.util.find_spec("onnxruntime") is not None


def test_encoder_exports_valid_graph(tiny_specs, tmp_path):
    cartoon, glass, regimen = tiny_specs
    encoder = CoupledEncoder(cartoon, glass, regimen).eval()
    patch = (16, 16, 16)
    path = export_encoder(encoder, patch, tmp_path / "enc.onnx")
    model = onnx.load(str(path))
    onnx.checker.check_model(model)
    assert {i.name for i in model.graph.input} == {"volume"}


def test_onnx_matches_torch(tiny_specs, tmp_path):
    if not _HAS_ORT:
        return
    import onnxruntime as ort

    cartoon, glass, regimen = tiny_specs
    torch.manual_seed(0)
    encoder = CoupledEncoder(cartoon, glass, regimen).eval()
    patch = (16, 16, 16)
    path = export_encoder(encoder, patch, tmp_path / "enc.onnx")
    probe = torch.randn(1, 1, *patch)
    reference = trace_reference(encoder, patch, probe).numpy()

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    produced = sess.run(["dense"], {"volume": probe.numpy()})[0]
    assert produced.shape == reference.shape
    assert np.allclose(produced, reference, atol=1e-4)
