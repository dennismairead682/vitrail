from __future__ import annotations

import torch

from vitrail.leading.crossattn import CameFusion


def test_token_permutation_invariance():
    torch.manual_seed(2)
    fusion = CameFusion(dense_dim=16, token_dim=24, heads=4, dropout=0.0).eval()
    dense = torch.randn(1, 16, 4, 4, 4)
    tokens = torch.randn(1, 6, 24)
    present = torch.ones(1, 6)
    perm = torch.randperm(6)
    with torch.no_grad():
        a = fusion(dense, tokens, present)
        b = fusion(dense, tokens[:, perm], present[:, perm])
    assert torch.allclose(a, b, atol=1e-5)


def test_absent_tokens_pass_dense_through():
    torch.manual_seed(3)
    fusion = CameFusion(dense_dim=16, token_dim=24, heads=4, dropout=0.0).eval()
    dense = torch.randn(2, 16, 4, 4, 4)
    tokens = torch.randn(2, 5, 24)
    present = torch.zeros(2, 5)
    with torch.no_grad():
        out = fusion(dense, tokens, present)
    assert torch.allclose(out, dense, atol=1e-6)


def test_query_source_is_dense_shape():
    fusion = CameFusion(dense_dim=8, token_dim=12, heads=2, dropout=0.0)
    dense = torch.randn(1, 8, 2, 3, 4)
    tokens = torch.randn(1, 4, 12)
    out = fusion(dense, tokens, torch.ones(1, 4))
    assert out.shape == dense.shape
