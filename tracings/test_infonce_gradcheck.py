from __future__ import annotations

import torch

from vitrail.soldering.infonce import cross_modal_infonce


def test_matched_pairs_score_low():
    z = torch.nn.functional.normalize(torch.randn(2, 4, 16), dim=-1)
    present = torch.ones(2, 4)
    temperature = torch.tensor(0.07)
    aligned = cross_modal_infonce(z, z, present, temperature)
    shuffled = z[:, torch.tensor([1, 0, 3, 2])]
    misaligned = cross_modal_infonce(z, shuffled, present, temperature)
    assert aligned.item() < misaligned.item()


def test_finite_difference_matches_autograd():
    torch.manual_seed(1)
    a = torch.nn.functional.normalize(torch.randn(5, 8, dtype=torch.double), dim=-1)
    b = torch.nn.functional.normalize(torch.randn(5, 8, dtype=torch.double), dim=-1)
    present = torch.ones(1, 5, dtype=torch.double)
    temperature = torch.tensor(0.1, dtype=torch.double)

    anchor = a.clone().requires_grad_(True)
    loss = cross_modal_infonce(anchor.view(1, 5, 8), b.view(1, 5, 8), present, temperature)
    loss.backward()
    analytic = anchor.grad[0].clone()

    eps = 1e-6
    numeric = torch.zeros(8, dtype=torch.double)
    for j in range(8):
        bump = a.clone()
        bump[0, j] += eps
        hi = cross_modal_infonce(bump.view(1, 5, 8), b.view(1, 5, 8), present, temperature)
        bump[0, j] -= 2 * eps
        lo = cross_modal_infonce(bump.view(1, 5, 8), b.view(1, 5, 8), present, temperature)
        numeric[j] = (hi - lo) / (2 * eps)
    assert torch.allclose(analytic, numeric, atol=1e-4)


def test_present_mask_drops_absent_regions():
    z = torch.nn.functional.normalize(torch.randn(1, 3, 16), dim=-1)
    present = torch.tensor([[1.0, 0.0, 0.0]])
    out = cross_modal_infonce(z, z, present, torch.tensor(0.07))
    assert out.item() == 0.0
