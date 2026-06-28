from __future__ import annotations

import torch

from vitrail.atelier.provenance import read_pane, stamp, write_pane


def test_pane_roundtrip_restores_weights_and_seed(tmp_path):
    model = torch.nn.Linear(4, 4)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    payload = stamp(model, optimizer, epoch=11, seed=4242, extra={"tag": "unit"})
    target = tmp_path / "weights.pane"
    write_pane(payload, target)

    restored = read_pane(target)
    assert restored["seed"] == 4242
    assert restored["epoch"] == 11
    assert restored["tag"] == "unit"
    for key, value in model.state_dict().items():
        assert torch.allclose(value, restored["model"][key])


def test_write_is_atomic(tmp_path):
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    target = tmp_path / "panes" / "x.pane"
    write_pane(stamp(model, optimizer, 0, 1), target)
    assert target.exists()
    assert not list(tmp_path.glob("**/*.inflight"))
