from __future__ import annotations

import torch

from nvs.conditional_nvs.subspace import (
    fit_centered_basis,
    fit_uncentered_basis,
    score_unified_deviation,
)


def test_centered_and_uncentered_basis_are_genuinely_different() -> None:
    values = torch.tensor([[10.0, -1.0], [10.0, 0.0], [10.0, 1.0]])
    _, centered = fit_centered_basis(values, rank=1)
    uncentered = fit_uncentered_basis(values, rank=1)
    assert abs(float(centered[0, 1])) > 0.99
    assert abs(float(uncentered[0, 0])) > 0.99


def test_d1_d2_receive_same_deviation() -> None:
    deviation = torch.tensor([[3.0, 4.0]])
    scores = score_unified_deviation(
        deviation,
        feature_basis=torch.tensor([[1.0, 0.0]]),
        delta_basis=torch.tensor([[0.0, 1.0]]),
    )
    assert scores["D1_Global"].item() == 4.0
    assert scores["D2_NVSGlobal"].item() == 3.0
