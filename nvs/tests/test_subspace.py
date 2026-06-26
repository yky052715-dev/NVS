from __future__ import annotations

import torch

from nvs.subspace import explained_ratio, fit_pca_basis, nvs_residual, sample_rows


def test_fit_pca_basis_shapes() -> None:
    values = torch.randn(20, 8)
    mean, basis = fit_pca_basis(values, rank=3)
    assert mean.shape == (8,)
    assert basis.shape == (3, 8)


def test_nvs_residual_removes_basis_direction() -> None:
    basis = torch.tensor([[1.0, 0.0]])
    values = torch.tensor([[3.0, 4.0]])
    residual = nvs_residual(values, basis)
    assert residual.item() == 4.0
    ratio = explained_ratio(values, basis)
    assert torch.allclose(ratio, torch.tensor([0.6]))


def test_sample_rows_is_deterministic() -> None:
    values = torch.arange(100, dtype=torch.float32).reshape(50, 2)
    a = sample_rows(values, max_rows=5, seed=42)
    b = sample_rows(values, max_rows=5, seed=42)
    assert torch.equal(a, b)

