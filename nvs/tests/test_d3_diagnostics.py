from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch

from nvs.conditional_nvs.conditional_subspace import PrototypeModel
from nvs.conditional_nvs.d3_diagnostics import (
    adaptive_patch_mask,
    basis_stability_rows,
    explained_energy,
    partial_residual_norm,
    routing_agreement,
    sample_image_ids,
    wrong_prototype_ids,
)
from nvs.conditional_nvs.subspace import residual_norm


def test_explained_energy_formula() -> None:
    values = torch.tensor([[3.0, 4.0], [0.0, 2.0]])
    basis = torch.tensor([[1.0, 0.0]])
    result = explained_energy(values, basis)
    assert torch.allclose(result, torch.tensor([9.0 / 25.0, 0.0]))


def test_partial_alpha_one_matches_existing_residual() -> None:
    values = torch.tensor([[3.0, 4.0], [1.0, 2.0]])
    basis = torch.tensor([[1.0, 0.0]])
    assert torch.allclose(
        partial_residual_norm(values, basis, alpha=1.0),
        residual_norm(values, basis),
    )


def test_partial_alpha_zero_is_deviation_norm() -> None:
    values = torch.tensor([[3.0, 4.0], [1.0, 2.0]])
    basis = torch.tensor([[1.0, 0.0]])
    assert torch.allclose(
        partial_residual_norm(values, basis, alpha=0.0),
        torch.linalg.norm(values, dim=-1),
    )


def test_wrong_prototype_control_is_reproducible_and_wrong() -> None:
    own = torch.tensor([0, 1, 2, 3, 0, 1])
    first = wrong_prototype_ids(own, prototype_count=4, seed=42)
    second = wrong_prototype_ids(own, prototype_count=4, seed=42)
    assert torch.equal(first, second)
    assert torch.all(first != own)


def test_routing_agreement() -> None:
    left = torch.tensor([0, 1, 2, 3])
    right = torch.tensor([0, 1, 0, 3])
    assert routing_agreement(left, right) == 0.75


def test_adaptive_max_pool_preserves_small_gt_defect() -> None:
    mask = torch.zeros(1, 17, 17)
    mask[0, 1, 1] = 1
    pooled = adaptive_patch_mask(mask, grid_side=4)
    assert pooled.shape == (1, 4, 4)
    assert pooled.any()


def test_gt_mask_cannot_enter_basis_fitting_api() -> None:
    assert "mask" not in inspect.signature(basis_stability_rows).parameters
    with pytest.raises(TypeError):
        basis_stability_rows(  # type: ignore[call-arg]
            torch.randn(4, 2, 3),
            torch.tensor([0, 0, 1, 1]),
            torch.tensor([0, 0, 0, 0]),
            object(),
            masks=torch.ones(4),
        )


def test_image_subsample_selects_unique_image_ids_not_rows() -> None:
    image_ids = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2])
    sampled = sample_image_ids(image_ids, fraction=2.0 / 3.0, seed=7)
    assert sampled.numel() == 2
    selected_rows = torch.zeros_like(image_ids, dtype=torch.bool)
    for image_id in sampled.tolist():
        selected_rows |= image_ids == image_id
    for image_id in sampled.tolist():
        assert torch.all(selected_rows[image_ids == image_id])


def test_fixed_seed_subsample_is_reproducible() -> None:
    image_ids = torch.arange(10).repeat_interleave(3)
    first = sample_image_ids(image_ids, fraction=0.8, seed=42)
    second = sample_image_ids(image_ids, fraction=0.8, seed=42)
    assert torch.equal(first, second)
