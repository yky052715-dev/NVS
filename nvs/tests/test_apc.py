from __future__ import annotations

import torch

from nvs.apc import decompose_deviation, spatial_consistency


def test_decompose_deviation_uses_energy_ratio() -> None:
    basis = torch.tensor([[1.0, 0.0]])
    delta = torch.tensor([[[3.0, 4.0]]])
    output = decompose_deviation(delta, basis)
    assert torch.allclose(output["parallel_norm"], torch.tensor([[3.0]]))
    assert torch.allclose(output["perpendicular_norm"], torch.tensor([[4.0]]))
    assert torch.allclose(output["rho"], torch.tensor([[9.0 / 25.0]]))


def test_decompose_deviation_boundary_cases() -> None:
    basis = torch.tensor([[1.0, 0.0]])
    delta = torch.tensor([[[2.0, 0.0], [0.0, 3.0], [0.0, 0.0]]])
    rho = decompose_deviation(delta, basis)["rho"]
    assert torch.allclose(rho, torch.tensor([[1.0, 0.0, 0.0]]))


def test_spatial_consistency_uses_median_direction() -> None:
    coefficients = torch.tensor(
        [[[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [-1.0, 0.0]]]
    )
    consistency = spatial_consistency(coefficients)
    assert torch.allclose(consistency[0, :3], torch.ones(3))
    assert consistency[0, 3].item() == 0.0


def test_spatial_consistency_handles_zero_center() -> None:
    coefficients = torch.zeros(2, 5, 3)
    consistency = spatial_consistency(coefficients)
    assert torch.equal(consistency, torch.zeros(2, 5))
