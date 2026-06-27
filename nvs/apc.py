from __future__ import annotations

import torch


def decompose_deviation(
    delta: torch.Tensor,
    basis: torch.Tensor,
    eps: float = 1.0e-8,
) -> dict[str, torch.Tensor]:
    """Split deviations into NVS-parallel/perpendicular components."""
    values = delta.float()
    components = basis.float()
    coefficients = values @ components.T
    parallel = coefficients @ components
    perpendicular = values - parallel
    parallel_energy = parallel.square().sum(dim=-1)
    perpendicular_energy = perpendicular.square().sum(dim=-1)
    total_energy = parallel_energy + perpendicular_energy
    return {
        "coefficients": coefficients,
        "parallel_norm": torch.sqrt(parallel_energy.clamp_min(0.0)),
        "perpendicular_norm": torch.sqrt(perpendicular_energy.clamp_min(0.0)),
        "total_norm": torch.sqrt(total_energy.clamp_min(0.0)),
        "rho": (
            parallel_energy / (total_energy + float(eps))
        ).clamp(0.0, 1.0),
    }


def spatial_consistency(
    coefficients: torch.Tensor,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    """Compute non-negative cosine agreement with the image median direction."""
    if coefficients.ndim < 2:
        raise ValueError("coefficients must have shape [..., patches, rank]")
    center = coefficients.median(dim=-2, keepdim=True).values
    numerator = (coefficients * center).sum(dim=-1)
    denominator = (
        torch.linalg.norm(coefficients, dim=-1)
        * torch.linalg.norm(center, dim=-1)
    )
    cosine = numerator / denominator.clamp_min(float(eps))
    return torch.where(
        denominator > float(eps),
        cosine,
        torch.zeros_like(cosine),
    ).clamp(0.0, 1.0)
