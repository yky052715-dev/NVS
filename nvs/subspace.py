from __future__ import annotations

import torch


def sample_rows(values: torch.Tensor, max_rows: int, seed: int) -> torch.Tensor:
    flat = values.reshape(-1, values.shape[-1]).float().cpu()
    if int(max_rows) <= 0 or flat.shape[0] <= int(max_rows):
        return flat
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    indices = torch.randperm(flat.shape[0], generator=generator)[: int(max_rows)]
    return flat[indices]


def fit_pca_basis(values: torch.Tensor, rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    if values.ndim != 2:
        raise ValueError("values must have shape [N, D]")
    if values.shape[0] < 2:
        raise ValueError("At least two rows are required for PCA")
    centered = values.float() - values.float().mean(dim=0, keepdim=True)
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    rank = min(max(1, int(rank)), vh.shape[0])
    basis = vh[:rank].contiguous()
    mean = values.float().mean(dim=0).contiguous()
    return mean, basis


def projection(values: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    if basis.ndim != 2:
        raise ValueError("basis must have shape [K, D]")
    return (values @ basis.T) @ basis


def residual_norm(values: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    residual = values.float() - projection(values.float(), basis.float())
    return torch.linalg.norm(residual, dim=-1)


def pca_feature_residual(features: torch.Tensor, mean: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return residual_norm(features.float() - mean.float(), basis.float())


def nvs_residual(delta: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return residual_norm(delta.float(), basis.float())


def explained_ratio(delta: torch.Tensor, basis: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    delta = delta.float()
    projected = projection(delta, basis.float())
    return torch.linalg.norm(projected, dim=-1) / (torch.linalg.norm(delta, dim=-1) + eps)

