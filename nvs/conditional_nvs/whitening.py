from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Whitener:
    reducer_mean: torch.Tensor
    reducer_components: torch.Tensor
    covariance_mean: float
    cholesky: torch.Tensor
    rho: float
    shrinkage: float
    relative_floor: float
    eigenvalue_floor: float
    jitter: float

    def transform(self, values: torch.Tensor) -> torch.Tensor:
        shape = values.shape
        device = values.device
        flat = values.reshape(-1, shape[-1]).float()
        mean = self.reducer_mean.to(device, non_blocking=True)
        components = self.reducer_components.to(device, non_blocking=True)
        cholesky = self.cholesky.to(device, non_blocking=True)
        reduced = (flat - mean) @ components.T
        whitened = torch.linalg.solve_triangular(
            cholesky, reduced.T, upper=False
        ).T
        return whitened.reshape(*shape[:-1], -1)

    def state_dict(self) -> dict:
        return {
            "reducer_mean": self.reducer_mean,
            "reducer_components": self.reducer_components,
            "covariance_mean": self.covariance_mean,
            "cholesky": self.cholesky,
            "rho": self.rho,
            "shrinkage": self.shrinkage,
            "relative_floor": self.relative_floor,
            "eigenvalue_floor": self.eigenvalue_floor,
            "jitter": self.jitter,
        }


def fit_whitener(
    candidate_features: torch.Tensor,
    rho: float = 0.99,
    shrinkage: float = 0.07,
    relative_floor: float = 1.0e-8,
    max_components: int | None = None,
) -> Whitener:
    values = candidate_features.reshape(-1, candidate_features.shape[-1]).double()
    device = values.device
    if values.shape[0] < 2:
        raise ValueError("At least two candidate features are required")
    mean = values.mean(dim=0)
    centered = values - mean
    covariance = centered.T @ centered / max(1, values.shape[0] - 1)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[order].clamp_min(0.0)
    eigenvectors = eigenvectors[:, order]
    total = eigenvalues.sum()
    if total <= 0:
        component_count = 1
    else:
        cumulative = torch.cumsum(eigenvalues, dim=0) / total
        component_count = int(
            torch.searchsorted(
                cumulative, torch.tensor(float(rho), device=device, dtype=cumulative.dtype)
            ).item()
        ) + 1
    if max_components is not None:
        component_count = min(component_count, int(max_components))
    components = eigenvectors[:, :component_count].T.contiguous()
    reduced = centered @ components.T
    reduced_covariance = reduced.T @ reduced / max(1, reduced.shape[0] - 1)
    covariance_mean = float(torch.trace(reduced_covariance) / component_count)
    shrunk = (1.0 - float(shrinkage)) * reduced_covariance + float(
        shrinkage
    ) * covariance_mean * torch.eye(
        component_count, dtype=torch.float64, device=device
    )
    floor = float(relative_floor) * max(abs(covariance_mean), 1.0e-12)
    evals, evecs = torch.linalg.eigh(shrunk)
    shrunk = (evecs * evals.clamp_min(floor).unsqueeze(0)) @ evecs.T
    jitter = 0.0
    identity = torch.eye(component_count, dtype=torch.float64, device=device)
    attempt = 0.0
    while True:
        try:
            cholesky = torch.linalg.cholesky(shrunk + attempt * identity)
            jitter = attempt
            break
        except RuntimeError:
            attempt = 1.0e-12 if attempt == 0.0 else attempt * 10.0
            if attempt > 1.0:
                raise RuntimeError("Adaptive Cholesky jitter exceeded 1")
    return Whitener(
        reducer_mean=mean.float(),
        reducer_components=components.float(),
        covariance_mean=covariance_mean,
        cholesky=cholesky.float(),
        rho=float(rho),
        shrinkage=float(shrinkage),
        relative_floor=float(relative_floor),
        eigenvalue_floor=floor,
        jitter=float(jitter),
    )
