from __future__ import annotations

import torch

from nvs.conditional_nvs.whitening import fit_whitener


def test_whitening_is_stable_for_nearly_singular_features() -> None:
    base = torch.linspace(-1, 1, 100).unsqueeze(1)
    values = torch.cat([base, base, base * 1.0e-10], dim=1)
    whitener = fit_whitener(
        values, rho=0.99, shrinkage=0.07, relative_floor=1.0e-8
    )
    transformed = whitener.transform(values)
    assert torch.isfinite(transformed).all()
    assert 0.0 <= whitener.jitter <= 1.0
    assert whitener.eigenvalue_floor > 0.0
    assert transformed.shape[1] >= 1
