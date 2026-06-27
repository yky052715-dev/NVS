from __future__ import annotations

import torch

from nvs.conditional_nvs.conditional_subspace import (
    fit_prototype_model,
    prototype_by_topk_vote_k5,
)


def test_sparse_prototype_falls_back_to_global_basis() -> None:
    generator = torch.Generator().manual_seed(9)
    original = torch.randn(20, 4, generator=generator)
    deltas = torch.randn(20, 13, 4, generator=generator)
    image_ids = torch.arange(20) // 5
    model = fit_prototype_model(
        original, deltas, image_ids, prototypes=2, rank=2, seed=42
    )
    assert model.fallback.all()
    for prototype in range(model.centers.shape[0]):
        assert torch.equal(model.delta_bases[prototype], model.global_delta_basis)
    assert model.statistics()["fallback_patch_fraction"] == 1.0


def test_top5_vote_has_deterministic_smallest_id_tie_break() -> None:
    memory_ids = torch.tensor([2, 1, 2, 1, 3])
    indices = torch.tensor([[0, 1, 2, 3, 4]])
    assert prototype_by_topk_vote_k5(memory_ids, indices).item() == 1
