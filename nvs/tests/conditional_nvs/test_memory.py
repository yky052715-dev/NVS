from __future__ import annotations

import torch

from nvs.conditional_nvs.memory import (
    build_memory,
    group_balanced_kcenter_indices,
    merge_reduce_kcenter_indices,
    shared_candidate_indices,
)


def test_memory_capacity_indices_and_reproducibility() -> None:
    values = torch.randn(200, 8, generator=torch.Generator().manual_seed(1))
    candidates = shared_candidate_indices(200, seed=42, size=100)
    first = build_memory(values, "kcenter", 20, 42, candidate_indices=candidates)
    second = build_memory(values, "kcenter", 20, 42, candidate_indices=candidates)
    assert first.memory_bank.shape == (20, 8)
    assert torch.equal(first.candidate_indices, candidates)
    assert torch.equal(first.selected_memory_indices, second.selected_memory_indices)
    assert set(first.selected_memory_indices.tolist()) <= set(candidates.tolist())


def test_random_and_kcenter_can_share_candidate_pool() -> None:
    values = torch.randn(120, 4, generator=torch.Generator().manual_seed(3))
    candidates = shared_candidate_indices(120, seed=7, size=60)
    random = build_memory(values, "random", 10, 7, candidate_indices=candidates)
    kcenter = build_memory(values, "kcenter", 10, 7, candidate_indices=candidates)
    assert torch.equal(random.candidate_indices, kcenter.candidate_indices)
    assert random.capacity == kcenter.capacity == 10


def test_merge_reduce_returns_unique_bounded_indices() -> None:
    values = torch.randn(101, 5, generator=torch.Generator().manual_seed(4))
    indices = merge_reduce_kcenter_indices(
        values, k=17, seed=42, block_size=25, batch_select=4
    )
    assert indices.shape == (17,)
    assert torch.unique(indices).numel() == 17
    assert int(indices.min()) >= 0 and int(indices.max()) < 101


def test_same_pool_global_merge_reduce_and_image_balanced_k10_variants() -> None:
    values = torch.randn(120, 6, generator=torch.Generator().manual_seed(9))
    groups = torch.arange(4).repeat_interleave(30)
    candidates = shared_candidate_indices(120, seed=42, size=80)
    common = {
        "features": values,
        "capacity": 20,
        "seed": 42,
        "candidate_indices": candidates,
        "group_indices": groups,
        "block_size": 20,
        "chunk_size": 32,
        "large_k_batch_select": 4,
    }

    global_k = build_memory(strategy="kcenter", **common)
    partitioned = build_memory(strategy="kcenter_merge_reduce", **common)
    balanced = build_memory(strategy="kcenter_image_balanced", **common)

    assert torch.equal(global_k.candidate_indices, candidates)
    assert torch.equal(partitioned.candidate_indices, candidates)
    assert torch.equal(balanced.candidate_indices, candidates)
    assert global_k.capacity == partitioned.capacity == balanced.capacity == 20
    assert partitioned.algorithm == "shared_candidate_merge_reduce_kcenter_gamma2"
    assert balanced.algorithm == "shared_candidate_image_balanced_kcenter"
    selected_counts = torch.bincount(
        groups[balanced.selected_memory_indices], minlength=4
    )
    assert int(selected_counts.max() - selected_counts.min()) <= 1


def test_group_balanced_kcenter_is_reproducible() -> None:
    values = torch.randn(60, 5, generator=torch.Generator().manual_seed(11))
    groups = torch.arange(3).repeat_interleave(20)
    first = group_balanced_kcenter_indices(values, groups, 15, seed=7)
    second = group_balanced_kcenter_indices(values, groups, 15, seed=7)
    assert torch.equal(first, second)
    assert torch.equal(
        torch.bincount(groups[first], minlength=3), torch.full((3,), 5)
    )
