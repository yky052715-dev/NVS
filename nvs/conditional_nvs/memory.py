from __future__ import annotations

import math
from dataclasses import dataclass
from time import perf_counter
from typing import Iterable

import torch
import torch.nn.functional as F


MEMORY_PROTOCOLS = {
    "M_R5": ("random", 5_000),
    "M_K5": ("kcenter", 5_000),
    "M_R10": ("random", 10_000),
    "M_K10": ("kcenter", 10_000),
    "M_MRK10": ("kcenter_merge_reduce", 10_000),
    "M_IBK10": ("kcenter_image_balanced", 10_000),
    "M_R30": ("random", 30_000),
    "M_K30": ("kcenter", 30_000),
    "M_F0": ("full", 0),
}


@dataclass(frozen=True)
class MemoryBuildResult:
    memory_bank: torch.Tensor
    candidate_indices: torch.Tensor
    selected_memory_indices: torch.Tensor
    strategy: str
    capacity: int
    build_seconds: float
    algorithm: str

    def state_dict(self) -> dict:
        return {
            "memory_bank": self.memory_bank,
            "candidate_indices": self.candidate_indices,
            "selected_memory_indices": self.selected_memory_indices,
            "strategy": self.strategy,
            "capacity": self.capacity,
            "build_seconds": self.build_seconds,
            "algorithm": self.algorithm,
        }


def shared_candidate_indices(total: int, seed: int, size: int = 50_000) -> torch.Tensor:
    if total <= 0:
        raise ValueError("total must be positive")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    return torch.randperm(int(total), generator=generator)[: min(int(total), int(size))]


def _squared_euclidean_to_centers(
    values: torch.Tensor, centers: torch.Tensor, chunk_size: int
) -> torch.Tensor:
    output: list[torch.Tensor] = []
    center_norm = (centers * centers).sum(dim=1)
    for start in range(0, values.shape[0], int(chunk_size)):
        block = values[start : start + int(chunk_size)]
        distances = (
            (block * block).sum(dim=1, keepdim=True)
            + center_norm.unsqueeze(0)
            - 2.0 * block @ centers.T
        ).clamp_min_(0.0)
        output.append(distances.min(dim=1).values)
    return torch.cat(output)


def greedy_kcenter_indices(
    values: torch.Tensor,
    k: int,
    seed: int,
    chunk_size: int = 8192,
    batch_select: int = 1,
) -> torch.Tensor:
    """Deterministic farthest-first selection.

    ``batch_select=1`` is exact greedy. Larger batches are the bounded-cost
    approximation used inside the 30k merge-reduce path.
    """

    values = values.float().contiguous()
    n = int(values.shape[0])
    k = min(max(1, int(k)), n)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    first = int(torch.randint(n, (1,), generator=generator).item())
    selected = [first]
    selected_mask = torch.zeros(n, dtype=torch.bool, device=values.device)
    selected_mask[first] = True
    min_distances = _squared_euclidean_to_centers(
        values, values[first : first + 1], chunk_size
    )
    min_distances[first] = -1.0
    batch = max(1, int(batch_select))
    while len(selected) < k:
        count = min(batch, k - len(selected))
        candidates = torch.topk(min_distances, k=count, largest=True).indices
        candidates = candidates[~selected_mask[candidates]]
        if candidates.numel() == 0:
            candidates = torch.nonzero(~selected_mask, as_tuple=True)[0][:count]
        selected.extend(candidates.detach().cpu().tolist())
        selected_mask[candidates] = True
        update = _squared_euclidean_to_centers(values, values[candidates], chunk_size)
        min_distances = torch.minimum(min_distances, update)
        min_distances[selected_mask] = -1.0
    return torch.tensor(selected[:k], dtype=torch.long, device="cpu")


def merge_reduce_kcenter_indices(
    values: torch.Tensor,
    k: int,
    seed: int,
    block_size: int = 50_000,
    oversample: float = 2.0,
    chunk_size: int = 8192,
    batch_select: int = 64,
) -> torch.Tensor:
    """Blockwise merge-reduce k-center for 30k banks.

    This function never calls full-pool exact greedy k-center. Each block is
    reduced to its proportional share of a 2K coreset and only that coreset is
    compressed to K with batched farthest-first updates.
    """

    values = values.float().contiguous()
    n = int(values.shape[0])
    k = min(max(1, int(k)), n)
    if k == n:
        return torch.arange(n)
    candidates: list[torch.Tensor] = []
    for block_number, start in enumerate(range(0, n, int(block_size))):
        stop = min(n, start + int(block_size))
        block_n = stop - start
        block_k = min(
            block_n,
            max(1, int(math.ceil(float(oversample) * k * block_n / n))),
        )
        local = greedy_kcenter_indices(
            values[start:stop],
            block_k,
            seed=int(seed) + 104729 * block_number,
            chunk_size=chunk_size,
            batch_select=max(1, int(batch_select)),
        )
        candidates.append(local + start)
    merged = torch.unique(torch.cat(candidates), sorted=True)
    if merged.numel() <= k:
        missing = torch.tensor(
            [index for index in range(n) if index not in set(merged.tolist())],
            dtype=torch.long,
        )
        return torch.cat([merged, missing[: k - merged.numel()]])
    final_local = greedy_kcenter_indices(
        values[merged.to(values.device, non_blocking=True)],
        k,
        seed=int(seed) + 1_000_003,
        chunk_size=chunk_size,
        batch_select=max(1, int(batch_select)),
    )
    return merged[final_local]


def group_balanced_kcenter_indices(
    values: torch.Tensor,
    group_indices: torch.Tensor,
    k: int,
    seed: int,
    chunk_size: int = 8192,
) -> torch.Tensor:
    """Select near-equal per-group quotas with within-group exact k-center."""

    values = values.float().contiguous()
    groups = group_indices.long().cpu().reshape(-1)
    n = int(values.shape[0])
    if groups.numel() != n:
        raise ValueError("group_indices must align with values")
    k = min(max(1, int(k)), n)
    unique, counts = torch.unique(groups, sorted=True, return_counts=True)
    quotas = torch.zeros_like(counts)
    allocated = 0
    while allocated < k:
        progressed = False
        for index in range(int(unique.numel())):
            if quotas[index] >= counts[index]:
                continue
            quotas[index] += 1
            allocated += 1
            progressed = True
            if allocated == k:
                break
        if not progressed:
            raise AssertionError("Unable to allocate balanced group quotas")

    selected: list[torch.Tensor] = []
    for ordinal, (group, quota) in enumerate(
        zip(unique.tolist(), quotas.tolist())
    ):
        if quota <= 0:
            continue
        members = torch.nonzero(groups == int(group), as_tuple=True)[0]
        local = greedy_kcenter_indices(
            values[members.to(values.device, non_blocking=True)],
            int(quota),
            seed=int(seed) + 104729 * ordinal,
            chunk_size=chunk_size,
            batch_select=1,
        )
        selected.append(members[local])
    output = torch.cat(selected).long().cpu()
    if output.numel() != k or torch.unique(output).numel() != k:
        raise AssertionError("Balanced k-center did not return K unique indices")
    return output


def build_memory(
    features: torch.Tensor,
    strategy: str,
    capacity: int,
    seed: int,
    candidate_indices: torch.Tensor | None = None,
    group_indices: torch.Tensor | None = None,
    candidate_size: int = 50_000,
    block_size: int = 50_000,
    chunk_size: int = 8192,
    large_k_batch_select: int = 64,
    normalize_features: bool = True,
) -> MemoryBuildResult:
    flat = features.reshape(-1, features.shape[-1]).float()
    if normalize_features:
        flat = F.normalize(flat, dim=-1)
    flat = flat.contiguous()
    n = int(flat.shape[0])
    if n == 0:
        raise ValueError("Cannot build an empty memory bank")
    strategy = str(strategy).lower()
    target = n if strategy == "full" or int(capacity) <= 0 else min(int(capacity), n)
    start = perf_counter()

    if strategy == "full":
        candidates = torch.arange(n)
        selected = candidates.clone()
        algorithm = "full"
    elif target >= 30_000 and n > target:
        candidates = torch.arange(n)
        if strategy == "random":
            generator = torch.Generator(device="cpu").manual_seed(int(seed))
            selected = torch.randperm(n, generator=generator)[:target]
            algorithm = "full_pool_random"
        elif strategy == "kcenter":
            selected = merge_reduce_kcenter_indices(
                flat,
                target,
                seed=seed,
                block_size=block_size,
                chunk_size=chunk_size,
                batch_select=large_k_batch_select,
            )
            algorithm = "merge_reduce_kcenter_gamma2"
        else:
            raise ValueError(f"Unsupported memory strategy: {strategy}")
    else:
        candidates = (
            shared_candidate_indices(n, seed, candidate_size)
            if candidate_indices is None
            else candidate_indices.long().cpu().clone()
        )
        if candidates.ndim != 1 or candidates.numel() == 0:
            raise ValueError("candidate_indices must be a non-empty vector")
        if int(candidates.min()) < 0 or int(candidates.max()) >= n:
            raise IndexError("candidate_indices out of range")
        target = min(target, int(candidates.numel()))
        if strategy == "random":
            generator = torch.Generator(device="cpu").manual_seed(int(seed))
            local = torch.randperm(candidates.numel(), generator=generator)[:target]
            algorithm = "shared_candidate_random"
        elif strategy == "kcenter":
            batch_select = 1 if target <= 10_000 else large_k_batch_select
            local = greedy_kcenter_indices(
                flat[candidates.to(flat.device, non_blocking=True)],
                target,
                seed=seed,
                chunk_size=chunk_size,
                batch_select=batch_select,
            )
            algorithm = "shared_candidate_greedy_kcenter"
        elif strategy == "kcenter_merge_reduce":
            local = merge_reduce_kcenter_indices(
                flat[candidates.to(flat.device, non_blocking=True)],
                target,
                seed=seed,
                block_size=block_size,
                chunk_size=chunk_size,
                batch_select=large_k_batch_select,
            )
            algorithm = "shared_candidate_merge_reduce_kcenter_gamma2"
        elif strategy == "kcenter_image_balanced":
            if group_indices is None:
                raise ValueError("Image-balanced k-center requires group_indices")
            groups = group_indices.long().cpu().reshape(-1)
            if groups.numel() != n:
                raise ValueError("group_indices must align with flattened features")
            local = group_balanced_kcenter_indices(
                flat[candidates.to(flat.device, non_blocking=True)],
                groups[candidates],
                target,
                seed=seed,
                chunk_size=chunk_size,
            )
            algorithm = "shared_candidate_image_balanced_kcenter"
        else:
            raise ValueError(f"Unsupported memory strategy: {strategy}")
        selected = candidates[local]
    return MemoryBuildResult(
        memory_bank=flat[selected.to(flat.device, non_blocking=True)].contiguous(),
        candidate_indices=candidates.contiguous(),
        selected_memory_indices=selected.contiguous(),
        strategy=strategy,
        capacity=int(selected.numel()),
        build_seconds=float(perf_counter() - start),
        algorithm=algorithm,
    )


def build_protocol_memory(
    features: torch.Tensor,
    protocol: str,
    seed: int,
    candidate_indices: torch.Tensor | None = None,
    **kwargs,
) -> MemoryBuildResult:
    if protocol not in MEMORY_PROTOCOLS:
        raise KeyError(f"Unknown memory protocol {protocol!r}")
    strategy, capacity = MEMORY_PROTOCOLS[protocol]
    return build_memory(
        features,
        strategy=strategy,
        capacity=capacity,
        seed=seed,
        candidate_indices=candidate_indices,
        **kwargs,
    )


def augmented_memory_candidates(
    original_features: torch.Tensor,
    transformed_features: Iterable[torch.Tensor],
) -> torch.Tensor:
    chunks = [original_features.reshape(-1, original_features.shape[-1])]
    chunks.extend(
        values.reshape(-1, values.shape[-1]) for values in transformed_features
    )
    return F.normalize(torch.cat(chunks, dim=0).float(), dim=-1)

def matched_augmented_memory_candidates(
    memory_original: torch.Tensor,
    nvs_fit_original: torch.Tensor,
    nvs_fit_transformed: Iterable[torch.Tensor],
) -> torch.Tensor:
    """Build AugMem from exactly the feature information available to D2.

    D2 retrieves from ``memory_original`` and fits its delta basis from the
    aligned ``nvs_fit_original`` plus 13 transformed nvs_fit tensors. AugMem
    receives those same tensors, but uses them as a direct retrieval pool.
    It must not receive transformed memory-split images.
    """

    transformed = tuple(nvs_fit_transformed)
    if len(transformed) != 13:
        raise ValueError("Matched AugMem requires exactly 13 nvs_fit transforms")
    if memory_original.ndim != 3 or nvs_fit_original.ndim != 3:
        raise ValueError("Original feature tensors must have shape [N,P,C]")
    if memory_original.shape[1:] != nvs_fit_original.shape[1:]:
        raise ValueError("memory and nvs_fit patch feature shapes must match")
    if any(values.shape != nvs_fit_original.shape for values in transformed):
        raise ValueError("Transformed nvs_fit features must align with originals")
    originals = torch.cat([memory_original, nvs_fit_original], dim=0)
    return augmented_memory_candidates(originals, transformed)
