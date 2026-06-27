from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F

from .subspace import fit_centered_basis, fit_uncentered_basis, residual_norm


@dataclass(frozen=True)
class PrototypeModel:
    centers: torch.Tensor
    labels: torch.Tensor
    feature_bases: torch.Tensor
    delta_bases: torch.Tensor
    fallback: torch.Tensor
    unique_patch_counts: torch.Tensor
    unique_image_counts: torch.Tensor
    global_feature_basis: torch.Tensor
    global_delta_basis: torch.Tensor
    rank: int

    def statistics(self) -> dict:
        counts = torch.bincount(self.labels, minlength=self.centers.shape[0])
        fallback_patch_count = int(counts[self.fallback].sum())
        return {
            "prototype_count": int(self.centers.shape[0]),
            "fallback_count": int(self.fallback.sum()),
            "fallback_fraction": float(self.fallback.float().mean()),
            "fallback_patch_fraction": float(
                fallback_patch_count / max(1, int(self.labels.numel()))
            ),
            "member_counts": counts.tolist(),
            "unique_original_patch_counts": self.unique_patch_counts.tolist(),
            "unique_image_counts": self.unique_image_counts.tolist(),
            "fallback_ids": torch.nonzero(self.fallback, as_tuple=True)[0].tolist(),
        }


def spherical_kmeans(
    values: torch.Tensor,
    clusters: int,
    seed: int,
    max_iter: int = 50,
    tolerance: float = 1.0e-5,
) -> tuple[torch.Tensor, torch.Tensor]:
    values = F.normalize(values.reshape(-1, values.shape[-1]).float().cpu(), dim=-1)
    n = int(values.shape[0])
    k = min(max(1, int(clusters)), n)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    initial = torch.randperm(n, generator=generator)[:k]
    centers = values[initial].clone()
    labels = torch.full((n,), -1, dtype=torch.long)
    for _ in range(int(max_iter)):
        new_labels = (values @ centers.T).argmax(dim=1)
        new_centers = torch.zeros_like(centers)
        new_centers.index_add_(0, new_labels, values)
        counts = torch.bincount(new_labels, minlength=k)
        empty = torch.nonzero(counts == 0, as_tuple=True)[0]
        if empty.numel():
            nearest_similarity = (values @ centers.T).max(dim=1).values
            replacements = torch.argsort(nearest_similarity)[: empty.numel()]
            new_centers[empty] = values[replacements]
        new_centers = F.normalize(new_centers, dim=-1)
        shift = torch.linalg.norm(new_centers - centers, dim=1).max().item()
        converged = torch.equal(labels, new_labels) or shift <= float(tolerance)
        labels, centers = new_labels, new_centers
        if converged:
            break
    return centers.contiguous(), labels.contiguous()


def assign_prototypes(values: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    normalized = F.normalize(values.reshape(-1, values.shape[-1]).float().cpu(), dim=-1)
    return (normalized @ F.normalize(centers.float().cpu(), dim=-1).T).argmax(dim=1)


def fit_prototype_model(
    original_features: torch.Tensor,
    deltas: torch.Tensor,
    image_ids: torch.Tensor,
    prototypes: int,
    rank: int,
    seed: int,
    max_iter: int = 50,
) -> PrototypeModel:
    """Fit shared appearance prototypes and D1/D3 bases.

    ``original_features`` is [N,D], ``deltas`` is [N,T,D], and ``image_ids``
    identifies the source image of each original patch.
    """

    originals = F.normalize(original_features.float().cpu(), dim=-1)
    deltas = deltas.float().cpu()
    image_ids = image_ids.long().cpu()
    if originals.ndim != 2 or deltas.ndim != 3:
        raise ValueError("Expected originals [N,D] and deltas [N,T,D]")
    if originals.shape[0] != deltas.shape[0] or image_ids.numel() != originals.shape[0]:
        raise ValueError("Original patch, delta, and image-id counts must match")
    centers, labels = spherical_kmeans(
        originals, clusters=prototypes, seed=seed, max_iter=max_iter
    )
    _, global_feature_basis = fit_centered_basis(originals, rank)
    global_delta_basis = fit_uncentered_basis(
        deltas.reshape(-1, deltas.shape[-1]), rank
    )
    k = int(centers.shape[0])
    feature_bases: list[torch.Tensor] = []
    delta_bases: list[torch.Tensor] = []
    fallback = torch.zeros(k, dtype=torch.bool)
    patch_counts = torch.zeros(k, dtype=torch.long)
    image_counts = torch.zeros(k, dtype=torch.long)
    minimum_patches = max(8 * int(rank), 64)
    for prototype in range(k):
        members = torch.nonzero(labels == prototype, as_tuple=True)[0]
        patch_counts[prototype] = members.numel()
        image_counts[prototype] = torch.unique(image_ids[members]).numel()
        use_fallback = (
            members.numel() < minimum_patches or image_counts[prototype].item() < 4
        )
        fallback[prototype] = use_fallback
        if use_fallback:
            feature_bases.append(global_feature_basis)
            delta_bases.append(global_delta_basis)
        else:
            _, feature_basis = fit_centered_basis(originals[members], rank)
            delta_basis = fit_uncentered_basis(
                deltas[members].reshape(-1, deltas.shape[-1]), rank
            )
            feature_bases.append(feature_basis)
            delta_bases.append(delta_basis)
    return PrototypeModel(
        centers=centers,
        labels=labels,
        feature_bases=torch.stack(feature_bases),
        delta_bases=torch.stack(delta_bases),
        fallback=fallback,
        unique_patch_counts=patch_counts,
        unique_image_counts=image_counts,
        global_feature_basis=global_feature_basis,
        global_delta_basis=global_delta_basis,
        rank=int(rank),
    )


def prototype_by_mstar(memory_prototype_ids: torch.Tensor, nn_indices: torch.Tensor) -> torch.Tensor:
    return memory_prototype_ids.long().cpu()[nn_indices.long().cpu()]


def prototype_by_topk_vote_k5(
    memory_prototype_ids: torch.Tensor, topk_indices: torch.Tensor
) -> torch.Tensor:
    if topk_indices.shape[-1] != 5:
        raise ValueError("topk vote protocol is fixed to k=5")
    votes = memory_prototype_ids.long().cpu()[topk_indices.long().cpu()]
    flat = votes.reshape(-1, 5)
    winners = []
    for row in flat:
        counts = torch.bincount(row)
        winners.append(int(torch.nonzero(counts == counts.max(), as_tuple=True)[0][0]))
    return torch.tensor(winners, dtype=torch.long).reshape(votes.shape[:-1])


def conditional_residual(
    deviation: torch.Tensor, bases: torch.Tensor, prototype_ids: torch.Tensor
) -> torch.Tensor:
    flat = deviation.reshape(-1, deviation.shape[-1]).float().cpu()
    ids = prototype_ids.reshape(-1).long().cpu()
    output = torch.empty(flat.shape[0], dtype=torch.float32)
    for prototype in torch.unique(ids).tolist():
        mask = ids == int(prototype)
        output[mask] = residual_norm(flat[mask], bases[int(prototype)])
    return output.reshape(deviation.shape[:-1])


def score_conditional_deviation(
    deviation: torch.Tensor,
    model: PrototypeModel,
    prototype_ids: torch.Tensor,
) -> dict[str, torch.Tensor]:
    return {
        "D1_Proto": conditional_residual(
            deviation, model.feature_bases, prototype_ids
        ),
        "D3_NVSProto": conditional_residual(
            deviation, model.delta_bases, prototype_ids
        ),
    }
