"""Post-hoc failure attribution for D3_NVSProto.

This module deliberately does not alter the core D0-D3 training or scoring
pipeline. Ground-truth masks are accepted only by diagnostic functions that
operate after the fitted state has been frozen.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .conditional_subspace import (
    PrototypeModel,
    assign_prototypes,
    prototype_by_mstar,
    prototype_by_topk_vote_k5,
)
from .protocol import fit_calibration, split_three_way
from .subspace import fit_uncentered_basis, residual_norm
from .transforms import FIT_TRANSFORMS, transform_name


EPS = 1.0e-12


def explained_energy(
    values: torch.Tensor,
    basis: torch.Tensor,
    eps: float = EPS,
) -> torch.Tensor:
    """Return ||U^T x||^2 / (||x||^2 + eps) for row-oriented U."""

    values = values.float()
    basis = basis.float()
    coefficients = values @ basis.T
    numerator = coefficients.square().sum(dim=-1)
    denominator = values.square().sum(dim=-1) + float(eps)
    return numerator / denominator


def conditional_explained_energy(
    values: torch.Tensor,
    bases: torch.Tensor,
    prototype_ids: torch.Tensor,
    eps: float = EPS,
) -> torch.Tensor:
    flat = values.reshape(-1, values.shape[-1]).float()
    ids = prototype_ids.reshape(-1).long()
    selected = bases.float()[ids]
    coefficients = torch.einsum("nd,nrd->nr", flat, selected)
    numerator = coefficients.square().sum(dim=-1)
    denominator = flat.square().sum(dim=-1) + float(eps)
    return (numerator / denominator).reshape(values.shape[:-1])


def partial_residual_norm(
    values: torch.Tensor,
    basis: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """Compute ||x - alpha U U^T x||."""

    alpha = float(alpha)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    values = values.float()
    projected = (values @ basis.float().T) @ basis.float()
    return torch.linalg.norm(values - alpha * projected, dim=-1)


def conditional_partial_residual_norm(
    values: torch.Tensor,
    bases: torch.Tensor,
    prototype_ids: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    alpha = float(alpha)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    flat = values.reshape(-1, values.shape[-1]).float()
    ids = prototype_ids.reshape(-1).long()
    selected = bases.float()[ids]
    coefficients = torch.einsum("nd,nrd->nr", flat, selected)
    projected = torch.einsum("nr,nrd->nd", coefficients, selected)
    result = torch.linalg.norm(flat - alpha * projected, dim=-1)
    return result.reshape(values.shape[:-1])


def wrong_prototype_ids(
    own_ids: torch.Tensor,
    prototype_count: int,
    seed: int,
) -> torch.Tensor:
    """Choose a deterministic random prototype that is never the own id."""

    if int(prototype_count) < 2:
        raise ValueError("wrong-prototype control requires at least two prototypes")
    ids = own_ids.long().cpu()
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    offsets = torch.randint(
        1,
        int(prototype_count),
        ids.shape,
        generator=generator,
        dtype=torch.long,
    )
    return (ids + offsets) % int(prototype_count)


def routing_agreement(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.reshape(-1).long()
    right = right.reshape(-1).long()
    if left.numel() != right.numel() or left.numel() == 0:
        raise ValueError("Routing vectors must have the same non-zero length")
    return float((left == right).float().mean())


def adaptive_patch_mask(
    masks: torch.Tensor | np.ndarray,
    grid_side: int,
) -> torch.Tensor:
    """Map pixel masks to the patch grid without dropping small defects."""

    tensor = torch.as_tensor(masks).float()
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(1)
    if tensor.ndim != 4 or tensor.shape[1] != 1:
        raise ValueError("masks must have shape [N,H,W] or [N,1,H,W]")
    pooled = F.adaptive_max_pool2d(tensor, (int(grid_side), int(grid_side)))
    return pooled[:, 0] > 0


def sample_image_ids(
    image_ids: torch.Tensor,
    fraction: float,
    seed: int,
) -> torch.Tensor:
    """Subsample unique images; never sample individual delta rows."""

    if not 0.0 < float(fraction) <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    unique = torch.unique(image_ids.long().cpu(), sorted=True)
    count = min(
        unique.numel(),
        max(1, int(math.ceil(unique.numel() * float(fraction)))),
    )
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    order = torch.randperm(unique.numel(), generator=generator)[:count]
    return unique[order].sort().values


def projector_distance(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    """Normalized projector distance ||UU^T-VV^T||F/sqrt(2r)."""

    reference = reference.float()
    candidate = candidate.float()
    if reference.ndim != 2 or candidate.ndim != 2:
        raise ValueError("Bases must have shape [rank, dimension]")
    if reference.shape != candidate.shape:
        raise ValueError("Projector distance requires equal basis shapes")
    rank = max(1, int(reference.shape[0]))
    p_ref = reference.T @ reference
    p_candidate = candidate.T @ candidate
    return float(torch.linalg.norm(p_ref - p_candidate) / math.sqrt(2.0 * rank))


def fit_uncentered_basis_covariance(
    values: torch.Tensor,
    rank: int,
    chunk_size: int = 32768,
) -> torch.Tensor:
    """Fit the same uncentered objective through a chunked X^T X eigensolve."""

    values = values.reshape(-1, values.shape[-1]).float().cpu()
    if values.shape[0] < 1:
        raise ValueError("At least one row is required")
    covariance = torch.zeros(
        values.shape[1], values.shape[1], dtype=torch.float32
    )
    for start in range(0, values.shape[0], int(chunk_size)):
        block = values[start : start + int(chunk_size)].float()
        covariance += block.T @ block
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    order = torch.argsort(eigenvalues, descending=True)
    effective_rank = min(max(1, int(rank)), values.shape[1])
    return eigenvectors[:, order[:effective_rank]].T.float().contiguous()

def safe_correlation(left: Sequence[float], right: Sequence[float]) -> float:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if x.size < 2 or np.std(x) <= 0.0 or np.std(y) <= 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _summary(values: torch.Tensor | np.ndarray | Sequence[float]) -> dict[str, float]:
    array = np.asarray(
        values.detach().cpu().numpy() if isinstance(values, torch.Tensor) else values,
        dtype=np.float64,
    ).reshape(-1)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "median": float("nan"),
            "q95": float("nan"),
            "q99": float("nan"),
        }
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "median": float(np.median(array)),
        "q95": float(np.quantile(array, 0.95)),
        "q99": float(np.quantile(array, 0.99)),
    }


def _mad(values: torch.Tensor | np.ndarray) -> float:
    array = np.asarray(
        values.detach().cpu().numpy() if isinstance(values, torch.Tensor) else values,
        dtype=np.float64,
    ).reshape(-1)
    median = np.median(array)
    return float(np.median(np.abs(array - median)))


def _mask_from_ids(values: torch.Tensor, selected_ids: torch.Tensor) -> torch.Tensor:
    mask = torch.zeros(values.shape, dtype=torch.bool)
    for image_id in selected_ids.tolist():
        mask |= values == int(image_id)
    return mask


def basis_stability_rows(
    deltas: torch.Tensor,
    image_ids: torch.Tensor,
    prototype_ids: torch.Tensor,
    model: PrototypeModel,
    repeats: int = 5,
    fraction: float = 0.80,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Refit bases on image-level subsamples with fixed prototype assignment."""

    if deltas.ndim != 3:
        raise ValueError("deltas must have shape [patch, transform, dimension]")
    if deltas.shape[0] != image_ids.numel() or image_ids.numel() != prototype_ids.numel():
        raise ValueError("Patch/image/prototype counts must match")
    rows: list[dict[str, Any]] = []
    rank = int(model.rank)
    for repeat in range(int(repeats)):
        sampled = sample_image_ids(image_ids, fraction, int(seed) + repeat)
        patch_mask = _mask_from_ids(image_ids.long().cpu(), sampled)
        global_basis = fit_uncentered_basis_covariance(
            deltas[patch_mask].reshape(-1, deltas.shape[-1]),
            rank,
        )
        rows.append(
            {
                "repeat": repeat,
                "scope": "global",
                "prototype": -1,
                "sampled_image_count": int(sampled.numel()),
                "member_patch_count": int(patch_mask.sum()),
                "distance": projector_distance(
                    model.global_delta_basis, global_basis
                ),
                "valid": True,
            }
        )
        for prototype in range(model.centers.shape[0]):
            members = patch_mask & (prototype_ids.long().cpu() == prototype)
            unique_images = torch.unique(image_ids[members]).numel()
            valid = (
                int(members.sum()) >= rank
                and int(unique_images) >= 2
                and not bool(model.fallback[prototype])
            )
            distance = float("nan")
            if valid:
                candidate = fit_uncentered_basis_covariance(
                    deltas[members].reshape(-1, deltas.shape[-1]),
                    rank,
                )
                distance = projector_distance(
                    model.delta_bases[prototype], candidate
                )
            rows.append(
                {
                    "repeat": repeat,
                    "scope": "prototype",
                    "prototype": prototype,
                    "sampled_image_count": int(unique_images),
                    "member_patch_count": int(members.sum()),
                    "distance": distance,
                    "valid": bool(valid),
                }
            )
    return rows


def prototype_calibration_rows(
    scores: torch.Tensor,
    prototype_ids: torch.Tensor,
    normalized_scores: torch.Tensor,
    category_threshold: float,
    prototype_count: int,
) -> list[dict[str, Any]]:
    raw = scores.reshape(-1).float().cpu()
    normalized = normalized_scores.reshape(-1).float().cpu()
    ids = prototype_ids.reshape(-1).long().cpu()
    rows = []
    for prototype in range(int(prototype_count)):
        mask = ids == prototype
        count = int(mask.sum())
        if count == 0:
            continue
        summary = _summary(raw[mask])
        rows.append(
            {
                "prototype": prototype,
                "patch_count": count,
                "score_median": summary["median"],
                "score_mad": _mad(raw[mask]),
                "score_q95": summary["q95"],
                "score_q99": summary["q99"],
                "category_threshold": float(category_threshold),
                "prototype_fpr": float(
                    (normalized[mask] >= float(category_threshold)).float().mean()
                ),
            }
        )
    return rows


def routing_rows(
    mstar_ids: torch.Tensor,
    vote_ids: torch.Tensor,
    direct_ids: torch.Tensor,
    margins: torch.Tensor,
    patch_types: torch.Tensor,
    positive_predictions: torch.Tensor,
) -> list[dict[str, Any]]:
    """Aggregate hard-routing disagreement without fitting any state."""

    mstar = mstar_ids.reshape(-1).long().cpu()
    vote = vote_ids.reshape(-1).long().cpu()
    direct = direct_ids.reshape(-1).long().cpu()
    margin = margins.reshape(-1).float().cpu()
    types = patch_types.reshape(-1).long().cpu()
    positives = positive_predictions.reshape(-1).bool().cpu()
    if not (
        mstar.numel()
        == vote.numel()
        == direct.numel()
        == margin.numel()
        == types.numel()
        == positives.numel()
    ):
        raise ValueError("Routing diagnostic vectors must align")
    rows: list[dict[str, Any]] = []
    for patch_type, label in ((0, "normal"), (1, "defect")):
        selected = types == patch_type
        if not bool(selected.any()):
            continue
        disagree = mstar[selected] != vote[selected]
        direct_disagree = direct[selected] != mstar[selected]
        selected_margin = margin[selected]
        selected_fp = positives[selected]
        rows.append(
            {
                "row_type": "summary",
                "patch_type": label,
                "count": int(selected.sum()),
                "mstar_vote_agreement": float((~disagree).float().mean()),
                "direct_mstar_agreement": float((~direct_disagree).float().mean()),
                "margin_mean": float(selected_margin.mean()),
                "margin_disagree_mean": float(selected_margin[disagree].mean())
                if bool(disagree.any())
                else float("nan"),
                "margin_agree_mean": float(selected_margin[~disagree].mean())
                if bool((~disagree).any())
                else float("nan"),
                "positive_rate_disagree": float(selected_fp[disagree].float().mean())
                if bool(disagree.any())
                else float("nan"),
                "positive_rate_agree": float(selected_fp[~disagree].float().mean())
                if bool((~disagree).any())
                else float("nan"),
            }
        )
        quantiles = torch.quantile(
            selected_margin,
            torch.tensor([0.0, 0.25, 0.50, 0.75, 1.0]),
        )
        for index in range(4):
            lower, upper = quantiles[index], quantiles[index + 1]
            in_bin = (selected_margin >= lower) & (
                selected_margin <= upper if index == 3 else selected_margin < upper
            )
            if not bool(in_bin.any()):
                continue
            rows.append(
                {
                    "row_type": "margin_bin",
                    "patch_type": label,
                    "margin_bin": index,
                    "margin_lower": float(lower),
                    "margin_upper": float(upper),
                    "count": int(in_bin.sum()),
                    "mstar_vote_disagreement": float(disagree[in_bin].float().mean()),
                    "positive_rate": float(selected_fp[in_bin].float().mean()),
                }
            )
    return rows


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _git_readonly_state(repo: Path) -> dict[str, Any]:
    state: dict[str, Any] = {}
    for key, args in {
        "commit": ["git", "rev-parse", "HEAD"],
        "branch": ["git", "branch", "--show-current"],
        "status_short": ["git", "status", "--short"],
    }.items():
        try:
            result = subprocess.run(
                args,
                cwd=repo,
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
            state[key] = result.stdout.strip()
        except Exception as error:
            state[key] = f"unavailable: {error}"
    return state


def _cached_or_extract(
    category: str,
    key: str,
    records,
    config: dict[str, Any],
    model,
    device: torch.device,
    transform_spec: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, int]:
    """Read a compatible feature cache if present; never writes cache files."""

    from .cli import _features

    cache = config.get("feature_cache", {}) or {}
    root_value = cache.get("root")
    if root_value:
        cache_path = Path(root_value) / category / f"{key}.pt"
        if cache_path.is_file():
            payload = torch.load(cache_path, map_location="cpu")
            expected_paths = [str(record.path) for record in records]
            if payload.get("paths") != expected_paths:
                raise RuntimeError(f"Feature cache path mismatch: {cache_path}")
            return payload["features"].float().cpu(), int(payload["grid_side"])
    return _features(records, config, model, device, transform_spec)


def _retrieval_context(
    features: torch.Tensor,
    pipeline,
) -> dict[str, torch.Tensor]:
    from .pipeline import _cosine_topk

    pipeline._require_fitted()
    query = F.normalize(features.float().cpu(), dim=-1)
    shape = query.shape
    flat = query.reshape(-1, shape[-1])
    similarities, indices = _cosine_topk(
        flat,
        pipeline.memory_result.memory_bank,
        5,
        pipeline.query_chunk_size,
        pipeline.bank_chunk_size,
    )
    nearest = pipeline.memory_result.memory_bank[indices[:, 0]]
    deviation = flat - nearest
    mstar_ids = prototype_by_mstar(
        pipeline.memory_prototype_ids, indices[:, 0]
    )
    vote_ids = prototype_by_topk_vote_k5(
        pipeline.memory_prototype_ids, indices
    )
    direct_similarity = (
        flat
        @ F.normalize(pipeline.prototype_model.centers.float().cpu(), dim=-1).T
    )
    top2 = torch.topk(direct_similarity, k=min(2, direct_similarity.shape[1]), dim=1)
    direct_ids = top2.indices[:, 0]
    margin = (
        top2.values[:, 0] - top2.values[:, 1]
        if top2.values.shape[1] > 1
        else torch.full_like(top2.values[:, 0], float("nan"))
    )
    return {
        "deviation": deviation.reshape(*shape[:-1], shape[-1]),
        "nearest_indices": indices.reshape(*shape[:-1], 5),
        "mstar_ids": mstar_ids.reshape(shape[:-1]),
        "vote_ids": vote_ids.reshape(shape[:-1]),
        "direct_ids": direct_ids.reshape(shape[:-1]),
        "margin": margin.reshape(shape[:-1]),
        "cosine_distance": (1.0 - similarities[:, 0]).reshape(shape[:-1]),
    }


def _aggregate_energy_row(
    category: str,
    transform_spec: Mapping[str, Any],
    prototype: int | str,
    global_energy: torch.Tensor,
    own_energy: torch.Tensor,
    wrong_energy: torch.Tensor,
) -> dict[str, Any]:
    global_summary = _summary(global_energy)
    own_summary = _summary(own_energy)
    wrong_summary = _summary(wrong_energy)
    return {
        "category": category,
        "transform": transform_name(dict(transform_spec)),
        "transform_type": str(transform_spec["name"]),
        "prototype": prototype,
        "count": int(global_energy.numel()),
        "global_mean": global_summary["mean"],
        "global_median": global_summary["median"],
        "proto_mean": own_summary["mean"],
        "proto_median": own_summary["median"],
        "wrong_mean": wrong_summary["mean"],
        "wrong_median": wrong_summary["median"],
        "proto_minus_global": own_summary["mean"] - global_summary["mean"],
        "proto_minus_wrong": own_summary["mean"] - wrong_summary["mean"],
    }


def _normal_delta_diagnostics(
    category: str,
    original: torch.Tensor,
    transformed: Sequence[torch.Tensor],
    model: PrototypeModel,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, float]]:
    originals = F.normalize(original.float().cpu(), dim=-1)
    flat_original = originals.reshape(-1, originals.shape[-1])
    own_ids = assign_prototypes(flat_original, model.centers)
    wrong_ids = wrong_prototype_ids(
        own_ids, model.centers.shape[0], seed
    )
    image_ids = torch.arange(originals.shape[0]).repeat_interleave(
        originals.shape[1]
    )
    energy_rows: list[dict[str, Any]] = []
    all_deltas, all_own, all_wrong, all_global = [], [], [], []
    transform_count = len(transformed)
    for transform_index, (spec, values) in enumerate(
        zip(FIT_TRANSFORMS, transformed)
    ):
        normalized = F.normalize(values.float().cpu(), dim=-1)
        delta = normalized.reshape_as(flat_original) - flat_original
        global_energy = explained_energy(delta, model.global_delta_basis)
        own_energy = conditional_explained_energy(
            delta, model.delta_bases, own_ids
        )
        wrong_energy = conditional_explained_energy(
            delta, model.delta_bases, wrong_ids
        )
        energy_rows.append(
            _aggregate_energy_row(
                category,
                spec,
                "__all__",
                global_energy,
                own_energy,
                wrong_energy,
            )
        )
        for prototype in torch.unique(own_ids).tolist():
            selected = own_ids == int(prototype)
            energy_rows.append(
                _aggregate_energy_row(
                    category,
                    spec,
                    int(prototype),
                    global_energy[selected],
                    own_energy[selected],
                    wrong_energy[selected],
                )
            )
        all_deltas.append(delta)
        all_global.append(global_energy)
        all_own.append(own_energy)
        all_wrong.append(wrong_energy)
    delta_tensor = torch.stack(all_deltas, dim=1)
    flat_delta = delta_tensor.reshape(-1, delta_tensor.shape[-1])
    repeated_own = own_ids.repeat_interleave(transform_count)
    repeated_images = image_ids.repeat_interleave(transform_count)
    unit_delta = F.normalize(flat_delta, dim=-1)
    global_centroid = unit_delta.mean(dim=0)
    global_dispersion = float(
        (unit_delta - global_centroid).square().sum(dim=-1).mean()
    )
    compactness_rows: list[dict[str, Any]] = [
        {
            "category": category,
            "prototype": -1,
            "heldout_patch_count": int(own_ids.numel()),
            "heldout_unique_image_count": int(originals.shape[0]),
            "fit_patch_count": int(model.labels.numel()),
            "fit_unique_image_count": int(model.unique_image_counts.max()),
            "global_dispersion": global_dispersion,
            "within_dispersion": global_dispersion,
            "within_global_ratio": 1.0,
            "cv_explained_energy": float(torch.stack(all_global, dim=1).mean()),
            "cv_residual_energy": float(1.0 - torch.stack(all_global, dim=1).mean()),
        }
    ]
    for prototype in range(model.centers.shape[0]):
        selected = repeated_own == prototype
        if not bool(selected.any()):
            continue
        group = unit_delta[selected]
        centroid = group.mean(dim=0)
        dispersion = float((group - centroid).square().sum(dim=-1).mean())
        original_members = own_ids == prototype
        compactness_rows.append(
            {
                "category": category,
                "prototype": prototype,
                "heldout_patch_count": int(original_members.sum()),
                "heldout_unique_image_count": int(
                    torch.unique(repeated_images[selected]).numel()
                ),
                "fit_patch_count": int(model.unique_patch_counts[prototype]),
                "fit_unique_image_count": int(model.unique_image_counts[prototype]),
                "global_dispersion": global_dispersion,
                "within_dispersion": dispersion,
                "within_global_ratio": dispersion / max(global_dispersion, EPS),
                "cv_explained_energy": float(
                    torch.stack(all_own, dim=1).reshape(-1)[selected].mean()
                ),
                "cv_residual_energy": float(
                    1.0 - torch.stack(all_own, dim=1).reshape(-1)[selected].mean()
                ),
            }
        )
    proto_rows = [row for row in compactness_rows if row["prototype"] != -1]
    correlations = {
        "prototype_size_vs_explained_correlation": safe_correlation(
            [row["fit_patch_count"] for row in proto_rows],
            [row["cv_explained_energy"] for row in proto_rows],
        ),
        "prototype_size_vs_dispersion_correlation": safe_correlation(
            [row["fit_patch_count"] for row in proto_rows],
            [row["within_global_ratio"] for row in proto_rows],
        ),
    }
    return energy_rows, compactness_rows, correlations


def _erasure_rows(
    category: str,
    deviation: torch.Tensor,
    mstar_ids: torch.Tensor,
    defect_mask: torch.Tensor,
    model: PrototypeModel,
) -> tuple[list[dict[str, Any]], torch.Tensor, torch.Tensor]:
    flat = deviation.reshape(-1, deviation.shape[-1])
    ids = mstar_ids.reshape(-1)
    defect = defect_mask.reshape(-1).bool()
    global_energy = explained_energy(flat, model.global_delta_basis)
    proto_energy = conditional_explained_energy(
        flat, model.delta_bases, ids
    ).reshape(-1)
    rows = []
    for patch_type, selected in (
        ("normal", ~defect),
        ("defect", defect),
    ):
        if not bool(selected.any()):
            continue
        for prototype in ["__all__", *torch.unique(ids[selected]).tolist()]:
            current = selected if prototype == "__all__" else selected & (ids == prototype)
            global_summary = _summary(global_energy[current])
            proto_summary = _summary(proto_energy[current])
            rows.append(
                {
                    "category": category,
                    "patch_type": patch_type,
                    "prototype": prototype,
                    "count": int(current.sum()),
                    "global_mean": global_summary["mean"],
                    "global_median": global_summary["median"],
                    "proto_mean": proto_summary["mean"],
                    "proto_median": proto_summary["median"],
                    "proto_minus_global": proto_summary["mean"]
                    - global_summary["mean"],
                }
            )
    return rows, global_energy, proto_energy


def _alpha_rows(
    category: str,
    calibration_context: Mapping[str, torch.Tensor],
    test_context: Mapping[str, torch.Tensor],
    pipeline,
    masks: np.ndarray,
    labels: np.ndarray,
    grid_side: int,
    output_size: int,
    alphas: Sequence[float],
    calibration_config: Mapping[str, Any],
    small_fraction: float,
) -> list[dict[str, Any]]:
    from .cli import patch_scores_to_maps
    from .metrics import evaluate_pixel_metrics

    rows = []
    for basis_kind in ("global", "prototype"):
        for alpha in alphas:
            if basis_kind == "global":
                calibration_scores = partial_residual_norm(
                    calibration_context["deviation"],
                    pipeline.prototype_model.global_delta_basis,
                    alpha,
                )
                test_scores = partial_residual_norm(
                    test_context["deviation"],
                    pipeline.prototype_model.global_delta_basis,
                    alpha,
                )
            else:
                calibration_scores = conditional_partial_residual_norm(
                    calibration_context["deviation"],
                    pipeline.prototype_model.delta_bases,
                    calibration_context["mstar_ids"],
                    alpha,
                )
                test_scores = conditional_partial_residual_norm(
                    test_context["deviation"],
                    pipeline.prototype_model.delta_bases,
                    test_context["mstar_ids"],
                    alpha,
                )
            state = fit_calibration(
                calibration_scores.numpy(),
                image_quantile=float(calibration_config.get("image_quantile", 0.95)),
                mad_epsilon=float(calibration_config.get("mad_epsilon", 1.0e-6)),
                scope="identity_calibration",
            )
            normalized = torch.from_numpy(
                state.normalize(test_scores.numpy())
            ).float()
            maps = patch_scores_to_maps(
                normalized,
                grid_side=grid_side,
                output_size=output_size,
            ).numpy()
            metrics = evaluate_pixel_metrics(
                masks,
                maps,
                labels,
                threshold=state.threshold,
                small_fraction=small_fraction,
            )
            rows.append(
                {
                    "category": category,
                    "basis": basis_kind,
                    "alpha": float(alpha),
                    "post_hoc_diagnostic": True,
                    "calibration_scope": "identity_calibration",
                    "threshold": state.threshold,
                    **metrics,
                }
            )
    return rows


def _category_summary(
    category: str,
    energy_rows: Sequence[Mapping[str, Any]],
    erasure_rows: Sequence[Mapping[str, Any]],
    compactness_rows: Sequence[Mapping[str, Any]],
    routing: Sequence[Mapping[str, Any]],
    calibration_rows: Sequence[Mapping[str, Any]],
    stability_rows: Sequence[Mapping[str, Any]],
    alpha_rows: Sequence[Mapping[str, Any]],
    correlations: Mapping[str, float],
) -> dict[str, Any]:
    energy_all = [
        row for row in energy_rows if str(row["prototype"]) == "__all__"
    ]
    normal_erasure = next(
        row
        for row in erasure_rows
        if row["patch_type"] == "normal" and str(row["prototype"]) == "__all__"
    )
    defect_erasure = next(
        (
            row
            for row in erasure_rows
            if row["patch_type"] == "defect"
            and str(row["prototype"]) == "__all__"
        ),
        None,
    )
    normal_route = next(
        row
        for row in routing
        if row.get("row_type") == "summary" and row.get("patch_type") == "normal"
    )
    proto_stability = [
        float(row["distance"])
        for row in stability_rows
        if row["scope"] == "prototype"
        and bool(row["valid"])
        and math.isfinite(float(row["distance"]))
    ]
    global_stability = [
        float(row["distance"])
        for row in stability_rows
        if row["scope"] == "global"
    ]
    result = {
        "category": category,
        "normal_delta_global_mean": float(
            np.mean([float(row["global_mean"]) for row in energy_all])
        ),
        "normal_delta_proto_mean": float(
            np.mean([float(row["proto_mean"]) for row in energy_all])
        ),
        "normal_delta_wrong_mean": float(
            np.mean([float(row["wrong_mean"]) for row in energy_all])
        ),
        "normal_delta_proto_minus_global": float(
            np.mean([float(row["proto_minus_global"]) for row in energy_all])
        ),
        "normal_delta_proto_minus_wrong": float(
            np.mean([float(row["proto_minus_wrong"]) for row in energy_all])
        ),
        "normal_extra_erasure": float(normal_erasure["proto_minus_global"]),
        "defect_extra_erasure": float(defect_erasure["proto_minus_global"])
        if defect_erasure
        else float("nan"),
        "mstar_vote_disagreement": 1.0
        - float(normal_route["mstar_vote_agreement"]),
        "direct_mstar_disagreement": 1.0
        - float(normal_route["direct_mstar_agreement"]),
        "routing_positive_rate_gap": float(
            normal_route["positive_rate_disagree"]
        )
        - float(normal_route["positive_rate_agree"]),
        "prototype_fpr_std": float(
            np.std([float(row["prototype_fpr"]) for row in calibration_rows])
        ),
        "prototype_fpr_max": float(
            np.max([float(row["prototype_fpr"]) for row in calibration_rows])
        ),
        "prototype_count_fpr_correlation": safe_correlation(
            [float(row["patch_count"]) for row in calibration_rows],
            [float(row["prototype_fpr"]) for row in calibration_rows],
        ),
        "prototype_count_mad_correlation": safe_correlation(
            [float(row["patch_count"]) for row in calibration_rows],
            [float(row["score_mad"]) for row in calibration_rows],
        ),
        "global_basis_distance_mean": float(np.mean(global_stability)),
        "prototype_basis_distance_mean": float(np.mean(proto_stability))
        if proto_stability
        else float("nan"),
        "prototype_basis_distance_std": float(np.std(proto_stability))
        if proto_stability
        else float("nan"),
        "prototype_basis_distance_max": float(np.max(proto_stability))
        if proto_stability
        else float("nan"),
        **dict(correlations),
    }
    for basis in ("global", "prototype"):
        for alpha in (0.8, 1.0):
            row = next(
                item
                for item in alpha_rows
                if item["basis"] == basis
                and abs(float(item["alpha"]) - alpha) < 1.0e-9
            )
            result[f"{basis}_alpha{alpha}_oracle_f1"] = float(
                row["pixel_F1_oracle"]
            )
            result[f"{basis}_alpha{alpha}_recall"] = float(row["Recall"])
    return result


def _aggregate_numeric(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    values = [
        float(row[key])
        for row in rows
        if key in row and math.isfinite(float(row[key]))
    ]
    return float(np.mean(values)) if values else float("nan")


def _diagnostic_conclusion(
    category_rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    thresholds = config.get("evidence_thresholds", {}) or {}
    own_global = _aggregate_numeric(
        category_rows, "normal_delta_proto_minus_global"
    )
    own_wrong = _aggregate_numeric(
        category_rows, "normal_delta_proto_minus_wrong"
    )
    global_stability = _aggregate_numeric(
        category_rows, "global_basis_distance_mean"
    )
    proto_stability = _aggregate_numeric(
        category_rows, "prototype_basis_distance_mean"
    )
    normal_erasure = _aggregate_numeric(category_rows, "normal_extra_erasure")
    defect_erasure = _aggregate_numeric(category_rows, "defect_extra_erasure")
    routing_disagreement = _aggregate_numeric(
        category_rows, "mstar_vote_disagreement"
    )
    routing_fp_gap = _aggregate_numeric(
        category_rows, "routing_positive_rate_gap"
    )
    fpr_std = _aggregate_numeric(category_rows, "prototype_fpr_std")
    alpha_recovery = _aggregate_numeric(
        [
            {
                "recovery": float(row["prototype_alpha0.8_oracle_f1"])
                - float(row["prototype_alpha1.0_oracle_f1"])
            }
            for row in category_rows
        ],
        "recovery",
    )
    evidence = {
        "H1_conditioning_variable": bool(
            own_global
            <= float(thresholds.get("min_own_over_global", 0.02))
            or own_wrong
            <= float(thresholds.get("min_own_over_wrong", 0.02))
        ),
        "H2_basis_instability": bool(
            math.isfinite(proto_stability)
            and proto_stability
            > max(
                float(thresholds.get("min_prototype_distance", 0.10)),
                global_stability
                * float(thresholds.get("stability_ratio", 1.5)),
            )
        ),
        "H3_anomaly_erasure": bool(
            math.isfinite(defect_erasure)
            and defect_erasure - normal_erasure
            > float(thresholds.get("min_excess_defect_erasure", 0.02))
            or alpha_recovery
            > float(thresholds.get("min_alpha_recovery", 0.005))
        ),
        "H4_hard_routing": bool(
            routing_disagreement
            > float(thresholds.get("min_routing_disagreement", 0.10))
            and routing_fp_gap
            > float(thresholds.get("min_routing_fp_gap", 0.01))
        ),
        "H5_calibration_heterogeneity": bool(
            fpr_std > float(thresholds.get("min_prototype_fpr_std", 0.02))
        ),
    }
    supported = [key for key, value in evidence.items() if value]
    if not supported:
        classification = "G. 证据不足"
        decision = "暂缓：保留D3失败结论，补充证据后再决定"
    elif len(supported) > 1:
        classification = "F. 多种原因共同导致"
        decision = "降级：D2作为主方法，D3仅保留为失败分析；修复后需重新预注册"
    else:
        mapping = {
            "H1_conditioning_variable": "A. conditioning variable错误",
            "H2_basis_instability": "B. basis估计不稳定",
            "H3_anomaly_erasure": "D. 完全投影导致异常误删除",
            "H4_hard_routing": "C. hard routing错误",
            "H5_calibration_heterogeneity": "E. 校准异质性",
        }
        classification = mapping[supported[0]]
        decision = "修复：仅针对已定位机制设计新版本，并重新预注册"
    return {
        "classification": classification,
        "decision": decision,
        "evidence": evidence,
        "aggregate": {
            "normal_delta_proto_minus_global": own_global,
            "normal_delta_proto_minus_wrong": own_wrong,
            "global_basis_distance_mean": global_stability,
            "prototype_basis_distance_mean": proto_stability,
            "normal_extra_erasure": normal_erasure,
            "defect_extra_erasure": defect_erasure,
            "mstar_vote_disagreement": routing_disagreement,
            "routing_positive_rate_gap": routing_fp_gap,
            "prototype_fpr_std": fpr_std,
            "prototype_alpha0.8_oracle_f1_recovery": alpha_recovery,
            "prototype_alpha0.8_recall_recovery": _aggregate_numeric(
                [
                    {
                        "recovery": float(row["prototype_alpha0.8_recall"])
                        - float(row["prototype_alpha1.0_recall"])
                    }
                    for row in category_rows
                ],
                "recovery",
            ),
            "prototype_count_fpr_correlation": _aggregate_numeric(
                category_rows, "prototype_count_fpr_correlation"
            ),
            "prototype_count_mad_correlation": _aggregate_numeric(
                category_rows, "prototype_count_mad_correlation"
            ),
            "prototype_size_vs_basis_instability_correlation": _aggregate_numeric(
                category_rows, "prototype_size_vs_basis_instability_correlation"
            ),
            "basis_instability_vs_prototype_fpr_correlation": _aggregate_numeric(
                category_rows, "basis_instability_vs_prototype_fpr_correlation"
            ),
        },
        "thresholds": dict(thresholds),
        "post_hoc_warning": (
            "All diagnostics are post-hoc failure attribution and cannot "
            "override the preregistered D3 failure."
        ),
    }


def _plot_outputs(output_dir: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as error:
        _write_json(
            output_dir / "visualization_error.json",
            {"error": str(error)},
        )
        return

    energy = [
        row
        for row in _read_csv(output_dir / "normal_delta_energy.csv")
        if row.get("prototype") == "__all__"
    ]
    if energy:
        fig, ax = plt.subplots(figsize=(7, 4))
        values = [
            [float(row[key]) for row in energy]
            for key in ("global_mean", "proto_mean", "wrong_mean")
        ]
        ax.boxplot(values, labels=["global", "own prototype", "wrong prototype"])
        ax.set_ylabel("Explained delta energy")
        ax.set_title("Held-out normal delta explanation")
        fig.tight_layout()
        fig.savefig(output_dir / "normal_delta_explained_energy.png", dpi=180)
        plt.close(fig)

    erasure = [
        row
        for row in _read_csv(output_dir / "anomaly_erasure.csv")
        if row.get("prototype") == "__all__"
    ]
    if erasure:
        fig, ax = plt.subplots(figsize=(7, 4))
        labels = ["normal-global", "normal-proto", "defect-global", "defect-proto"]
        values = []
        for patch_type in ("normal", "defect"):
            selected = [row for row in erasure if row["patch_type"] == patch_type]
            values.extend(
                [
                    [float(row["global_mean"]) for row in selected],
                    [float(row["proto_mean"]) for row in selected],
                ]
            )
        ax.boxplot(values, labels=labels)
        ax.set_ylabel("Projected deviation energy")
        ax.tick_params(axis="x", rotation=20)
        ax.set_title("Normal vs defect direction erasure")
        fig.tight_layout()
        fig.savefig(output_dir / "normal_defect_projection_energy.png", dpi=180)
        plt.close(fig)

    compactness = [
        row
        for row in _read_csv(output_dir / "prototype_compactness.csv")
        if row.get("prototype") not in {"-1", ""}
    ]
    stability = [
        row
        for row in _read_csv(output_dir / "basis_stability.csv")
        if row.get("scope") == "prototype"
        and row.get("valid", "").lower() == "true"
    ]
    if compactness and stability:
        distances: dict[tuple[str, str], list[float]] = defaultdict(list)
        for row in stability:
            distances[(row["category"], row["prototype"])].append(
                float(row["distance"])
            )
        x, y = [], []
        for row in compactness:
            key = (row["category"], row["prototype"])
            if key in distances:
                x.append(float(row["fit_patch_count"]))
                y.append(float(np.mean(distances[key])))
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(x, y, s=12, alpha=0.65)
        ax.set_xscale("log")
        ax.set_xlabel("Prototype fit member count")
        ax.set_ylabel("Mean projector distance")
        ax.set_title("Prototype size vs basis instability")
        fig.tight_layout()
        fig.savefig(output_dir / "prototype_size_basis_stability.png", dpi=180)
        plt.close(fig)

    routing = [
        row
        for row in _read_csv(output_dir / "routing_stability.csv")
        if row.get("row_type") == "margin_bin"
        and row.get("patch_type") == "normal"
    ]
    if routing:
        fig, ax = plt.subplots(figsize=(6, 4))
        x = [
            (float(row["margin_lower"]) + float(row["margin_upper"])) / 2.0
            for row in routing
        ]
        y = [float(row["positive_rate"]) for row in routing]
        color = [float(row["mstar_vote_disagreement"]) for row in routing]
        scatter = ax.scatter(x, y, c=color, cmap="viridis", s=35)
        fig.colorbar(scatter, ax=ax, label="m* vs vote disagreement")
        ax.set_xlabel("Prototype top1-top2 cosine margin")
        ax.set_ylabel("D3 positive rate")
        ax.set_title("Routing margin and normal false triggers")
        fig.tight_layout()
        fig.savefig(output_dir / "routing_margin_false_trigger.png", dpi=180)
        plt.close(fig)

    alpha = _read_csv(output_dir / "alpha_sweep.csv")
    if alpha:
        metrics = ("pixel_AUROC", "pixel_AUPR", "pixel_AUPRO", "pixel_F1_oracle")
        fig, axes = plt.subplots(2, 2, figsize=(9, 7))
        for axis, metric in zip(axes.reshape(-1), metrics):
            for basis in ("global", "prototype"):
                points = []
                for value in sorted({float(row["alpha"]) for row in alpha}):
                    selected = [
                        float(row[metric])
                        for row in alpha
                        if row["basis"] == basis
                        and float(row["alpha"]) == value
                    ]
                    points.append((value, float(np.mean(selected))))
                axis.plot(
                    [point[0] for point in points],
                    [point[1] for point in points],
                    marker="o",
                    label=basis,
                )
            axis.set_title(metric)
            axis.set_xlabel("alpha")
        axes[0, 0].legend()
        fig.suptitle("Post-hoc partial-removal diagnostics")
        fig.tight_layout()
        fig.savefig(output_dir / "alpha_sweep_metrics.png", dpi=180)
        plt.close(fig)


def render_failure_report(
    summary: Mapping[str, Any] | None,
    category_rows: Sequence[Mapping[str, Any]],
    path: Path,
) -> None:
    pending = summary is None
    conclusion = {} if pending else dict(summary["conclusion"])
    aggregate = {} if pending else dict(conclusion["aggregate"])
    evidence = {} if pending else dict(conclusion["evidence"])

    def value(key: str) -> str:
        current = aggregate.get(key, float("nan"))
        return f"{float(current):.6f}" if math.isfinite(float(current)) else "待计算"

    classification = (
        "G. 证据不足（等待GPU诊断结果）"
        if pending
        else str(conclusion["classification"])
    )
    decision = (
        "等待诊断数据，不提前判断D3有效"
        if pending
        else str(conclusion["decision"])
    )
    lines = [
        "# D3失败归因实验报告",
        "",
        "> 本报告用于解释既有 D3_NVSProto128_r8 失败，不用于事后调参，",
        "> 不得覆盖原预注册 D3 失败结论。",
        "",
        "## 1. 实验目的",
        "",
        "区分外观条件变量、局部basis稳定性、异常方向误删除、hard routing和原型级校准异质性。",
        "",
        "## 2. D3与IDEAL的关键差异",
        "",
        "IDEAL采用查询相关top-k正常邻域、centered PCA、较低rank和部分投影，且使用异常参考与mask学习判别方向；当前D3采用离线外观原型、人工delta的uncentered SVD、hard routing和完全投影，无异常参考。",
        "",
        "## 3. 外观原型是否能预测delta响应",
        "",
        f"own-prototype 相对 global 的 held-out 解释率增量：`{value('normal_delta_proto_minus_global')}`；相对 wrong-prototype 的增量：`{value('normal_delta_proto_minus_wrong')}`。H1证据：`{evidence.get('H1_conditioning_variable', '待计算')}`。",
        "",
        "## 4. 是否存在异常方向误删除",
        "",
        f"D3相对D2多删除的正常能量：`{value('normal_extra_erasure')}`；多删除的缺陷能量：`{value('defect_extra_erasure')}`。GT仅用于事后分组。H3证据：`{evidence.get('H3_anomaly_erasure', '待计算')}`。",
        "",
        "## 5. hard routing是否不稳定",
        "",
        f"m*与top-5 vote disagreement：`{value('mstar_vote_disagreement')}`；disagreement patch相对agreement patch的正常误触发差：`{value('routing_positive_rate_gap')}`。H4证据：`{evidence.get('H4_hard_routing', '待计算')}`。",
        "",
        "## 6. 局部basis是否比global basis更不稳定",
        "",
        f"global projector distance：`{value('global_basis_distance_mean')}`；prototype projector distance：`{value('prototype_basis_distance_mean')}`；原型规模与不稳定性的相关性：`{value('prototype_size_vs_basis_instability_correlation')}`。H2证据：`{evidence.get('H2_basis_instability', '待计算')}`。",
        "",
        "## 7. prototype分数异质性",
        "",
        f"逐原型FPR标准差：`{value('prototype_fpr_std')}`；占用量与FPR相关性：`{value('prototype_count_fpr_correlation')}`；占用量与MAD相关性：`{value('prototype_count_mad_correlation')}`；basis不稳定性与FPR相关性：`{value('basis_instability_vs_prototype_fpr_correlation')}`。H5证据：`{evidence.get('H5_calibration_heterogeneity', '待计算')}`。",
        "",
        "## 8. alpha部分消除诊断",
        "",
        f"prototype alpha=0.8 相对 alpha=1.0 的 Oracle F1恢复量：`{value('prototype_alpha0.8_oracle_f1_recovery')}`；Recall恢复量：`{value('prototype_alpha0.8_recall_recovery')}`。固定alpha结果均为post-hoc diagnostic，不能成为新主配置。",
        "",
        "## 9. 逐类别结果",
        "",
    ]
    if category_rows:
        headers = [
            "category",
            "normal_delta_proto_minus_global",
            "defect_extra_erasure",
            "mstar_vote_disagreement",
            "prototype_basis_distance_mean",
            "prototype_fpr_std",
        ]
        lines.extend(
            [
                "|" + "|".join(headers) + "|",
                "|" + "|".join(["---"] * len(headers)) + "|",
            ]
        )
        for row in category_rows:
            lines.append(
                "|"
                + "|".join(
                    str(row.get(key, ""))
                    if key == "category"
                    else f"{float(row.get(key, float('nan'))):.6f}"
                    for key in headers
                )
                + "|"
            )
    else:
        lines.append("尚未运行GPU诊断。")
    lines.extend(
        [
            "",
            "## 10. 最终失败归因",
            "",
            f"当前分类：**{classification}**。",
            "",
            "## 11. D3应该修复、降级还是放弃",
            "",
            f"当前建议：**{decision}**。",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")

def _run(args: argparse.Namespace) -> None:
    from nvs.common import load_dinov2

    from .cli import (
        _load_config,
        _masks_and_labels,
        _pipeline,
        patch_scores_to_maps,
    )
    from .datasets import build_mvtec_dataset

    config = _load_config(args.config)
    diagnostics = config["diagnostics"]
    seed = int(config["experiment"].get("seed", 42))
    if args.seed is not None:
        seed = int(args.seed)
    if seed != 42:
        raise ValueError("First-round D3 failure diagnostics are fixed to seed42")
    if args.memory_protocol:
        config["memory"]["protocol"] = args.memory_protocol
    categories = args.categories or list(config["data"]["categories"])
    if args.smoke:
        categories = categories[:1]
        diagnostics["bootstrap_repeats"] = 1
    output_dir = Path(args.output_dir or config["experiment"]["output_dir"])
    if "core_proto128_r8" in str(output_dir).replace("\\", "/"):
        raise ValueError("Diagnostic output must not target the core result directory")
    summary_path = output_dir / "diagnostic_summary.json"
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Diagnostic output directory is non-empty; refusing to overwrite: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    config["experiment"]["seed"] = seed
    config["data"]["categories"] = categories
    resolved = {
        **config,
        "diagnostic_metadata": {
            "model_seed": seed,
            "memory_protocol": config["memory"]["protocol"],
            "prototype_count": config["subspace"]["prototypes"],
            "rank": config["subspace"]["rank"],
            "alphas": diagnostics["alphas"],
            "gt_usage": "post_hoc_diagnostics_only",
            "git": _git_readonly_state(Path(__file__).resolve().parents[2]),
        },
    }
    _write_json(output_dir / "resolved_config.json", resolved)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(args.device)
    model = load_dinov2(
        config["model"]["name"], device, config["model"].get("hub_dir")
    )

    all_outputs: dict[str, list[dict[str, Any]]] = {
        "normal_delta_energy": [],
        "anomaly_erasure": [],
        "prototype_compactness": [],
        "routing_stability": [],
        "prototype_calibration": [],
        "alpha_sweep": [],
        "basis_stability": [],
        "per_category_summary": [],
    }
    for category in categories:
        category_dir = output_dir / category
        category_dir.mkdir(parents=True, exist_ok=True)
        train_records, test_records = build_mvtec_dataset(args.data_root, category)
        split = split_three_way(
            train_records,
            split_seed=seed,
            nvs_split_seed=seed,
            calibration_fraction=0.20,
            nvs_fit_fraction_of_remainder=0.30,
            key=lambda record: str(record.path),
        )
        manifest = split.manifest(key=lambda record: str(record.path))
        _write_json(category_dir / "sample_manifest.json", manifest)

        memory_features, grid_side = _cached_or_extract(
            category, "memory_identity", split.memory, config, model, device
        )
        nvs_original, _ = _cached_or_extract(
            category, "nvs_fit_identity", split.nvs_fit, config, model, device
        )
        nvs_transformed = tuple(
            _cached_or_extract(
                category,
                f"nvs_fit_{transform_name(dict(spec))}",
                split.nvs_fit,
                config,
                model,
                device,
                dict(spec),
            )[0]
            for spec in FIT_TRANSFORMS
        )
        pipeline_memory = memory_features
        reused_core_memory = False
        core_root_value = diagnostics.get("reuse_core_output")
        if core_root_value:
            core_seed_dir = Path(core_root_value) / category / f"seed{seed}"
            core_state_path = core_seed_dir / "state_summary.json"
            core_manifest_path = core_seed_dir / "sample_manifest.json"
            if core_state_path.is_file() and core_manifest_path.is_file():
                core_manifest = json.loads(
                    core_manifest_path.read_text(encoding="utf-8")
                )
                if core_manifest.get("manifest_hash") != manifest["manifest_hash"]:
                    raise RuntimeError(
                        f"Core/diagnostic split mismatch for {category}"
                    )
                core_state = json.loads(
                    core_state_path.read_text(encoding="utf-8")
                )
                selected = torch.tensor(
                    core_state["selected_memory_indices"], dtype=torch.long
                )
                flat_memory = memory_features.reshape(
                    -1, memory_features.shape[-1]
                )
                if selected.numel() == 0 or int(selected.max()) >= flat_memory.shape[0]:
                    raise RuntimeError(
                        f"Invalid core selected_memory_indices for {category}"
                    )
                pipeline_memory = flat_memory[selected]
                reused_core_memory = True
        pipeline = _pipeline(config, seed)
        if reused_core_memory:
            # The selected entries and their order are identical to the core run;
            # only the expensive k-center reconstruction is skipped.
            pipeline.memory_strategy = "full"
            pipeline.memory_capacity = 0
        pipeline.fit(
            __import__(
                "nvs.conditional_nvs.pipeline",
                fromlist=["FeatureSplit"],
            ).FeatureSplit(
                memory=pipeline_memory,
                nvs_fit_original=nvs_original,
                nvs_fit_transformed=nvs_transformed,
            )
        )
        normalized_nvs = F.normalize(nvs_original.float().cpu(), dim=-1)
        flat_nvs = normalized_nvs.reshape(-1, normalized_nvs.shape[-1])
        nvs_deltas = torch.stack(
            [
                F.normalize(values.float().cpu(), dim=-1).reshape_as(flat_nvs)
                - flat_nvs
                for values in nvs_transformed
            ],
            dim=1,
        )
        nvs_image_ids = torch.arange(nvs_original.shape[0]).repeat_interleave(
            nvs_original.shape[1]
        )
        stability_rows = basis_stability_rows(
            nvs_deltas,
            nvs_image_ids,
            pipeline.prototype_model.labels,
            pipeline.prototype_model,
            repeats=int(diagnostics.get("bootstrap_repeats", 5)),
            fraction=float(diagnostics.get("bootstrap_fraction", 0.80)),
            seed=seed,
        )
        for row in stability_rows:
            row["category"] = category
        del nvs_transformed, nvs_original, memory_features, pipeline_memory, nvs_deltas, normalized_nvs, flat_nvs, nvs_image_ids

        calibration_original, _ = _cached_or_extract(
            category,
            "calibration_identity",
            split.calibration,
            config,
            model,
            device,
        )
        calibration_transformed = tuple(
            _cached_or_extract(
                category,
                f"calibration_{transform_name(dict(spec))}",
                split.calibration,
                config,
                model,
                device,
                dict(spec),
            )[0]
            for spec in FIT_TRANSFORMS
        )
        pipeline.calibrate(
            calibration_original,
            image_quantile=float(config["calibration"]["image_quantile"]),
            mad_epsilon=float(config["calibration"]["mad_epsilon"]),
        )
        test_features, test_grid = _cached_or_extract(
            category,
            "test_identity",
            test_records,
            config,
            model,
            device,
        )
        if test_grid != grid_side:
            raise AssertionError("Train/test feature grids differ")
        masks, labels = _masks_and_labels(
            test_records, int(config["data"]["input_size"])
        )
        defect_patch_mask = adaptive_patch_mask(masks, grid_side)

        category_seed = seed + sum(category.encode("utf-8"))
        energy_rows, compactness_rows, compactness_correlations = (
            _normal_delta_diagnostics(
                category,
                calibration_original,
                calibration_transformed,
                pipeline.prototype_model,
                category_seed,
            )
        )
        calibration_context = _retrieval_context(
            calibration_original, pipeline
        )
        test_context = _retrieval_context(test_features, pipeline)
        erasure_rows, _, _ = _erasure_rows(
            category,
            test_context["deviation"],
            test_context["mstar_ids"],
            defect_patch_mask,
            pipeline.prototype_model,
        )

        calibration_raw = pipeline.score_patch_features(calibration_original)
        calibration_normalized = pipeline.normalize_scores(calibration_raw)
        d3_state = pipeline.calibrations["D3_NVSProto"]
        calibration_rows = prototype_calibration_rows(
            calibration_raw["D3_NVSProto"],
            calibration_context["mstar_ids"],
            calibration_normalized["D3_NVSProto"],
            d3_state.threshold,
            pipeline.prototype_model.centers.shape[0],
        )
        for row in calibration_rows:
            row["category"] = category

        test_raw = pipeline.score_patch_features(test_features)
        test_normalized = pipeline.normalize_scores(test_raw)
        test_positive = (
            test_normalized["D3_NVSProto"] >= d3_state.threshold
        )
        patch_types = defect_patch_mask.long()
        route_rows = routing_rows(
            test_context["mstar_ids"],
            test_context["vote_ids"],
            test_context["direct_ids"],
            test_context["margin"],
            patch_types,
            test_positive,
        )
        for row in route_rows:
            row["category"] = category
            row["scope"] = "test"
        original_direct = assign_prototypes(
            F.normalize(
                calibration_original.float().cpu(), dim=-1
            ).reshape(-1, calibration_original.shape[-1]),
            pipeline.prototype_model.centers,
        )
        for spec, transformed in zip(FIT_TRANSFORMS, calibration_transformed):
            transformed_direct = assign_prototypes(
                F.normalize(transformed.float().cpu(), dim=-1).reshape(
                    -1, transformed.shape[-1]
                ),
                pipeline.prototype_model.centers,
            )
            route_rows.append(
                {
                    "category": category,
                    "scope": "heldout_transform",
                    "row_type": "transform_consistency",
                    "patch_type": "normal",
                    "transform": transform_name(dict(spec)),
                    "transform_type": spec["name"],
                    "count": int(original_direct.numel()),
                    "original_transformed_direct_agreement": routing_agreement(
                        original_direct, transformed_direct
                    ),
                }
            )
        del calibration_transformed



        alpha_rows = _alpha_rows(
            category,
            calibration_context,
            test_context,
            pipeline,
            masks,
            labels,
            grid_side,
            int(config["data"]["input_size"]),
            [float(value) for value in diagnostics["alphas"]],
            config["calibration"],
            float(config["metrics"]["small_defect_area_fraction"]),
        )
        # Required numerical regression: alpha=1 equals the existing residual.
        alpha_one_global = partial_residual_norm(
            test_context["deviation"],
            pipeline.prototype_model.global_delta_basis,
            1.0,
        )
        alpha_one_proto = conditional_partial_residual_norm(
            test_context["deviation"],
            pipeline.prototype_model.delta_bases,
            test_context["mstar_ids"],
            1.0,
        )
        if not torch.allclose(
            alpha_one_global,
            test_raw["D2_NVSGlobal"],
            atol=1.0e-5,
            rtol=1.0e-5,
        ):
            raise AssertionError("alpha=1 global score differs from D2")
        if not torch.allclose(
            alpha_one_proto,
            test_raw["D3_NVSProto"],
            atol=1.0e-5,
            rtol=1.0e-5,
        ):
            raise AssertionError("alpha=1 prototype score differs from D3")

        proto_stability_mean: dict[int, float] = {}
        for prototype in range(pipeline.prototype_model.centers.shape[0]):
            values = [
                float(row["distance"])
                for row in stability_rows
                if row["scope"] == "prototype"
                and int(row["prototype"]) == prototype
                and bool(row["valid"])
                and math.isfinite(float(row["distance"]))
            ]
            if values:
                proto_stability_mean[prototype] = float(np.mean(values))
        calibration_fpr = {
            int(row["prototype"]): float(row["prototype_fpr"])
            for row in calibration_rows
        }
        compactness_correlations = {
            **compactness_correlations,
            "prototype_size_vs_basis_instability_correlation": safe_correlation(
                [
                    float(pipeline.prototype_model.unique_patch_counts[prototype])
                    for prototype in sorted(proto_stability_mean)
                ],
                [
                    proto_stability_mean[prototype]
                    for prototype in sorted(proto_stability_mean)
                ],
            ),
            "basis_instability_vs_prototype_fpr_correlation": safe_correlation(
                [
                    proto_stability_mean[prototype]
                    for prototype in sorted(
                        set(proto_stability_mean) & set(calibration_fpr)
                    )
                ],
                [
                    calibration_fpr[prototype]
                    for prototype in sorted(
                        set(proto_stability_mean) & set(calibration_fpr)
                    )
                ],
            ),
        }
        category_summary = _category_summary(
            category,
            energy_rows,
            erasure_rows,
            compactness_rows,
            route_rows,
            calibration_rows,
            stability_rows,
            alpha_rows,
            compactness_correlations,
        )
        category_payloads = {
            "normal_delta_energy": energy_rows,
            "anomaly_erasure": erasure_rows,
            "prototype_compactness": compactness_rows,
            "routing_stability": route_rows,
            "prototype_calibration": calibration_rows,
            "alpha_sweep": alpha_rows,
            "basis_stability": stability_rows,
            "per_category_summary": [category_summary],
        }
        for name, rows in category_payloads.items():
            _write_csv(category_dir / f"{name}.csv", rows)
            all_outputs[name].extend(rows)
        _write_json(
            category_dir / "category_complete.json",
            {
                "status": "complete",
                "category": category,
                "seed": seed,
                "sample_manifest_hash": manifest["manifest_hash"],
                "gt_usage": "post_hoc_diagnostics_only",
                "reused_core_memory_indices": reused_core_memory,
            },
        )

    for name, rows in all_outputs.items():
        _write_csv(output_dir / f"{name}.csv", rows)
    conclusion = _diagnostic_conclusion(
        all_outputs["per_category_summary"], diagnostics
    )
    summary = {
        "status": "complete",
        "seed": seed,
        "categories": categories,
        "memory_protocol": config["memory"]["protocol"],
        "prototype_count": config["subspace"]["prototypes"],
        "rank": config["subspace"]["rank"],
        "alphas": diagnostics["alphas"],
        "conclusion": conclusion,
        "existing_d3_conclusion": "failed_and_unchanged",
        "output_files": [
            "normal_delta_energy.csv",
            "anomaly_erasure.csv",
            "prototype_compactness.csv",
            "routing_stability.csv",
            "prototype_calibration.csv",
            "alpha_sweep.csv",
            "basis_stability.csv",
            "per_category_summary.csv",
        ],
    }
    _write_json(summary_path, summary)
    _plot_outputs(output_dir)
    render_failure_report(
        summary,
        all_outputs["per_category_summary"],
        output_dir / "D3失败归因实验报告.md",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Post-hoc D3 failure attribution diagnostics"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--memory-protocol")
    parser.add_argument("--categories", nargs="+")
    parser.add_argument("--output-dir")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def main() -> None:
    _run(parse_args())


if __name__ == "__main__":
    main()
