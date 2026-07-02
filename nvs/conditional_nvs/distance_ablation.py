from __future__ import annotations

from time import perf_counter
from typing import Mapping

import numpy as np
import torch
import torch.nn.functional as F

from .memory import MemoryBuildResult, build_memory, shared_candidate_indices
from .pipeline import _euclidean_topk
from .protocol import CalibrationState, fit_calibration, stable_hash
from .whitening import LedoitWolfWhitener, fit_ledoit_wolf_whitener


DISTANCE_BRANCHES = {
    "E0": {"selection_space": "raw_l2", "retrieval_space": "raw_euclidean"},
    "E1": {"selection_space": "ledoit_wolf_whitened", "retrieval_space": "raw_euclidean"},
    "E2": {"selection_space": "raw_l2", "retrieval_space": "mahalanobis"},
    "E3": {"selection_space": "ledoit_wolf_whitened", "retrieval_space": "mahalanobis"},
}


def _gini(values: torch.Tensor) -> float:
    data = values.detach().double().cpu().clamp_min(0.0).sort().values
    total = data.sum()
    if data.numel() == 0 or total <= 0:
        return 0.0
    ranks = torch.arange(1, data.numel() + 1, dtype=torch.float64)
    return float(((2.0 * ranks - data.numel() - 1.0) * data).sum() / (data.numel() * total))


def distance_concentration_and_hubness(
    queries: torch.Tensor,
    bank: torch.Tensor,
    *,
    query_chunk_size: int = 512,
    bank_chunk_size: int = 8192,
    epsilon: float = 1.0e-12,
) -> dict[str, float]:
    """Compute source-normal distance concentration and 1-NN hubness."""

    queries = queries.float()
    bank = bank.float().to(queries.device, non_blocking=True)
    minimum_parts: list[torch.Tensor] = []
    mean_parts: list[torch.Tensor] = []
    index_parts: list[torch.Tensor] = []
    for query_start in range(0, queries.shape[0], int(query_chunk_size)):
        query = queries[query_start : query_start + int(query_chunk_size)]
        minimum = torch.full((query.shape[0],), float("inf"), device=query.device)
        nearest = torch.full((query.shape[0],), -1, dtype=torch.long, device=query.device)
        distance_sum = torch.zeros(query.shape[0], dtype=torch.float64, device=query.device)
        for bank_start in range(0, bank.shape[0], int(bank_chunk_size)):
            distances = torch.cdist(
                query, bank[bank_start : bank_start + int(bank_chunk_size)]
            )
            local_minimum, local_index = distances.min(dim=1)
            replace = local_minimum < minimum
            minimum = torch.where(replace, local_minimum, minimum)
            nearest = torch.where(replace, local_index + bank_start, nearest)
            distance_sum += distances.double().sum(dim=1)
        minimum_parts.append(minimum)
        mean_parts.append((distance_sum / max(1, bank.shape[0])).float())
        index_parts.append(nearest)
    minimum = torch.cat(minimum_parts)
    mean_distance = torch.cat(mean_parts)
    nearest_indices = torch.cat(index_parts)
    relative = (mean_distance - minimum) / minimum.clamp_min(float(epsilon))
    counts = torch.bincount(nearest_indices, minlength=bank.shape[0]).double()
    count_mean = counts.mean()
    count_std = counts.std(unbiased=False)
    skewness = (
        float((((counts - count_mean) / count_std).pow(3)).mean())
        if count_std > 0
        else 0.0
    )
    return {
        "query_count": int(queries.shape[0]),
        "bank_count": int(bank.shape[0]),
        "nearest_distance_mean": float(minimum.mean()),
        "mean_distance_mean": float(mean_distance.mean()),
        "relative_contrast_mean": float(relative.mean()),
        "relative_contrast_median": float(torch.quantile(relative, 0.50)),
        "relative_contrast_p05": float(torch.quantile(relative, 0.05)),
        "relative_contrast_p95": float(torch.quantile(relative, 0.95)),
        "hubness_skewness": skewness,
        "hubness_gini": _gini(counts),
        "hubness_max_fraction": float(counts.max() / max(1, queries.shape[0])),
        "hubness_occupied_fraction": float((counts > 0).double().mean()),
    }


class DistanceAblationDetector:
    """Four-way D0-only selection/retrieval distance ablation."""

    def __init__(
        self,
        *,
        memory_capacity: int,
        seed: int,
        compute_device: torch.device | str,
        query_chunk_size: int,
        bank_chunk_size: int,
        candidate_size: int = 50_000,
        kcenter_chunk_size: int = 8192,
        relative_floor: float = 1.0e-8,
        diagnostic_queries: int = 2048,
        diagnostic_query_chunk_size: int = 512,
    ) -> None:
        self.memory_capacity = int(memory_capacity)
        self.seed = int(seed)
        self.compute_device = torch.device(compute_device)
        self.query_chunk_size = int(query_chunk_size)
        self.bank_chunk_size = int(bank_chunk_size)
        self.candidate_size = int(candidate_size)
        self.kcenter_chunk_size = int(kcenter_chunk_size)
        self.relative_floor = float(relative_floor)
        self.diagnostic_queries = int(diagnostic_queries)
        self.diagnostic_query_chunk_size = int(diagnostic_query_chunk_size)
        self.whitener: LedoitWolfWhitener | None = None
        self.whitener_fit_seconds = 0.0
        self.memory_result: MemoryBuildResult | None = None
        self.branch_results: dict[str, MemoryBuildResult] = {}
        self.search_banks: dict[str, torch.Tensor] = {}
        self.calibrations: dict[str, CalibrationState] = {}
        self.distance_diagnostics: dict[str, dict[str, float]] = {}
        self.diagnostic_query_indices: torch.Tensor | None = None
        self.last_method_inference_seconds: dict[str, float] = {}

    def fit(self, candidates: torch.Tensor) -> "DistanceAblationDetector":
        raw = F.normalize(
            candidates.reshape(-1, candidates.shape[-1])
            .float()
            .to(self.compute_device, non_blocking=True),
            dim=-1,
        ).contiguous()
        candidate_indices = shared_candidate_indices(
            raw.shape[0], self.seed, self.candidate_size
        )
        fit_values = raw[candidate_indices.to(raw.device, non_blocking=True)]
        start = perf_counter()
        self.whitener = fit_ledoit_wolf_whitener(
            fit_values, relative_floor=self.relative_floor
        )
        if self.compute_device.type == "cuda":
            torch.cuda.synchronize(self.compute_device)
        self.whitener_fit_seconds = float(perf_counter() - start)
        whitened = self.whitener.transform(raw).contiguous()
        raw_selection = build_memory(
            raw,
            strategy="kcenter",
            capacity=self.memory_capacity,
            seed=self.seed,
            candidate_indices=candidate_indices,
            candidate_size=self.candidate_size,
            chunk_size=self.kcenter_chunk_size,
            normalize_features=False,
        )
        white_selection = build_memory(
            whitened,
            strategy="kcenter",
            capacity=self.memory_capacity,
            seed=self.seed,
            candidate_indices=candidate_indices,
            candidate_size=self.candidate_size,
            chunk_size=self.kcenter_chunk_size,
            normalize_features=False,
        )
        self.branch_results = {
            "E0": raw_selection,
            "E1": white_selection,
            "E2": raw_selection,
            "E3": white_selection,
        }
        self.memory_result = raw_selection
        for method, result in self.branch_results.items():
            indices = result.selected_memory_indices.to(raw.device, non_blocking=True)
            retrieval = DISTANCE_BRANCHES[method]["retrieval_space"]
            self.search_banks[method] = (
                raw[indices] if retrieval == "raw_euclidean" else whitened[indices]
            ).contiguous()
        return self

    def _require_fitted(self) -> tuple[MemoryBuildResult, LedoitWolfWhitener]:
        if self.memory_result is None or self.whitener is None or not self.search_banks:
            raise RuntimeError("Distance ablation detector has not been fitted")
        return self.memory_result, self.whitener

    def score_patch_features(self, query_features: torch.Tensor) -> dict[str, torch.Tensor]:
        _, whitener = self._require_fitted()
        shape = query_features.shape
        raw = F.normalize(
            query_features.reshape(-1, shape[-1])
            .float()
            .to(self.compute_device, non_blocking=True),
            dim=-1,
        )
        if self.compute_device.type == "cuda":
            torch.cuda.synchronize(self.compute_device)
        whitening_start = perf_counter()
        whitened = whitener.transform(raw)
        if self.compute_device.type == "cuda":
            torch.cuda.synchronize(self.compute_device)
        whitening_seconds = float(perf_counter() - whitening_start)
        output: dict[str, torch.Tensor] = {}
        self.last_method_inference_seconds = {}
        for method, specification in DISTANCE_BRANCHES.items():
            query = raw if specification["retrieval_space"] == "raw_euclidean" else whitened
            if self.compute_device.type == "cuda":
                torch.cuda.synchronize(self.compute_device)
            start = perf_counter()
            distances, _ = _euclidean_topk(
                query,
                self.search_banks[method],
                1,
                self.query_chunk_size,
                self.bank_chunk_size,
            )
            if self.compute_device.type == "cuda":
                torch.cuda.synchronize(self.compute_device)
            retrieval_seconds = float(perf_counter() - start)
            self.last_method_inference_seconds[method] = retrieval_seconds + (
                whitening_seconds
                if specification["retrieval_space"] == "mahalanobis"
                else 0.0
            )
            output[method] = distances[:, 0].sqrt().reshape(shape[:-1]).float()
        return output

    def calibrate(
        self,
        calibration_features: torch.Tensor,
        image_quantile: float = 0.95,
        mad_epsilon: float = 1.0e-6,
    ) -> Mapping[str, CalibrationState]:
        scores = self.score_patch_features(calibration_features)
        self.calibrations = {
            method: fit_calibration(
                values.detach().cpu().numpy(),
                image_quantile=image_quantile,
                mad_epsilon=mad_epsilon,
                scope="identity_calibration",
            )
            for method, values in scores.items()
        }
        self._fit_distance_diagnostics(calibration_features)
        return dict(self.calibrations)

    def _fit_distance_diagnostics(self, calibration_features: torch.Tensor) -> None:
        _, whitener = self._require_fitted()
        raw = F.normalize(
            calibration_features.reshape(-1, calibration_features.shape[-1])
            .float()
            .to(self.compute_device, non_blocking=True),
            dim=-1,
        )
        count = min(max(1, self.diagnostic_queries), raw.shape[0])
        generator = torch.Generator(device="cpu").manual_seed(self.seed + 8_675_309)
        indices = torch.randperm(raw.shape[0], generator=generator)[:count]
        self.diagnostic_query_indices = indices
        raw = raw[indices.to(raw.device, non_blocking=True)]
        whitened = whitener.transform(raw)
        self.distance_diagnostics = {}
        for method, specification in DISTANCE_BRANCHES.items():
            query = raw if specification["retrieval_space"] == "raw_euclidean" else whitened
            self.distance_diagnostics[method] = distance_concentration_and_hubness(
                query,
                self.search_banks[method],
                query_chunk_size=self.diagnostic_query_chunk_size,
                bank_chunk_size=self.bank_chunk_size,
            )

    def normalize_scores(
        self, scores: Mapping[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        missing = set(self.method_names()) - set(self.calibrations)
        if missing:
            raise RuntimeError(f"Missing independent calibration for {sorted(missing)}")
        output = {}
        for method in self.method_names():
            values = scores[method]
            normalized = self.calibrations[method].normalize(
                values.detach().cpu().numpy()
            )
            output[method] = torch.from_numpy(np.asarray(normalized)).to(
                values.device, non_blocking=True
            ).float()
        return output

    def method_names(self) -> tuple[str, ...]:
        return tuple(DISTANCE_BRANCHES)

    def sr_weight_rows(self) -> list[dict]:
        return []

    def state_summary(self) -> dict:
        memory, whitener = self._require_fitted()
        candidate_indices = memory.candidate_indices.detach().cpu()
        branches = {}
        for method, result in self.branch_results.items():
            branches[method] = {
                **DISTANCE_BRANCHES[method],
                "memory_entries": int(result.capacity),
                "memory_algorithm": str(result.algorithm),
                "memory_build_seconds": float(result.build_seconds),
                "selected_memory_indices": result.selected_memory_indices.detach().cpu().tolist(),
                "selected_memory_hash": stable_hash(
                    result.selected_memory_indices.detach().cpu().tolist()
                ),
            }
        query_indices = (
            []
            if self.diagnostic_query_indices is None
            else self.diagnostic_query_indices.tolist()
        )
        return {
            "detector": "DistanceAblationDetector",
            "compute_device": str(self.compute_device),
            "methods": list(self.method_names()),
            "memory_entries": int(memory.capacity),
            "memory_strategy": "kcenter",
            "memory_algorithm": "distance_space_controlled_kcenter",
            "candidate_size": int(candidate_indices.numel()),
            "candidate_indices": candidate_indices.tolist(),
            "candidate_indices_hash": stable_hash(candidate_indices.tolist()),
            "whitener_fit_scope": "memory_split_shared_candidate_pool",
            "whitener_fit_seconds": float(self.whitener_fit_seconds),
            "whitener": whitener.summary(),
            "branches": branches,
            "diagnostic_query_indices": query_indices,
            "diagnostic_query_indices_hash": stable_hash(query_indices),
            "distance_diagnostics": self.distance_diagnostics,
        }