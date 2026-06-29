from __future__ import annotations

from typing import Mapping

import numpy as np
import torch
import torch.nn.functional as F

from .memory import MemoryBuildResult, build_memory
from .pipeline import _cosine_topk
from .protocol import CalibrationState, fit_calibration


class AugMemDetector:
    """Cosine-NN detector backed only by an augmented normal memory bank."""

    def __init__(
        self,
        *,
        memory_strategy: str,
        memory_capacity: int,
        seed: int,
        compute_device: torch.device | str,
        query_chunk_size: int,
        bank_chunk_size: int,
        candidate_size: int = 50_000,
        kcenter_chunk_size: int = 8192,
    ) -> None:
        self.memory_strategy = str(memory_strategy)
        self.memory_capacity = int(memory_capacity)
        self.seed = int(seed)
        self.compute_device = torch.device(compute_device)
        self.query_chunk_size = int(query_chunk_size)
        self.bank_chunk_size = int(bank_chunk_size)
        self.candidate_size = int(candidate_size)
        self.kcenter_chunk_size = int(kcenter_chunk_size)
        self.memory_result: MemoryBuildResult | None = None
        self.calibrations: dict[str, CalibrationState] = {}

    def fit(self, candidates: torch.Tensor) -> "AugMemDetector":
        values = F.normalize(
            candidates.reshape(-1, candidates.shape[-1])
            .float()
            .to(self.compute_device, non_blocking=True),
            dim=-1,
        )
        self.memory_result = build_memory(
            values,
            strategy=self.memory_strategy,
            capacity=self.memory_capacity,
            seed=self.seed,
            candidate_size=self.candidate_size,
            chunk_size=self.kcenter_chunk_size,
        )
        return self

    def _require_fitted(self) -> MemoryBuildResult:
        if self.memory_result is None:
            raise RuntimeError("AugMem detector has not been fitted")
        return self.memory_result

    def score_patch_features(self, query_features: torch.Tensor) -> dict[str, torch.Tensor]:
        memory = self._require_fitted()
        shape = query_features.shape
        query = F.normalize(
            query_features.reshape(-1, shape[-1])
            .float()
            .to(self.compute_device, non_blocking=True),
            dim=-1,
        )
        similarities, _ = _cosine_topk(
            query,
            memory.memory_bank,
            1,
            self.query_chunk_size,
            self.bank_chunk_size,
        )
        return {"D0_NN": (1.0 - similarities[:, 0]).reshape(shape[:-1]).float()}

    def calibrate(
        self,
        calibration_features: torch.Tensor,
        image_quantile: float = 0.95,
        mad_epsilon: float = 1.0e-6,
    ) -> Mapping[str, CalibrationState]:
        scores = self.score_patch_features(calibration_features)["D0_NN"]
        self.calibrations = {
            "D0_NN": fit_calibration(
                scores.detach().cpu().numpy(),
                image_quantile=image_quantile,
                mad_epsilon=mad_epsilon,
                scope="identity_calibration",
            )
        }
        return dict(self.calibrations)

    def normalize_scores(
        self, scores: Mapping[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        if "D0_NN" not in self.calibrations:
            raise RuntimeError("AugMem requires identity calibration before evaluation")
        values = scores["D0_NN"]
        normalized = self.calibrations["D0_NN"].normalize(
            values.detach().cpu().numpy()
        )
        return {
            "D0_NN": torch.from_numpy(np.asarray(normalized))
            .to(values.device, non_blocking=True)
            .float()
        }

    def method_names(self) -> tuple[str, ...]:
        return ("D0_NN",)

    def sr_weight_rows(self) -> list[dict]:
        return []

    def state_summary(self) -> dict:
        memory = self._require_fitted()
        return {
            "detector": "AugMemDetector",
            "compute_device": str(self.compute_device),
            "methods": ["D0_NN"],
            "memory_entries": memory.capacity,
            "memory_strategy": memory.strategy,
            "memory_algorithm": memory.algorithm,
            "memory_build_seconds": memory.build_seconds,
            "candidate_indices": memory.candidate_indices.detach().cpu().tolist(),
            "selected_memory_indices": memory.selected_memory_indices.detach().cpu().tolist(),
            "candidate_size": self.candidate_size,
        }
