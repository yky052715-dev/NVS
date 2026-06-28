from __future__ import annotations

import pytest
import torch

from nvs.conditional_nvs.pipeline import (
    CORE_METHODS,
    ConditionalNVSPipeline,
    FeatureSplit,
)


def _fixture() -> FeatureSplit:
    generator = torch.Generator().manual_seed(5)
    memory = torch.randn(40, 4, generator=generator)
    original = torch.randn(4, 20, 4, generator=generator)
    transformed = tuple(
        original + 0.01 * (index + 1) * torch.randn(original.shape, generator=generator)
        for index in range(13)
    )
    return FeatureSplit(memory, original, transformed)


def test_pipeline_outputs_new_fields_and_independent_calibration() -> None:
    pipeline = ConditionalNVSPipeline(
        rank=1,
        prototypes=2,
        memory_strategy="random",
        memory_capacity=20,
        seed=42,
    ).fit(_fixture())
    query = torch.randn(3, 5, 4, generator=torch.Generator().manual_seed(8))
    scores = pipeline.score_patch_features(query)
    assert tuple(scores) == CORE_METHODS
    assert all(score.shape == (3, 5) for score in scores.values())
    calibrations = pipeline.calibrate(query)
    assert set(calibrations) == set(CORE_METHODS)
    assert all(state.fit_scope == "identity_calibration" for state in calibrations.values())


def test_top5_prototype_selection_smoke() -> None:
    pipeline = ConditionalNVSPipeline(
        rank=1,
        prototypes=2,
        memory_strategy="random",
        memory_capacity=20,
        prototype_selection="proto_by_topk_vote_k5",
    ).fit(_fixture())
    scores = pipeline.score_patch_features(torch.randn(2, 3, 4))
    assert scores["D3_NVSProto"].shape == (2, 3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_pipeline_keeps_heavy_state_and_scores_on_gpu() -> None:
    pipeline = ConditionalNVSPipeline(
        rank=1,
        prototypes=2,
        memory_strategy="full",
        memory_capacity=0,
        seed=42,
        compute_device="cuda",
        query_chunk_size=64,
        bank_chunk_size=64,
    ).fit(_fixture())
    query = torch.randn(2, 3, 4)
    scores = pipeline.score_patch_features(query)
    pipeline.calibrate(query)
    normalized = pipeline.normalize_scores(scores)

    assert pipeline.memory_result is not None
    assert pipeline.prototype_model is not None
    assert pipeline.memory_result.memory_bank.is_cuda
    assert pipeline.prototype_model.centers.is_cuda
    assert pipeline.prototype_model.delta_bases.is_cuda
    assert all(value.is_cuda and torch.isfinite(value).all() for value in scores.values())
    assert all(value.is_cuda for value in normalized.values())
