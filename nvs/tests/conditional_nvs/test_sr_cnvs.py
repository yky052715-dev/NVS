from __future__ import annotations

import torch

from nvs.conditional_nvs.conditional_subspace import (
    PrototypeModel,
    estimate_sr_weights,
    sr_conditional_residual,
)
from nvs.conditional_nvs.pipeline import ConditionalNVSPipeline, FeatureSplit


def _model_with_weights(weights: torch.Tensor) -> PrototypeModel:
    centers = torch.eye(2)
    labels = torch.tensor([0, 1])
    global_delta = torch.tensor([[1.0, 0.0]])
    local_delta = torch.tensor([[[0.0, 1.0]], [[0.0, 1.0]], [[0.0, 1.0]]])
    feature_bases = local_delta.clone()
    return PrototypeModel(
        centers=centers,
        labels=labels,
        feature_bases=feature_bases,
        delta_bases=local_delta,
        fallback=torch.zeros(3, dtype=torch.bool),
        unique_patch_counts=torch.tensor([10, 10, 10]),
        unique_image_counts=torch.tensor([4, 4, 4]),
        global_feature_basis=global_delta,
        global_delta_basis=global_delta,
        rank=1,
        sr_gain=torch.ones(3),
        sr_instability=torch.zeros(3),
        sr_weights=weights,
    )


def test_sr_projection_shrinks_between_global_and_local_projectors() -> None:
    deviation = torch.tensor([[3.0, 4.0], [3.0, 4.0], [3.0, 4.0]])
    prototypes = torch.tensor([0, 1, 2])
    scores = sr_conditional_residual(
        deviation, _model_with_weights(torch.tensor([0.0, 1.0, 0.25])), prototypes
    )

    assert torch.allclose(scores[0], torch.tensor(4.0))
    assert torch.allclose(scores[1], torch.tensor(3.0))
    assert torch.allclose(scores[2], torch.linalg.norm(torch.tensor([0.75, 3.0])))


def test_sr_weights_positive_gain_and_nonpositive_gain_rules() -> None:
    deltas = torch.zeros(80, 13, 3)
    deltas[..., 0] = 1.0
    labels = torch.zeros(80, dtype=torch.long)
    image_ids = torch.arange(4).repeat_interleave(20)
    fallback = torch.tensor([False])
    local_basis = torch.tensor([[[1.0, 0.0, 0.0]]])

    gain, instability, weights = estimate_sr_weights(
        deltas=deltas,
        labels=labels,
        image_ids=image_ids,
        global_basis=torch.tensor([[0.0, 1.0, 0.0]]),
        local_bases=local_basis,
        fallback=fallback,
        rank=1,
        seed=42,
        bootstrap_repeats=2,
    )
    assert gain.item() > 0.99
    assert instability.item() < 1.0e-4
    assert weights.item() > 0.99

    gain_same, _, weights_same = estimate_sr_weights(
        deltas=deltas,
        labels=labels,
        image_ids=image_ids,
        global_basis=torch.tensor([[1.0, 0.0, 0.0]]),
        local_bases=local_basis,
        fallback=fallback,
        rank=1,
        seed=42,
        bootstrap_repeats=2,
    )
    assert gain_same.item() <= 1.0e-6
    assert weights_same.item() == 0.0

    _, _, weights_fallback = estimate_sr_weights(
        deltas=deltas,
        labels=labels,
        image_ids=image_ids,
        global_basis=torch.tensor([[0.0, 1.0, 0.0]]),
        local_bases=local_basis,
        fallback=torch.tensor([True]),
        rank=1,
        seed=42,
        bootstrap_repeats=2,
    )
    assert weights_fallback.item() == 0.0


def test_pipeline_outputs_and_calibrates_sr_cnvs_when_enabled() -> None:
    generator = torch.Generator().manual_seed(7)
    memory = torch.randn(60, 4, generator=generator)
    original = torch.randn(4, 20, 4, generator=generator)
    transformed = tuple(
        original + 0.01 * (index + 1) * torch.randn(original.shape, generator=generator)
        for index in range(13)
    )
    pipeline = ConditionalNVSPipeline(
        rank=1,
        prototypes=1,
        memory_strategy="random",
        memory_capacity=20,
        seed=42,
        stability_regularization=True,
        sr_bootstrap_repeats=2,
    ).fit(FeatureSplit(memory, original, transformed))
    query = torch.randn(2, 5, 4, generator=torch.Generator().manual_seed(9))

    scores = pipeline.score_patch_features(query)
    assert "SR_CNVS" in scores
    assert scores["SR_CNVS"].shape == (2, 5)
    calibrations = pipeline.calibrate(query)
    assert "SR_CNVS" in calibrations
    normalized = pipeline.normalize_scores(scores)
    assert "SR_CNVS" in normalized
    assert pipeline.sr_weight_rows()