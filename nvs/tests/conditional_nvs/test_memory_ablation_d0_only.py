from __future__ import annotations

import pytest
import torch

from nvs.conditional_nvs.augmem import AugMemDetector
from nvs.conditional_nvs.cli import _validate_core_config
from nvs.conditional_nvs.pipeline import ConditionalNVSPipeline, FeatureSplit
from nvs.conditional_nvs.transforms import FIT_TRANSFORMS


def _config() -> dict:
    return {
        "core_experiment": {"enabled": True},
        "fusion": {"enabled": False},
        "postprocess": {"enabled": False},
        "augmem": {"enabled": False},
        "memory_ablation": {"d0_only": True},
        "report_methods": ["D0_NN"],
        "fit_transforms": [dict(item) for item in FIT_TRANSFORMS],
    }


def test_d0_only_memory_ablation_config_is_strict() -> None:
    config = _config()
    _validate_core_config(config)
    config["report_methods"] = ["D0_NN", "D2_NVSGlobal"]
    with pytest.raises(ValueError, match="report only D0_NN"):
        _validate_core_config(config)

    config = _config()
    config["augmem"]["enabled"] = True
    with pytest.raises(ValueError, match="cannot enable AugMem"):
        _validate_core_config(config)


def test_memory_only_detector_preserves_pipeline_d0_definition() -> None:
    generator = torch.Generator().manual_seed(42)
    memory = torch.randn(30, 4, generator=generator)
    original = torch.randn(4, 5, 4, generator=generator)
    transformed = tuple(original + 0.01 * index for index in range(13))
    query = torch.randn(3, 6, 4, generator=generator)

    full = ConditionalNVSPipeline(
        rank=1,
        prototypes=2,
        memory_strategy="kcenter",
        memory_capacity=10,
        seed=42,
        query_chunk_size=32,
        bank_chunk_size=32,
    ).fit(FeatureSplit(memory, original, transformed))
    memory_only = AugMemDetector(
        memory_strategy="kcenter",
        memory_capacity=10,
        seed=42,
        compute_device="cpu",
        query_chunk_size=32,
        bank_chunk_size=32,
    ).fit(memory)

    assert full.memory_result is not None
    assert memory_only.memory_result is not None
    assert torch.equal(
        full.memory_result.selected_memory_indices,
        memory_only.memory_result.selected_memory_indices,
    )
    assert torch.allclose(
        full.score_patch_features(query)["D0_NN"],
        memory_only.score_patch_features(query)["D0_NN"],
    )
