from __future__ import annotations

import pytest
import torch

from nvs.conditional_nvs.augmem import AugMemDetector
from nvs.conditional_nvs.augmem_comparison import (
    assemble_comparison_rows,
    validate_augmem_protocol,
)
from nvs.conditional_nvs.cli import _configured_methods
from nvs.conditional_nvs.memory import matched_augmented_memory_candidates


def _config(protocol: str, augmem: bool) -> dict:
    return {
        "experiment": {"seed": 42},
        "data": {
            "categories": ["bottle"],
            "input_size": 518,
            "calibration_fraction": 0.20,
            "nvs_fit_fraction_of_remainder": 0.30,
        },
        "model": {"name": "dinov2_vits14", "hub_dir": None},
        "memory": {"protocol": protocol},
        "calibration": {"mad_epsilon": 1.0e-6, "image_quantile": 0.95},
        "metrics": {"small_defect_area_fraction": 0.01},
        "fit_transforms": [{"name": "noise", "value": index} for index in range(13)],
        "robustness": {
            "enabled": True,
            "transforms": [{"name": "noise", "value": 5}],
        },
        "fusion": {"enabled": False},
        "postprocess": {"enabled": False},
        "augmem": {
            "enabled": augmem,
            "detection_mode": "memory_only",
            "candidate_source": "matched_d2_information",
        },
    }


def test_matched_augmem_uses_exactly_d2_feature_information() -> None:
    memory = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    nvs_original = torch.tensor([[[1.0, 1.0], [1.0, -1.0]]])
    transformed = tuple(nvs_original + float(index + 1) for index in range(13))

    candidates = matched_augmented_memory_candidates(
        memory, nvs_original, transformed
    )

    assert candidates.shape == (30, 2)
    expected_originals = torch.cat([memory, nvs_original], dim=0).reshape(-1, 2)
    assert torch.allclose(
        candidates[:4],
        torch.nn.functional.normalize(expected_originals, dim=-1),
    )
    assert torch.allclose(candidates.norm(dim=-1), torch.ones(30), atol=1.0e-6)


def test_matched_augmem_requires_thirteen_aligned_transforms() -> None:
    original = torch.ones(2, 3, 4)
    with pytest.raises(ValueError, match="exactly 13"):
        matched_augmented_memory_candidates(original, original, [original] * 12)


def test_augmem_protocol_validation_rejects_unmatched_changes() -> None:
    d2 = _config("M_K10", False)
    augmem = _config("M_K10", True)
    assert validate_augmem_protocol(d2, augmem)["status"] == "matched"
    augmem["calibration"]["image_quantile"] = 0.90
    with pytest.raises(ValueError, match="calibration.image_quantile"):
        validate_augmem_protocol(d2, augmem)


def test_comparison_requires_identical_evaluation_coverage() -> None:
    base = {
        "category": "bottle",
        "seed": "42",
        "transform": "identity_0",
        "pixel_AUROC": "0.9",
        "memory_entries": "10000",
    }
    d2 = [
        {**base, "method": "D0_NN"},
        {**base, "method": "D2_NVSGlobal"},
    ]
    augmem = [{**base, "method": "AugMem_K10"}]
    assert len(assemble_comparison_rows(d2, augmem)) == 3
    augmem[0]["transform"] = "noise_5"
    with pytest.raises(ValueError, match="coverage"):
        assemble_comparison_rows(d2, augmem)


def test_augmem_result_alias_does_not_enable_d3_or_sr() -> None:
    config = {
        "report_methods": ["D0_NN"],
        "result_method_aliases": {"D0_NN": "AugMem_K10"},
    }
    assert _configured_methods(config) == ("AugMem_K10",)

def test_augmem_detector_exposes_only_cosine_nn() -> None:
    detector = AugMemDetector(
        memory_strategy="kcenter",
        memory_capacity=3,
        seed=42,
        compute_device="cpu",
        query_chunk_size=16,
        bank_chunk_size=16,
        candidate_size=6,
    ).fit(torch.randn(2, 3, 4))
    query = torch.randn(2, 5, 4)
    scores = detector.score_patch_features(query)
    detector.calibrate(query, image_quantile=0.5)

    assert detector.method_names() == ("D0_NN",)
    assert set(scores) == {"D0_NN"}
    assert set(detector.calibrations) == {"D0_NN"}
    assert detector.memory_result is not None
    assert detector.memory_result.capacity == 3
