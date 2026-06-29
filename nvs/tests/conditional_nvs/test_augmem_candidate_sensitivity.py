from __future__ import annotations

import pytest

from nvs.conditional_nvs.augmem_candidate_sensitivity import (
    AUGMEM_100K_METHOD,
    AUGMEM_50K_METHOD,
    D2_METHOD,
    _method_summary,
    assemble_sensitivity_rows,
    validate_candidate_sizes,
)


def _row(method: str, transform: str = "identity_0") -> dict[str, str]:
    return {
        "category": "bottle",
        "seed": "42",
        "transform": transform,
        "method": method,
        "memory_entries": "10000",
        "pixel_AUROC": "0.90",
        "pixel_AUPR": "0.80",
        "pixel_AUPRO": "0.70",
        "pixel_F1_calibrated": "0.60",
        "pixel_F1_oracle": "0.65",
        "Recall": "0.50",
        "small_recall": "0.40",
        "normal_image_fp_rate": "0.10",
        "inference_ms_per_image": "1.0",
    }


def test_candidate_sensitivity_requires_exact_50k_and_100k() -> None:
    config_50k = {"memory": {"candidate_size": 50_000}}
    config_100k = {"memory": {"candidate_size": 100_000}}
    assert validate_candidate_sizes(config_50k, config_100k) == {
        "AugMem-50k": 50_000,
        "AugMem-100k": 100_000,
    }
    config_100k["memory"]["candidate_size"] = 99_999
    with pytest.raises(ValueError, match="100000"):
        validate_candidate_sizes(config_50k, config_100k)


def test_sensitivity_assembles_three_equal_capacity_methods() -> None:
    d2 = [_row(D2_METHOD)]
    augmem_50k = [_row("AugMem_K10")]
    augmem_100k = [_row(AUGMEM_100K_METHOD)]

    rows = assemble_sensitivity_rows(d2, augmem_50k, augmem_100k)

    assert {row["method"] for row in rows} == {
        D2_METHOD,
        AUGMEM_50K_METHOD,
        AUGMEM_100K_METHOD,
    }
    assert len(rows) == 3
    assert set(_method_summary(rows)["methods"]) == {
        D2_METHOD,
        AUGMEM_50K_METHOD,
        AUGMEM_100K_METHOD,
    }


def test_sensitivity_rejects_coverage_or_capacity_mismatch() -> None:
    d2 = [_row(D2_METHOD)]
    augmem_50k = [_row("AugMem_K10")]
    augmem_100k = [_row(AUGMEM_100K_METHOD, transform="noise_5")]
    with pytest.raises(ValueError, match="coverage"):
        assemble_sensitivity_rows(d2, augmem_50k, augmem_100k)

    augmem_100k = [_row(AUGMEM_100K_METHOD)]
    augmem_100k[0]["memory_entries"] = "9999"
    with pytest.raises(ValueError, match="exactly 10000"):
        assemble_sensitivity_rows(d2, augmem_50k, augmem_100k)
