from __future__ import annotations

import pytest

from nvs.conditional_nvs.cli import _validate_core_config
from nvs.conditional_nvs.memory_partition_audit import (
    _runtime_signature,
    balance_statistics,
)
from nvs.conditional_nvs.transforms import FIT_TRANSFORMS


def test_balance_statistics_identifies_equal_and_skewed_selection() -> None:
    equal = balance_statistics([5, 5, 5, 5])
    skewed = balance_statistics([17, 1, 1, 1])
    assert equal["selected_per_image_cv"] == pytest.approx(0.0)
    assert equal["selected_per_image_gini"] == pytest.approx(0.0)
    assert equal["selected_max_image_fraction"] == pytest.approx(0.25)
    assert skewed["selected_per_image_cv"] > equal["selected_per_image_cv"]
    assert skewed["selected_per_image_gini"] > equal["selected_per_image_gini"]
    assert skewed["selected_max_image_fraction"] == pytest.approx(0.85)


def test_partition_audit_protocols_are_d0_only() -> None:
    config = {
        "core_experiment": {"enabled": True},
        "fusion": {"enabled": False},
        "postprocess": {"enabled": False},
        "augmem": {"enabled": False},
        "memory": {"protocol": "M_MRK10"},
        "memory_ablation": {"d0_only": False},
        "fit_transforms": [dict(item) for item in FIT_TRANSFORMS],
    }
    with pytest.raises(ValueError, match="D0-only"):
        _validate_core_config(config)

def test_runtime_signature_controls_batch_and_chunk_settings() -> None:
    first = {
        "data": {"input_size": 518, "num_workers": 8},
        "model": {"name": "dinov2_vits14", "gpu_batch_size": 16},
        "memory": {
            "protocol": "M_K10",
            "candidate_size": 50_000,
            "gpu_query_chunk_size": 32_768,
            "gpu_bank_chunk_size": 65_536,
            "kcenter_chunk_size": 8192,
        },
        "calibration": {"image_quantile": 0.95},
    }
    protocol_only = {
        **first,
        "memory": {**first["memory"], "protocol": "M_MRK10"},
    }
    changed_chunk = {
        **first,
        "memory": {**first["memory"], "gpu_bank_chunk_size": 131_072},
    }
    assert _runtime_signature(first) == _runtime_signature(protocol_only)
    assert _runtime_signature(first) != _runtime_signature(changed_chunk)
