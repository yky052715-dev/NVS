from __future__ import annotations

import json

import numpy as np
import pytest

from nvs.conditional_nvs.protocol import (
    completion_is_valid,
    config_fingerprint,
    fit_calibration,
    protocol_metadata,
    select_memory_protocol,
    split_three_way,
    write_completion,
)
from nvs.conditional_nvs.robustness import frozen_stress_predictions


def test_three_way_split_is_reproducible_and_disjoint() -> None:
    records = [f"image_{index}" for index in range(100)]
    first = split_three_way(records, 42, 42)
    second = split_three_way(records, 42, 42)
    assert first == second
    first.assert_disjoint()
    assert len(first.calibration) == 20
    assert len(first.nvs_fit) == 24
    assert len(first.memory) == 56


def test_calibration_rejects_stress_fit_and_remains_frozen() -> None:
    identity = np.arange(16, dtype=np.float64).reshape(2, 2, 4)
    state = fit_calibration(identity, image_quantile=0.5)
    before = (state.median, state.mad, state.threshold)
    frozen_stress_predictions(identity + 100.0, state)
    assert (state.median, state.mad, state.threshold) == before
    with pytest.raises(ValueError):
        fit_calibration(identity, scope="stress")


def test_completion_requires_protocol_identity(tmp_path) -> None:
    manifest = split_three_way(list(range(20)), 42, 42).manifest()
    expected = protocol_metadata("bottle", 42, manifest, {"rank": 8}, ["D0_NN"])
    marker = tmp_path / "complete.json"
    write_completion(marker, expected)
    assert completion_is_valid(marker, expected)
    changed = dict(expected, seed=43)
    assert not completion_is_valid(marker, changed)

def test_runtime_metadata_does_not_change_config_fingerprint(tmp_path) -> None:
    manifest = split_three_way(list(range(20)), 42, 42).manifest()
    config = {"model": {"name": "dinov2_vits14"}, "data": {}, "_runtime": [1]}
    before = config_fingerprint(config)
    expected = protocol_metadata("bottle", 42, manifest, config, ["D0_NN"])
    marker = tmp_path / "complete.json"
    write_completion(marker, expected)

    config["_runtime"] = [1, 2, 3]
    config["data"] = {"_perturbed_cache_usage": [{"hits": 10}]}
    assert config_fingerprint(config) == before
    assert completion_is_valid(
        marker,
        protocol_metadata("bottle", 42, manifest, config, ["D0_NN"]),
    )


def test_memory_lock_applies_auroc_tolerance_to_every_eligible_protocol() -> None:
    rows = [
        {
            "memory_protocol": protocol,
            "seed": 42,
            "pixel_AUROC": auroc,
            "normal_FP": normal_fp,
            "capacity": capacity,
            "inference_ms": inference_ms,
        }
        for protocol, auroc, normal_fp, capacity, inference_ms in (
            ("M_K30", 0.987228, 0.069442, 30_000, 1.744),
            ("M_K10", 0.985990, 0.037662, 10_000, 0.604),
            ("M_R30", 0.985293, 0.033017, 30_000, 1.742),
            ("M_K5", 0.984794, 0.037662, 5_000, 0.326),
            ("M_R10", 0.983988, 0.051631, 10_000, 0.602),
            ("M_R5", 0.982599, 0.004878, 5_000, 0.326),
        )
    ]

    result = select_memory_protocol(rows, auroc_tolerance=0.002)

    assert result["top_two"] == ["M_R30", "M_K10"]
    assert [row["memory_protocol"] for row in result["ranking"][:3]] == [
        "M_R30",
        "M_K10",
        "M_K30",
    ]
