from __future__ import annotations

import json

import numpy as np
import pytest

from nvs.conditional_nvs.protocol import (
    completion_is_valid,
    fit_calibration,
    protocol_metadata,
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
