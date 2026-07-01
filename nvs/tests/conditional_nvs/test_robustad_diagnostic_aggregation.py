from __future__ import annotations

import pytest

from nvs.conditional_nvs.robustad_failure_summary import aggregate_fields


def test_diagnostic_fields_are_aggregated_across_seeds() -> None:
    rows = [
        {
            "seed": 42,
            "category": "PCB",
            "shift": "lighting",
            "method": "D2_NVSGlobal",
            "mean": 1.0,
            "false_positive_rate": 0.2,
        },
        {
            "seed": 43,
            "category": "PCB",
            "shift": "lighting",
            "method": "D2_NVSGlobal",
            "mean": 3.0,
            "false_positive_rate": 0.4,
        },
    ]
    result = aggregate_fields(
        rows,
        ("category", "shift", "method"),
        ("mean", "false_positive_rate"),
    )
    assert len(result) == 1
    assert result[0]["seeds"] == 2
    assert result[0]["mean_mean"] == pytest.approx(2.0)
    assert result[0]["false_positive_rate_mean"] == pytest.approx(0.3)
