from __future__ import annotations

import pytest

from nvs.conditional_nvs.robustad_failure_summary import (
    aggregate_rows,
    paired_d2_minus_augmem,
)


def _row(seed: int, method: str, value: float) -> dict:
    return {
        "seed": seed,
        "category": "PCB",
        "domain": "target",
        "shift": "lighting",
        "shift_family": "photometric",
        "method": method,
        "image_AUROC": value,
        "image_AUPR": value - 0.1,
        "ARD_image_AUROC": value - 0.9,
        "ARD_image_AUPR": value - 0.8,
    }


def test_summary_keeps_seed_pairing_for_d2_minus_augmem() -> None:
    rows = [
        _row(42, "D2_NVSGlobal", 0.8),
        _row(42, "AugMem_K10", 0.7),
        _row(43, "D2_NVSGlobal", 0.9),
        _row(43, "AugMem_K10", 0.85),
    ]
    paired = paired_d2_minus_augmem(rows)
    assert len(paired) == 1
    assert paired[0]["seeds"] == 2
    assert paired[0]["D2_minus_AugMem_image_AUROC_mean"] == pytest.approx(
        0.075
    )


def test_summary_reports_sample_standard_deviation() -> None:
    rows = [_row(42, "D2_NVSGlobal", 0.8), _row(43, "D2_NVSGlobal", 0.9)]
    summary = aggregate_rows(
        rows, ("category", "domain", "shift", "shift_family", "method")
    )
    assert summary[0]["image_AUROC_mean"] == pytest.approx(0.85)
    assert summary[0]["image_AUROC_std"] == pytest.approx(0.070710678)
