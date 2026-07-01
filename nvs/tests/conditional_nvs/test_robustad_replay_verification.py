from __future__ import annotations

import pytest

from nvs.conditional_nvs.robustad_replay_verification import (
    verify_replayed_metrics,
)


def _row(value: float) -> dict:
    return {
        "category": "PCB",
        "domain": "target",
        "shift": "lighting",
        "method": "D2_NVSGlobal",
        "image_AUROC": value,
        "pixel_AUROC": value - 0.1,
        "pixel_AUPR": value - 0.2,
    }


def test_replay_verification_accepts_numerical_tolerance() -> None:
    checked = verify_replayed_metrics([_row(0.9000001)], [_row(0.9)])
    assert len(checked) == 3
    assert all(row["within_tolerance"] for row in checked)


def test_replay_verification_rejects_protocol_drift() -> None:
    with pytest.raises(ValueError, match="does not reproduce"):
        verify_replayed_metrics([_row(0.8)], [_row(0.9)])
