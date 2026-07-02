from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from nvs.conditional_nvs.t0_tail_attribution import (
    ConsistencyAccumulator,
    decompose_patch_energies,
    fit_calibration_views,
    macro_region_statistics,
    paired_bootstrap_delta,
    score_views,
    tail_retention_rows,
)
from nvs.conditional_nvs.t0_tail_summary import attribution_decisions


class _FakePipeline:
    def __init__(self, bank: torch.Tensor, basis: torch.Tensor) -> None:
        bank = F.normalize(bank.float(), dim=-1)
        self.memory_result = SimpleNamespace(memory_bank=bank)
        self.search_bank = bank
        self.prototype_model = SimpleNamespace(global_delta_basis=basis.float())
        self.compute_device = torch.device("cpu")
        self.query_chunk_size = 2
        self.bank_chunk_size = 2
        self.whitener = None

    def _require_fitted(self) -> None:
        return None

    def _normalized(self, values: torch.Tensor) -> torch.Tensor:
        return F.normalize(values.float(), dim=-1)


def test_energy_decomposition_reuses_nn_and_passes_all_identities() -> None:
    pipeline = _FakePipeline(
        torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        torch.tensor([[0.0, 0.0, 1.0]]),
    )
    features = torch.tensor(
        [[[0.8, 0.6, 0.0], [0.0, 0.8, 0.6]]], dtype=torch.float32
    )
    checks = ConsistencyAccumulator()
    result = decompose_patch_energies(
        features, pipeline, image_batch_size=1, checks=checks
    )
    assert checks.batches == 1
    assert checks.max_abs_d0_minus_e0 < 1.0e-6
    assert checks.max_abs_d2_minus_sqrt_2e2 < 1.0e-6
    assert checks.max_relative_energy_identity_error < 1.0e-6
    assert torch.allclose(result["e0"], result["eP"] + result["e2"], atol=1e-6)
    assert torch.allclose(
        result["D2_NVSGlobal"], torch.sqrt(2.0 * result["e2"]), atol=1e-6
    )


def test_nonorthonormal_basis_hard_stops_before_aggregation() -> None:
    pipeline = _FakePipeline(
        torch.tensor([[1.0, 0.0]]), torch.tensor([[2.0, 0.0]])
    )
    features = torch.tensor([[[0.8, 0.6]]], dtype=torch.float32)
    with pytest.raises(AssertionError, match="aggregation is forbidden"):
        decompose_patch_energies(features, pipeline)


def test_shared_and_legacy_calibration_use_predeclared_physical_quantities() -> None:
    energy = {
        "e0": torch.tensor([[0.0, 1.0], [2.0, 4.0]]),
        "e2": torch.tensor([[0.0, 0.5], [1.0, 2.0]]),
        "D0_NN": torch.tensor([[0.0, 1.0], [2.0, 4.0]]),
        "D2_NVSGlobal": torch.tensor(
            [[0.0, 1.0], [2.0**0.5, 2.0]]
        ),
    }
    calibrations = fit_calibration_views(
        energy, image_quantile=0.95, mad_epsilon=1.0e-6
    )
    views = score_views(energy, calibrations)
    assert torch.equal(views["raw_energy"]["D0_NN"], energy["e0"])
    assert torch.equal(views["raw_energy"]["D2_NVSGlobal"], energy["e2"])
    assert torch.allclose(
        views["shared_calibration"]["D2_NVSGlobal"],
        torch.clamp(
            (energy["e2"] - calibrations["shared_e0"].median)
            / calibrations["shared_e0"].mad,
            min=0.0,
        ),
    )
    assert calibrations["legacy_D2_NVSGlobal"].median != pytest.approx(
        calibrations["shared_e0"].median
    )


def test_region_statistics_are_per_image_macro_not_pixel_pooled() -> None:
    values = np.asarray([[0.0, 0.0, 0.0, 100.0], [10.0, 10.0, 10.0, 10.0]])
    regions = np.asarray(
        [[False, False, False, True], [True, True, True, True]], dtype=bool
    )
    stats, per_image = macro_region_statistics(values, regions)
    assert len(per_image) == 2
    assert stats["mean"] == pytest.approx(55.0)
    assert np.mean(values[regions]) == pytest.approx(28.0)


def _paired_rows(method: str, offset: float) -> list[dict]:
    return [
        {
            "image_id": f"image-{index}",
            "label": index % 2,
            "method": method,
            "image_score": float(index % 2) + offset,
            "pixel_AUPR": 0.2 + offset,
        }
        for index in range(8)
    ]


def test_paired_bootstrap_preserves_images_and_reports_signed_delta() -> None:
    result = paired_bootstrap_delta(
        _paired_rows("D0_NN", 0.0),
        _paired_rows("D2_NVSGlobal", -0.1),
        "pixel_AUPR",
        repeats=100,
        seed=42,
    )
    assert result["images"] == 8
    assert result["delta"] == pytest.approx(-0.1)
    assert result["ci_high"] < 0.0


def test_tail_retention_uses_macro_gaps() -> None:
    rows = [
        {
            "category": "PCB",
            "domain": "target",
            "shift": "lighting",
            "view": "raw_energy",
            "method": method,
            "patch_normal_p99": normal,
            "patch_defect_p99": defect,
            "patch_G99": defect - normal,
            "pixel_normal_p99": normal,
            "pixel_defect_p99": defect,
            "pixel_G99": defect - normal,
        }
        for method, normal, defect in (
            ("D0_NN", 1.0, 5.0),
            ("D2_NVSGlobal", 1.0, 3.0),
        )
    ]
    result = tail_retention_rows(rows)
    assert {row["level"] for row in result} == {"patch", "pixel"}
    assert all(row["R_G99"] == pytest.approx(0.5) for row in result)


def test_summary_rule_distinguishes_projection_calibration_mixed_and_t2() -> None:
    def row(
        shift: str,
        comparison: str,
        view: str,
        metric: str,
        low: float,
        high: float,
    ) -> dict:
        return {
            "category": "PCB",
            "domain": "target",
            "shift": shift,
            "comparison": comparison,
            "view": view,
            "metric": metric,
            "delta": (low + high) / 2.0,
            "ci_low": low,
            "ci_high": high,
        }

    rows = []
    for shift in ("projection", "calibration", "mixed", "neither"):
        raw = shift in {"projection", "mixed"}
        calibration = shift in {"calibration", "mixed"}
        rows.extend(
            [
                row(
                    shift,
                    "D2_minus_D0",
                    "raw_energy",
                    "pixel_AUPR",
                    -0.3 if raw else -0.1,
                    -0.1 if raw else 0.1,
                ),
                row(
                    shift,
                    "D2_minus_D0",
                    "raw_energy",
                    "pixel_G99",
                    -0.1,
                    0.1,
                ),
                row(
                    shift,
                    "D2_minus_D0",
                    "legacy_independent",
                    "pixel_F1_calibrated",
                    -0.3 if calibration else -0.1,
                    -0.1 if calibration else 0.1,
                ),
                row(
                    shift,
                    "D2_minus_D0",
                    "legacy_independent",
                    "normal_image_fp",
                    -0.1,
                    0.1,
                ),
                row(
                    shift,
                    "D2_minus_D0",
                    "legacy_independent",
                    "pixel_G99",
                    -0.1,
                    0.1,
                ),
                row(
                    shift,
                    "shared_minus_legacy",
                    "D2_NVSGlobal",
                    "pixel_F1_calibrated",
                    0.1 if calibration else -0.1,
                    0.3 if calibration else 0.1,
                ),
                row(
                    shift,
                    "shared_minus_legacy",
                    "D2_NVSGlobal",
                    "normal_image_fp",
                    -0.1,
                    0.1,
                ),
                row(
                    shift,
                    "shared_minus_legacy",
                    "D2_NVSGlobal",
                    "pixel_G99",
                    -0.1,
                    0.1,
                ),
            ]
        )
    decisions = {row["shift"]: row["decision"] for row in attribution_decisions(rows)}
    assert decisions == {
        "projection": "projection_dominant",
        "calibration": "calibration_dominant",
        "mixed": "mixed",
        "neither": "neither",
    }
