from __future__ import annotations

import numpy as np
import pytest
import torch

from nvs.conditional_nvs.cli import _validate_core_config
from nvs.conditional_nvs.distance_ablation import (
    DISTANCE_BRANCHES,
    DistanceAblationDetector,
    distance_concentration_and_hubness,
)
from nvs.conditional_nvs.distance_ablation_summary import gate_decision
from nvs.conditional_nvs.transforms import FIT_TRANSFORMS
from nvs.conditional_nvs.whitening import fit_ledoit_wolf_whitener


def test_ledoit_wolf_whitening_is_finite_for_anisotropic_features() -> None:
    generator = torch.Generator().manual_seed(4)
    base = torch.randn(80, 5, generator=generator)
    values = torch.stack(
        [base[:, 0], 0.01 * base[:, 1], base[:, 0] + 1.0e-4 * base[:, 2], base[:, 3], base[:, 4]],
        dim=1,
    )
    whitener = fit_ledoit_wolf_whitener(values)
    transformed = whitener.transform(values)
    assert torch.isfinite(transformed).all()
    assert 0.0 <= whitener.shrinkage <= 1.0
    assert whitener.eigenvalue_floor > 0.0
    assert whitener.summary()["empirical_spectrum"]["condition_number"] > 1.0


def test_ledoit_wolf_matches_independent_formula_and_mahalanobis_distance() -> None:
    generator = torch.Generator().manual_seed(17)
    values = torch.randn(120, 7, generator=generator)
    values[:, 1] *= 0.03
    whitener = fit_ledoit_wolf_whitener(values)
    array = values.numpy().astype(np.float64)
    array -= array.mean(axis=0)
    samples, dimensions = array.shape
    empirical = array.T @ array / samples
    mu = np.trace(empirical) / dimensions
    delta = ((empirical - mu * np.eye(dimensions)) ** 2).sum() / dimensions
    squared = array**2
    beta_raw = (
        (squared.T @ squared / samples - empirical**2).sum()
        / (dimensions * samples)
    )
    expected = min(beta_raw, delta) / delta
    assert whitener.shrinkage == pytest.approx(expected, abs=2.0e-5)
    difference = values[3] - values[8]
    transformed = whitener.transform(values[[3, 8]])
    whitened_distance = (transformed[0] - transformed[1]).double().square().sum()
    inverse = torch.cholesky_inverse(whitener.cholesky.double())
    mahalanobis = difference.double() @ inverse @ difference.double()
    assert whitened_distance == pytest.approx(float(mahalanobis), rel=2.0e-5)



def test_distance_detector_shares_selection_indices_and_calibrates_independently() -> None:
    generator = torch.Generator().manual_seed(9)
    memory = torch.randn(12, 4, 6, generator=generator)
    calibration = torch.randn(3, 4, 6, generator=generator)
    detector = DistanceAblationDetector(
        memory_capacity=12,
        seed=42,
        compute_device="cpu",
        query_chunk_size=16,
        bank_chunk_size=16,
        candidate_size=30,
        kcenter_chunk_size=16,
        diagnostic_queries=8,
        diagnostic_query_chunk_size=4,
    ).fit(memory)
    detector.calibrate(calibration)
    scores = detector.score_patch_features(calibration)
    state = detector.state_summary()
    assert set(scores) == set(DISTANCE_BRANCHES)
    assert set(detector.calibrations) == set(DISTANCE_BRANCHES)
    assert state["branches"]["E0"]["selected_memory_indices"] == state["branches"]["E2"]["selected_memory_indices"]
    assert state["branches"]["E1"]["selected_memory_indices"] == state["branches"]["E3"]["selected_memory_indices"]
    assert state["whitener_fit_scope"] == "memory_split_shared_candidate_pool"
    assert set(state["distance_diagnostics"]) == set(DISTANCE_BRANCHES)


def test_distance_diagnostics_report_concentration_and_hubness() -> None:
    queries = torch.tensor([[0.0, 0.0], [0.1, 0.0], [4.0, 4.0]])
    bank = torch.tensor([[0.0, 0.0], [1.0, 0.0], [4.0, 4.0], [8.0, 8.0]])
    result = distance_concentration_and_hubness(
        queries, bank, query_chunk_size=2, bank_chunk_size=2
    )
    assert result["relative_contrast_mean"] >= 0.0
    assert 0.0 <= result["hubness_gini"] <= 1.0
    assert 0.0 <= result["hubness_occupied_fraction"] <= 1.0
    assert result["query_count"] == 3
    assert result["bank_count"] == 4


def _macro(auroc: float, aupro: float, fp: float) -> dict[str, float]:
    return {
        "pixel_AUROC": auroc,
        "pixel_AUPRO": aupro,
        "localization_test_normal_image_positive_rate": fp,
    }


def test_gate_requires_clear_gain_and_risk_guard() -> None:
    passing = gate_decision(
        {
            "E0": _macro(0.90, 0.80, 0.10),
            "E2": _macro(0.9012, 0.8001, 0.10),
            "E3": _macro(0.8995, 0.8030, 0.09),
        }
    )
    assert passing["advance_to_three_seed"]
    assert set(passing["eligible_variants"]) == {"E2", "E3"}
    stopped = gate_decision(
        {
            "E0": _macro(0.90, 0.80, 0.10),
            "E2": _macro(0.9002, 0.8002, 0.095),
            "E3": _macro(0.8980, 0.8030, 0.08),
        }
    )
    assert not stopped["advance_to_three_seed"]
    assert stopped["status"] == "stop_mahalanobis_route"


def test_distance_ablation_config_is_locked() -> None:
    config = {
        "core_experiment": {"enabled": True},
        "fusion": {"enabled": False},
        "postprocess": {"enabled": False},
        "augmem": {"enabled": False},
        "robustness": {"enabled": False},
        "memory": {"protocol": "M_K10"},
        "memory_ablation": {"d0_only": True},
        "distance_ablation": {
            "enabled": True,
            "whitening_estimator": "ledoit_wolf",
        },
        "report_methods": ["E0", "E1", "E2", "E3"],
        "fit_transforms": [dict(item) for item in FIT_TRANSFORMS],
    }
    _validate_core_config(config)
    config["memory"] = {"protocol": "M_R10"}
    with pytest.raises(ValueError, match="fixed to M_K10"):
        _validate_core_config(config)