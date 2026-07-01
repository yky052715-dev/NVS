from __future__ import annotations

import numpy as np
import pytest
import torch

from nvs.conditional_nvs.robustad_failure_diagnostics import (
    add_per_shift_ard,
    masks_to_patch_regions,
    pooling_diagnostics,
    projection_energy_ratio,
    score_distribution,
    shift_family,
)


def test_shift_families_are_predeclared_and_complete() -> None:
    assert shift_family("lighting") == "photometric"
    assert shift_family("white_balancing") == "photometric"
    assert shift_family("shadow") == "photometric"
    assert shift_family("rotation") == "geometry"
    assert shift_family("position") == "geometry"
    assert shift_family("scale") == "geometry"
    assert shift_family("position_rotation") == "geometry"
    assert shift_family("background_1") == "background_global_configuration"
    assert (
        shift_family("background_box_color")
        == "background_global_configuration"
    )
    with pytest.raises(ValueError, match="Unclassified"):
        shift_family("unknown_shift")


def test_projection_energy_ratio_matches_known_subspace() -> None:
    deviation = torch.tensor([[3.0, 4.0], [0.0, 2.0]])
    basis = torch.tensor([[1.0, 0.0]])
    ratio = projection_energy_ratio(deviation, basis)
    assert ratio.tolist() == pytest.approx([9.0 / 25.0, 0.0])


def test_adaptive_max_pool_keeps_small_defect() -> None:
    masks = np.zeros((1, 8, 8), dtype=np.uint8)
    masks[0, 1, 1] = 1
    patches = masks_to_patch_regions(masks, grid_side=2)
    assert patches.shape == (1, 4)
    assert patches.sum() == 1


def test_per_shift_ard_uses_matching_category_method_and_metric() -> None:
    rows = [
        {
            "category": "PCB",
            "method": "D2_NVSGlobal",
            "domain": "source",
            "image_AUROC": 0.9,
            "image_AUPR": 0.8,
        },
        {
            "category": "PCB",
            "method": "D2_NVSGlobal",
            "domain": "target",
            "image_AUROC": 0.7,
            "image_AUPR": 0.85,
        },
    ]
    add_per_shift_ard(rows)
    assert rows[0]["ARD_image_AUROC"] == 0.0
    assert rows[1]["ARD_image_AUROC"] == pytest.approx(-0.2)
    assert rows[1]["ARD_image_AUPR"] == 0.0


def test_piledbags_pooling_is_fixed_diagnostic_not_selection() -> None:
    labels = np.asarray([0, 0, 1, 1])
    patch_scores = np.asarray(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.5, 0.5, 0.5, 0.5],
            [0.6, 0.6, 0.6, 0.6],
        ]
    )
    rows = pooling_diagnostics(labels, patch_scores)
    assert {row["aggregation"] for row in rows} == {
        "max",
        "mean",
        "top_1pct_mean",
        "top_5pct_mean",
        "top_10pct_mean",
    }
    assert all(row["diagnostic_only"] is True for row in rows)
    by_name = {row["aggregation"]: row for row in rows}
    assert by_name["mean"]["image_AUROC"] > by_name["max"]["image_AUROC"]


def test_score_distribution_has_requested_quantiles() -> None:
    result = score_distribution(np.arange(100, dtype=np.float64))
    assert result["count"] == 100
    assert result["mean"] == pytest.approx(49.5)
    assert result["p95"] == pytest.approx(94.05)
    assert result["p99"] == pytest.approx(98.01)
