from __future__ import annotations

import numpy as np

from nvs.conditional_nvs.metrics import (
    average_relative_drop,
    pixel_aupr,
    pixel_aupro,
)
from nvs.conditional_nvs.robustness import aggregate_ard


def test_aupr_and_aupro_reward_perfect_map() -> None:
    masks = np.zeros((2, 8, 8), dtype=np.uint8)
    masks[:, 2:4, 2:4] = 1
    perfect = masks.astype(np.float64)
    random = np.random.default_rng(2).random(masks.shape)
    assert pixel_aupr(masks, perfect) == 1.0
    assert pixel_aupro(masks, perfect) > pixel_aupro(masks, random)
    assert 0.0 <= pixel_aupro(masks, perfect) <= 1.0


def test_ard_formula_is_nonpositive_and_ignores_improvements() -> None:
    assert average_relative_drop(0.90, [0.85, 0.95, 0.80]) == np.mean(
        [-0.05, 0.0, -0.10]
    )

def test_aggregate_ard_separates_categories_and_seeds() -> None:
    rows = []
    for category, source, target in (
        ("bottle", 0.90, 0.85),
        ("grid", 0.80, 0.70),
    ):
        rows.extend(
            [
                {
                    "category": category,
                    "seed": 42,
                    "method": "D0_NN",
                    "transform": "identity_0",
                    "pixel_AUROC": source,
                },
                {
                    "category": category,
                    "seed": 42,
                    "method": "D0_NN",
                    "transform": "noise_5",
                    "pixel_AUROC": target,
                },
            ]
        )

    summary = aggregate_ard(rows)

    assert len(summary) == 2
    assert [row["category"] for row in summary] == ["bottle", "grid"]
    assert all(row["seed"] == "42" for row in summary)
    assert np.isclose(summary[0]["ARD_pixel_AUROC"], -0.05)
    assert np.isclose(summary[1]["ARD_pixel_AUROC"], -0.10)
