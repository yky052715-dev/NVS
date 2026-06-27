from __future__ import annotations

import numpy as np

from nvs.conditional_nvs.metrics import (
    average_relative_drop,
    pixel_aupr,
    pixel_aupro,
)


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
