from __future__ import annotations

import numpy as np

from nvs.conditional_nvs.metrics import (
    average_relative_drop,
    binary_f1,
    oracle_f1,
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

def test_optimized_oracle_f1_matches_threshold_bruteforce() -> None:
    rng = np.random.default_rng(17)
    masks = rng.random((3, 9, 7)) > 0.82
    maps = rng.random(masks.shape)
    thresholds = np.unique(
        np.quantile(maps, np.linspace(0.0, 1.0, min(37, maps.size)))
    )
    expected = max(
        ((binary_f1(masks, maps >= threshold), float(threshold)) for threshold in thresholds),
        key=lambda item: item[0],
    )
    actual = oracle_f1(masks, maps, max_thresholds=37)
    assert np.allclose(actual, expected)

def test_optimized_aupro_matches_bruteforce() -> None:
    from scipy import ndimage

    rng = np.random.default_rng(23)
    masks = rng.random((2, 7, 8)) > 0.84
    maps = rng.random(masks.shape)
    max_fpr = 0.30
    thresholds = np.unique(
        np.quantile(maps, np.linspace(0.0, 1.0, min(29, maps.size)))
    )[::-1]
    background = ~masks
    regions = []
    for image_index, mask in enumerate(masks):
        labels, count = ndimage.label(mask)
        regions.extend(
            (image_index, labels == region_index)
            for region_index in range(1, count + 1)
        )
    points = [(0.0, 0.0)]
    for threshold in thresholds:
        prediction = maps >= threshold
        fpr = np.logical_and(prediction, background).sum() / background.sum()
        pro = np.mean(
            [prediction[image_index][region].mean() for image_index, region in regions]
        )
        points.append((float(fpr), float(pro)))
    points.append((1.0, 1.0))
    points.sort(key=lambda pair: pair[0])
    fpr = np.asarray([point[0] for point in points])
    pro = np.asarray([point[1] for point in points])
    unique_fpr = np.unique(fpr)
    max_pro = np.asarray([pro[fpr == value].max() for value in unique_fpr])
    if max_fpr not in unique_fpr:
        interpolated = np.interp(max_fpr, unique_fpr, max_pro)
        keep = unique_fpr < max_fpr
        unique_fpr = np.append(unique_fpr[keep], max_fpr)
        max_pro = np.append(max_pro[keep], interpolated)
    else:
        keep = unique_fpr <= max_fpr
        unique_fpr, max_pro = unique_fpr[keep], max_pro[keep]
    expected = np.trapz(max_pro, unique_fpr) / max_fpr

    assert np.isclose(pixel_aupro(masks, maps, max_fpr=max_fpr, max_thresholds=29), expected)
