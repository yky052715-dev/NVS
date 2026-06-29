from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from nvs.conditional_nvs.locked_validation_summary import (
    DEV5,
    METHODS,
    SEEDS,
    VALIDATION10,
    run_summary,
)
from nvs.conditional_nvs.perturbed_cache_audit import (
    audit_fit_transform_cache,
    populate_missing_fit_transform_cache,
)
from nvs.conditional_nvs.transforms import FIT_TRANSFORMS, transform_name


TRANSFORMS = (
    "identity_0", "noise_5", "jpeg_70", "jpeg_50",
    "gamma_0p7", "gamma_1p3",
    "rgb_gain_1p1_1_0p9", "rgb_gain_0p9_1_1p1",
)


def _write_comparison(path: Path, categories: set[str], seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    values = {"D0_NN": 0.80, "AugMem_K10": 0.85, "D2_NVSGlobal": 0.90}
    for category in sorted(categories):
        for transform in TRANSFORMS:
            for method in METHODS:
                value = values[method] - (0.01 if transform != "identity_0" else 0.0)
                rows.append(
                    {
                        "category": category,
                        "seed": seed,
                        "transform": transform,
                        "method": method,
                        "memory_entries": 10000,
                        "pixel_AUROC": value,
                        "pixel_AUPR": value - 0.1,
                        "pixel_AUPRO": value - 0.02,
                        "pixel_F1_calibrated": value - 0.2,
                        "pixel_F1_oracle": value - 0.15,
                        "Recall": value - 0.1,
                        "small_recall": value - 0.12,
                        "normal_image_fp_rate": 1.0 - value,
                    }
                )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_locked_configs_freeze_validation10_protocol() -> None:
    root = Path("nvs/configs/conditional_nvs")
    d2 = yaml.safe_load((root / "validation10_d2_locked.yaml").read_text())
    augmem = yaml.safe_load((root / "validation10_augmem_locked.yaml").read_text())
    expected = sorted(VALIDATION10)
    assert sorted(d2["data"]["categories"]) == expected
    assert sorted(augmem["data"]["categories"]) == expected
    assert d2["report_methods"] == ["D0_NN", "D2_NVSGlobal"]
    assert augmem["result_method_aliases"] == {"D0_NN": "AugMem_K10"}
    assert d2["memory"]["protocol"] == augmem["memory"]["protocol"] == "M_K10"
    assert len(d2["fit_transforms"]) == len(augmem["fit_transforms"]) == 13
    assert len(d2["robustness"]["transforms"]) == 7
    assert d2["fusion"]["enabled"] is False
    assert d2["postprocess"]["enabled"] is False
    assert d2["validation_label"] == "locked_cross_category_coverage"


def test_cache_audit_detects_missing_and_complete_fit_transforms(tmp_path) -> None:
    data = tmp_path / "mvtec"
    perturbed = tmp_path / "perturbed"
    original = data / "cable" / "train" / "good" / "000.png"
    original.parent.mkdir(parents=True)
    Image.new("RGB", (8, 8), color=(64, 96, 128)).save(original)
    (data / "cable" / "test").mkdir(parents=True)

    # A same-named test image must never satisfy a train/good cache lookup.
    wrong_split = (
        perturbed / "cable" / transform_name(dict(FIT_TRANSFORMS[0]))
        / "test" / "good" / "000.png"
    )
    wrong_split.parent.mkdir(parents=True)
    Image.new("RGB", (8, 8), color=(255, 0, 0)).save(wrong_split)
    incomplete = audit_fit_transform_cache(data, perturbed, ["cable"])
    assert incomplete["status"] == "incomplete"
    assert all(row["misses"] == 1 for row in incomplete["rows"])

    population = populate_missing_fit_transform_cache(
        data, perturbed, ["cable"], workers=1
    )
    assert population["created"] == 13
    assert population["overwrites_existing"] is False
    complete = audit_fit_transform_cache(data, perturbed, ["cable"])
    assert complete["status"] == "complete"
    assert all(row["hits"] == 1 for row in complete["rows"])


def test_locked_summary_builds_validation10_and_full15(tmp_path) -> None:
    for seed in SEEDS:
        _write_comparison(
            tmp_path / f"dev5_{seed}" / "comparison_metrics.csv", DEV5, seed
        )
        _write_comparison(
            tmp_path / f"validation10_{seed}" / "comparison_metrics.csv",
            VALIDATION10,
            seed,
        )

    output = tmp_path / "summary"
    result = run_summary(
        str(tmp_path / "dev5_{seed}"),
        str(tmp_path / "validation10_{seed}"),
        output,
    )

    validation = result["summaries"]["Validation10_locked"]
    full = result["summaries"]["Full15_coverage"]
    delta = validation["paired_deltas"]["D2_minus_AugMem_K10"]
    assert np.isclose(delta["unseen"]["pixel_AUROC"]["mean"], 0.05)
    assert len(validation["categories"]) == 10
    assert len(full["categories"]) == 15
    assert validation["category_wins"]["AugMem_K10"]["unseen"][
        "pixel_AUROC"
    ]["wins"] == 10
    assert validation["transform_wins"]["AugMem_K10"]["unseen"][
        "pixel_AUROC"
    ]["wins"] == 7
    marker = json.loads(
        (output / "locked_validation_complete.json").read_text(encoding="utf-8")
    )
    assert marker["status"] == "complete"
    assert len(marker["full15_categories"]) == 15
