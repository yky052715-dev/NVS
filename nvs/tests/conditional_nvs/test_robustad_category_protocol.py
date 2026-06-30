from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from nvs.conditional_nvs.robustad_category_protocol import (
    RobustADCategoryRecord,
    assert_source_only,
    category_protocol_payload,
    evaluation_groups,
    mask_scope,
    parse_category_manifest,
    parse_official_directory,
    robustad_ard_rows,
    robustad_macro_summary,
    source_training,
)


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(path)


def _write_split(
    directory: Path,
    normal_name: str,
    anomaly_name: str,
    with_mask: bool,
) -> None:
    normal = directory / "normal" / normal_name
    anomaly = directory / "anomaly" / anomaly_name
    _write_image(normal)
    _write_image(anomaly)
    rows = [
        {"file_name": f"normal/{normal_name}", "label": 0},
        {"file_name": f"anomaly/{anomaly_name}", "label": 1},
    ]
    if with_mask:
        mask = directory / "masks" / f"{Path(anomaly_name).stem}.png"
        _write_image(mask)
        rows[0]["mask"] = None
        rows[1]["mask"] = f"masks/{mask.name}"
    (directory / "metadata.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows),
        encoding="utf-8",
    )


def test_official_directory_maps_train_test0_and_targets(tmp_path: Path) -> None:
    category_root = tmp_path / "PiledBags"
    train = category_root / "piled_bags_data_dir_train"
    train_rows = []
    for index in range(3):
        path = train / "normal" / f"normal_{index}.png"
        _write_image(path)
        train_rows.append({"file_name": f"normal/{path.name}", "label": 0})
    anomaly = train / "anomaly" / "supervised_only.png"
    _write_image(anomaly)
    train_rows.append({"file_name": f"anomaly/{anomaly.name}", "label": 1})
    (train / "metadata.jsonl").write_text(
        "\n".join(json.dumps(row) for row in train_rows),
        encoding="utf-8",
    )
    for index in range(6):
        _write_split(
            category_root / f"piled_bags_data_dir_test{index}",
            f"normal_{index}.png",
            f"anomaly_{index}.png",
            with_mask=False,
        )

    records = parse_official_directory(
        tmp_path, categories=["PiledBags"], require_files=True
    )
    training = source_training(records, "PiledBags")
    groups = evaluation_groups(records, "PiledBags")

    assert len(training) == 3
    assert all(record.label == 0 for record in training)
    assert ("PiledBags", "source", "source") in groups
    assert ("PiledBags", "target", "lighting") in groups
    assert ("PiledBags", "target", "shadow") in groups
    assert len(groups) == 6
    assert {mask_scope(group) for group in groups.values()} == {"image"}


def test_explicit_manifest_requires_category_and_keeps_target_out_of_fit(
    tmp_path: Path,
) -> None:
    rows = []
    for index in range(3):
        path = tmp_path / f"train_{index}.png"
        _write_image(path)
        rows.append(
            {
                "path": path.name,
                "label": 0,
                "category": "widget",
                "domain": "source",
                "role": "train",
                "shift": "source_train",
            }
        )
    for domain, shift in (("source", "source"), ("target", "lighting")):
        normal = tmp_path / f"{domain}_normal.png"
        anomaly = tmp_path / f"{domain}_anomaly.png"
        mask = tmp_path / f"{domain}_mask.png"
        for path in (normal, anomaly, mask):
            _write_image(path)
        rows.extend(
            [
                {
                    "path": normal.name,
                    "label": 0,
                    "category": "widget",
                    "domain": domain,
                    "role": "evaluation",
                    "shift": shift,
                },
                {
                    "path": anomaly.name,
                    "mask_path": mask.name,
                    "label": 1,
                    "category": "widget",
                    "domain": domain,
                    "role": "evaluation",
                    "shift": shift,
                },
            ]
        )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(rows), encoding="utf-8")

    records = parse_category_manifest(manifest)
    fit = source_training(records, "widget")
    targets = [
        record
        for record in records
        if record.domain == "target" and record.role == "evaluation"
    ]

    assert len(fit) == 3
    assert_source_only(fit)
    with pytest.raises(AssertionError):
        assert_source_only(targets)

    rows[0].pop("category")
    manifest.write_text(json.dumps(rows), encoding="utf-8")
    with pytest.raises(ValueError, match="category"):
        parse_category_manifest(manifest)


def test_partial_masks_are_rejected() -> None:
    group = [
        RobustADCategoryRecord(
            Path("normal.png"), 0, "widget", "target", "evaluation", "shift"
        ),
        RobustADCategoryRecord(
            Path("a.png"),
            1,
            "widget",
            "target",
            "evaluation",
            "shift",
            Path("a_mask.png"),
        ),
        RobustADCategoryRecord(
            Path("b.png"), 1, "widget", "target", "evaluation", "shift"
        ),
    ]
    with pytest.raises(ValueError, match="partial"):
        mask_scope(group)


def test_ard_is_computed_per_category_before_macro() -> None:
    rows = [
        {"category": "a", "method": "D2", "domain": "source", "image_AUROC": 0.9},
        {"category": "a", "method": "D2", "domain": "target", "image_AUROC": 0.8},
        {"category": "a", "method": "D2", "domain": "target", "image_AUROC": 0.95},
        {"category": "b", "method": "D2", "domain": "source", "image_AUROC": 0.7},
        {"category": "b", "method": "D2", "domain": "target", "image_AUROC": 0.6},
    ]
    ard = robustad_ard_rows(rows)
    summary = robustad_macro_summary(rows, ard)

    by_category = {row["category"]: row for row in ard}
    assert by_category["a"]["ARD"] == pytest.approx(-0.05)
    assert by_category["b"]["ARD"] == pytest.approx(-0.10)
    assert summary["D2"]["image_AUROC"]["ARD_category_macro"] == pytest.approx(
        -0.075
    )


def test_protocol_hash_covers_target_sample_list() -> None:
    training = [
        RobustADCategoryRecord(
            Path(f"train_{index}.png"),
            0,
            "widget",
            "source",
            "train",
            "source_train",
        )
        for index in range(3)
    ]
    evaluation = [
        RobustADCategoryRecord(
            Path("source_normal.png"),
            0,
            "widget",
            "source",
            "evaluation",
            "source",
        ),
        RobustADCategoryRecord(
            Path("source_anomaly.png"),
            1,
            "widget",
            "source",
            "evaluation",
            "source",
        ),
        RobustADCategoryRecord(
            Path("target_normal.png"),
            0,
            "widget",
            "target",
            "evaluation",
            "lighting",
        ),
        RobustADCategoryRecord(
            Path("target_anomaly.png"),
            1,
            "widget",
            "target",
            "evaluation",
            "lighting",
        ),
    ]
    first = category_protocol_payload(training + evaluation, "widget", 42)
    changed = list(training + evaluation)
    changed[-1] = RobustADCategoryRecord(
        Path("different_target.png"),
        1,
        "widget",
        "target",
        "evaluation",
        "lighting",
    )
    second = category_protocol_payload(changed, "widget", 42)

    assert first["manifest_hash"] != second["manifest_hash"]
