from __future__ import annotations

import csv
from pathlib import Path

import pytest

from nvs.common import ImageRecord
from nvs.perturbation_cache import PerturbationCache


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "category",
        "transform",
        "transform_name",
        "transform_value",
        "source",
        "output",
        "split",
        "defect_type",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_cache_maps_source_records_and_preserves_masks(tmp_path: Path) -> None:
    source = tmp_path / "mvtec" / "bottle" / "test" / "broken" / "001.png"
    output = tmp_path / "cache" / "bottle" / "blur_1" / "test" / "broken" / "001.png"
    mask = tmp_path / "mvtec" / "bottle" / "ground_truth" / "broken" / "001_mask.png"
    for path in (source, output, mask):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")

    manifest = tmp_path / "manifest.csv"
    _write_manifest(
        manifest,
        [
            {
                "category": "bottle",
                "transform": "blur_1",
                "transform_name": "blur",
                "transform_value": "1",
                "source": str(source),
                "output": str(output),
                "split": "test",
                "defect_type": "broken",
            }
        ],
    )
    cache = PerturbationCache(manifest)
    records = [ImageRecord(source, 1, "broken", mask)]

    cached = cache.records_for(
        records,
        category="bottle",
        transform_spec={"name": "blur", "value": 1.0},
    )

    assert len(cache) == 1
    assert cached == [ImageRecord(output.resolve(), 1, "broken", mask)]


def test_identity_uses_original_records_without_manifest_lookup(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    _write_manifest(
        manifest,
        [
            {
                "category": "bottle",
                "transform": "blur_1",
                "transform_name": "blur",
                "transform_value": "1",
                "source": str(tmp_path / "source.png"),
                "output": str(tmp_path / "output.png"),
                "split": "test",
                "defect_type": "good",
            }
        ],
    )
    record = ImageRecord(tmp_path / "identity.png", 0, "good", None)

    assert PerturbationCache(manifest).records_for(
        [record],
        category="bottle",
        transform_spec={"name": "identity", "value": 0},
    ) == [record]


def test_cache_fails_on_missing_transform_entry(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    _write_manifest(
        manifest,
        [
            {
                "category": "bottle",
                "transform": "blur_1",
                "transform_name": "blur",
                "transform_value": "1",
                "source": str(tmp_path / "other.png"),
                "output": str(tmp_path / "output.png"),
                "split": "test",
                "defect_type": "good",
            }
        ],
    )

    with pytest.raises(KeyError, match="no blur=1 cache entry"):
        PerturbationCache(manifest).records_for(
            [ImageRecord(tmp_path / "wanted.png", 0, "good", None)],
            category="bottle",
            transform_spec={"name": "blur", "value": 1.0},
        )
