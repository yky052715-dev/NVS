from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from nvs.common import ImageRecord
from nvs.conditional_nvs.cli import _records_from_perturbed_cache


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")


def test_perturbed_cache_replaces_fit_transform_records() -> None:
    with TemporaryDirectory() as tmp:
        root = Path(tmp) / "mvtec"
        perturbed = Path(tmp) / "mvtec_perturbed"
        original = root / "bottle" / "train" / "good" / "000.png"
        cached = perturbed / "bottle" / "brightness_m30" / "train" / "good" / "000.png"
        _touch(original)
        _touch(cached)
        config = {
            "data": {
                "root": str(root),
                "perturbed_root": str(perturbed),
                "require_perturbed_fit_transforms": True,
            }
        }

        records, usage, used_cache = _records_from_perturbed_cache(
            [ImageRecord(original, 0, "good")],
            config,
            {"name": "brightness", "value": -30},
        )

        assert used_cache is True
        assert usage is not None and usage["hits"] == 1 and usage["misses"] == 0
        assert records[0].path == cached


def test_required_fit_cache_fails_on_missing_records() -> None:
    with TemporaryDirectory() as tmp:
        root = Path(tmp) / "mvtec"
        perturbed = Path(tmp) / "mvtec_perturbed"
        original = root / "bottle" / "train" / "good" / "000.png"
        _touch(original)
        config = {
            "data": {
                "root": str(root),
                "perturbed_root": str(perturbed),
                "require_perturbed_fit_transforms": True,
            }
        }

        with pytest.raises(FileNotFoundError):
            _records_from_perturbed_cache(
                [ImageRecord(original, 0, "good")],
                config,
                {"name": "noise", "value": 1},
            )

def test_train_cache_never_falls_back_to_same_named_test_image() -> None:
    with TemporaryDirectory() as tmp:
        root = Path(tmp) / "mvtec"
        perturbed = Path(tmp) / "mvtec_perturbed"
        original = root / "bottle" / "train" / "good" / "000.png"
        wrong = perturbed / "bottle" / "brightness_m30" / "test" / "good" / "000.png"
        _touch(original)
        _touch(wrong)
        config = {
            "data": {
                "root": str(root),
                "perturbed_root": str(perturbed),
                "require_perturbed_fit_transforms": True,
            }
        }

        with pytest.raises(FileNotFoundError, match="Missing cached perturbed images"):
            _records_from_perturbed_cache(
                [ImageRecord(original, 0, "good")],
                config,
                {"name": "brightness", "value": -30},
            )
