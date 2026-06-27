from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from nvs.conditional_nvs.datasets import (
    assert_fit_records_source_only,
    mask_capability,
    parse_robustad_manifest,
    source_normal_training,
    target_evaluation,
)


def test_robustad_parser_and_source_target_isolation(tmp_path) -> None:
    for name in ("source.png", "target.png", "mask.png"):
        Image.fromarray(np.zeros((4, 4), dtype=np.uint8)).save(tmp_path / name)
    rows = [
        {
            "path": "source.png",
            "label": 0,
            "domain": "source",
            "role": "train",
            "subset": "source",
        },
        {
            "path": "target.png",
            "mask_path": "mask.png",
            "label": 1,
            "domain": "target",
            "role": "test",
            "subset": "shift_a",
        },
    ]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(rows), encoding="utf-8")
    records = parse_robustad_manifest(manifest)
    fit = source_normal_training(records)
    target = target_evaluation(records)
    assert len(fit) == len(target) == 1
    assert mask_capability(target) == "pixel"
    with pytest.raises(AssertionError):
        assert_fit_records_source_only(target)
