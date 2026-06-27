from __future__ import annotations

import numpy as np
from PIL import Image

from nvs.conditional_nvs.transforms import (
    FIT_TRANSFORMS,
    UNSEEN_TRANSFORMS,
    apply_transform,
    is_spatially_aligned,
    transform_name,
)


def _image() -> Image.Image:
    array = np.arange(16 * 16 * 3, dtype=np.uint8).reshape(16, 16, 3)
    return Image.fromarray(array, "RGB")


def test_fit_protocol_has_exactly_thirteen_transforms() -> None:
    assert len(FIT_TRANSFORMS) == 13
    assert sum(spec["name"] == "noise" for spec in FIT_TRANSFORMS) == 2


def test_jpeg_gamma_and_rgb_gain_are_deterministic_and_aligned() -> None:
    for spec in UNSEEN_TRANSFORMS:
        first = apply_transform(_image(), spec, seed=7)
        second = apply_transform(_image(), spec, seed=7)
        assert np.array_equal(np.asarray(first), np.asarray(second))
        assert first.size == _image().size
        assert is_spatially_aligned(spec)


def test_rgb_transform_name_is_path_safe() -> None:
    name = transform_name({"name": "rgb_gain", "value": [1.1, 1.0, 0.9]})
    assert name == "rgb_gain_1p1_1_0p9"
