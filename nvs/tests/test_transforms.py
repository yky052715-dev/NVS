from __future__ import annotations

import numpy as np
from PIL import Image

from nvs.transforms import apply_transform, is_spatially_aligned, transform_name


def _image() -> Image.Image:
    return Image.fromarray(np.full((16, 16, 3), 128, dtype=np.uint8), mode="RGB")


def test_brightness_transform_changes_pixels() -> None:
    out = apply_transform(_image(), {"name": "brightness", "value": 15})
    assert np.asarray(out).mean() == 143


def test_noise_transform_is_deterministic_for_seed() -> None:
    a = apply_transform(_image(), {"name": "noise", "value": 3}, seed=7)
    b = apply_transform(_image(), {"name": "noise", "value": 3}, seed=7)
    assert np.array_equal(np.asarray(a), np.asarray(b))


def test_transform_name_is_path_safe() -> None:
    assert transform_name({"name": "brightness", "value": -15}) == "brightness_m15"
    assert transform_name({"name": "blur", "value": 0.5}) == "blur_0p5"


def test_spatial_alignment_flags() -> None:
    assert is_spatially_aligned({"name": "contrast", "value": 1.1})
    assert not is_spatially_aligned({"name": "rotation", "value": 10})

