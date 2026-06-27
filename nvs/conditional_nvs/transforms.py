from __future__ import annotations

from io import BytesIO
from typing import Any

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


FIT_TRANSFORMS = (
    *({"name": "brightness", "value": value} for value in (-30, -15, 15, 30)),
    *({"name": "contrast", "value": value} for value in (0.75, 0.90, 1.10, 1.25)),
    *({"name": "blur", "value": value} for value in (0.5, 1.0, 1.5)),
    *({"name": "noise", "value": value} for value in (1, 3)),
)

UNSEEN_TRANSFORMS = (
    {"name": "noise", "value": 5},
    {"name": "jpeg", "value": 70},
    {"name": "jpeg", "value": 50},
    {"name": "gamma", "value": 0.7},
    {"name": "gamma", "value": 1.3},
    {"name": "rgb_gain", "value": [1.10, 1.00, 0.90]},
    {"name": "rgb_gain", "value": [0.90, 1.00, 1.10]},
)


def transform_name(spec: dict[str, Any]) -> str:
    name = str(spec.get("name", "identity")).lower()
    value = spec.get("value", 0)
    if isinstance(value, (list, tuple)):
        encoded = "_".join(f"{float(item):g}" for item in value)
    else:
        encoded = f"{value}"
    return f"{name}_{encoded}".replace(".", "p").replace("-", "m")


def apply_transform(
    image: Image.Image, spec: dict[str, Any], seed: int = 0
) -> Image.Image:
    name = str(spec.get("name", "identity")).lower()
    value = spec.get("value", 0)
    image = image.convert("RGB")
    if name == "identity":
        return image
    if name == "brightness":
        array = np.asarray(image, dtype=np.float32) + float(value)
        return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), "RGB")
    if name == "contrast":
        return ImageEnhance.Contrast(image).enhance(float(value))
    if name == "blur":
        return image.filter(ImageFilter.GaussianBlur(radius=float(value)))
    if name == "noise":
        rng = np.random.default_rng(int(seed))
        array = np.asarray(image, dtype=np.float32)
        noise = rng.normal(0.0, float(value), size=array.shape)
        return Image.fromarray(np.clip(array + noise, 0, 255).astype(np.uint8), "RGB")
    if name in {"jpeg", "jpeg_quality"}:
        quality = int(value)
        if not 1 <= quality <= 100:
            raise ValueError("JPEG quality must be in [1, 100]")
        buffer = BytesIO()
        image.save(
            buffer,
            format="JPEG",
            quality=quality,
            subsampling=0,
            optimize=False,
            progressive=False,
        )
        buffer.seek(0)
        with Image.open(buffer) as decoded:
            return decoded.convert("RGB").copy()
    if name == "gamma":
        gamma = float(value)
        if gamma <= 0:
            raise ValueError("gamma must be positive")
        array = np.asarray(image, dtype=np.float32) / 255.0
        return Image.fromarray(
            np.clip(np.power(array, gamma) * 255.0, 0, 255).astype(np.uint8),
            "RGB",
        )
    if name == "rgb_gain":
        gains = np.asarray(value, dtype=np.float32)
        if gains.shape != (3,) or np.any(gains <= 0):
            raise ValueError("rgb_gain requires three positive values")
        array = np.asarray(image, dtype=np.float32) * gains.reshape(1, 1, 3)
        return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), "RGB")
    raise ValueError(f"Unsupported aligned transform: {name}")


def is_spatially_aligned(spec: dict[str, Any]) -> bool:
    return str(spec.get("name", "identity")).lower() in {
        "identity",
        "brightness",
        "contrast",
        "blur",
        "noise",
        "jpeg",
        "jpeg_quality",
        "gamma",
        "rgb_gain",
    }
