from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


def transform_name(spec: dict[str, Any]) -> str:
    name = str(spec.get("name", "identity"))
    value = spec.get("value", 0)
    return f"{name}_{value}".replace(".", "p").replace("-", "m")


def apply_transform(image: Image.Image, spec: dict[str, Any], seed: int = 0) -> Image.Image:
    name = str(spec.get("name", "identity")).lower()
    value = spec.get("value", 0)
    image = image.convert("RGB")
    if name == "identity":
        return image
    if name == "brightness":
        delta = float(value)
        array = np.asarray(image, dtype=np.float32)
        array = np.clip(array + delta, 0, 255).astype(np.uint8)
        return Image.fromarray(array, mode="RGB")
    if name == "contrast":
        return ImageEnhance.Contrast(image).enhance(float(value))
    if name == "blur":
        return image.filter(ImageFilter.GaussianBlur(radius=float(value)))
    if name == "noise":
        rng = np.random.default_rng(seed)
        array = np.asarray(image, dtype=np.float32)
        noise = rng.normal(0.0, float(value), size=array.shape)
        return Image.fromarray(np.clip(array + noise, 0, 255).astype(np.uint8), mode="RGB")
    if name == "rotation":
        return image.rotate(float(value), resample=Image.Resampling.BICUBIC, fillcolor=(0, 0, 0))
    if name == "scale":
        scale = float(value)
        if scale <= 0:
            raise ValueError("scale transform value must be positive")
        width, height = image.size
        new_width = max(1, int(round(width * scale)))
        new_height = max(1, int(round(height * scale)))
        resized = image.resize((new_width, new_height), Image.Resampling.BICUBIC)
        if scale >= 1.0:
            left = (new_width - width) // 2
            top = (new_height - height) // 2
            return resized.crop((left, top, left + width, top + height))
        canvas = Image.new("RGB", (width, height), color=(0, 0, 0))
        left = (width - new_width) // 2
        top = (height - new_height) // 2
        canvas.paste(resized, (left, top))
        return canvas
    raise ValueError(f"Unsupported transform: {name}")


def is_spatially_aligned(spec: dict[str, Any]) -> bool:
    return str(spec.get("name", "identity")).lower() in {
        "identity",
        "brightness",
        "contrast",
        "blur",
        "noise",
    }

