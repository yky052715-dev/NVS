"""Cache deterministic perturbed MVTec images for NVS robustness experiments.

This utility materializes the image transforms used by the robustness
diagnostics so repeated runs do not spend most of their time re-applying the
same CPU-side PIL operations.

The output is intentionally kept outside the original MVTec tree:

    <output-root>/<category>/<transform-slug>/<split>/<defect-type>/<image>

For example:

    /home/ubuntu/yyk/datasets/mvtec_perturbed/bottle/blur_0p5/test/good/000.png

Masks are not copied because perturbations here are photometric only by
default. Geometric transforms are deliberately excluded from the default set;
if you later enable them, masks must be transformed with the same geometry.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".JPG", ".JPEG", ".PNG"}


DEFAULT_TRANSFORMS: tuple[tuple[str, float], ...] = (
    ("blur", 0.5),
    ("blur", 1.0),
    ("blur", 1.5),
    ("brightness", -30.0),
    ("brightness", -15.0),
    ("brightness", 15.0),
    ("brightness", 30.0),
    ("contrast", 0.75),
    ("contrast", 0.90),
    ("contrast", 1.10),
    ("contrast", 1.25),
    ("noise", 1.0),
    ("noise", 3.0),
    ("noise", 5.0),
)


@dataclass(frozen=True)
class TransformSpec:
    name: str
    value: float

    @property
    def slug(self) -> str:
        value = ("%g" % self.value).replace("-", "m").replace(".", "p")
        return f"{self.name}_{value}"


def _stable_seed(*parts: str, base_seed: int) -> int:
    payload = "::".join([str(base_seed), *parts]).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False) % (2**32)


def _apply_transform(image: Image.Image, spec: TransformSpec, *, seed: int) -> Image.Image:
    image = image.convert("RGB")
    if spec.name == "blur":
        return image.filter(ImageFilter.GaussianBlur(radius=float(spec.value)))
    if spec.name == "brightness":
        arr = np.asarray(image).astype(np.float32)
        arr = np.clip(arr + float(spec.value), 0.0, 255.0).astype(np.uint8)
        return Image.fromarray(arr, mode="RGB")
    if spec.name == "contrast":
        return ImageEnhance.Contrast(image).enhance(float(spec.value))
    if spec.name == "noise":
        rng = np.random.default_rng(seed)
        arr = np.asarray(image).astype(np.float32)
        arr = arr + rng.normal(loc=0.0, scale=float(spec.value), size=arr.shape)
        arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
        return Image.fromarray(arr, mode="RGB")
    raise ValueError(f"Unsupported transform: {spec.name!r}")


def _parse_transform(text: str) -> TransformSpec:
    if ":" not in text:
        raise argparse.ArgumentTypeError(
            f"Transform must be NAME:VALUE, got {text!r}. Example: blur:1.0"
        )
    name, value = text.split(":", 1)
    name = name.strip()
    if name not in {"blur", "brightness", "contrast", "noise"}:
        raise argparse.ArgumentTypeError(f"Unsupported transform name: {name!r}")
    try:
        return TransformSpec(name=name, value=float(value))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid transform value in {text!r}") from exc


def _iter_images(category_root: Path, splits: Iterable[str]) -> Iterable[Path]:
    for split in splits:
        split_root = category_root / split
        if not split_root.exists():
            continue
        for path in sorted(split_root.rglob("*")):
            if path.is_file() and path.suffix in IMAGE_EXTENSIONS:
                yield path


def _write_manifest(manifest_path: Path, rows: list[dict[str, str]]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
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
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def cache_category(
    *,
    data_root: Path,
    output_root: Path,
    category: str,
    transforms: list[TransformSpec],
    splits: list[str],
    seed: int,
    overwrite: bool,
) -> list[dict[str, str]]:
    category_root = data_root / category
    if not category_root.exists():
        raise FileNotFoundError(f"Category directory not found: {category_root}")

    image_paths = list(_iter_images(category_root, splits))
    if not image_paths:
        raise FileNotFoundError(f"No images found under {category_root} for splits {splits}")

    rows: list[dict[str, str]] = []
    for image_path in image_paths:
        relative = image_path.relative_to(category_root)
        split = relative.parts[0] if relative.parts else ""
        defect_type = relative.parts[1] if len(relative.parts) > 1 else ""
        with Image.open(image_path) as source_image:
            for spec in transforms:
                output_path = output_root / category / spec.slug / relative
                if overwrite or not output_path.exists():
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    image_seed = _stable_seed(
                        category,
                        spec.slug,
                        str(relative).replace("\\", "/"),
                        base_seed=seed,
                    )
                    perturbed = _apply_transform(source_image, spec, seed=image_seed)
                    save_kwargs = {}
                    if output_path.suffix.lower() in {".jpg", ".jpeg"}:
                        save_kwargs = {"quality": 95, "subsampling": 0}
                    perturbed.save(output_path, **save_kwargs)

                rows.append(
                    {
                        "category": category,
                        "transform": spec.slug,
                        "transform_name": spec.name,
                        "transform_value": "%g" % spec.value,
                        "source": str(image_path),
                        "output": str(output_path),
                        "split": split,
                        "defect_type": defect_type,
                    }
                )
    return rows


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True, help="Original MVTec root.")
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Directory where perturbed images will be materialized.",
    )
    parser.add_argument("--categories", nargs="+", required=True)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "test"],
        help="MVTec splits to cache. Default: train test.",
    )
    parser.add_argument(
        "--transform",
        dest="transforms",
        action="append",
        type=_parse_transform,
        help=(
            "Transform as NAME:VALUE. May be repeated. "
            "If omitted, the current NVS robustness photometric set is used."
        ),
    )
    parser.add_argument("--seed", type=int, default=42, help="Base seed for deterministic noise.")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate existing files.")
    parser.add_argument(
        "--manifest-name",
        default="manifest.csv",
        help="Manifest filename written under output-root.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    transforms = args.transforms
    if not transforms:
        transforms = [TransformSpec(name=name, value=value) for name, value in DEFAULT_TRANSFORMS]

    all_rows: list[dict[str, str]] = []
    for category in args.categories:
        print(f"[cache] {category}: {len(transforms)} transforms")
        rows = cache_category(
            data_root=args.data_root,
            output_root=args.output_root,
            category=category,
            transforms=transforms,
            splits=args.splits,
            seed=args.seed,
            overwrite=args.overwrite,
        )
        all_rows.extend(rows)
        print(f"[cache] {category}: wrote/indexed {len(rows)} transformed images")

    manifest_path = args.output_root / args.manifest_name
    _write_manifest(manifest_path, all_rows)
    print(f"[cache] manifest: {manifest_path}")
    print(f"[cache] total rows: {len(all_rows)}")


if __name__ == "__main__":
    main()
