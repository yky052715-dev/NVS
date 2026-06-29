from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Sequence

from .cli import _cached_transform_path
from .datasets import build_mvtec_dataset
from .transforms import FIT_TRANSFORMS, apply_transform, transform_name


VALIDATION10_CATEGORIES = (
    "cable",
    "capsule",
    "carpet",
    "hazelnut",
    "pill",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
)


def audit_fit_transform_cache(
    data_root: str | Path,
    perturbed_root: str | Path,
    categories: Sequence[str],
) -> dict[str, Any]:
    data_root = Path(data_root)
    perturbed_root = Path(perturbed_root)
    rows = []
    complete = True
    for category in categories:
        train, _ = build_mvtec_dataset(data_root, category)
        config = {
            "data": {
                "root": str(data_root),
                "perturbed_root": str(perturbed_root),
            }
        }
        for spec in FIT_TRANSFORMS:
            hits = 0
            missing = []
            for record in train:
                cached, _, _ = _cached_transform_path(record, config, dict(spec))
                if cached is None:
                    missing.append(str(record.path))
                else:
                    hits += 1
            row = {
                "category": str(category),
                "transform": transform_name(dict(spec)),
                "train_images": len(train),
                "hits": hits,
                "misses": len(missing),
                "complete": len(missing) == 0,
                "missing_examples": missing[:5],
            }
            rows.append(row)
            complete = complete and bool(row["complete"])
    return {
        "status": "complete" if complete else "incomplete",
        "data_root": str(data_root),
        "perturbed_root": str(perturbed_root),
        "categories": list(categories),
        "transform_count": len(FIT_TRANSFORMS),
        "checks": len(rows),
        "rows": rows,
    }


def _populate_one(task: tuple[str, str, dict[str, Any], int]) -> str:
    from PIL import Image

    source_value, output_value, spec, seed = task
    source, output = Path(source_value), Path(output_value)
    if output.is_file():
        return "existing"
    output.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as handle:
        transformed = apply_transform(handle.convert("RGB"), spec, seed=seed)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    transformed.save(temporary, format="PNG")
    os.replace(temporary, output)
    return "created"


def populate_missing_fit_transform_cache(
    data_root: str | Path,
    perturbed_root: str | Path,
    categories: Sequence[str],
    workers: int = 12,
) -> dict[str, Any]:
    """Populate exact train/good paths without overwriting existing cache files.

    Noise seeds are the indices of images in the category's sorted full
    train/good list. Transforms are applied at source resolution, matching the
    existing MVTec perturbed-image cache convention; the feature loader later
    resizes cached images to the configured DINO input size.
    """

    data_root, perturbed_root = Path(data_root), Path(perturbed_root)
    tasks: list[tuple[str, str, dict[str, Any], int]] = []
    existing = 0
    for category in categories:
        train, _ = build_mvtec_dataset(data_root, category)
        for spec in FIT_TRANSFORMS:
            directory = perturbed_root / category / transform_name(dict(spec))
            for image_index, record in enumerate(train):
                output = directory / "train" / "good" / record.path.name
                if output.is_file():
                    existing += 1
                else:
                    tasks.append(
                        (str(record.path), str(output), dict(spec), image_index)
                    )
    created = 0
    worker_count = max(1, int(workers))
    if tasks and worker_count == 1:
        for task in tasks:
            created += int(_populate_one(task) == "created")
    elif tasks:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            for status in executor.map(_populate_one, tasks, chunksize=8):
                created += int(status == "created")
    return {
        "status": "complete",
        "categories": list(categories),
        "transform_count": len(FIT_TRANSFORMS),
        "existing": existing,
        "created": created,
        "total": existing + created,
        "workers": worker_count,
        "seed_rule": "sorted_full_train_good_image_index",
        "transform_order": "source_resolution_then_loader_resize",
        "overwrites_existing": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit cached MVTec fit transforms")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--perturbed-root", required=True)
    parser.add_argument("--categories", nargs="+", default=list(VALIDATION10_CATEGORIES))
    parser.add_argument("--output")
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--populate-missing", action="store_true")
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    population = None
    if args.populate_missing:
        population = populate_missing_fit_transform_cache(
            args.data_root,
            args.perturbed_root,
            args.categories,
            workers=args.workers,
        )
    result = audit_fit_transform_cache(
        args.data_root, args.perturbed_root, args.categories
    )
    result["population"] = population
    for row in result["rows"]:
        print(
            f"{row['category']:12s} {row['transform']:18s} "
            f"hits={row['hits']}/{row['train_images']} misses={row['misses']}",
            flush=True,
        )
    print(f"cache_status={result['status']} checks={result['checks']}", flush=True)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    if args.require_complete and result["status"] != "complete":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
