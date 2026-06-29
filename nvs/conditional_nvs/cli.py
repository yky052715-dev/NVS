from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from nvs.common import (
    ImageRecord,
    collate_records,
    extract_features,
    image_to_dino_tensor,
    load_dinov2,
    mask_to_tensor,
    patch_scores_to_maps,
    resize_pil,
)

from .augmem import AugMemDetector
from .datasets import (
    IMG_EXTS,
    RobustADRecord,
    assert_fit_records_source_only,
    build_mvtec_dataset,
    group_robustad_evaluation,
    mask_capability,
    parse_robustad_manifest,
    source_normal_training,
)
from .memory import (
    MEMORY_PROTOCOLS,
    augmented_memory_candidates,
    matched_augmented_memory_candidates,
)
from .metrics import evaluate_pixel_metrics, safe_auroc
from .pipeline import CORE_METHODS, ConditionalNVSPipeline, FeatureSplit
from .protocol import (
    completion_is_valid,
    protocol_metadata,
    select_memory_protocol,
    split_three_way,
    write_completion,
    write_split_manifest,
)
from .robustness import aggregate_ard
from .transforms import FIT_TRANSFORMS, UNSEEN_TRANSFORMS, apply_transform, transform_name


class _Dataset(Dataset):
    def __init__(
        self,
        records: Sequence[ImageRecord],
        input_size: int,
        transform_spec: dict[str, Any] | None = None,
    ) -> None:
        self.records = list(records)
        self.input_size = int(input_size)
        self.transform_spec = transform_spec

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        with Image.open(record.path) as handle:
            image = resize_pil(handle, self.input_size, is_mask=False)
        if self.transform_spec is not None:
            image = apply_transform(image, self.transform_spec, seed=index)
        return {
            "image": image_to_dino_tensor(image),
            "label": int(record.label),
            "path": str(record.path),
            "defect_type": str(record.defect_type),
        }


def _load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration root must be a mapping")
    return config


def _save_json(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)



def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


_PERTURBED_INDEX_CACHE: dict[tuple[str, str, str], dict[str, Path]] = {}


def _is_fit_transform(spec: dict[str, Any] | None) -> bool:
    if spec is None:
        return False
    target = transform_name(dict(spec))
    return target in {transform_name(dict(item)) for item in FIT_TRANSFORMS}


def _record_category_and_relative(
    record: ImageRecord, data_root: str | Path | None
) -> tuple[str | None, Path | None]:
    path = Path(record.path)
    if data_root:
        try:
            relative = path.resolve().relative_to(Path(data_root).resolve())
            if len(relative.parts) >= 2:
                return relative.parts[0], Path(*relative.parts[1:])
        except ValueError:
            pass
    parts = path.parts
    for marker in ("train", "test"):
        if marker in parts:
            index = parts.index(marker)
            if index > 0:
                return parts[index - 1], Path(*parts[index:])
    return None, None


def _perturbed_index(root: Path, category: str, transform: str) -> dict[str, Path]:
    key = (str(root), str(category), str(transform))
    cached = _PERTURBED_INDEX_CACHE.get(key)
    if cached is not None:
        return cached
    directory = root / category / transform
    index: dict[str, Path] = {}
    if directory.is_dir():
        for path in sorted(directory.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in IMG_EXTS:
                continue
            index.setdefault(path.name, path)
            index.setdefault(path.stem, path)
    _PERTURBED_INDEX_CACHE[key] = index
    return index


def _cached_transform_path(
    record: ImageRecord,
    config: dict[str, Any],
    transform_spec: dict[str, Any],
) -> tuple[Path | None, str | None, str | None]:
    data_cfg = config.get("data", {}) or {}
    root_value = data_cfg.get("perturbed_root")
    if not root_value:
        return None, None, None
    root = Path(root_value)
    category, relative = _record_category_and_relative(record, data_cfg.get("root"))
    if category is None:
        return None, None, None
    transform = transform_name(transform_spec)
    base = root / category / transform
    original = Path(record.path)
    candidates: list[Path] = []
    if relative is not None:
        candidates.append(base / relative)
        candidates.append(base / relative.name)
    candidates.extend(
        [
            base / "train" / "good" / original.name,
            base / "good" / original.name,
            base / original.name,
        ]
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate, category, transform
    index = _perturbed_index(root, category, transform)
    return index.get(original.name) or index.get(original.stem), category, transform


def _records_from_perturbed_cache(
    records: Sequence[ImageRecord],
    config: dict[str, Any],
    transform_spec: dict[str, Any] | None,
) -> tuple[list[ImageRecord], dict[str, Any] | None, bool]:
    if transform_spec is None:
        return list(records), None, False
    data_cfg = config.get("data", {}) or {}
    require_fit = bool(data_cfg.get("require_perturbed_fit_transforms", False))
    root_value = data_cfg.get("perturbed_root")
    if not root_value:
        if require_fit and _is_fit_transform(transform_spec):
            raise FileNotFoundError(
                "data.require_perturbed_fit_transforms=true but data.perturbed_root is not set"
            )
        return list(records), None, False

    converted: list[ImageRecord] = []
    missing: list[str] = []
    category = transform = None
    for record in records:
        cached_path, record_category, record_transform = _cached_transform_path(
            record, config, transform_spec
        )
        category = category or record_category
        transform = transform or record_transform
        if cached_path is None:
            missing.append(str(record.path))
            continue
        converted.append(
            ImageRecord(
                path=cached_path,
                label=int(record.label),
                defect_type=str(record.defect_type),
                mask_path=record.mask_path,
            )
        )

    usage = {
        "transform": transform_name(transform_spec),
        "category": category,
        "perturbed_root": str(root_value),
        "records": len(records),
        "hits": len(converted),
        "misses": len(missing),
        "used_cache": len(converted) == len(records),
        "fallback_online": len(converted) != len(records),
        "missing_examples": missing[:5],
    }
    config.setdefault("_perturbed_cache_usage", []).append(usage)
    if len(converted) == len(records):
        return converted, usage, True
    if require_fit and _is_fit_transform(transform_spec):
        raise FileNotFoundError(
            "Missing cached perturbed images for fit transform "
            f"{usage['transform']}: {len(missing)}/{len(records)} missing; "
            f"examples={missing[:3]}"
        )
    return list(records), usage, False


def _loader(
    records: Sequence[ImageRecord],
    config: dict[str, Any],
    transform_spec: dict[str, Any] | None = None,
    device: torch.device | None = None,
) -> DataLoader:
    batch_size = int(config["model"]["batch_size"])
    if device is not None and device.type == "cuda":
        batch_size = int(
            config["model"].get("gpu_batch_size", max(batch_size, 16))
        )
    return DataLoader(
        _Dataset(records, config["data"]["input_size"], transform_spec),
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(config["data"].get("num_workers", 4)),
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_records,
    )


def _features(
    records: Sequence[ImageRecord],
    config: dict[str, Any],
    model,
    device: torch.device,
    transform_spec: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, int]:
    feature_records, _, used_cache = _records_from_perturbed_cache(
        records, config, transform_spec
    )
    effective_transform = None if used_cache else transform_spec
    values, _, grid_side = extract_features(
        model,
        _loader(feature_records, config, effective_transform, device=device),
        device,
        keep_on_device=device.type == "cuda",
    )
    return values.float(), int(grid_side)


def _masks_and_labels(
    records: Sequence[ImageRecord], input_size: int
) -> tuple[np.ndarray, np.ndarray]:
    masks = []
    labels = []
    for record in records:
        labels.append(int(record.label))
        if record.mask_path is None:
            mask = Image.new("L", (int(input_size), int(input_size)), 0)
        else:
            with Image.open(record.mask_path) as handle:
                mask = resize_pil(handle, int(input_size), is_mask=True)
        masks.append(mask_to_tensor(mask).numpy().astype(np.uint8))
    return np.stack(masks), np.asarray(labels, dtype=np.int64)


def _pipeline(
    config: dict[str, Any],
    seed: int,
    device: torch.device | str | None = None,
) -> ConditionalNVSPipeline:
    memory = config["memory"]
    protocol = str(memory.get("protocol", "M_K30"))
    if protocol not in MEMORY_PROTOCOLS:
        raise KeyError(f"Unknown memory protocol {protocol}")
    strategy, capacity = MEMORY_PROTOCOLS[protocol]
    subspace = config["subspace"]
    whitening = config.get("whitening", {}) or {}
    sr_config = config.get("stability_regularization", {}) or {}
    compute_device = torch.device(device or "cpu")
    query_chunk_size = int(memory.get("query_chunk_size", 4096))
    bank_chunk_size = int(memory.get("bank_chunk_size", 8192))
    if compute_device.type == "cuda":
        query_chunk_size = int(
            memory.get("gpu_query_chunk_size", max(query_chunk_size, 16_384))
        )
        bank_chunk_size = int(
            memory.get("gpu_bank_chunk_size", max(bank_chunk_size, 32_768))
        )
    return ConditionalNVSPipeline(
        rank=int(subspace.get("rank", 8)),
        prototypes=int(subspace.get("prototypes", 128)),
        memory_strategy=strategy,
        memory_capacity=capacity,
        seed=int(seed),
        prototype_selection=str(
            subspace.get("prototype_selection", "proto_by_mstar")
        ),
        search_mode=str(whitening.get("mode", "cosine")),
        query_chunk_size=query_chunk_size,
        bank_chunk_size=bank_chunk_size,
        compute_device=compute_device,
        whitener_rho=float(whitening.get("rho", 0.99)),
        whitener_shrinkage=float(whitening.get("lambda", 0.07)),
        whitener_relative_floor=float(whitening.get("relative_floor", 1.0e-8)),
        whitener_max_components=whitening.get("max_components"),
        stability_regularization=bool(sr_config.get("enabled", False)),
        sr_bootstrap_repeats=int(sr_config.get("bootstrap_repeats", 5)),
        sr_bootstrap_fraction=float(sr_config.get("bootstrap_fraction", 0.8)),
        sr_weight_epsilon=float(sr_config.get("weight_epsilon", 1.0e-8)),
    )


def _configured_methods(config: dict[str, Any]) -> tuple[str, ...]:
    report_methods = config.get("report_methods")
    if report_methods:
        methods = [str(method) for method in report_methods]
    else:
        methods = list(CORE_METHODS)
        if bool(
            (config.get("stability_regularization", {}) or {}).get("enabled", False)
        ):
            methods.append("SR_CNVS")
    aliases = {
        str(method): str(alias)
        for method, alias in (config.get("result_method_aliases", {}) or {}).items()
    }
    return tuple(aliases.get(method, method) for method in methods)


def _validate_core_config(config: dict[str, Any]) -> None:
    if bool((config.get("core_experiment", {}) or {}).get("enabled", True)):
        if bool((config.get("fusion", {}) or {}).get("enabled", False)):
            raise ValueError("Core 2x2 experiment requires fusion.enabled=false")
        if bool((config.get("postprocess", {}) or {}).get("enabled", False)):
            raise ValueError("Core 2x2 experiment requires postprocess.enabled=false")
    transforms = config.get("fit_transforms", list(FIT_TRANSFORMS))
    if [dict(item) for item in transforms] != [dict(item) for item in FIT_TRANSFORMS]:
        raise ValueError("NVS fit transform protocol must be the fixed 13 transforms")


def _fit_category(
    train_records: Sequence[ImageRecord],
    config: dict[str, Any],
    model,
    device: torch.device,
    seed: int,
    category_dir: Path,
    category: str,
) -> tuple[Any, dict[str, Any], int]:
    config["_perturbed_cache_usage"] = []
    split = split_three_way(
        train_records,
        split_seed=seed,
        nvs_split_seed=seed,
        calibration_fraction=0.20,
        nvs_fit_fraction_of_remainder=0.30,
        key=lambda record: str(record.path),
    )
    manifest = write_split_manifest(
        split,
        category_dir / "sample_manifest.json",
        category,
        key=lambda record: str(record.path),
    )
    print(
        f"[{category}] split calibration={len(split.calibration)} "
        f"nvs_fit={len(split.nvs_fit)} memory={len(split.memory)}",
        flush=True,
    )
    print(f"[{category}] extracting original memory/nvs_fit features", flush=True)
    memory_features, grid_side = _features(split.memory, config, model, device)
    nvs_original, _ = _features(split.nvs_fit, config, model, device)
    transformed_values = []
    for index, spec in enumerate(FIT_TRANSFORMS, start=1):
        print(
            f"[{category}] fit transform {index}/{len(FIT_TRANSFORMS)} "
            f"{transform_name(dict(spec))}",
            flush=True,
        )
        transformed_values.append(
            _features(split.nvs_fit, config, model, device, dict(spec))[0]
        )
    nvs_transformed = tuple(transformed_values)
    augmem_config = config.get("augmem", {}) or {}
    if bool(augmem_config.get("enabled", False)):
        candidate_source = str(
            augmem_config.get("candidate_source", "legacy_all_transformed")
        )
        if candidate_source == "matched_d2_information":
            memory_features = matched_augmented_memory_candidates(
                memory_features,
                nvs_original,
                nvs_transformed,
            )
        elif candidate_source == "legacy_all_transformed":
            memory_transformed = [
                _features(split.memory, config, model, device, dict(spec))[0]
                for spec in FIT_TRANSFORMS
            ]
            memory_features = augmented_memory_candidates(
                torch.cat([memory_features, nvs_original], dim=0),
                [
                    torch.cat([memory_value, nvs_value], dim=0)
                    for memory_value, nvs_value in zip(
                        memory_transformed, nvs_transformed
                    )
                ],
            )
        else:
            raise ValueError(f"Unsupported augmem.candidate_source={candidate_source!r}")
        candidate_layout = []
        if candidate_source == "matched_d2_information":
            patch_count = int(nvs_original.shape[1])
            offset = 0
            chunks = [
                ("memory_original", len(split.memory) * patch_count),
                ("nvs_fit_original", len(split.nvs_fit) * patch_count),
                *[
                    (
                        f"nvs_fit_{transform_name(dict(spec))}",
                        len(split.nvs_fit) * patch_count,
                    )
                    for spec in FIT_TRANSFORMS
                ],
            ]
            for name, count in chunks:
                candidate_layout.append(
                    {"name": name, "start": offset, "stop": offset + count}
                )
                offset += count
            if offset != int(memory_features.shape[0]):
                raise AssertionError("AugMem candidate layout does not cover the pool")
        _save_json(
            category_dir / "augmem_candidate_pool.json",
            {
                "candidate_source": candidate_source,
                "memory_images": len(split.memory),
                "nvs_fit_images": len(split.nvs_fit),
                "fit_transform_count": len(nvs_transformed),
                "candidate_entries": int(
                    memory_features.reshape(-1, memory_features.shape[-1]).shape[0]
                ),
                "feature_dimension": int(memory_features.shape[-1]),
                "uses_transformed_memory_split": candidate_source
                == "legacy_all_transformed",
                "candidate_layout": candidate_layout,
            },
        )
    if bool(augmem_config.get("enabled", False)) and str(
        augmem_config.get("detection_mode", "conditional_pipeline")
    ) == "memory_only":
        memory_config = config["memory"]
        strategy, capacity = MEMORY_PROTOCOLS[str(memory_config["protocol"])]
        query_chunk_size = int(memory_config.get("query_chunk_size", 4096))
        bank_chunk_size = int(memory_config.get("bank_chunk_size", 8192))
        if device.type == "cuda":
            query_chunk_size = int(
                memory_config.get("gpu_query_chunk_size", query_chunk_size)
            )
            bank_chunk_size = int(
                memory_config.get("gpu_bank_chunk_size", bank_chunk_size)
            )
        print(
            f"[{category}] building memory-only AugMem bank "
            f"strategy={strategy} capacity={capacity or 'full'} "
            f"candidates={memory_features.reshape(-1, memory_features.shape[-1]).shape[0]}",
            flush=True,
        )
        pipeline = AugMemDetector(
            memory_strategy=strategy,
            memory_capacity=capacity,
            seed=seed,
            compute_device=device,
            query_chunk_size=query_chunk_size,
            bank_chunk_size=bank_chunk_size,
            candidate_size=int(memory_config.get("candidate_size", 50_000)),
            kcenter_chunk_size=int(memory_config.get("kcenter_chunk_size", 8192)),
        ).fit(memory_features)
    else:
        pipeline = _pipeline(config, seed, device).fit(
            FeatureSplit(
                memory=memory_features,
                nvs_fit_original=nvs_original,
                nvs_fit_transformed=nvs_transformed,
            )
        )
    print(
        f"[{category}] memory ready entries={pipeline.memory_result.capacity}; "
        "calibrating on original normal images",
        flush=True,
    )
    calibration_features, _ = _features(split.calibration, config, model, device)
    pipeline.calibrate(
        calibration_features,
        image_quantile=float(config["calibration"].get("image_quantile", 0.95)),
        mad_epsilon=float(config["calibration"].get("mad_epsilon", 1.0e-6)),
    )
    fusion = config.get("fusion", {}) or {}
    if bool(fusion.get("enabled", False)):
        from .protocol import CalibrationState

        normalized_calibration = pipeline.normalize_scores(
            pipeline.score_patch_features(calibration_features)
        )
        for alpha in fusion.get("alphas", [0.25]):
            method = f"F_D3_D0_alpha{float(alpha):.2f}"
            fused = (
                pipeline.fuse(normalized_calibration, float(alpha))
                .detach()
                .cpu()
                .numpy()
            )
            image_max = fused.reshape(fused.shape[0], -1).max(axis=1)
            threshold = float(
                np.quantile(
                    image_max,
                    float(config["calibration"].get("image_quantile", 0.95)),
                )
            )
            pipeline.calibrations[method] = CalibrationState(
                median=0.0,
                mad=1.0,
                threshold=threshold,
                fit_scope="identity_calibration",
            )
    _save_json(category_dir / "state_summary.json", pipeline.state_summary())
    sr_rows = pipeline.sr_weight_rows()
    if sr_rows:
        _write_csv(category_dir / "sr_cnvs_prototype_weights.csv", sr_rows)
    if config.get("_perturbed_cache_usage"):
        _save_json(
            category_dir / "perturbed_cache_usage.json",
            config.get("_perturbed_cache_usage", []),
        )
    _save_json(
        category_dir / "calibration.json",
        {
            method: {
                "median": value.median,
                "mad": value.mad,
                "threshold": value.threshold,
                "fit_scope": value.fit_scope,
            }
            for method, value in pipeline.calibrations.items()
        },
    )
    return pipeline, manifest, grid_side


def _evaluate_records(
    records: Sequence[ImageRecord],
    pipeline: ConditionalNVSPipeline,
    config: dict[str, Any],
    model,
    device: torch.device,
    transform_spec: dict[str, Any] | None = None,
    pixel_metrics: bool = True,
) -> list[dict[str, Any]]:
    features, grid_side = _features(
        records, config, model, device, transform_spec=transform_spec
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = perf_counter()
    normalized = pipeline.normalize_scores(pipeline.score_patch_features(features))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = perf_counter() - start
    labels = np.asarray([int(record.label) for record in records], dtype=np.int64)
    masks = None
    if pixel_metrics:
        masks, labels = _masks_and_labels(records, int(config["data"]["input_size"]))
    rows: list[dict[str, Any]] = []
    scores = dict(normalized)
    report_methods = config.get("report_methods")
    if report_methods:
        keep = {str(method) for method in report_methods}
        scores = {method: value for method, value in scores.items() if method in keep}
    fusion = config.get("fusion", {}) or {}
    if bool(fusion.get("enabled", False)):
        for alpha in fusion.get("alphas", [0.25]):
            scores[f"F_D3_D0_alpha{float(alpha):.2f}"] = pipeline.fuse(
                normalized, float(alpha)
            )
    for method, patch_scores in scores.items():
        maps = patch_scores_to_maps(
            patch_scores,
            grid_side=grid_side,
            output_size=int(config["data"]["input_size"]),
        ).detach().cpu().numpy()
        calibration = (
            pipeline.calibrations[method]
            if method in pipeline.calibrations
            else pipeline.calibrations["D3_NVSProto"]
        )
        aliases = config.get("result_method_aliases", {}) or {}
        reported_method = str(aliases.get(method, method))
        row: dict[str, Any] = {
            "method": reported_method,
            "images": len(records),
            "inference_ms_per_image": elapsed * 1000.0 / max(1, len(records)),
            "memory_entries": pipeline.memory_result.capacity,
            "threshold": calibration.threshold,
            "image_AUROC": safe_auroc(
                labels, maps.reshape(maps.shape[0], -1).max(axis=1)
            ),
        }
        if pixel_metrics:
            row.update(
                evaluate_pixel_metrics(
                    masks,
                    maps,
                    labels,
                    threshold=calibration.threshold,
                    small_fraction=float(
                        config.get("metrics", {}).get(
                            "small_defect_area_fraction", 0.01
                        )
                    ),
                )
            )
        rows.append(row)
    return rows


def _run_mvtec(args: argparse.Namespace, config: dict[str, Any]) -> None:
    if not args.data_root:
        raise ValueError("--data-root is required for MVTec")
    _validate_core_config(config)
    seed = int(args.seed if args.seed is not None else config["experiment"]["seed"])
    categories = args.categories or config["data"]["categories"]
    output_dir = Path(args.output_dir or config["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    config.setdefault("data", {})["root"] = str(args.data_root)
    if getattr(args, "perturbed_root", None):
        config["data"]["perturbed_root"] = str(args.perturbed_root)
    config["experiment"]["seed"] = seed
    config["data"]["categories"] = list(categories)
    _save_json(output_dir / "resolved_config.json", config)
    _save_json(
        output_dir / "method_mapping.json",
        {
            "D0_NN": "cosine nearest-neighbor distance",
            "D1_Global": "global centered-feature PCA residual on d=q-m*",
            "D1_Proto": "prototype centered-feature PCA residual on d=q-m*",
            "D2_NVSGlobal": "global uncentered-delta SVD residual on d=q-m*",
            "D3_NVSProto": "prototype uncentered-delta SVD residual on d=q-m*",
            "SR_CNVS": "stability-regularized shrinkage between global and prototype NVS projectors",
            "AugMem_K10": "cosine NN over matched-information augmented k-center memory (10k)",
            "AugMem_K10_100k": "cosine NN over 100k matched-information candidates compressed to 10k",
            "AugMem_Full": "cosine NN over matched-information uncompressed augmented memory",
        },
    )
    device = torch.device(args.device)
    model = load_dinov2(
        config["model"]["name"], device, config["model"].get("hub_dir")
    )
    methods = _configured_methods(config)
    all_rows: list[dict[str, Any]] = []
    robustness_rows: list[dict[str, Any]] = []
    for category in categories:
        category_dir = output_dir / str(category) / f"seed{seed}"
        train_records, test_records = build_mvtec_dataset(args.data_root, category)
        prospective = split_three_way(
            train_records,
            seed,
            seed,
            key=lambda record: str(record.path),
        ).manifest(key=lambda record: str(record.path))
        metadata = protocol_metadata(category, seed, prospective, config, methods)
        marker = category_dir / "experiment_complete.json"
        if not args.force and completion_is_valid(marker, metadata):
            metrics_path = category_dir / "metrics_summary.csv"
            robustness_path = category_dir / "unseen_robustness.csv"
            robustness_enabled = bool(
                (config.get("robustness", {}) or {}).get("enabled", False)
            )
            required_outputs_exist = metrics_path.is_file() and (
                not robustness_enabled or robustness_path.is_file()
            )
            if required_outputs_exist:
                print(f"[{category}] valid completion marker; loading saved CSVs", flush=True)
                all_rows.extend(_read_csv(metrics_path))
                if robustness_enabled:
                    robustness_rows.extend(_read_csv(robustness_path))
                continue
        pipeline, manifest, _ = _fit_category(
            train_records,
            config,
            model,
            device,
            seed,
            category_dir,
            str(category),
        )
        rows = _evaluate_records(
            test_records, pipeline, config, model, device, pixel_metrics=True
        )
        for row in rows:
            row.update(
                category=str(category),
                seed=seed,
                memory_protocol=config["memory"]["protocol"],
            )
        _write_csv(category_dir / "metrics_summary.csv", rows)
        all_rows.extend(rows)
        if bool((config.get("robustness", {}) or {}).get("enabled", False)):
            specs = [{"name": "identity", "value": 0}] + [
                dict(spec)
                for spec in config["robustness"].get(
                    "transforms", list(UNSEEN_TRANSFORMS)
                )
            ]
            for spec in specs:
                print(
                    f"[{category}] evaluating {transform_name(spec)}",
                    flush=True,
                )
                shifted = _evaluate_records(
                    test_records,
                    pipeline,
                    config,
                    model,
                    device,
                    transform_spec=None if spec["name"] == "identity" else spec,
                    pixel_metrics=True,
                )
                for row in shifted:
                    row.update(
                        category=str(category),
                        seed=seed,
                        transform=transform_name(spec),
                    )
                    row["normal_image_fp_rate"] = row.get(
                        "localization_test_normal_image_positive_rate"
                    )
                robustness_rows.extend(shifted)
            _write_csv(
                category_dir / "unseen_robustness.csv",
                [row for row in robustness_rows if row["category"] == category],
            )
        metadata = protocol_metadata(category, seed, manifest, config, methods)
        write_completion(marker, metadata)
        _write_csv(output_dir / "category_metrics.csv", all_rows)
    _write_csv(output_dir / "category_metrics.csv", all_rows)
    if robustness_rows:
        _write_csv(output_dir / "robustness_metrics.csv", robustness_rows)
        _save_json(
            output_dir / "ard_summary.json",
            aggregate_ard(robustness_rows, metric="pixel_AUROC"),
        )


def _run_robustad(args: argparse.Namespace, config: dict[str, Any]) -> None:
    manifest_path = args.manifest or config["data"].get("manifest")
    if not manifest_path:
        raise ValueError("--manifest is required for RobustAD")
    records = parse_robustad_manifest(
        manifest_path, data_root=args.data_root, require_files=True
    )
    source = source_normal_training(records)
    assert_fit_records_source_only(source)
    source_images = [record.as_image_record() for record in source]
    seed = int(args.seed if args.seed is not None else config["experiment"]["seed"])
    output_dir = Path(args.output_dir or config["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model = load_dinov2(
        config["model"]["name"], device, config["model"].get("hub_dir")
    )
    pipeline, manifest, _ = _fit_category(
        source_images,
        config,
        model,
        device,
        seed,
        output_dir / "source_fit",
        "robustad_source",
    )
    rows = []
    for subset, group in group_robustad_evaluation(records).items():
        pixel = mask_capability(group) == "pixel"
        evaluated = _evaluate_records(
            [record.as_image_record() for record in group],
            pipeline,
            config,
            model,
            device,
            pixel_metrics=pixel,
        )
        for row in evaluated:
            row.update(
                subset=subset,
                domain=group[0].domain,
                metric_scope="pixel" if pixel else "image",
                seed=seed,
            )
        rows.extend(evaluated)
    _write_csv(output_dir / "robustad_metrics.csv", rows)
    from .metrics import average_relative_drop

    ard_rows = []
    for method in sorted({str(row["method"]) for row in rows}):
        method_rows = [row for row in rows if str(row["method"]) == method]
        for metric in ("image_AUROC", "pixel_AUROC"):
            source_values = [
                float(row[metric])
                for row in method_rows
                if row["domain"] == "source" and metric in row
            ]
            target_values = [
                float(row[metric])
                for row in method_rows
                if row["domain"] == "target" and metric in row
            ]
            if source_values and target_values:
                source_value = float(np.nanmean(source_values))
                ard_rows.append(
                    {
                        "method": method,
                        "metric": metric,
                        "source": source_value,
                        "target_count": len(target_values),
                        "ARD": average_relative_drop(source_value, target_values),
                    }
                )
    _save_json(output_dir / "robustad_ard.json", ard_rows)
    _save_json(output_dir / "source_sample_manifest.json", manifest)


def _lock_memory(args: argparse.Namespace) -> None:
    with Path(args.seed42_csv).open("r", encoding="utf-8-sig", newline="") as handle:
        seed42 = list(csv.DictReader(handle))
    confirmation = None
    if args.confirmation_csv:
        with Path(args.confirmation_csv).open(
            "r", encoding="utf-8-sig", newline=""
        ) as handle:
            confirmation = list(csv.DictReader(handle))
    result = select_memory_protocol(seed42, confirmation)
    _save_json(args.output, result)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Conditional NVS experiments")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--config", required=True)
    run.add_argument("--dataset", choices=["mvtec", "robustad"], default="mvtec")
    run.add_argument("--data-root")
    run.add_argument("--perturbed-root")
    run.add_argument("--manifest")
    run.add_argument("--categories", nargs="+")
    run.add_argument("--output-dir")
    run.add_argument("--seed", type=int)
    run.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    run.add_argument("--force", action="store_true")
    lock = subparsers.add_parser("lock-memory")
    lock.add_argument("--seed42-csv", required=True)
    lock.add_argument("--confirmation-csv")
    lock.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "lock-memory":
        _lock_memory(args)
        return
    config = _load_config(args.config)
    seed = int(args.seed if args.seed is not None else config["experiment"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.dataset == "robustad":
        _run_robustad(args, config)
    else:
        _run_mvtec(args, config)


if __name__ == "__main__":
    main()
