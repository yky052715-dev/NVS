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

from .datasets import (
    RobustADRecord,
    assert_fit_records_source_only,
    build_mvtec_dataset,
    group_robustad_evaluation,
    mask_capability,
    parse_robustad_manifest,
    source_normal_training,
)
from .memory import MEMORY_PROTOCOLS, augmented_memory_candidates
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


def _loader(
    records: Sequence[ImageRecord],
    config: dict[str, Any],
    transform_spec: dict[str, Any] | None = None,
) -> DataLoader:
    return DataLoader(
        _Dataset(records, config["data"]["input_size"], transform_spec),
        batch_size=int(config["model"]["batch_size"]),
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
    values, _, grid_side = extract_features(
        model, _loader(records, config, transform_spec), device
    )
    return values.float().cpu(), int(grid_side)


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


def _pipeline(config: dict[str, Any], seed: int) -> ConditionalNVSPipeline:
    memory = config["memory"]
    protocol = str(memory.get("protocol", "M_K30"))
    if protocol not in MEMORY_PROTOCOLS:
        raise KeyError(f"Unknown memory protocol {protocol}")
    strategy, capacity = MEMORY_PROTOCOLS[protocol]
    subspace = config["subspace"]
    whitening = config.get("whitening", {}) or {}
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
        query_chunk_size=int(memory.get("query_chunk_size", 4096)),
        bank_chunk_size=int(memory.get("bank_chunk_size", 8192)),
        whitener_rho=float(whitening.get("rho", 0.99)),
        whitener_shrinkage=float(whitening.get("lambda", 0.07)),
        whitener_relative_floor=float(whitening.get("relative_floor", 1.0e-8)),
        whitener_max_components=whitening.get("max_components"),
    )


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
) -> tuple[ConditionalNVSPipeline, dict[str, Any], int]:
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
    memory_features, grid_side = _features(split.memory, config, model, device)
    nvs_original, _ = _features(split.nvs_fit, config, model, device)
    nvs_transformed = tuple(
        _features(split.nvs_fit, config, model, device, dict(spec))[0]
        for spec in FIT_TRANSFORMS
    )
    if bool((config.get("augmem", {}) or {}).get("enabled", False)):
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
    pipeline = _pipeline(config, seed).fit(
        FeatureSplit(
            memory=memory_features,
            nvs_fit_original=nvs_original,
            nvs_fit_transformed=nvs_transformed,
        )
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
            fused = pipeline.fuse(normalized_calibration, float(alpha)).numpy()
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
    start = perf_counter()
    normalized = pipeline.normalize_scores(pipeline.score_patch_features(features))
    elapsed = perf_counter() - start
    labels = np.asarray([int(record.label) for record in records], dtype=np.int64)
    masks = None
    if pixel_metrics:
        masks, labels = _masks_and_labels(records, int(config["data"]["input_size"]))
    rows: list[dict[str, Any]] = []
    scores = dict(normalized)
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
        ).numpy()
        calibration = (
            pipeline.calibrations[method]
            if method in pipeline.calibrations
            else pipeline.calibrations["D3_NVSProto"]
        )
        row: dict[str, Any] = {
            "method": method,
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
        },
    )
    device = torch.device(args.device)
    model = load_dinov2(
        config["model"]["name"], device, config["model"].get("hub_dir")
    )
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
        metadata = protocol_metadata(
            category, seed, prospective, config, CORE_METHODS
        )
        marker = category_dir / "experiment_complete.json"
        if not args.force and completion_is_valid(marker, metadata):
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
        metadata = protocol_metadata(category, seed, manifest, config, CORE_METHODS)
        write_completion(marker, metadata)
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
