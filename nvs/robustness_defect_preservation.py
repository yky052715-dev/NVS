from __future__ import annotations

import argparse
import random
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from nvs.common import (
    RecordDataset,
    build_mvtec_records,
    collate_records,
    load_config,
    load_dinov2,
    patch_scores_to_maps,
    save_json,
    write_csv,
)
from nvs.detection import (
    _add_fusion_maps,
    _all_methods,
    _calibrate_methods,
    _fit_state,
    _normalize_base_maps,
    _score_batch,
)
from nvs.metrics import (
    binary_f1,
    keep_topk_components,
    localization_metrics_from_prediction,
    oracle_pixel_f1,
    safe_auroc,
)
from nvs.perturbation_cache import PerturbationCache
from nvs.transforms import transform_name


DEFAULT_METHODS = ["R0_nn_distance", "R2_nvs_residual", "P_topk3_r2"]


def _loader(records, config: dict[str, Any], transform_spec: dict[str, Any] | None = None, include_mask: bool = True) -> DataLoader:
    dataset = RecordDataset(
        records,
        input_size=int(config["data"]["input_size"]),
        include_mask=include_mask,
        transform_spec=transform_spec,
    )
    return DataLoader(
        dataset,
        batch_size=int(config["model"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["data"]["num_workers"]),
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_records,
    )


@torch.inference_mode()
def _score_records(
    records,
    transform_spec: dict[str, Any],
    state: dict[str, Any],
    config: dict[str, Any],
    model,
    device: torch.device,
    *,
    cached_images: bool = False,
) -> dict[str, Any]:
    from dinov2_mvtec_nn import extract_patch_tokens

    method_maps: dict[str, list[np.ndarray]] = {method: [] for method in ["R0_nn_distance", "R1_feature_pca_residual", "R2_nvs_residual"]}
    masks: list[np.ndarray] = []
    labels: list[int] = []
    paths: list[str] = []
    defect_types: list[str] = []
    start = perf_counter()
    image_count = 0
    loader_transform = None if cached_images else transform_spec
    for batch in tqdm(_loader(records, config, transform_spec=loader_transform, include_mask=True), desc=f"defect {state['category']} {transform_name(transform_spec)}"):
        features, grid_side = extract_patch_tokens(model, batch["image"], device)
        scores_by_method = _score_batch(features, state, config, device)
        for method, scores in scores_by_method.items():
            maps = patch_scores_to_maps(
                scores,
                grid_side=int(grid_side),
                output_size=int(config["data"]["input_size"]),
            ).numpy()
            method_maps[method].extend([value.astype(np.float32) for value in maps])
        masks.extend([value.numpy().astype(np.uint8) for value in batch["mask"]])
        labels.extend([int(value) for value in batch["label"].tolist()])
        paths.extend(batch["path"])
        defect_types.extend(batch["defect_type"])
        image_count += int(batch["image"].shape[0])
    return {
        "maps": {method: np.stack(values) for method, values in method_maps.items()},
        "masks": np.stack(masks).astype(bool),
        "labels": np.asarray(labels, dtype=np.int64),
        "paths": paths,
        "defect_types": defect_types,
        "seconds": perf_counter() - start,
        "images": image_count,
    }


def _robustness_transforms(config: dict[str, Any]) -> list[dict[str, Any]]:
    specs = [{"name": "identity", "value": 0}]
    seen = {transform_name(specs[0])}
    for spec in config.get("robustness_transforms", config.get("transforms", [])):
        key = transform_name(spec)
        if key not in seen:
            specs.append(dict(spec))
            seen.add(key)
    return specs


def _method_maps_and_predictions(
    raw_maps: dict[str, np.ndarray],
    calibrations: dict[str, dict[str, float]],
    config: dict[str, Any],
    methods: list[str],
) -> dict[str, dict[str, Any]]:
    base_maps = _normalize_base_maps(raw_maps, calibrations)
    maps_by_method = _add_fusion_maps(base_maps, config)
    outputs: dict[str, dict[str, Any]] = {}
    for method in methods:
        if method not in calibrations:
            raise KeyError(f"Method {method!r} is not available. Available: {sorted(calibrations)}")
        cal = calibrations[method]
        threshold = float(cal["pixel_threshold"])
        if cal.get("kind") == "topk_component":
            source = str(cal["source_method"])
            maps = maps_by_method[source]
            predictions = keep_topk_components(maps >= threshold, k=int(cal["topk_components"]))
        else:
            maps = maps_by_method[method]
            predictions = maps >= threshold
        outputs[method] = {
            "maps": maps,
            "predictions": predictions,
            "threshold": threshold,
            "method_kind": str(cal.get("kind", "base")),
            "source_method": cal.get("source_method"),
            "topk_components": cal.get("topk_components"),
        }
    return outputs


def _summary_for_method(
    category: str,
    transform_spec: dict[str, Any],
    method: str,
    maps: np.ndarray,
    predictions: np.ndarray,
    masks: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    elapsed_seconds: float,
    image_count: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    loc = localization_metrics_from_prediction(
        masks,
        predictions,
        labels,
        small_defect_area_fraction=float(config["metrics"]["small_defect_area_fraction"]),
    )
    oracle_f1, oracle_threshold = oracle_pixel_f1(masks, maps)
    return {
        "category": category,
        "transform": transform_name(transform_spec),
        "transform_name": str(transform_spec.get("name", "identity")),
        "transform_value": transform_spec.get("value", 0),
        "method": method,
        "images": int(image_count),
        "pixel_AUROC": safe_auroc(masks.reshape(-1), maps.reshape(-1)),
        "pixel_F1_calibrated": binary_f1(masks, predictions),
        "pixel_F1_oracle": oracle_f1,
        "pixel_threshold_calibrated": float(threshold),
        "pixel_threshold_oracle": oracle_threshold,
        "inference_ms_per_image": float(elapsed_seconds * 1000.0 / max(1, image_count)),
        **loc,
    }


def _aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = sorted({(row["transform_name"], row["transform_value"], row["transform"], row["method"]) for row in rows}, key=lambda item: (str(item[0]), float(item[1]), str(item[3])))
    metrics = [
        "pixel_AUROC",
        "pixel_F1_calibrated",
        "pixel_F1_oracle",
        "localization_overseg_anomaly_macro",
        "localization_recall_anomaly_macro",
        "localization_small_defect_recall_macro",
        "localization_test_normal_image_positive_rate",
        "inference_ms_per_image",
    ]
    outputs: list[dict[str, Any]] = []
    for transform_name_value, transform_value, transform_key, method in keys:
        selected = [row for row in rows if row["transform"] == transform_key and row["method"] == method]
        out: dict[str, Any] = {
            "transform_name": transform_name_value,
            "transform_value": transform_value,
            "transform": transform_key,
            "method": method,
            "categories": len({row["category"] for row in selected}),
        }
        for metric in metrics:
            out[metric] = float(np.nanmean([float(row[metric]) for row in selected]))
        outputs.append(out)
    return outputs


def _write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    headers = [
        "transform",
        "value",
        "method",
        "AUROC",
        "F1",
        "oracle F1",
        "OverSeg",
        "Recall",
        "small recall",
        "normal FP",
    ]
    lines = [
        "# NVS defect-preservation robustness summary",
        "",
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["transform_name"]),
                    str(row["transform_value"]),
                    str(row["method"]),
                    f"{float(row['pixel_AUROC']):.6f}",
                    f"{float(row['pixel_F1_calibrated']):.6f}",
                    f"{float(row['pixel_F1_oracle']):.6f}",
                    f"{float(row['localization_overseg_anomaly_macro']):.6f}",
                    f"{float(row['localization_recall_anomaly_macro']):.6f}",
                    f"{float(row['localization_small_defect_recall_macro']):.6f}",
                    f"{float(row['localization_test_normal_image_positive_rate']):.6f}",
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Defect-preservation robustness benchmark for NVS")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--categories", nargs="+")
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--perturbation-manifest",
        help=(
            "Optional manifest.csv produced by nvs.cache_perturbed_mvtec. "
            "Masks remain sourced from the original MVTec tree."
        ),
    )
    parser.add_argument("--seed", type=int, help="Override experiment.seed from config")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.categories:
        config["data"]["categories"] = args.categories
    if args.output_dir:
        config["experiment"]["output_dir"] = args.output_dir
    else:
        config["experiment"]["output_dir"] = "outputs/nvs/robustness_dev5_defect_preservation"
    if args.seed is not None:
        config["experiment"]["seed"] = int(args.seed)
    seed = int(config["experiment"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(args.device)
    model = load_dinov2(config["model"]["name"], device, config["model"].get("hub_dir"))
    output_dir = Path(config["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(config, output_dir / "resolved_config.json")
    perturbation_cache = (
        PerturbationCache(args.perturbation_manifest)
        if args.perturbation_manifest
        else None
    )

    unavailable = sorted(set(args.methods) - set(_all_methods(config)))
    if unavailable:
        raise ValueError(f"Requested unavailable methods: {unavailable}. Available methods: {_all_methods(config)}")

    summary_rows: list[dict[str, Any]] = []
    transforms = _robustness_transforms(config)
    for category in config["data"]["categories"]:
        train_records, test_records = build_mvtec_records(args.data_root, category)
        anomaly_records = [record for record in test_records if int(record.label) == 1]
        state = _fit_state(category, train_records, config, model, device)
        calibrations = _calibrate_methods(state, config, model, device)
        save_json(
            {
                key: value
                for key, value in state.items()
                if key
                not in {
                    "memory_bank",
                    "feature_mean",
                    "feature_basis",
                    "nvs_basis",
                    "calibration_records",
                    "threshold_fit_records",
                }
            }
            | {"calibrations": calibrations},
            output_dir / category / "state_summary.json",
        )
        for spec in transforms:
            cached_images = (
                perturbation_cache is not None
                and str(spec.get("name", "identity")).lower() != "identity"
            )
            score_records = (
                perturbation_cache.records_for(
                    anomaly_records,
                    category=category,
                    transform_spec=spec,
                )
                if cached_images
                else anomaly_records
            )
            scored = _score_records(
                score_records,
                spec,
                state,
                config,
                model,
                device,
                cached_images=cached_images,
            )
            outputs = _method_maps_and_predictions(scored["maps"], calibrations, config, args.methods)
            for method in args.methods:
                payload = outputs[method]
                summary_rows.append(
                    _summary_for_method(
                        category,
                        spec,
                        method,
                        payload["maps"],
                        payload["predictions"],
                        scored["masks"],
                        scored["labels"],
                        payload["threshold"],
                        scored["seconds"],
                        scored["images"],
                        config,
                    )
                )

    write_csv(summary_rows, output_dir / "category_transform_method_summary.csv")
    aggregate_rows = _aggregate_rows(summary_rows)
    write_csv(aggregate_rows, output_dir / "defect_preservation_group_summary.csv")
    _write_markdown(aggregate_rows, output_dir / "defect_preservation_summary.md")
    save_json(
        {
            "status": "complete",
            "categories": list(config["data"]["categories"]),
            "transforms": [transform_name(spec) for spec in transforms],
            "methods": list(args.methods),
            "summary_rows": len(summary_rows),
            "perturbation_manifest": (
                str(perturbation_cache.manifest_path)
                if perturbation_cache is not None
                else None
            ),
            "perturbation_cache_rows": (
                len(perturbation_cache) if perturbation_cache is not None else 0
            ),
        },
        output_dir / "experiment_complete.json",
    )


if __name__ == "__main__":
    main()