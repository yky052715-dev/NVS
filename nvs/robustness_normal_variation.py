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
    patch_scores_to_maps,
    load_dinov2,
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
from nvs.metrics import image_score_from_map, keep_topk_components
from nvs.perturbation_cache import PerturbationCache
from nvs.transforms import transform_name


DEFAULT_METHODS = ["R0_nn_distance", "R2_nvs_residual", "P_topk3_r2"]


def _loader(records, config: dict[str, Any], transform_spec: dict[str, Any] | None = None) -> DataLoader:
    dataset = RecordDataset(
        records,
        input_size=int(config["data"]["input_size"]),
        include_mask=False,
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
def _score_normal_records(
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
    paths: list[str] = []
    start = perf_counter()
    image_count = 0
    loader_transform = None if cached_images else transform_spec
    for batch in tqdm(_loader(records, config, transform_spec=loader_transform), desc=f"robust {state['category']} {transform_name(transform_spec)}"):
        features, grid_side = extract_patch_tokens(model, batch["image"], device)
        scores_by_method = _score_batch(features, state, config, device)
        for method, scores in scores_by_method.items():
            maps = patch_scores_to_maps(
                scores,
                grid_side=int(grid_side),
                output_size=int(config["data"]["input_size"]),
            ).numpy()
            method_maps[method].extend([value.astype(np.float32) for value in maps])
        paths.extend(batch["path"])
        image_count += int(batch["image"].shape[0])
    return {
        "maps": {method: np.stack(values) for method, values in method_maps.items()},
        "paths": paths,
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
    threshold: float,
    config: dict[str, Any],
) -> dict[str, Any]:
    flattened_predictions = predictions.reshape(predictions.shape[0], -1)
    positive_area = flattened_predictions.mean(axis=1)
    image_fp = flattened_predictions.any(axis=1).astype(np.float32)
    scoring = config.get("robustness", {}) or {}
    image_scores = image_score_from_map(
        maps,
        method=str(scoring.get("image_score", "topk_mean")),
        topk_fraction=float(scoring.get("image_topk_fraction", 0.01)),
    )
    return {
        "category": category,
        "transform": transform_name(transform_spec),
        "transform_name": str(transform_spec.get("name", "identity")),
        "transform_value": transform_spec.get("value", 0),
        "method": method,
        "normal_images": int(predictions.shape[0]),
        "pixel_threshold": float(threshold),
        "normal_image_fp_rate": float(image_fp.mean()),
        "normal_positive_area_mean": float(positive_area.mean()),
        "normal_positive_area_p95": float(np.quantile(positive_area, 0.95)),
        "normal_positive_area_max": float(np.max(positive_area)),
        "image_score_mean": float(image_scores.mean()),
        "image_score_p95": float(np.quantile(image_scores, 0.95)),
        "image_score_max": float(np.max(image_scores)),
    }


def _per_image_rows(
    category: str,
    transform_spec: dict[str, Any],
    method: str,
    paths: list[str],
    maps: np.ndarray,
    predictions: np.ndarray,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    scoring = config.get("robustness", {}) or {}
    image_scores = image_score_from_map(
        maps,
        method=str(scoring.get("image_score", "topk_mean")),
        topk_fraction=float(scoring.get("image_topk_fraction", 0.01)),
    )
    positive_area = predictions.reshape(predictions.shape[0], -1).mean(axis=1)
    image_fp = predictions.reshape(predictions.shape[0], -1).any(axis=1).astype(np.uint8)
    return [
        {
            "category": category,
            "transform": transform_name(transform_spec),
            "transform_name": str(transform_spec.get("name", "identity")),
            "transform_value": transform_spec.get("value", 0),
            "method": method,
            "path": path,
            "image_score": float(score),
            "normal_positive_area": float(area),
            "normal_image_positive": int(fp),
        }
        for path, score, area, fp in zip(paths, image_scores, positive_area, image_fp)
    ]


def _aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = sorted({(row["transform_name"], row["transform_value"], row["transform"], row["method"]) for row in rows}, key=lambda item: (str(item[0]), float(item[1]), str(item[3])))
    outputs: list[dict[str, Any]] = []
    metrics = [
        "normal_image_fp_rate",
        "normal_positive_area_mean",
        "normal_positive_area_p95",
        "normal_positive_area_max",
        "image_score_mean",
        "image_score_p95",
        "image_score_max",
    ]
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
        "normal FP",
        "area mean",
        "area p95",
        "score p95",
    ]
    lines = [
        "# NVS normal-variation robustness summary",
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
                    f"{float(row['normal_image_fp_rate']):.6f}",
                    f"{float(row['normal_positive_area_mean']):.6f}",
                    f"{float(row['normal_positive_area_p95']):.6f}",
                    f"{float(row['image_score_p95']):.6f}",
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normal variation robustness benchmark for NVS")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--categories", nargs="+")
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--perturbation-manifest",
        help=(
            "Optional manifest.csv produced by nvs.cache_perturbed_mvtec. "
            "Identity images still come from --data-root."
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
        config["experiment"]["output_dir"] = "outputs/nvs/robustness_dev5_normal_variation"
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
    per_image_rows: list[dict[str, Any]] = []
    transforms = _robustness_transforms(config)
    for category in config["data"]["categories"]:
        train_records, test_records = build_mvtec_records(args.data_root, category)
        normal_test_records = [record for record in test_records if int(record.label) == 0]
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
                    normal_test_records,
                    category=category,
                    transform_spec=spec,
                )
                if cached_images
                else normal_test_records
            )
            scored = _score_normal_records(
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
                        payload["threshold"],
                        config,
                    )
                )
                per_image_rows.extend(
                    _per_image_rows(
                        category,
                        spec,
                        method,
                        scored["paths"],
                        payload["maps"],
                        payload["predictions"],
                        config,
                    )
                )
    write_csv(summary_rows, output_dir / "category_transform_method_summary.csv")
    write_csv(per_image_rows, output_dir / "per_image.csv")
    aggregate_rows = _aggregate_rows(summary_rows)
    write_csv(aggregate_rows, output_dir / "robustness_group_summary.csv")
    _write_markdown(aggregate_rows, output_dir / "robustness_summary.md")

    save_json(
        {
            "status": "complete",
            "categories": list(config["data"]["categories"]),
            "transforms": [transform_name(spec) for spec in transforms],
            "methods": list(args.methods),
            "summary_rows": len(summary_rows),
            "per_image_rows": len(per_image_rows),
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