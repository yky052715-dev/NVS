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
    build_memory_bank,
    build_mvtec_records,
    collate_records,
    extract_features,
    load_config,
    load_dinov2,
    nearest_with_indices,
    patch_scores_to_maps,
    save_json,
    split_normal_records,
    split_records,
    write_csv,
)
from nvs.metrics import (
    binary_f1,
    localization_metrics,
    oracle_pixel_f1,
    safe_auroc,
    threshold_image_max,
)
from nvs.subspace import fit_pca_basis, nvs_residual, pca_feature_residual, sample_rows
from nvs.transforms import is_spatially_aligned


BASE_METHODS = ("R0_nn_distance", "R1_feature_pca_residual", "R2_nvs_residual")


def _fusion_method_name(alpha: float) -> str:
    return f"F_alpha{int(round(float(alpha) * 100)):03d}_r0_r2"


def _fusion_alphas(config: dict[str, Any]) -> list[float]:
    fusion = config.get("fusion", {}) or {}
    if not bool(fusion.get("enabled", False)):
        return []
    return [float(value) for value in fusion.get("alphas", [])]


def _all_methods(config: dict[str, Any]) -> list[str]:
    return list(BASE_METHODS) + [_fusion_method_name(alpha) for alpha in _fusion_alphas(config)]


def _loader(records, config: dict[str, Any], include_mask: bool = False) -> DataLoader:
    dataset = RecordDataset(
        records,
        input_size=int(config["data"]["input_size"]),
        include_mask=include_mask,
    )
    return DataLoader(
        dataset,
        batch_size=int(config["model"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["data"]["num_workers"]),
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_records,
    )


def _transformed_loader(records, config: dict[str, Any], transform_spec: dict[str, Any]) -> DataLoader:
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


def _fit_state(category: str, train_records, config: dict[str, Any], model, device: torch.device) -> dict[str, Any]:
    memory_records, calibration_records = split_normal_records(
        train_records,
        calibration_fraction=float(config["data"]["calibration_fraction"]),
        seed=int(config["experiment"]["seed"]),
    )
    threshold_fit_records, _ = split_records(
        calibration_records,
        fraction=float(config["data"]["threshold_fit_fraction"]),
        seed=int(config["experiment"]["seed"]) + 1009,
    )
    memory_features, _, grid_side = extract_features(model, _loader(memory_records, config), device)
    memory_bank = build_memory_bank(
        memory_features,
        max_bank_size=int(config["memory"]["max_bank_size"]),
        seed=int(config["experiment"]["seed"]),
    )
    feature_samples = sample_rows(
        memory_features,
        max_rows=int(config["subspace"]["max_feature_samples"]),
        seed=int(config["experiment"]["seed"]) + 11,
    )
    feature_mean, feature_basis = fit_pca_basis(
        feature_samples,
        rank=int(config["subspace"]["feature_pca_rank"]),
    )

    delta_chunks: list[torch.Tensor] = []
    aligned_transforms = [spec for spec in config["transforms"] if is_spatially_aligned(spec)]
    original_features, _, _ = extract_features(model, _loader(calibration_records, config), device)
    for spec in aligned_transforms:
        transformed_features, _, _ = extract_features(
            model,
            _transformed_loader(calibration_records, config, spec),
            device,
        )
        delta_chunks.append(transformed_features - original_features)
    delta_samples = sample_rows(
        torch.cat(delta_chunks, dim=0),
        max_rows=int(config["subspace"]["max_delta_samples"]),
        seed=int(config["experiment"]["seed"]) + 23,
    )
    _, nvs_basis = fit_pca_basis(delta_samples, rank=int(config["subspace"]["nvs_rank"]))

    return {
        "category": category,
        "memory_bank": memory_bank,
        "grid_side": int(grid_side),
        "calibration_records": calibration_records,
        "threshold_fit_records": threshold_fit_records,
        "feature_mean": feature_mean,
        "feature_basis": feature_basis,
        "nvs_basis": nvs_basis,
        "memory_count": len(memory_records),
        "calibration_count": len(calibration_records),
        "threshold_fit_count": len(threshold_fit_records),
        "memory_entries": int(memory_bank.shape[0]),
        "feature_samples": int(feature_samples.shape[0]),
        "delta_samples": int(delta_samples.shape[0]),
    }


def _score_batch(features: torch.Tensor, state: dict[str, Any], config: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    distances, indices = nearest_with_indices(
        features.cpu(),
        state["memory_bank"],
        query_chunk_size=int(config["memory"]["query_chunk_size"]),
        bank_chunk_size=int(config["memory"]["bank_chunk_size"]),
        device=device,
    )
    flat_features = features.cpu().reshape(-1, features.shape[-1]).float()
    feature_residual = pca_feature_residual(
        flat_features,
        state["feature_mean"],
        state["feature_basis"],
    ).reshape(features.shape[0], features.shape[1])
    nearest_features = state["memory_bank"][indices.reshape(-1)].float()
    delta = flat_features - nearest_features
    residual = nvs_residual(delta, state["nvs_basis"]).reshape(features.shape[0], features.shape[1])
    return {
        "R0_nn_distance": distances.float(),
        "R1_feature_pca_residual": feature_residual.float(),
        "R2_nvs_residual": residual.float(),
    }


@torch.inference_mode()
def _score_records(records, state: dict[str, Any], config: dict[str, Any], model, device: torch.device, include_mask: bool) -> dict[str, Any]:
    method_maps: dict[str, list[np.ndarray]] = {method: [] for method in BASE_METHODS}
    labels: list[int] = []
    masks: list[np.ndarray] = []
    paths: list[str] = []
    defect_types: list[str] = []
    start = perf_counter()
    image_count = 0
    for batch in tqdm(_loader(records, config, include_mask=include_mask), desc=f"score {state['category']}"):
        from dinov2_mvtec_nn import extract_patch_tokens

        features, grid_side = extract_patch_tokens(model, batch["image"], device)
        scores_by_method = _score_batch(features, state, config, device)
        for method, scores in scores_by_method.items():
            maps = patch_scores_to_maps(
                scores,
                grid_side=int(grid_side),
                output_size=int(config["data"]["input_size"]),
            ).numpy()
            method_maps[method].extend([value.astype(np.float32) for value in maps])
        labels.extend([int(value) for value in batch["label"].tolist()])
        paths.extend(batch["path"])
        defect_types.extend(batch["defect_type"])
        if include_mask:
            masks.extend([value.numpy().astype(np.uint8) for value in batch["mask"]])
        image_count += int(batch["image"].shape[0])
    elapsed = perf_counter() - start
    return {
        "maps": {method: np.stack(values) for method, values in method_maps.items()},
        "labels": np.asarray(labels, dtype=np.int64),
        "masks": np.stack(masks) if include_mask else None,
        "paths": paths,
        "defect_types": defect_types,
        "seconds": elapsed,
        "images": image_count,
    }


def _normalize_base_maps(raw_maps_by_method: dict[str, np.ndarray], calibrations: dict[str, dict[str, float]]) -> dict[str, np.ndarray]:
    normalized: dict[str, np.ndarray] = {}
    for method in BASE_METHODS:
        cal = calibrations[method]
        normalized[method] = np.maximum((raw_maps_by_method[method] - cal["median"]) / cal["mad"], 0.0)
    return normalized


def _add_fusion_maps(maps_by_method: dict[str, np.ndarray], config: dict[str, Any]) -> dict[str, np.ndarray]:
    output = dict(maps_by_method)
    r0 = maps_by_method["R0_nn_distance"]
    r2 = maps_by_method["R2_nvs_residual"]
    for alpha in _fusion_alphas(config):
        output[_fusion_method_name(alpha)] = float(alpha) * r0 + (1.0 - float(alpha)) * r2
    return output


def _calibrate_methods(state: dict[str, Any], config: dict[str, Any], model, device: torch.device) -> dict[str, dict[str, float]]:
    calibration = _score_records(
        state["calibration_records"],
        state,
        config,
        model,
        device,
        include_mask=False,
    )
    fit_set = {str(record.path) for record in state["threshold_fit_records"]}
    fit_indices = [index for index, path in enumerate(calibration["paths"]) if path in fit_set]
    if not fit_indices:
        raise RuntimeError(f"No threshold-fit records found for {state['category']}")
    calibrations: dict[str, dict[str, float]] = {}
    for method in BASE_METHODS:
        raw_maps = calibration["maps"][method]
        flat = raw_maps.reshape(-1)
        median = float(np.median(flat))
        mad = float(np.median(np.abs(flat - median)))
        mad = max(mad, float(config["calibration"]["mad_epsilon"]))
        maps = np.maximum((raw_maps - median) / mad, 0.0)
        threshold = threshold_image_max(
            maps[np.asarray(fit_indices, dtype=np.int64)],
            image_quantile=float(config["calibration"]["image_quantile"]),
        )
        calibrations[method] = {
            "kind": "base",
            "median": median,
            "mad": mad,
            "pixel_threshold": float(threshold),
        }

    normalized = _normalize_base_maps(calibration["maps"], calibrations)
    fused = _add_fusion_maps(normalized, config)
    for alpha in _fusion_alphas(config):
        method = _fusion_method_name(alpha)
        threshold = threshold_image_max(
            fused[method][np.asarray(fit_indices, dtype=np.int64)],
            image_quantile=float(config["calibration"]["image_quantile"]),
        )
        calibrations[method] = {
            "kind": "fusion",
            "alpha_r0": float(alpha),
            "alpha_r2": float(1.0 - float(alpha)),
            "median": 0.0,
            "mad": 1.0,
            "pixel_threshold": float(threshold),
        }
    return calibrations


def _evaluate_category(category: str, train_records, test_records, config: dict[str, Any], model, device: torch.device, output_dir: Path) -> list[dict[str, Any]]:
    state = _fit_state(category, train_records, config, model, device)
    calibrations = _calibrate_methods(state, config, model, device)
    scored = _score_records(test_records, state, config, model, device, include_mask=True)
    base_maps = _normalize_base_maps(scored["maps"], calibrations)
    maps_by_method = _add_fusion_maps(base_maps, config)
    rows: list[dict[str, Any]] = []
    category_dir = output_dir / category
    category_dir.mkdir(parents=True, exist_ok=True)
    for method in _all_methods(config):
        cal = calibrations[method]
        maps = maps_by_method[method]
        masks = scored["masks"].astype(bool)
        labels = scored["labels"]
        threshold = float(cal["pixel_threshold"])
        pixel_auroc = safe_auroc(masks.reshape(-1), maps.reshape(-1))
        pixel_f1 = binary_f1(masks, maps >= threshold)
        oracle_f1, oracle_threshold = oracle_pixel_f1(masks, maps)
        loc = localization_metrics(
            masks,
            maps,
            labels,
            threshold,
            small_defect_area_fraction=float(config["metrics"]["small_defect_area_fraction"]),
        )
        row = {
            "category": category,
            "method": method,
            "method_kind": str(cal.get("kind", "base")),
            "alpha_r0": cal.get("alpha_r0"),
            "alpha_r2": cal.get("alpha_r2"),
            "pixel_AUROC": pixel_auroc,
            "pixel_F1_calibrated": pixel_f1,
            "pixel_F1_oracle": oracle_f1,
            "pixel_threshold_calibrated": threshold,
            "pixel_threshold_oracle": oracle_threshold,
            "inference_ms_per_image": float(scored["seconds"] * 1000.0 / max(1, scored["images"])),
            "memory_entries": int(state["memory_entries"]),
            "feature_samples": int(state["feature_samples"]),
            "delta_samples": int(state["delta_samples"]),
            **loc,
        }
        rows.append(row)
        save_json(row, category_dir / f"{method}.json")
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
        category_dir / "state_summary.json",
    )
    return rows


def _mean_rows(rows: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for method in _all_methods(config):
        method_rows = [row for row in rows if row["method"] == method]
        if not method_rows:
            continue
        summary: dict[str, Any] = {
            "method": method,
            "method_kind": method_rows[0].get("method_kind", "base"),
            "alpha_r0": method_rows[0].get("alpha_r0"),
            "alpha_r2": method_rows[0].get("alpha_r2"),
            "categories": len(method_rows),
        }
        for key in method_rows[0]:
            if key in {"category", "method", "method_kind", "alpha_r0", "alpha_r2"}:
                continue
            values = np.asarray([float(row[key]) for row in method_rows], dtype=np.float64)
            summary[key] = float(np.nanmean(values))
        output.append(summary)
    return output


def _write_markdown(summary_rows: list[dict[str, Any]], path: Path) -> None:
    keys = [
        "method",
        "pixel_AUROC",
        "pixel_F1_calibrated",
        "pixel_F1_oracle",
        "localization_overseg_anomaly_macro",
        "localization_recall_anomaly_macro",
        "localization_small_defect_recall_macro",
        "localization_test_normal_image_positive_rate",
        "inference_ms_per_image",
    ]
    lines = [
        "# NVS detection Dev5 summary",
        "",
        "|" + "|".join(keys) + "|",
        "|" + "|".join(["---"] * len(keys)) + "|",
    ]
    for row in summary_rows:
        values = []
        for key in keys:
            value = row[key]
            if isinstance(value, float):
                values.append(f"{value:.6f}")
            else:
                values.append(str(value))
        lines.append("|" + "|".join(values) + "|")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NVS detection prototype on MVTec Dev5")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--categories", nargs="+")
    parser.add_argument("--output-dir")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.categories:
        config["data"]["categories"] = args.categories
    if args.output_dir:
        config["experiment"]["output_dir"] = args.output_dir
    seed = int(config["experiment"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(args.device)
    model = load_dinov2(config["model"]["name"], device, config["model"].get("hub_dir"))
    output_dir = Path(config["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(config, output_dir / "resolved_config.json")
    rows: list[dict[str, Any]] = []
    for category in config["data"]["categories"]:
        train_records, test_records = build_mvtec_records(args.data_root, category)
        rows.extend(_evaluate_category(category, train_records, test_records, config, model, device, output_dir))
        write_csv(rows, output_dir / "category_metrics.csv")
        mean_rows = _mean_rows(rows, config)
        write_csv(mean_rows, output_dir / "method_summary.csv")
        _write_markdown(mean_rows, output_dir / "detection_summary.md")
    save_json(
        {
            "status": "complete",
            "categories": list(config["data"]["categories"]),
            "completed_count": len(config["data"]["categories"]),
            "methods": _all_methods(config),
            "fusion_alphas": _fusion_alphas(config),
        },
        output_dir / "experiment_complete.json",
    )


if __name__ == "__main__":
    main()