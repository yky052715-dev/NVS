from __future__ import annotations

import argparse
import random
from pathlib import Path

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
from nvs.metrics import image_score_from_map, summarize_values, threshold_image_max
from nvs.transforms import transform_name


def _loader(records, config: dict, transform_spec=None, include_mask: bool = False) -> DataLoader:
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


def _calibrate_category(
    category: str,
    train_records,
    config: dict,
    model,
    device: torch.device,
) -> dict:
    memory_records, calibration_records = split_normal_records(
        train_records,
        calibration_fraction=float(config["data"]["calibration_fraction"]),
        seed=int(config["experiment"]["seed"]),
    )
    threshold_fit_records, normal_validation_records = split_records(
        calibration_records,
        fraction=float(config["data"]["threshold_fit_fraction"]),
        seed=int(config["experiment"]["seed"]) + 1009,
    )
    memory_features, _, grid_side = extract_features(
        model,
        _loader(memory_records, config),
        device,
    )
    memory_bank = build_memory_bank(
        memory_features,
        max_bank_size=int(config["memory"]["max_bank_size"]),
        seed=int(config["experiment"]["seed"]),
    )
    calibration_features, calibration_paths, _ = extract_features(
        model,
        _loader(calibration_records, config),
        device,
    )
    calibration_scores, _ = nearest_with_indices(
        calibration_features,
        memory_bank,
        query_chunk_size=int(config["memory"]["query_chunk_size"]),
        bank_chunk_size=int(config["memory"]["bank_chunk_size"]),
        device=device,
    )
    median = float(torch.median(calibration_scores.reshape(-1)).item())
    mad = float(torch.median(torch.abs(calibration_scores.reshape(-1) - median)).item())
    mad = max(mad, float(config["calibration"]["mad_epsilon"]))
    calibration_scores = ((calibration_scores - median) / mad).clamp_min(0.0)
    calibration_maps = patch_scores_to_maps(
        calibration_scores,
        grid_side=grid_side,
        output_size=int(config["data"]["input_size"]),
    ).numpy()
    fit_set = {str(record.path) for record in threshold_fit_records}
    fit_indices = [index for index, path in enumerate(calibration_paths) if path in fit_set]
    threshold = threshold_image_max(
        calibration_maps[np.asarray(fit_indices, dtype=np.int64)],
        image_quantile=float(config["calibration"]["image_quantile"]),
    )
    return {
        "memory_bank": memory_bank,
        "grid_side": grid_side,
        "median": median,
        "mad": mad,
        "pixel_threshold": threshold,
        "memory_count": len(memory_records),
        "calibration_count": len(calibration_records),
        "threshold_fit_count": len(threshold_fit_records),
        "normal_validation_count": len(normal_validation_records),
    }


def _evaluate_transform(
    category: str,
    normal_records,
    transform_spec: dict,
    state: dict,
    config: dict,
    model,
    device: torch.device,
) -> tuple[dict, list[dict]]:
    loader = _loader(normal_records, config, transform_spec=transform_spec)
    maps_all: list[np.ndarray] = []
    path_all: list[str] = []
    for batch in tqdm(loader, desc=f"{category} {transform_name(transform_spec)}"):
        features, _ = model_forward(model, batch["image"], device)
        scores, _ = nearest_with_indices(
            features.cpu(),
            state["memory_bank"],
            query_chunk_size=int(config["memory"]["query_chunk_size"]),
            bank_chunk_size=int(config["memory"]["bank_chunk_size"]),
            device=device,
        )
        scores = ((scores - state["median"]) / state["mad"]).clamp_min(0.0)
        maps = patch_scores_to_maps(
            scores,
            grid_side=int(state["grid_side"]),
            output_size=int(config["data"]["input_size"]),
        ).numpy()
        maps_all.extend([value.astype(np.float32) for value in maps])
        path_all.extend(batch["path"])
    maps_np = np.stack(maps_all)
    threshold = float(state["pixel_threshold"])
    predictions = maps_np >= threshold
    positive_area = predictions.reshape(predictions.shape[0], -1).mean(axis=1)
    image_fp = predictions.reshape(predictions.shape[0], -1).any(axis=1).astype(np.float32)
    image_scores = image_score_from_map(
        maps_np,
        method=str(config["calibration"]["image_score"]),
        topk_fraction=float(config["calibration"]["image_topk_fraction"]),
    )
    summary = {
        "category": category,
        "transform": transform_name(transform_spec),
        "transform_name": str(transform_spec["name"]),
        "transform_value": transform_spec.get("value", 0),
        "normal_images": len(maps_np),
        "pixel_threshold": threshold,
        "normal_image_fp_rate": float(image_fp.mean()),
        "normal_positive_area_mean": float(positive_area.mean()),
        "normal_positive_area_p95": float(np.quantile(positive_area, 0.95)),
        "image_score_mean": float(image_scores.mean()),
        "image_score_p95": float(np.quantile(image_scores, 0.95)),
    }
    per_image = [
        {
            "category": category,
            "transform": transform_name(transform_spec),
            "path": path,
            "image_score": float(score),
            "normal_positive_area": float(area),
            "normal_image_positive": int(fp),
        }
        for path, score, area, fp in zip(path_all, image_scores, positive_area, image_fp)
    ]
    return summary, per_image


@torch.inference_mode()
def model_forward(model, images: torch.Tensor, device: torch.device):
    from dinov2_mvtec_nn import extract_patch_tokens

    return extract_patch_tokens(model, images, device)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="E1: normal transform false-positive curves")
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
    random.seed(int(config["experiment"]["seed"]))
    np.random.seed(int(config["experiment"]["seed"]))
    torch.manual_seed(int(config["experiment"]["seed"]))
    device = torch.device(args.device)
    model = load_dinov2(
        config["model"]["name"],
        device,
        config["model"].get("hub_dir"),
    )
    output_dir = Path(config["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(config, output_dir / "resolved_config.json")
    summary_rows: list[dict] = []
    per_image_rows: list[dict] = []
    for category in config["data"]["categories"]:
        train_records, test_records = build_mvtec_records(args.data_root, category)
        normal_test_records = [record for record in test_records if record.label == 0]
        state = _calibrate_category(category, train_records, config, model, device)
        save_json(
            {key: value for key, value in state.items() if key != "memory_bank"},
            output_dir / category / "calibration.json",
        )
        for transform_spec in config["transforms"]:
            summary, per_image = _evaluate_transform(
                category,
                normal_test_records,
                transform_spec,
                state,
                config,
                model,
                device,
            )
            summary_rows.append(summary)
            per_image_rows.extend(per_image)
            write_csv(summary_rows, output_dir / "summary.csv")
            write_csv(per_image_rows, output_dir / "per_image.csv")
    save_json(
        {
            "status": "complete",
            "categories": list(config["data"]["categories"]),
            "summary_rows": len(summary_rows),
            "per_image_rows": len(per_image_rows),
        },
        output_dir / "experiment_complete.json",
    )


if __name__ == "__main__":
    main()

