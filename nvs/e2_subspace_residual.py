from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
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
    save_json,
    split_normal_records,
    write_csv,
)
from nvs.metrics import safe_auroc, summarize_values
from nvs.subspace import (
    explained_ratio,
    fit_pca_basis,
    nvs_residual,
    pca_feature_residual,
    sample_rows,
)
from nvs.transforms import is_spatially_aligned, transform_name


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


def _fit_state(category: str, train_records, config: dict, model, device: torch.device) -> dict:
    memory_records, calibration_records = split_normal_records(
        train_records,
        calibration_fraction=float(config["data"]["calibration_fraction"]),
        seed=int(config["experiment"]["seed"]),
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
    transforms = [
        spec for spec in config["transforms"] if is_spatially_aligned(spec)
    ]
    for spec in transforms:
        original_features, _, _ = extract_features(
            model,
            _loader(calibration_records, config),
            device,
        )
        transformed_features, _, _ = extract_features(
            model,
            _loader(calibration_records, config, transform_spec=spec),
            device,
        )
        delta_chunks.append(transformed_features - original_features)
    delta_samples = sample_rows(
        torch.cat(delta_chunks, dim=0),
        max_rows=int(config["subspace"]["max_delta_samples"]),
        seed=int(config["experiment"]["seed"]) + 23,
    )
    _, nvs_basis = fit_pca_basis(
        delta_samples,
        rank=int(config["subspace"]["nvs_rank"]),
    )
    return {
        "category": category,
        "memory_bank": memory_bank,
        "grid_side": int(grid_side),
        "calibration_records": calibration_records,
        "feature_mean": feature_mean,
        "feature_basis": feature_basis,
        "nvs_basis": nvs_basis,
        "memory_entries": int(memory_bank.shape[0]),
        "delta_samples": int(delta_samples.shape[0]),
        "feature_samples": int(feature_samples.shape[0]),
    }


def _collect_transformed_normal_rows(
    category: str,
    state: dict,
    config: dict,
    model,
    device: torch.device,
) -> list[dict]:
    rows: list[dict] = []
    max_rows = int(config["subspace"]["max_normal_samples"])
    per_transform_budget = max(1, max_rows // max(1, len(config["transforms"])))
    for spec in config["transforms"]:
        if not is_spatially_aligned(spec):
            continue
        features, paths, _ = extract_features(
            model,
            _loader(state["calibration_records"], config, transform_spec=spec),
            device,
        )
        distances, indices = nearest_with_indices(
            features,
            state["memory_bank"],
            query_chunk_size=int(config["memory"]["query_chunk_size"]),
            bank_chunk_size=int(config["memory"]["bank_chunk_size"]),
            device=device,
        )
        flat_features = features.reshape(-1, features.shape[-1]).float()
        flat_indices = indices.reshape(-1)
        nn_features = state["memory_bank"][flat_indices].float()
        delta = flat_features - nn_features
        seed = int(config["experiment"]["seed"]) + abs(hash(transform_name(spec))) % 10000
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        take = min(per_transform_budget, flat_features.shape[0])
        selected = torch.randperm(flat_features.shape[0], generator=generator)[:take]
        nn_distance = distances.reshape(-1)[selected].numpy()
        feature_residual = pca_feature_residual(
            flat_features[selected],
            state["feature_mean"],
            state["feature_basis"],
        ).numpy()
        residual = nvs_residual(delta[selected], state["nvs_basis"]).numpy()
        ratio = explained_ratio(delta[selected], state["nvs_basis"]).numpy()
        for values in zip(nn_distance, feature_residual, residual, ratio):
            rows.append(
                {
                    "category": category,
                    "group": "transformed_normal",
                    "transform": transform_name(spec),
                    "label": 0,
                    "nn_distance": float(values[0]),
                    "feature_pca_residual": float(values[1]),
                    "nvs_residual": float(values[2]),
                    "nvs_explained_ratio": float(values[3]),
                }
            )
    return rows


def _patch_defect_mask(masks: torch.Tensor, grid_side: int, threshold: float) -> torch.Tensor:
    if masks.ndim == 3:
        masks = masks[:, None].float()
    pooled = F.interpolate(
        masks.float(),
        size=(int(grid_side), int(grid_side)),
        mode="area",
    )[:, 0]
    return pooled.reshape(pooled.shape[0], -1) >= float(threshold)


def _collect_defect_rows(category: str, state: dict, test_records, config: dict, model, device: torch.device) -> list[dict]:
    rows: list[dict] = []
    max_rows = int(config["subspace"]["max_defect_samples"])
    for batch in tqdm(_loader(test_records, config, include_mask=True), desc=f"E2 defects {category}"):
        labels = batch["label"].numpy()
        if int(labels.sum()) == 0:
            continue
        from dinov2_mvtec_nn import extract_patch_tokens

        features, grid_side = extract_patch_tokens(model, batch["image"], device)
        distances, indices = nearest_with_indices(
            features.cpu(),
            state["memory_bank"],
            query_chunk_size=int(config["memory"]["query_chunk_size"]),
            bank_chunk_size=int(config["memory"]["bank_chunk_size"]),
            device=device,
        )
        mask = _patch_defect_mask(
            batch["mask"],
            grid_side=grid_side,
            threshold=float(config["subspace"]["defect_mask_threshold"]),
        )
        flat_features = features.cpu().reshape(-1, features.shape[-1]).float()
        flat_indices = indices.reshape(-1)
        flat_mask = mask.reshape(-1)
        if not flat_mask.any():
            continue
        selected_all = torch.nonzero(flat_mask, as_tuple=False).reshape(-1)
        if len(rows) + selected_all.numel() > max_rows:
            selected_all = selected_all[: max(0, max_rows - len(rows))]
        if selected_all.numel() == 0:
            break
        nn_features = state["memory_bank"][flat_indices[selected_all]].float()
        delta = flat_features[selected_all] - nn_features
        nn_distance = distances.reshape(-1)[selected_all].numpy()
        feature_residual = pca_feature_residual(
            flat_features[selected_all],
            state["feature_mean"],
            state["feature_basis"],
        ).numpy()
        residual = nvs_residual(delta, state["nvs_basis"]).numpy()
        ratio = explained_ratio(delta, state["nvs_basis"]).numpy()
        for values in zip(nn_distance, feature_residual, residual, ratio):
            rows.append(
                {
                    "category": category,
                    "group": "real_defect",
                    "transform": "none",
                    "label": 1,
                    "nn_distance": float(values[0]),
                    "feature_pca_residual": float(values[1]),
                    "nvs_residual": float(values[2]),
                    "nvs_explained_ratio": float(values[3]),
                }
            )
        if len(rows) >= max_rows:
            break
    return rows


def _summary_from_rows(category: str, rows: list[dict], state: dict) -> dict:
    labels = np.asarray([row["label"] for row in rows], dtype=np.uint8)
    nn = np.asarray([row["nn_distance"] for row in rows], dtype=np.float32)
    feature = np.asarray([row["feature_pca_residual"] for row in rows], dtype=np.float32)
    nvs = np.asarray([row["nvs_residual"] for row in rows], dtype=np.float32)
    ratio = np.asarray([row["nvs_explained_ratio"] for row in rows], dtype=np.float32)
    normal = labels == 0
    defect = labels == 1
    summary = {
        "category": category,
        "rows": len(rows),
        "normal_rows": int(normal.sum()),
        "defect_rows": int(defect.sum()),
        "memory_entries": state["memory_entries"],
        "feature_samples": state["feature_samples"],
        "delta_samples": state["delta_samples"],
        "nn_distance_auc": safe_auroc(labels, nn),
        "feature_pca_residual_auc": safe_auroc(labels, feature),
        "nvs_residual_auc": safe_auroc(labels, nvs),
    }
    for prefix, values in [
        ("normal_nn", nn[normal]),
        ("defect_nn", nn[defect]),
        ("normal_feature_pca", feature[normal]),
        ("defect_feature_pca", feature[defect]),
        ("normal_nvs", nvs[normal]),
        ("defect_nvs", nvs[defect]),
        ("normal_explained", ratio[normal]),
        ("defect_explained", ratio[defect]),
    ]:
        for key, value in summarize_values(values).items():
            summary[f"{prefix}_{key}"] = value
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="E2: NVS residual distribution experiment")
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
    all_rows: list[dict] = []
    for category in config["data"]["categories"]:
        train_records, test_records = build_mvtec_records(args.data_root, category)
        state = _fit_state(category, train_records, config, model, device)
        normal_rows = _collect_transformed_normal_rows(category, state, config, model, device)
        defect_rows = _collect_defect_rows(category, state, test_records, config, model, device)
        rows = normal_rows + defect_rows
        summary = _summary_from_rows(category, rows, state)
        summary_rows.append(summary)
        all_rows.extend(rows)
        category_dir = output_dir / category
        write_csv(rows, category_dir / "residual_rows.csv")
        save_json(summary, category_dir / "summary.json")
        write_csv(summary_rows, output_dir / "residual_summary.csv")
    write_csv(all_rows, output_dir / "residual_distribution.csv")
    save_json(
        {
            "status": "complete",
            "categories": list(config["data"]["categories"]),
            "summary_rows": len(summary_rows),
            "residual_rows": len(all_rows),
        },
        output_dir / "experiment_complete.json",
    )


if __name__ == "__main__":
    main()

