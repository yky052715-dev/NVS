from __future__ import annotations

import argparse
import random
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from nvs.apc import apc_residual_score, rho_gate
from nvs.apc_separability import _stats
from nvs.common import (
    build_mvtec_records,
    load_config,
    load_dinov2,
    patch_scores_to_maps,
    save_json,
    write_csv,
)
from nvs.detection import _fit_state, _loader
from nvs.metrics import (
    binary_f1,
    localization_metrics_from_prediction,
    oracle_pixel_f1,
    safe_auroc,
    threshold_image_max,
)
from nvs.transforms import transform_name


METHODS = ("C0_r0", "C1_r2", "C2_apc_rho")


@torch.inference_mode()
def _score_records(
    records,
    transform_spec: dict[str, Any] | None,
    state: dict[str, Any],
    config: dict[str, Any],
    model,
    device: torch.device,
    include_mask: bool,
) -> dict[str, Any]:
    from dinov2_mvtec_nn import extract_patch_tokens

    r0_chunks: list[np.ndarray] = []
    parallel_chunks: list[np.ndarray] = []
    perpendicular_chunks: list[np.ndarray] = []
    rho_chunks: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    labels: list[int] = []
    paths: list[str] = []
    defect_types: list[str] = []
    dataset = _loader(records, config, include_mask=include_mask).dataset
    dataset.transform_spec = transform_spec
    start = perf_counter()
    image_count = 0
    name = "identity" if transform_spec is None else transform_name(transform_spec)
    for batch in tqdm(
        torch.utils.data.DataLoader(
            dataset,
            batch_size=int(config["model"]["batch_size"]),
            shuffle=False,
            num_workers=int(config["data"]["num_workers"]),
            pin_memory=torch.cuda.is_available(),
            collate_fn=_loader(records[:1], config).collate_fn,
        ),
        desc=f"C2 score {state['category']} {name}",
    ):
        features, grid_side = extract_patch_tokens(model, batch["image"], device)
        stats = _stats(features, state, config, device)
        r0_chunks.append(stats["r0_nn_distance"].cpu().numpy())
        parallel_chunks.append(stats["parallel_norm"].cpu().numpy())
        perpendicular_chunks.append(stats["perpendicular_norm"].cpu().numpy())
        rho_chunks.append(stats["rho"].cpu().numpy())
        labels.extend(int(value) for value in batch["label"].tolist())
        paths.extend(batch["path"])
        defect_types.extend(batch["defect_type"])
        if include_mask:
            masks.extend(value.numpy().astype(np.uint8) for value in batch["mask"])
        image_count += int(batch["image"].shape[0])
    return {
        "r0": np.concatenate(r0_chunks),
        "parallel": np.concatenate(parallel_chunks),
        "perpendicular": np.concatenate(perpendicular_chunks),
        "rho": np.concatenate(rho_chunks),
        "grid_side": int(grid_side),
        "labels": np.asarray(labels, dtype=np.int64),
        "masks": np.stack(masks) if include_mask else None,
        "paths": paths,
        "defect_types": defect_types,
        "seconds": perf_counter() - start,
        "images": image_count,
    }


def _raw_maps(
    scored: dict[str, Any],
    output_size: int,
    tau_rho: float,
    temperature: float,
) -> dict[str, np.ndarray]:
    r0 = torch.from_numpy(scored["r0"])
    parallel = torch.from_numpy(scored["parallel"])
    perpendicular = torch.from_numpy(scored["perpendicular"])
    rho = torch.from_numpy(scored["rho"])
    gate = rho_gate(rho, tau_rho, temperature)
    c2 = apc_residual_score(parallel, perpendicular, gate)
    return {
        "C0_r0": patch_scores_to_maps(
            r0, scored["grid_side"], output_size
        ).numpy(),
        "C1_r2": patch_scores_to_maps(
            perpendicular, scored["grid_side"], output_size
        ).numpy(),
        "C2_apc_rho": patch_scores_to_maps(
            c2, scored["grid_side"], output_size
        ).numpy(),
    }


def _fit_calibration(
    maps: np.ndarray,
    fit_indices: list[int],
    config: dict[str, Any],
) -> dict[str, float]:
    flat = maps.reshape(-1)
    median = float(np.median(flat))
    mad = max(
        float(np.median(np.abs(flat - median))),
        float(config["calibration"]["mad_epsilon"]),
    )
    normalized = np.maximum((maps - median) / mad, 0.0)
    threshold = threshold_image_max(
        normalized[np.asarray(fit_indices, dtype=np.int64)],
        image_quantile=float(config["calibration"]["image_quantile"]),
    )
    return {"median": median, "mad": mad, "pixel_threshold": threshold}


def _normalize(
    maps: np.ndarray,
    calibration: dict[str, float],
) -> np.ndarray:
    return np.maximum(
        (maps - calibration["median"]) / calibration["mad"],
        0.0,
    )


def _normal_fp(
    maps: list[np.ndarray],
    calibration: dict[str, float],
    indices: list[int],
) -> tuple[float, float]:
    selected = np.concatenate(
        [_normalize(values, calibration)[indices] for values in maps],
        axis=0,
    )
    predictions = selected >= float(calibration["pixel_threshold"])
    return (
        float(predictions.reshape(predictions.shape[0], -1).any(axis=1).mean()),
        float(predictions.mean()),
    )


def _calibrate_and_select(
    state: dict[str, Any],
    config: dict[str, Any],
    model,
    device: torch.device,
) -> tuple[dict[str, dict[str, float]], dict[str, Any], list[dict[str, Any]]]:
    records = state["calibration_records"]
    identity = _score_records(
        records, None, state, config, model, device, include_mask=False
    )
    fit_set = {str(record.path) for record in state["threshold_fit_records"]}
    fit_indices = [
        index for index, path in enumerate(identity["paths"]) if path in fit_set
    ]
    holdout_indices = [
        index for index, path in enumerate(identity["paths"]) if path not in fit_set
    ]
    if not holdout_indices:
        holdout_indices = list(range(len(identity["paths"])))

    known_scores = [
        _score_records(records, spec, state, config, model, device, False)
        for spec in config["transforms"]
    ]
    quantiles = [float(value) for value in config["apc_c2"]["rho_quantiles"]]
    temperature = float(config["apc_c2"]["temperature"])
    tau_by_q = {
        q: float(np.quantile(identity["rho"].reshape(-1), q))
        for q in quantiles
    }

    base_maps = _raw_maps(
        identity,
        int(config["data"]["input_size"]),
        tau_by_q[quantiles[0]],
        temperature,
    )
    calibrations = {
        "C0_r0": _fit_calibration(base_maps["C0_r0"], fit_indices, config),
        "C1_r2": _fit_calibration(base_maps["C1_r2"], fit_indices, config),
    }
    candidate_rows: list[dict[str, Any]] = []
    candidate_calibrations: dict[float, dict[str, float]] = {}
    for q in quantiles:
        tau = tau_by_q[q]
        identity_maps = _raw_maps(
            identity,
            int(config["data"]["input_size"]),
            tau,
            temperature,
        )
        calibration = _fit_calibration(
            identity_maps["C2_apc_rho"], fit_indices, config
        )
        candidate_calibrations[q] = calibration
        known_maps = [
            _raw_maps(
                scored,
                int(config["data"]["input_size"]),
                tau,
                temperature,
            )["C2_apc_rho"]
            for scored in known_scores
        ]
        identity_fp, identity_area = _normal_fp(
            [identity_maps["C2_apc_rho"]],
            calibration,
            holdout_indices,
        )
        known_fp, known_area = _normal_fp(
            known_maps, calibration, holdout_indices
        )
        candidate_rows.append(
            {
                "rho_quantile": q,
                "tau_rho": tau,
                "identity_normal_fp": identity_fp,
                "identity_positive_area": identity_area,
                "known_transformed_normal_fp": known_fp,
                "known_transformed_positive_area": known_area,
            }
        )

    r0_identity_fp, _ = _normal_fp(
        [base_maps["C0_r0"]],
        calibrations["C0_r0"],
        holdout_indices,
    )
    tolerance = float(config["apc_c2"]["identity_fp_tolerance"])
    feasible = [
        row
        for row in candidate_rows
        if row["identity_normal_fp"] <= r0_identity_fp + tolerance
    ]
    pool = feasible or candidate_rows
    selected = min(
        pool,
        key=lambda row: (
            row["known_transformed_normal_fp"],
            row["known_transformed_positive_area"],
            -row["rho_quantile"],
        ),
    )
    calibrations["C2_apc_rho"] = candidate_calibrations[
        float(selected["rho_quantile"])
    ]
    selection = {
        **selected,
        "r0_identity_normal_fp": r0_identity_fp,
        "identity_fp_tolerance": tolerance,
        "selection_feasible": bool(feasible),
    }
    return calibrations, selection, candidate_rows


def _evaluate(
    category: str,
    shift_group: str,
    condition: str,
    scored: dict[str, Any],
    calibrations: dict[str, dict[str, float]],
    selection: dict[str, Any],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    raw_maps = _raw_maps(
        scored,
        int(config["data"]["input_size"]),
        float(selection["tau_rho"]),
        float(config["apc_c2"]["temperature"]),
    )
    masks = scored["masks"].astype(bool)
    labels = scored["labels"]
    rows: list[dict[str, Any]] = []
    for method in METHODS:
        maps = _normalize(raw_maps[method], calibrations[method])
        threshold = float(calibrations[method]["pixel_threshold"])
        predictions = maps >= threshold
        oracle_f1, oracle_threshold = oracle_pixel_f1(masks, maps)
        rows.append(
            {
                "category": category,
                "shift_group": shift_group,
                "condition": condition,
                "method": method,
                "rho_quantile": (
                    selection["rho_quantile"]
                    if method == "C2_apc_rho"
                    else None
                ),
                "tau_rho": (
                    selection["tau_rho"]
                    if method == "C2_apc_rho"
                    else None
                ),
                "pixel_AUROC": safe_auroc(
                    masks.reshape(-1), maps.reshape(-1)
                ),
                "pixel_F1_calibrated": binary_f1(masks, predictions),
                "pixel_F1_oracle": oracle_f1,
                "pixel_threshold_calibrated": threshold,
                "pixel_threshold_oracle": oracle_threshold,
                "inference_ms_per_image": float(
                    scored["seconds"] * 1000.0 / max(1, scored["images"])
                ),
                **localization_metrics_from_prediction(
                    masks,
                    predictions,
                    labels,
                    small_defect_area_fraction=float(
                        config["metrics"]["small_defect_area_fraction"]
                    ),
                ),
            }
        )
    return rows


def _mean_rows(
    rows: list[dict[str, Any]],
    group_key: str,
) -> list[dict[str, Any]]:
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
    keys = sorted({(row[group_key], row["method"]) for row in rows})
    output: list[dict[str, Any]] = []
    for group, method in keys:
        selected = [
            row
            for row in rows
            if row[group_key] == group and row["method"] == method
        ]
        result: dict[str, Any] = {
            group_key: group,
            "method": method,
            "rows": len(selected),
        }
        for metric in metrics:
            values = np.asarray([float(row[metric]) for row in selected])
            result[metric] = (
                float(np.nanmean(values))
                if np.isfinite(values).any()
                else float("nan")
            )
        output.append(result)
    return output


def _write_report(
    group_rows: list[dict[str, Any]],
    path: Path,
) -> None:
    keys = [
        "shift_group",
        "method",
        "pixel_AUROC",
        "pixel_F1_calibrated",
        "pixel_F1_oracle",
        "localization_overseg_anomaly_macro",
        "localization_recall_anomaly_macro",
        "localization_small_defect_recall_macro",
        "localization_test_normal_image_positive_rate",
    ]
    lines = [
        "# APC-NVS C2 Dev5 summary",
        "",
        "|" + "|".join(keys) + "|",
        "|" + "|".join(["---"] * len(keys)) + "|",
    ]
    for row in group_rows:
        values = []
        for key in keys:
            value = row[key]
            values.append(
                f"{value:.6f}" if isinstance(value, float) else str(value)
            )
        lines.append("|" + "|".join(values) + "|")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="APC-NVS C2 experiment")
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
    model = load_dinov2(
        config["model"]["name"], device, config["model"].get("hub_dir")
    )
    output_dir = Path(config["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(config, output_dir / "resolved_config.json")

    rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    conditions = [("identity", "identity", None)]
    conditions += [
        ("known_shift", transform_name(spec), spec)
        for spec in config["transforms"]
    ]
    conditions += [
        ("unseen_strength", transform_name(spec), spec)
        for spec in config["apc_c2"]["unseen_strength_transforms"]
    ]
    for category in config["data"]["categories"]:
        print(f"[C2 fit] {category}")
        train_records, test_records = build_mvtec_records(
            args.data_root, category
        )
        state = _fit_state(category, train_records, config, model, device)
        calibrations, selection, candidates = _calibrate_and_select(
            state, config, model, device
        )
        selection_rows.append({"category": category, **selection})
        candidate_rows += [
            {"category": category, **candidate} for candidate in candidates
        ]
        category_dir = output_dir / category
        category_dir.mkdir(parents=True, exist_ok=True)
        save_json(
            {
                "category": category,
                "selection": selection,
                "calibrations": calibrations,
            },
            category_dir / "c2_state.json",
        )
        for shift_group, condition, spec in conditions:
            scored = _score_records(
                test_records,
                spec,
                state,
                config,
                model,
                device,
                include_mask=True,
            )
            rows += _evaluate(
                category,
                shift_group,
                condition,
                scored,
                calibrations,
                selection,
                config,
            )
            write_csv(rows, output_dir / "category_condition_metrics.csv")

    condition_rows = _mean_rows(rows, "condition")
    group_rows = _mean_rows(rows, "shift_group")
    write_csv(selection_rows, output_dir / "selected_rho_thresholds.csv")
    write_csv(candidate_rows, output_dir / "rho_candidate_selection.csv")
    write_csv(condition_rows, output_dir / "condition_summary.csv")
    write_csv(group_rows, output_dir / "shift_group_summary.csv")
    _write_report(group_rows, output_dir / "apc_c2_summary.md")
    save_json(
        {
            "status": "complete",
            "categories": list(config["data"]["categories"]),
            "completed_count": len(config["data"]["categories"]),
            "methods": list(METHODS),
            "shift_groups": ["identity", "known_shift", "unseen_strength"],
        },
        output_dir / "experiment_complete.json",
    )


if __name__ == "__main__":
    main()
