from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from nvs.apc_c2_experiment import (
    _fit_calibration,
    _mean_rows,
    _normal_fp,
    _normalize,
    _raw_maps,
    _score_records,
)
from nvs.common import build_mvtec_records, load_config, load_dinov2, save_json, write_csv
from nvs.detection import _fit_state
from nvs.metrics import (
    binary_f1,
    localization_metrics_from_prediction,
    oracle_pixel_f1,
    safe_auroc,
)
from nvs.transforms import transform_name


BASE_METHODS = ("C0_r0", "C1_r2")


def quantile_method_name(quantile: float) -> str:
    return f"C2_q{int(round(float(quantile) * 100)):03d}"


def _all_methods(config: dict[str, Any]) -> list[str]:
    return list(BASE_METHODS) + [
        quantile_method_name(value)
        for value in config["apc_c2"]["rho_quantiles"]
    ]


def _maps_for_all_quantiles(
    scored: dict[str, Any],
    tau_by_method: dict[str, float],
    config: dict[str, Any],
) -> dict[str, np.ndarray]:
    output_size = int(config["data"]["input_size"])
    temperature = float(config["apc_c2"]["temperature"])
    first_tau = next(iter(tau_by_method.values()))
    base = _raw_maps(scored, output_size, first_tau, temperature)
    output = {
        "C0_r0": base["C0_r0"],
        "C1_r2": base["C1_r2"],
    }
    for method, tau in tau_by_method.items():
        output[method] = _raw_maps(
            scored,
            output_size,
            tau,
            temperature,
        )["C2_apc_rho"]
    return output


def _image_score_p95(
    maps: list[np.ndarray],
    calibration: dict[str, float],
    indices: list[int],
) -> float:
    selected = np.concatenate(
        [_normalize(values, calibration)[indices] for values in maps],
        axis=0,
    )
    maxima = selected.reshape(selected.shape[0], -1).max(axis=1)
    return float(np.quantile(maxima, 0.95))


def _calibrate_all_quantiles(
    state: dict[str, Any],
    config: dict[str, Any],
    model,
    device: torch.device,
) -> tuple[
    dict[str, dict[str, float]],
    dict[str, float],
    list[dict[str, Any]],
]:
    records = state["calibration_records"]
    identity = _score_records(
        records,
        None,
        state,
        config,
        model,
        device,
        include_mask=False,
    )
    known_scores = [
        _score_records(
            records,
            spec,
            state,
            config,
            model,
            device,
            include_mask=False,
        )
        for spec in config["transforms"]
    ]
    fit_set = {str(record.path) for record in state["threshold_fit_records"]}
    fit_indices = [
        index for index, path in enumerate(identity["paths"]) if path in fit_set
    ]
    holdout_indices = [
        index for index, path in enumerate(identity["paths"]) if path not in fit_set
    ]
    if not holdout_indices:
        holdout_indices = list(range(len(identity["paths"])))

    quantiles = [float(value) for value in config["apc_c2"]["rho_quantiles"]]
    tau_by_method = {
        quantile_method_name(q): float(
            np.quantile(identity["rho"].reshape(-1), q)
        )
        for q in quantiles
    }
    identity_maps = _maps_for_all_quantiles(
        identity,
        tau_by_method,
        config,
    )
    calibrations: dict[str, dict[str, float]] = {
        method: _fit_calibration(identity_maps[method], fit_indices, config)
        for method in _all_methods(config)
    }
    known_maps_by_method: dict[str, list[np.ndarray]] = {
        method: [] for method in _all_methods(config)
    }
    for scored in known_scores:
        maps = _maps_for_all_quantiles(scored, tau_by_method, config)
        for method, values in maps.items():
            known_maps_by_method[method].append(values)

    calibration_rows: list[dict[str, Any]] = []
    for method in _all_methods(config):
        identity_fp, identity_area = _normal_fp(
            [identity_maps[method]],
            calibrations[method],
            holdout_indices,
        )
        known_fp, known_area = _normal_fp(
            known_maps_by_method[method],
            calibrations[method],
            holdout_indices,
        )
        q = None
        tau = None
        if method.startswith("C2_q"):
            q = next(
                value
                for value in quantiles
                if quantile_method_name(value) == method
            )
            tau = tau_by_method[method]
        calibration_rows.append(
            {
                "method": method,
                "rho_quantile": q,
                "tau_rho": tau,
                "identity_normal_fp": identity_fp,
                "identity_positive_area": identity_area,
                "identity_image_score_p95": _image_score_p95(
                    [identity_maps[method]],
                    calibrations[method],
                    holdout_indices,
                ),
                "known_transformed_normal_fp": known_fp,
                "known_transformed_positive_area": known_area,
                "known_transformed_image_score_p95": _image_score_p95(
                    known_maps_by_method[method],
                    calibrations[method],
                    holdout_indices,
                ),
            }
        )
    return calibrations, tau_by_method, calibration_rows


def _evaluate(
    category: str,
    shift_group: str,
    condition: str,
    scored: dict[str, Any],
    calibrations: dict[str, dict[str, float]],
    tau_by_method: dict[str, float],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    raw_maps = _maps_for_all_quantiles(scored, tau_by_method, config)
    masks = scored["masks"].astype(bool)
    labels = scored["labels"]
    quantiles = [float(value) for value in config["apc_c2"]["rho_quantiles"]]
    rows: list[dict[str, Any]] = []
    for method in _all_methods(config):
        maps = _normalize(raw_maps[method], calibrations[method])
        threshold = float(calibrations[method]["pixel_threshold"])
        predictions = maps >= threshold
        oracle_f1, oracle_threshold = oracle_pixel_f1(masks, maps)
        q = None
        tau = None
        if method.startswith("C2_q"):
            q = next(
                value
                for value in quantiles
                if quantile_method_name(value) == method
            )
            tau = tau_by_method[method]
        rows.append(
            {
                "category": category,
                "shift_group": shift_group,
                "condition": condition,
                "method": method,
                "rho_quantile": q,
                "tau_rho": tau,
                "pixel_AUROC": safe_auroc(
                    masks.reshape(-1),
                    maps.reshape(-1),
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


def _sensitivity_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = [
        "pixel_AUROC",
        "pixel_F1_calibrated",
        "pixel_F1_oracle",
        "localization_overseg_anomaly_macro",
        "localization_recall_anomaly_macro",
        "localization_small_defect_recall_macro",
        "localization_test_normal_image_positive_rate",
    ]
    output: list[dict[str, Any]] = []
    keys = sorted({(row["shift_group"], row["category"]) for row in rows})
    for shift_group, category in keys:
        selected = [
            row
            for row in rows
            if row["shift_group"] == shift_group
            and row["category"] == category
            and row["method"].startswith("C2_q")
        ]
        result: dict[str, Any] = {
            "shift_group": shift_group,
            "category": category,
            "quantile_methods": len(selected),
        }
        for metric in metrics:
            values = np.asarray([float(row[metric]) for row in selected])
            finite = values[np.isfinite(values)]
            result[f"{metric}_min"] = (
                float(finite.min()) if finite.size else float("nan")
            )
            result[f"{metric}_max"] = (
                float(finite.max()) if finite.size else float("nan")
            )
            result[f"{metric}_range"] = (
                float(finite.max() - finite.min())
                if finite.size
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
        "# APC-NVS C2 rho-quantile sensitivity",
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
    lines.extend(
        [
            "",
            "All C2 quantiles share the same DINO features. "
            "No anomaly metric is used to select a category-specific quantile.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="APC-NVS C2 rho-quantile sensitivity experiment"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--categories", nargs="+")
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
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
        config["model"]["name"],
        device,
        config["model"].get("hub_dir"),
    )
    output_dir = Path(config["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(config, output_dir / "resolved_config.json")

    rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
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
        print(f"[C2 sensitivity] {category}")
        train_records, test_records = build_mvtec_records(
            args.data_root,
            category,
        )
        state = _fit_state(category, train_records, config, model, device)
        calibrations, tau_by_method, category_calibration = (
            _calibrate_all_quantiles(state, config, model, device)
        )
        calibration_rows.extend(
            {"category": category, **row}
            for row in category_calibration
        )
        category_dir = output_dir / category
        category_dir.mkdir(parents=True, exist_ok=True)
        save_json(
            {
                "category": category,
                "tau_by_method": tau_by_method,
                "calibrations": calibrations,
            },
            category_dir / "c2_sensitivity_state.json",
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
            rows.extend(
                _evaluate(
                    category,
                    shift_group,
                    condition,
                    scored,
                    calibrations,
                    tau_by_method,
                    config,
                )
            )
            write_csv(
                rows,
                output_dir / "category_condition_metrics.csv",
            )

    condition_rows = _mean_rows(rows, "condition")
    group_rows = _mean_rows(rows, "shift_group")
    sensitivity = _sensitivity_rows(rows)
    write_csv(
        calibration_rows,
        output_dir / "normal_calibration_by_quantile.csv",
    )
    write_csv(condition_rows, output_dir / "condition_summary.csv")
    write_csv(group_rows, output_dir / "shift_group_summary.csv")
    write_csv(
        sensitivity,
        output_dir / "quantile_sensitivity_ranges.csv",
    )
    _write_report(
        group_rows,
        output_dir / "apc_c2_quantile_sensitivity_summary.md",
    )
    save_json(
        {
            "status": "complete",
            "categories": list(config["data"]["categories"]),
            "completed_count": len(config["data"]["categories"]),
            "methods": _all_methods(config),
            "rho_quantiles": list(config["apc_c2"]["rho_quantiles"]),
            "uses_shared_features": True,
            "automatic_quantile_selection": False,
        },
        output_dir / "experiment_complete.json",
    )


if __name__ == "__main__":
    main()
