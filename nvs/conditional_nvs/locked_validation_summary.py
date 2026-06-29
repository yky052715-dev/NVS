from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Mapping, Sequence

from .robustness import aggregate_ard


DEV5 = {"bottle", "grid", "leather", "metal_nut", "screw"}
VALIDATION10 = {
    "cable", "capsule", "carpet", "hazelnut", "pill",
    "tile", "toothbrush", "transistor", "wood", "zipper",
}
METHODS = ("D0_NN", "AugMem_K10", "D2_NVSGlobal")
SEEDS = (42, 43, 44)
METRICS = (
    "pixel_AUROC",
    "pixel_AUPR",
    "pixel_AUPRO",
    "pixel_F1_calibrated",
    "pixel_F1_oracle",
    "Recall",
    "small_recall",
    "normal_image_fp_rate",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty table: {path}")
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _float(row: Mapping[str, Any], metric: str) -> float:
    try:
        return float(row[metric])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def _finite_mean(values: Sequence[float]) -> float:
    valid = [float(value) for value in values if math.isfinite(float(value))]
    return mean(valid) if valid else float("nan")


def _mean_std(values: Sequence[float]) -> dict[str, float]:
    valid = [float(value) for value in values if math.isfinite(float(value))]
    return {
        "mean": mean(valid) if valid else float("nan"),
        "sample_std": stdev(valid) if len(valid) >= 2 else float("nan"),
        "n": len(valid),
    }


def _validate_rows(
    rows: Sequence[Mapping[str, Any]],
    categories: set[str],
    seed: int,
    label: str,
) -> set[str]:
    actual_categories = {str(row["category"]) for row in rows}
    actual_methods = {str(row["method"]) for row in rows}
    actual_seeds = {int(row["seed"]) for row in rows}
    transforms = {str(row["transform"]) for row in rows}
    if actual_categories != categories:
        raise ValueError(f"{label} categories differ: {sorted(actual_categories)}")
    if actual_methods != set(METHODS):
        raise ValueError(f"{label} methods differ: {sorted(actual_methods)}")
    if actual_seeds != {int(seed)}:
        raise ValueError(f"{label} seeds differ: {sorted(actual_seeds)}")
    if len(transforms) != 8 or "identity_0" not in transforms:
        raise ValueError(f"{label} must contain identity plus seven transforms")
    keys = [
        (str(row["category"]), int(row["seed"]), str(row["transform"]), str(row["method"]))
        for row in rows
    ]
    expected = len(categories) * len(transforms) * len(METHODS)
    if len(keys) != expected or len(set(keys)) != expected:
        raise ValueError(f"{label} has duplicate or missing rows: {len(keys)} vs {expected}")
    capacities = {
        str(row["method"]): {
            int(float(item["memory_entries"]))
            for item in rows
            if str(item["method"]) == str(row["method"])
        }
        for row in rows
    }
    if any(values != {10_000} for values in capacities.values()):
        raise ValueError(f"{label} contains non-10k memory: {capacities}")
    return transforms


def load_locked_rows(
    dev5_template: str,
    validation10_template: str,
) -> tuple[list[dict[str, str]], dict[str, str]]:
    combined = []
    hashes = {}
    expected_transforms = None
    for seed in SEEDS:
        dev_path = Path(dev5_template.format(seed=seed)) / "comparison_metrics.csv"
        val_path = Path(validation10_template.format(seed=seed)) / "comparison_metrics.csv"
        for scope, path, categories in (
            ("Dev5", dev_path, DEV5),
            ("Validation10_locked", val_path, VALIDATION10),
        ):
            if not path.is_file():
                raise FileNotFoundError(path)
            rows = _read_csv(path)
            transforms = _validate_rows(rows, categories, seed, f"{scope}/seed{seed}")
            if expected_transforms is None:
                expected_transforms = transforms
            elif transforms != expected_transforms:
                raise ValueError(f"Transform coverage differs at {scope}/seed{seed}")
            for row in rows:
                row = dict(row)
                row["category_scope"] = scope
                combined.append(row)
            hashes[f"{scope}/seed{seed}"] = _sha256(path)
    return combined, hashes


def _seed_macro(
    rows: Sequence[Mapping[str, Any]],
    method: str,
    seed: int,
    condition: str,
    metric: str,
) -> float:
    selected = [
        row for row in rows
        if str(row["method"]) == method
        and int(row["seed"]) == int(seed)
        and ((str(row["transform"]) == "identity_0") == (condition == "identity"))
    ]
    return _finite_mean([_float(row, metric) for row in selected])


def summarize_scope(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    method_summary = {}
    per_seed: dict[str, dict[str, dict[int, dict[str, float]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    ard_rows = aggregate_ard(rows, metric="pixel_AUROC")
    for method in METHODS:
        method_summary[method] = {}
        for condition in ("identity", "unseen"):
            method_summary[method][condition] = {}
            for metric in METRICS:
                values = [
                    _seed_macro(rows, method, seed, condition, metric)
                    for seed in SEEDS
                ]
                method_summary[method][condition][metric] = _mean_std(values)
                for seed, value in zip(SEEDS, values):
                    per_seed[method][condition].setdefault(seed, {})[metric] = value
        ard_values = []
        for seed in SEEDS:
            selected = [
                float(row["ARD_pixel_AUROC"])
                for row in ard_rows
                if str(row["method"]) == method and int(row["seed"]) == seed
            ]
            value = _finite_mean(selected)
            ard_values.append(value)
            per_seed[method].setdefault("ARD", {}).setdefault(seed, {})[
                "ARD_pixel_AUROC"
            ] = value
        method_summary[method]["ARD_pixel_AUROC"] = _mean_std(ard_values)

    paired = {}
    for baseline in ("D0_NN", "AugMem_K10"):
        name = f"D2_minus_{baseline}"
        paired[name] = {}
        for condition in ("identity", "unseen"):
            paired[name][condition] = {}
            for metric in METRICS:
                deltas = [
                    per_seed["D2_NVSGlobal"][condition][seed][metric]
                    - per_seed[baseline][condition][seed][metric]
                    for seed in SEEDS
                ]
                paired[name][condition][metric] = _mean_std(deltas)
        ard_deltas = [
            per_seed["D2_NVSGlobal"]["ARD"][seed]["ARD_pixel_AUROC"]
            - per_seed[baseline]["ARD"][seed]["ARD_pixel_AUROC"]
            for seed in SEEDS
        ]
        paired[name]["ARD_pixel_AUROC"] = _mean_std(ard_deltas)
    return {
        "methods": method_summary,
        "paired_deltas": paired,
        "ard_by_category_seed": ard_rows,
    }


def _comparison_rows(
    rows: Sequence[Mapping[str, Any]],
    dimension: str,
    dimension_values: Sequence[str],
    summary_scope: str,
) -> list[dict[str, Any]]:
    output = []
    for value in dimension_values:
        selected_dimension = [row for row in rows if str(row[dimension]) == value]
        conditions = ("identity", "unseen")
        if dimension == "transform":
            conditions = ("identity",) if value == "identity_0" else ("unseen",)
        for condition in conditions:
            selected = [
                row for row in selected_dimension
                if ((str(row["transform"]) == "identity_0") == (condition == "identity"))
            ]
            row: dict[str, Any] = {
                "summary_scope": summary_scope,
                dimension: value,
                "condition": condition,
            }
            for method in METHODS:
                method_rows = [item for item in selected if str(item["method"]) == method]
                for metric in METRICS:
                    row[f"{method}__{metric}"] = _finite_mean(
                        [_float(item, metric) for item in method_rows]
                    )
            for baseline in ("D0_NN", "AugMem_K10"):
                for metric in METRICS:
                    row[f"D2_minus_{baseline}__{metric}"] = (
                        float(row[f"D2_NVSGlobal__{metric}"])
                        - float(row[f"{baseline}__{metric}"])
                    )
            output.append(row)
    return output


def _win_counts(rows: Sequence[Mapping[str, Any]], dimension: str) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for baseline in ("D0_NN", "AugMem_K10"):
        output[baseline] = {}
        for condition in ("identity", "unseen"):
            selected = [row for row in rows if str(row["condition"]) == condition]
            output[baseline][condition] = {}
            for metric in METRICS:
                wins = ties = losses = 0
                for row in selected:
                    delta = float(row[f"D2_minus_{baseline}__{metric}"])
                    oriented = -delta if metric == "normal_image_fp_rate" else delta
                    if abs(oriented) <= 1.0e-12:
                        ties += 1
                    elif oriented > 0:
                        wins += 1
                    else:
                        losses += 1
                output[baseline][condition][metric] = {
                    "wins": wins,
                    "ties": ties,
                    "losses": losses,
                    "dimension": dimension,
                }
    return output


def run_summary(
    dev5_template: str,
    validation10_template: str,
    output_root: Path,
) -> dict[str, Any]:
    rows, hashes = load_locked_rows(dev5_template, validation10_template)
    validation_rows = [row for row in rows if row["category_scope"] == "Validation10_locked"]
    full_rows = list(rows)
    output_root.mkdir(parents=True, exist_ok=True)
    _write_csv(output_root / "full15_metrics.csv", full_rows)
    _write_csv(output_root / "validation10_metrics.csv", validation_rows)

    category_rows = []
    transform_rows = []
    summaries = {}
    for label, selected in (
        ("Validation10_locked", validation_rows),
        ("Full15_coverage", full_rows),
    ):
        categories = sorted({str(row["category"]) for row in selected})
        transforms = sorted({str(row["transform"]) for row in selected})
        current_category_rows = _comparison_rows(
            selected, "category", categories, label
        )
        current_transform_rows = _comparison_rows(
            selected, "transform", transforms, label
        )
        category_rows.extend(current_category_rows)
        transform_rows.extend(current_transform_rows)
        summaries[label] = {
            **summarize_scope(selected),
            "category_wins": _win_counts(current_category_rows, "category"),
            "transform_wins": _win_counts(current_transform_rows, "transform"),
            "categories": categories,
            "transforms": transforms,
        }
    _write_csv(output_root / "per_category_comparison.csv", category_rows)
    _write_csv(output_root / "per_transform_comparison.csv", transform_rows)
    payload = {
        "validation_label": "locked_cross_category_coverage_not_fully_unseen",
        "input_hashes": hashes,
        "summaries": summaries,
    }
    summary_path = output_root / "locked_validation_summary.json"
    summary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    marker = {
        "status": "complete",
        "validation10_categories": sorted(VALIDATION10),
        "full15_categories": sorted(DEV5 | VALIDATION10),
        "seeds": list(SEEDS),
        "methods": list(METHODS),
        "input_hashes": hashes,
        "summary_hash": _sha256(summary_path),
    }
    (output_root / "locked_validation_complete.json").write_text(
        json.dumps(marker, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize locked Validation10 and Full15")
    parser.add_argument(
        "--dev5-template",
        default="outputs/conditional_nvs/augmem_comparison_seed{seed}",
    )
    parser.add_argument(
        "--validation10-template",
        default="outputs/conditional_nvs/validation10_comparison_seed{seed}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/conditional_nvs/locked_validation_full15"),
    )
    args = parser.parse_args()
    result = run_summary(
        args.dev5_template, args.validation10_template, args.output_dir
    )
    print(json.dumps(result["summaries"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
