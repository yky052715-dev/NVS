from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

from .augmem_comparison import (
    SUMMARY_METRICS,
    _coverage,
    _finite_mean,
    _load_json,
    _read_csv,
    _write_csv,
    validate_augmem_protocol,
    validate_split_manifests,
)
from .robustness import aggregate_ard


D2_METHOD = "D2_NVSGlobal"
AUGMEM_50K_INPUT_METHOD = "AugMem_K10"
AUGMEM_50K_METHOD = "AugMem_K10_50k"
AUGMEM_100K_METHOD = "AugMem_K10_100k"


def validate_candidate_sizes(
    augmem_50k_config: Mapping[str, Any],
    augmem_100k_config: Mapping[str, Any],
) -> dict[str, int]:
    expected = (("AugMem-50k", augmem_50k_config, 50_000),
                ("AugMem-100k", augmem_100k_config, 100_000))
    output = {}
    for label, config, size in expected:
        actual = int(config["memory"]["candidate_size"])
        if actual != size:
            raise ValueError(f"{label} candidate_size must be {size}, got {actual}")
        output[label] = actual
    return output


def assemble_sensitivity_rows(
    d2_rows: Sequence[Mapping[str, Any]],
    augmem_50k_rows: Sequence[Mapping[str, Any]],
    augmem_100k_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    selected_d2 = [dict(row) for row in d2_rows if str(row["method"]) == D2_METHOD]
    selected_50k = [
        {**dict(row), "method": AUGMEM_50K_METHOD}
        for row in augmem_50k_rows
        if str(row["method"]) == AUGMEM_50K_INPUT_METHOD
    ]
    selected_100k = [
        dict(row)
        for row in augmem_100k_rows
        if str(row["method"]) == AUGMEM_100K_METHOD
    ]
    if not selected_d2 or not selected_50k or not selected_100k:
        raise ValueError("Missing D2, AugMem-50k, or AugMem-100k rows")
    reference = _coverage(selected_d2)
    for method, rows in (
        (AUGMEM_50K_METHOD, selected_50k),
        (AUGMEM_100K_METHOD, selected_100k),
    ):
        if _coverage(rows) != reference:
            raise ValueError(f"{method} evaluation coverage differs from D2")
    expected_count = len(reference)
    combined = selected_d2 + selected_50k + selected_100k
    for method in (D2_METHOD, AUGMEM_50K_METHOD, AUGMEM_100K_METHOD):
        selected = [row for row in combined if str(row["method"]) == method]
        if len(selected) != expected_count:
            raise ValueError(f"{method} has duplicate or missing rows")
        capacities = {int(float(row["memory_entries"])) for row in selected}
        if capacities != {10_000}:
            raise ValueError(f"{method} final memory must be exactly 10000")
    return combined


def _method_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    methods = sorted({str(row["method"]) for row in rows})
    ard_rows = aggregate_ard(rows, metric="pixel_AUROC")
    ard_by_method: dict[str, list[float]] = defaultdict(list)
    for row in ard_rows:
        ard_by_method[str(row["method"])].append(float(row["ARD_pixel_AUROC"]))
    output = {}
    for method in methods:
        selected = [row for row in rows if str(row["method"]) == method]
        identity = [row for row in selected if str(row["transform"]) == "identity_0"]
        unseen = [row for row in selected if str(row["transform"]) != "identity_0"]
        output[method] = {
            "identity": {metric: _finite_mean(identity, metric) for metric in SUMMARY_METRICS},
            "unseen_macro": {metric: _finite_mean(unseen, metric) for metric in SUMMARY_METRICS},
            "ARD_pixel_AUROC": mean(ard_by_method[method]),
            "row_count": len(selected),
        }
    return {"methods": output, "ard_by_category": ard_rows}


def _paired_deltas(summary: Mapping[str, Any]) -> dict[str, Any]:
    methods = summary["methods"]
    comparisons = {
        "AugMem100k_minus_50k": (AUGMEM_100K_METHOD, AUGMEM_50K_METHOD),
        "D2_minus_AugMem100k": (D2_METHOD, AUGMEM_100K_METHOD),
        "D2_minus_AugMem50k": (D2_METHOD, AUGMEM_50K_METHOD),
    }
    output = {}
    for name, (left, right) in comparisons.items():
        output[name] = {
            scope: {
                metric: float(methods[left][scope][metric])
                - float(methods[right][scope][metric])
                for metric in SUMMARY_METRICS
                if metric != "inference_ms_per_image"
            }
            for scope in ("identity", "unseen_macro")
        }
        output[name]["ARD_pixel_AUROC"] = (
            float(methods[left]["ARD_pixel_AUROC"])
            - float(methods[right]["ARD_pixel_AUROC"])
        )
    return output


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_sensitivity(
    d2_root: Path,
    augmem_50k_root: Path,
    augmem_100k_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    d2_config = _load_json(d2_root / "resolved_config.json")
    config_50k = _load_json(augmem_50k_root / "resolved_config.json")
    config_100k = _load_json(augmem_100k_root / "resolved_config.json")
    protocol_50k = validate_augmem_protocol(d2_config, config_50k)
    protocol_100k = validate_augmem_protocol(d2_config, config_100k)
    sizes = validate_candidate_sizes(config_50k, config_100k)
    split_50k = validate_split_manifests(d2_root, augmem_50k_root)
    split_100k = validate_split_manifests(d2_root, augmem_100k_root)

    d2_csv = d2_root / "robustness_metrics.csv"
    csv_50k = augmem_50k_root / "robustness_metrics.csv"
    csv_100k = augmem_100k_root / "robustness_metrics.csv"
    rows = assemble_sensitivity_rows(
        _read_csv(d2_csv), _read_csv(csv_50k), _read_csv(csv_100k)
    )
    output_root.mkdir(parents=True, exist_ok=True)
    comparison_csv = output_root / "sensitivity_metrics.csv"
    _write_csv(comparison_csv, rows)
    summary = {
        "protocol_validation": {
            "AugMem_50k": protocol_50k,
            "AugMem_100k": protocol_100k,
            "candidate_sizes": sizes,
            "split_50k": split_50k,
            "split_100k": split_100k,
        },
        **_method_summary(rows),
    }
    summary["paired_deltas"] = _paired_deltas(summary)
    (output_root / "sensitivity_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    marker = {
        "status": "complete",
        "d2_root": str(d2_root),
        "augmem_50k_root": str(augmem_50k_root),
        "augmem_100k_root": str(augmem_100k_root),
        "input_hashes": {
            "D2": _sha256(d2_csv),
            "AugMem_50k": _sha256(csv_50k),
            "AugMem_100k": _sha256(csv_100k),
        },
        "sensitivity_hash": _sha256(comparison_csv),
    }
    (output_root / "sensitivity_complete.json").write_text(
        json.dumps(marker, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit AugMem 50k/100k candidate sensitivity")
    parser.add_argument("--d2-root", required=True, type=Path)
    parser.add_argument("--augmem-50k-root", required=True, type=Path)
    parser.add_argument("--augmem-100k-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    summary = run_sensitivity(
        args.d2_root,
        args.augmem_50k_root,
        args.augmem_100k_root,
        args.output_dir,
    )
    print(json.dumps({
        "methods": summary["methods"],
        "paired_deltas": summary["paired_deltas"],
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
