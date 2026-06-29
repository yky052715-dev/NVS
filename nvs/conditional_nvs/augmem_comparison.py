from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

from .protocol import config_fingerprint
from .robustness import aggregate_ard


D2_METHODS = ("D0_NN", "D2_NVSGlobal")
AUGMEM_K10_METHOD = "AugMem_K10"
AUGMEM_FULL_METHOD = "AugMem_Full"
SUMMARY_METRICS = (
    "pixel_AUROC",
    "pixel_AUPR",
    "pixel_AUPRO",
    "pixel_F1_calibrated",
    "pixel_F1_oracle",
    "Recall",
    "small_recall",
    "normal_image_fp_rate",
    "inference_ms_per_image",
    "memory_entries",
)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty comparison")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _nested(config: Mapping[str, Any], path: str) -> Any:
    value: Any = config
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise KeyError(f"Missing config field {path!r}")
        value = value[part]
    return value


def validate_augmem_protocol(
    d2_config: Mapping[str, Any],
    augmem_config: Mapping[str, Any],
    *,
    full: bool = False,
) -> dict[str, Any]:
    """Validate all scientific protocol fields while ignoring compute tuning."""

    shared_fields = (
        "experiment.seed",
        "data.categories",
        "data.input_size",
        "data.calibration_fraction",
        "data.nvs_fit_fraction_of_remainder",
        "model.name",
        "model.hub_dir",
        "calibration.mad_epsilon",
        "calibration.image_quantile",
        "metrics.small_defect_area_fraction",
        "fit_transforms",
        "robustness.transforms",
    )
    mismatches = []
    for field in shared_fields:
        left, right = _nested(d2_config, field), _nested(augmem_config, field)
        if left != right:
            mismatches.append(f"{field}: D2={left!r}, AugMem={right!r}")
    if _nested(d2_config, "memory.protocol") != "M_K10":
        mismatches.append("D2 reference must use memory.protocol=M_K10")
    expected_augmem_protocol = "M_F0" if full else "M_K10"
    if _nested(augmem_config, "memory.protocol") != expected_augmem_protocol:
        mismatches.append(
            f"AugMem must use memory.protocol={expected_augmem_protocol}"
        )
    if bool(_nested(d2_config, "augmem.enabled")):
        mismatches.append("D2 reference must have augmem.enabled=false")
    if not bool(_nested(augmem_config, "augmem.enabled")):
        mismatches.append("AugMem run must have augmem.enabled=true")
    if _nested(augmem_config, "augmem.candidate_source") != "matched_d2_information":
        mismatches.append("AugMem candidate_source must be matched_d2_information")
    if _nested(augmem_config, "augmem.detection_mode") != "memory_only":
        mismatches.append("AugMem detection_mode must be memory_only")
    for label, config in (("D2", d2_config), ("AugMem", augmem_config)):
        if bool(_nested(config, "fusion.enabled")):
            mismatches.append(f"{label} fusion.enabled must be false")
        if bool(_nested(config, "postprocess.enabled")):
            mismatches.append(f"{label} postprocess.enabled must be false")
        if not bool(_nested(config, "robustness.enabled")):
            mismatches.append(f"{label} robustness.enabled must be true")
    if mismatches:
        raise ValueError("AugMem protocol mismatch:\n- " + "\n- ".join(mismatches))
    return {
        "status": "matched",
        "shared_fields": list(shared_fields),
        "d2_config_fingerprint": config_fingerprint(d2_config),
        "augmem_config_fingerprint": config_fingerprint(augmem_config),
        "augmem_protocol": expected_augmem_protocol,
        "candidate_source": "matched_d2_information",
    }


def _manifests(root: Path) -> dict[tuple[str, int], dict[str, Any]]:
    output = {}
    for path in root.glob("*/seed*/sample_manifest.json"):
        payload = _load_json(path)
        category = str(payload["category"])
        seed = int(path.parent.name.removeprefix("seed"))
        output[(category, seed)] = payload
    if not output:
        raise FileNotFoundError(f"No sample manifests found under {root}")
    return output


def validate_split_manifests(d2_root: Path, augmem_root: Path) -> dict[str, Any]:
    d2, augmem = _manifests(d2_root), _manifests(augmem_root)
    if set(d2) != set(augmem):
        raise ValueError(
            f"Manifest category/seed sets differ: D2={sorted(d2)}, AugMem={sorted(augmem)}"
        )
    mismatches = [
        key
        for key in sorted(d2)
        if str(d2[key]["manifest_hash"]) != str(augmem[key]["manifest_hash"])
    ]
    if mismatches:
        raise ValueError(f"Sample manifests differ for {mismatches}")
    return {
        "status": "matched",
        "category_seed_count": len(d2),
        "manifest_hashes": {
            f"{category}/seed{seed}": str(d2[(category, seed)]["manifest_hash"])
            for category, seed in sorted(d2)
        },
    }


def _coverage(rows: Sequence[Mapping[str, Any]]) -> set[tuple[str, str, str]]:
    return {
        (str(row["category"]), str(row["seed"]), str(row["transform"]))
        for row in rows
    }


def assemble_comparison_rows(
    d2_rows: Sequence[Mapping[str, Any]],
    augmem_rows: Sequence[Mapping[str, Any]],
    full_rows: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    selected_d2 = [dict(row) for row in d2_rows if str(row["method"]) in D2_METHODS]
    selected_augmem = [
        dict(row)
        for row in augmem_rows
        if str(row["method"]) == AUGMEM_K10_METHOD
    ]
    if not selected_d2 or not selected_augmem:
        raise ValueError("Missing D0/D2 or AugMem-K10 rows")
    reference_coverage = _coverage(selected_d2)
    if _coverage(selected_augmem) != reference_coverage:
        raise ValueError("D2 and AugMem-K10 evaluation coverage differs")
    methods = {str(row["method"]) for row in selected_d2}
    if methods != set(D2_METHODS):
        raise ValueError(f"D2 reference methods differ: {sorted(methods)}")
    expected_per_method = len(reference_coverage)
    for method in (*D2_METHODS, AUGMEM_K10_METHOD):
        source = selected_augmem if method == AUGMEM_K10_METHOD else selected_d2
        count = sum(str(row["method"]) == method for row in source)
        if count != expected_per_method:
            raise ValueError(f"{method} has {count}, expected {expected_per_method} rows")
    combined = selected_d2 + selected_augmem
    for method in (*D2_METHODS, AUGMEM_K10_METHOD):
        selected = [row for row in combined if str(row["method"]) == method]
        if any("memory_entries" not in row for row in selected):
            raise ValueError(f"{method} rows are missing memory_entries")
        capacities = {int(float(row["memory_entries"])) for row in selected}
        if capacities != {10_000}:
            raise ValueError(
                f"{method} memory capacity must be exactly 10000, got {capacities}"
            )
    if full_rows is not None:
        selected_full = [
            dict(row)
            for row in full_rows
            if str(row["method"]) == AUGMEM_FULL_METHOD
        ]
        if _coverage(selected_full) != reference_coverage:
            raise ValueError("D2 and AugMem-Full evaluation coverage differs")
        if len(selected_full) != expected_per_method:
            raise ValueError("AugMem-Full has duplicate or missing rows")
        combined.extend(selected_full)
    return combined


def _finite_mean(rows: Sequence[Mapping[str, Any]], metric: str) -> float:
    values = []
    for row in rows:
        try:
            value = float(row[metric])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return mean(values) if values else float("nan")


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    methods = sorted({str(row["method"]) for row in rows})
    by_method = {}
    ard_rows = aggregate_ard(rows, metric="pixel_AUROC")
    ard_by_method: dict[str, list[float]] = defaultdict(list)
    for row in ard_rows:
        ard_by_method[str(row["method"])].append(float(row["ARD_pixel_AUROC"]))
    for method in methods:
        selected = [row for row in rows if str(row["method"]) == method]
        identity = [row for row in selected if str(row["transform"]) == "identity_0"]
        unseen = [row for row in selected if str(row["transform"]) != "identity_0"]
        by_method[method] = {
            "identity": {metric: _finite_mean(identity, metric) for metric in SUMMARY_METRICS},
            "unseen_macro": {metric: _finite_mean(unseen, metric) for metric in SUMMARY_METRICS},
            "ARD_pixel_AUROC": mean(ard_by_method[method]),
            "row_count": len(selected),
        }
    return {
        "methods": by_method,
        "ard_by_category": ard_rows,
        "cost_interpretation": {
            "memory_entries_comparable": True,
            "inference_timing_comparable": False,
            "reason": (
                "The D2 reference timed joint five-method scoring and copied that elapsed "
                "time to each row; do not use it as isolated D0/D2 latency. AugMem timing "
                "is memory-only. Run a dedicated timing benchmark only if accuracy is tied."
            ),
        },
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_comparison(
    d2_root: Path,
    augmem_root: Path,
    output_root: Path,
    full_root: Path | None = None,
) -> dict[str, Any]:
    d2_config = _load_json(d2_root / "resolved_config.json")
    augmem_config = _load_json(augmem_root / "resolved_config.json")
    protocol = validate_augmem_protocol(d2_config, augmem_config)
    splits = validate_split_manifests(d2_root, augmem_root)
    d2_csv = d2_root / "robustness_metrics.csv"
    augmem_csv = augmem_root / "robustness_metrics.csv"
    full_csv = None
    full_validation = None
    full_rows = None
    if full_root is not None:
        full_config = _load_json(full_root / "resolved_config.json")
        full_validation = validate_augmem_protocol(d2_config, full_config, full=True)
        validate_split_manifests(d2_root, full_root)
        full_csv = full_root / "robustness_metrics.csv"
        full_rows = _read_csv(full_csv)
    rows = assemble_comparison_rows(
        _read_csv(d2_csv),
        _read_csv(augmem_csv),
        full_rows,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    comparison_csv = output_root / "comparison_metrics.csv"
    _write_csv(comparison_csv, rows)
    summary = {
        "protocol_validation": protocol,
        "split_validation": splits,
        "full_validation": full_validation,
        **summarize(rows),
    }
    (output_root / "comparison_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    marker = {
        "status": "complete",
        "d2_root": str(d2_root),
        "augmem_k10_root": str(augmem_root),
        "augmem_full_root": None if full_root is None else str(full_root),
        "input_hashes": {
            "d2": _file_sha256(d2_csv),
            "augmem_k10": _file_sha256(augmem_csv),
            "augmem_full": None if full_csv is None else _file_sha256(full_csv),
        },
        "comparison_hash": _file_sha256(comparison_csv),
    }
    (output_root / "comparison_complete.json").write_text(
        json.dumps(marker, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit and merge matched AugMem comparison")
    parser.add_argument("--d2-root", required=True, type=Path)
    parser.add_argument("--augmem-k10-root", required=True, type=Path)
    parser.add_argument("--augmem-full-root", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    summary = run_comparison(
        args.d2_root,
        args.augmem_k10_root,
        args.output_dir,
        args.augmem_full_root,
    )
    print(json.dumps(summary["methods"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
