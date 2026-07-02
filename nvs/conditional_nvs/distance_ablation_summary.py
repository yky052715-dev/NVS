from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

from .distance_ablation import DISTANCE_BRANCHES
from .protocol import protocol_metadata, stable_hash


METHODS = tuple(DISTANCE_BRANCHES)
METRICS = (
    "pixel_AUROC",
    "pixel_AUPR",
    "pixel_AUPRO",
    "pixel_F1_calibrated",
    "pixel_F1_oracle",
    "Recall",
    "small_recall",
    "localization_test_normal_image_positive_rate",
    "inference_ms_per_image",
    "memory_entries",
)
DIAGNOSTICS = (
    "nearest_distance_mean",
    "mean_distance_mean",
    "relative_contrast_mean",
    "relative_contrast_median",
    "relative_contrast_p05",
    "relative_contrast_p95",
    "hubness_skewness",
    "hubness_gini",
    "hubness_max_fraction",
    "hubness_occupied_fraction",
)
DEFAULT_CATEGORIES = ("bottle", "metal_nut", "grid", "leather", "screw")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({field for row in rows for field in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def gate_decision(
    macro: Mapping[str, Mapping[str, float]],
    *,
    min_auroc_gain: float = 0.001,
    min_aupro_gain: float = 0.002,
    min_normal_fp_reduction: float = 0.01,
    max_auroc_drop: float = 0.001,
    max_aupro_drop: float = 0.002,
) -> dict[str, Any]:
    baseline = macro["E0"]
    variants: dict[str, Any] = {}
    for method in ("E2", "E3"):
        row = macro[method]
        auroc = float(row["pixel_AUROC"] - baseline["pixel_AUROC"])
        aupro = float(row["pixel_AUPRO"] - baseline["pixel_AUPRO"])
        normal_fp = float(
            row["localization_test_normal_image_positive_rate"]
            - baseline["localization_test_normal_image_positive_rate"]
        )
        clear_gain = (
            auroc >= float(min_auroc_gain)
            or aupro >= float(min_aupro_gain)
            or normal_fp <= -float(min_normal_fp_reduction)
        )
        guard = auroc >= -float(max_auroc_drop) and aupro >= -float(max_aupro_drop)
        variants[method] = {
            "delta_pixel_AUROC": auroc,
            "delta_pixel_AUPRO": aupro,
            "delta_normal_image_fp_rate": normal_fp,
            "clear_gain": bool(clear_gain),
            "risk_guard": bool(guard),
            "pass": bool(clear_gain and guard),
        }
    advance = any(value["pass"] for value in variants.values())
    return {
        "status": "advance_to_three_seed" if advance else "stop_mahalanobis_route",
        "advance_to_three_seed": bool(advance),
        "baseline": "E0",
        "eligible_variants": [method for method, row in variants.items() if row["pass"]],
        "thresholds": {
            "min_auroc_gain": float(min_auroc_gain),
            "min_aupro_gain": float(min_aupro_gain),
            "min_normal_fp_reduction": float(min_normal_fp_reduction),
            "max_auroc_drop": float(max_auroc_drop),
            "max_aupro_drop": float(max_aupro_drop),
        },
        "variants": variants,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.result_root)
    categories = tuple(args.categories)
    resolved = _read_json(root / "resolved_config.json")
    if int(resolved.get("experiment", {}).get("seed", -1)) != int(args.seed):
        raise RuntimeError("Resolved config seed mismatch")
    if tuple(resolved.get("data", {}).get("categories", [])) != categories:
        raise RuntimeError("Resolved config category mismatch")
    if resolved.get("memory", {}).get("protocol") != "M_K10":
        raise RuntimeError("Distance ablation must remain M_K10")
    if resolved.get("report_methods") != list(METHODS):
        raise RuntimeError("Resolved method coverage mismatch")
    distance_config = resolved.get("distance_ablation", {}) or {}
    if not bool(distance_config.get("enabled", False)):
        raise RuntimeError("Resolved distance ablation is disabled")
    if distance_config.get("whitening_estimator") != "ledoit_wolf":
        raise RuntimeError("Resolved whitening estimator mismatch")
    resolved_gate = distance_config.get("gate", {}) or {}
    for key, value in args.gate_config.items():
        if not math.isclose(float(resolved_gate.get(key, float("nan"))), float(value)):
            raise RuntimeError(f"Gate threshold mismatch for {key}")
    metric_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    spectrum_rows: list[dict[str, Any]] = []
    manifest_hashes: dict[str, str] = {}
    for category in categories:
        category_dir = root / category / f"seed{int(args.seed)}"
        marker = _read_json(category_dir / "experiment_complete.json")
        if marker.get("status") != "complete":
            raise RuntimeError(f"Incomplete result: {category_dir}")
        if marker.get("category") != category or int(marker.get("seed", -1)) != int(args.seed):
            raise RuntimeError(f"Completion identity mismatch: {category_dir}")
        manifest = _read_json(category_dir / "sample_manifest.json")
        expected_metadata = protocol_metadata(
            category, int(args.seed), manifest, resolved, METHODS
        )
        for field in (
            "category",
            "seed",
            "sample_manifest_hash",
            "config_fingerprint",
            "protocol_hash",
        ):
            if marker.get(field) != expected_metadata.get(field):
                raise RuntimeError(f"Completion {field} mismatch: {category_dir}")
        manifest_hash = str(manifest["manifest_hash"])
        manifest_hashes[category] = manifest_hash
        state = _read_json(category_dir / "state_summary.json")
        if state.get("detector") != "DistanceAblationDetector":
            raise RuntimeError(f"Unexpected detector: {category_dir}")
        if state.get("methods") != list(METHODS):
            raise RuntimeError(f"Method mismatch: {category_dir}")
        if int(state.get("candidate_size", -1)) != 50_000:
            raise RuntimeError(f"Expected shared 50k candidates: {category_dir}")
        candidate_indices = [int(value) for value in state.get("candidate_indices", [])]
        if len(candidate_indices) != 50_000 or len(set(candidate_indices)) != 50_000:
            raise RuntimeError(f"Invalid shared candidate indices: {category_dir}")
        if state.get("candidate_indices_hash") != stable_hash(candidate_indices):
            raise RuntimeError(f"Candidate index hash mismatch: {category_dir}")
        candidate_set = set(candidate_indices)
        if state.get("whitener_fit_scope") != "memory_split_shared_candidate_pool":
            raise RuntimeError(f"Whitening fit scope violation: {category_dir}")
        whitener = state.get("whitener", {})
        if whitener.get("estimator") != "ledoit_wolf":
            raise RuntimeError(f"Expected Ledoit-Wolf: {category_dir}")
        if int(whitener.get("fit_samples", -1)) != 50_000:
            raise RuntimeError(f"Whitener did not use shared candidates: {category_dir}")
        shrinkage = float(whitener.get("shrinkage", float("nan")))
        if not 0.0 <= shrinkage <= 1.0:
            raise RuntimeError(f"Invalid Ledoit-Wolf shrinkage: {category_dir}")
        branches = state.get("branches", {})
        if set(branches) != set(METHODS):
            raise RuntimeError(f"Branch mismatch: {category_dir}")
        for method, expected in DISTANCE_BRANCHES.items():
            branch = branches[method]
            if any(branch.get(key) != value for key, value in expected.items()):
                raise RuntimeError(f"Space mismatch for {method}: {category_dir}")
            if int(branch.get("memory_entries", -1)) != 10_000:
                raise RuntimeError(f"Expected 10k memory for {method}: {category_dir}")
            selected_indices = [
                int(value) for value in branch.get("selected_memory_indices", [])
            ]
            if len(selected_indices) != 10_000 or len(set(selected_indices)) != 10_000:
                raise RuntimeError(f"Invalid selected indices for {method}: {category_dir}")
            if not set(selected_indices).issubset(candidate_set):
                raise RuntimeError(f"Selected indices escape candidates for {method}: {category_dir}")
            if branch.get("selected_memory_hash") != stable_hash(selected_indices):
                raise RuntimeError(f"Selected index hash mismatch for {method}: {category_dir}")
        if branches["E0"]["selected_memory_indices"] != branches["E2"]["selected_memory_indices"]:
            raise RuntimeError(f"E0/E2 must share raw selection: {category_dir}")
        if branches["E1"]["selected_memory_indices"] != branches["E3"]["selected_memory_indices"]:
            raise RuntimeError(f"E1/E3 must share whitened selection: {category_dir}")
        calibration = _read_json(category_dir / "calibration.json")
        if set(calibration) != set(METHODS) or any(
            calibration[method].get("fit_scope") != "identity_calibration"
            for method in METHODS
        ):
            raise RuntimeError(f"Independent calibration violation: {category_dir}")
        rows = _read_rows(category_dir / "metrics_summary.csv")
        by_method = {row["method"]: row for row in rows}
        if set(by_method) != set(METHODS):
            raise RuntimeError(f"Metric method mismatch: {category_dir}")
        query_indices = [int(value) for value in state.get("diagnostic_query_indices", [])]
        if not query_indices or state.get("diagnostic_query_indices_hash") != stable_hash(query_indices):
            raise RuntimeError(f"Diagnostic query hash mismatch: {category_dir}")
        diagnostics = state.get("distance_diagnostics", {})
        if set(diagnostics) != set(METHODS):
            raise RuntimeError(f"Missing distance diagnostics: {category_dir}")
        for method in METHODS:
            row: dict[str, Any] = {"category": category, "seed": int(args.seed), "method": method}
            for field in METRICS:
                row[field] = float(by_method[method][field])
            metric_rows.append(row)
            diagnostic_rows.append(
                {
                    "category": category,
                    "seed": int(args.seed),
                    "method": method,
                    **{field: float(diagnostics[method][field]) for field in DIAGNOSTICS},
                }
            )
        for spectrum_name in ("empirical_spectrum", "shrunk_spectrum"):
            spectrum_rows.append(
                {
                    "category": category,
                    "seed": int(args.seed),
                    "spectrum": spectrum_name,
                    "shrinkage": shrinkage,
                    **{
                        key: float(value)
                        for key, value in whitener[spectrum_name].items()
                    },
                }
            )
    macro_rows: list[dict[str, Any]] = []
    diagnostic_macro: list[dict[str, Any]] = []
    macro_index: dict[str, dict[str, float]] = {}
    for method in METHODS:
        selected = [row for row in metric_rows if row["method"] == method]
        macro = {"method": method, "seed": int(args.seed), "categories": len(selected)}
        for field in METRICS:
            macro[field] = float(mean(float(row[field]) for row in selected))
        macro_rows.append(macro)
        macro_index[method] = macro
        selected_diagnostics = [row for row in diagnostic_rows if row["method"] == method]
        diagnostic_macro.append(
            {
                "method": method,
                "seed": int(args.seed),
                "categories": len(selected_diagnostics),
                **{
                    field: float(
                        mean(float(row[field]) for row in selected_diagnostics)
                    )
                    for field in DIAGNOSTICS
                },
            }
        )
    deltas = []
    baseline = macro_index["E0"]
    for method in ("E1", "E2", "E3"):
        deltas.append(
            {
                "method": method,
                "baseline": "E0",
                **{
                    f"delta_{field}": float(macro_index[method][field] - baseline[field])
                    for field in METRICS
                },
            }
        )
    gate_config = args.gate_config
    decision = gate_decision(macro_index, **gate_config)
    summary = {
        "status": "complete",
        "protocol": "distance_ablation_gate1_v1",
        "retrospective": True,
        "seed": int(args.seed),
        "categories": list(categories),
        "methods": list(METHODS),
        "shared_candidate_entries": 50_000,
        "final_memory_entries": 10_000,
        "whitener_fit_scope": "memory_split_shared_candidate_pool",
        "macro_summary": macro_rows,
        "diagnostic_macro": diagnostic_macro,
        "delta_vs_E0": deltas,
        "gate_decision": decision,
        "manifest_hashes": manifest_hashes,
        "config_fingerprint": stable_hash(resolved),
    }
    summary["summary_hash"] = stable_hash(
        {
            "metrics": metric_rows,
            "diagnostics": diagnostic_rows,
            "spectra": spectrum_rows,
            "gate": decision,
        }
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "category_results.csv", metric_rows)
    _write_csv(output / "macro_summary.csv", macro_rows)
    _write_csv(output / "delta_vs_E0.csv", deltas)
    _write_csv(output / "distance_diagnostics.csv", diagnostic_rows)
    _write_csv(output / "distance_diagnostic_macro.csv", diagnostic_macro)
    _write_csv(output / "covariance_spectra.csv", spectrum_rows)
    (output / "comparison_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "gate_decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "comparison_complete.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "protocol": summary["protocol"],
                "seed": int(args.seed),
                "summary_hash": summary["summary_hash"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--categories", nargs="+", default=list(DEFAULT_CATEGORIES))
    parser.add_argument("--min-auroc-gain", type=float, default=0.001)
    parser.add_argument("--min-aupro-gain", type=float, default=0.002)
    parser.add_argument("--min-normal-fp-reduction", type=float, default=0.01)
    parser.add_argument("--max-auroc-drop", type=float, default=0.001)
    parser.add_argument("--max-aupro-drop", type=float, default=0.002)
    args = parser.parse_args()
    args.gate_config = {
        "min_auroc_gain": args.min_auroc_gain,
        "min_aupro_gain": args.min_aupro_gain,
        "min_normal_fp_reduction": args.min_normal_fp_reduction,
        "max_auroc_drop": args.max_auroc_drop,
        "max_aupro_drop": args.max_aupro_drop,
    }
    return args


def main() -> None:
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()