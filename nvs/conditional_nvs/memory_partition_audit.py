"""Strict retrospective audit of whole, merge-reduce, and image-balanced K10."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Mapping, Sequence

from .protocol import stable_hash


PROTOCOLS = ("M_K10", "M_MRK10", "M_IBK10")
EXPECTED_ALGORITHMS = {
    "M_K10": "shared_candidate_greedy_kcenter",
    "M_MRK10": "shared_candidate_merge_reduce_kcenter_gamma2",
    "M_IBK10": "shared_candidate_image_balanced_kcenter",
}
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
DEFAULT_CATEGORIES = ("bottle", "metal_nut", "grid", "leather", "screw")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_d0_row(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row.get("method") == "D0_NN"
        ]
    if len(rows) != 1:
        raise RuntimeError(f"Expected one D0_NN row in {path}, got {len(rows)}")
    return rows[0]


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def balance_statistics(counts: Sequence[int]) -> dict[str, float | int]:
    values = [int(value) for value in counts]
    if not values or any(value < 0 for value in values):
        raise ValueError("counts must be a non-empty non-negative sequence")
    total = sum(values)
    if total <= 0:
        raise ValueError("counts must contain selected entries")
    avg = mean(values)
    absolute = sum(abs(left - right) for left in values for right in values)
    gini = absolute / (2.0 * len(values) * total)
    return {
        "selected_images": sum(value > 0 for value in values),
        "selected_per_image_min": min(values),
        "selected_per_image_max": max(values),
        "selected_per_image_mean": float(avg),
        "selected_per_image_std": float(pstdev(values)),
        "selected_per_image_cv": float(pstdev(values) / avg) if avg else 0.0,
        "selected_per_image_gini": float(gini),
        "selected_max_image_fraction": float(max(values) / total),
    }


def _runtime_signature(config: Mapping[str, Any]) -> str:
    data = config.get("data", {}) or {}
    model = config.get("model", {}) or {}
    memory = config.get("memory", {}) or {}
    calibration = config.get("calibration", {}) or {}
    return stable_hash(
        {
            "data": {
                key: data.get(key)
                for key in (
                    "categories",
                    "input_size",
                    "calibration_fraction",
                    "nvs_fit_fraction_of_remainder",
                    "num_workers",
                )
            },
            "model": {
                key: model.get(key)
                for key in ("name", "batch_size", "gpu_batch_size")
            },
            "memory_compute": {
                key: memory.get(key)
                for key in (
                    "candidate_size",
                    "query_chunk_size",
                    "bank_chunk_size",
                    "gpu_query_chunk_size",
                    "gpu_bank_chunk_size",
                    "kcenter_chunk_size",
                    "kcenter_block_size",
                    "large_k_batch_select",
                )
            },
            "calibration": calibration,
        }
    )


def _protocol_root(args: argparse.Namespace, protocol: str) -> Path:
    if protocol == "M_K10":
        return Path(args.global_root)
    return Path(args.audit_root) / protocol


def _load_category(
    root: Path,
    protocol: str,
    category: str,
    seed: int,
) -> tuple[dict[str, Any], str, str, str]:
    category_dir = root / category / f"seed{seed}"
    marker = _read_json(category_dir / "experiment_complete.json")
    if marker.get("status") != "complete":
        raise RuntimeError(f"Incomplete result: {category_dir}")
    if str(marker.get("category")) != category or int(marker.get("seed")) != seed:
        raise RuntimeError(f"Completion identity mismatch: {category_dir}")
    manifest = _read_json(category_dir / "sample_manifest.json")
    if str(marker.get("sample_manifest_hash")) != str(
        manifest.get("manifest_hash")
    ):
        raise RuntimeError(f"Manifest hash mismatch: {category_dir}")
    state = _read_json(category_dir / "state_summary.json")
    if int(state.get("memory_entries", -1)) != 10_000:
        raise RuntimeError(f"Expected 10k memory: {category_dir}")
    if str(state.get("memory_algorithm")) != EXPECTED_ALGORITHMS[protocol]:
        raise RuntimeError(
            f"Unexpected algorithm for {protocol}: {state.get('memory_algorithm')}"
        )
    if protocol == "M_MRK10" and (
        int(state.get("kcenter_block_size", -1)) != 10_000
        or int(state.get("large_k_batch_select", -1)) != 64
    ):
        raise RuntimeError("M_MRK10 must use 10k blocks and batch-select 64")
    candidate_indices = [int(value) for value in state["candidate_indices"]]
    selected_indices = [int(value) for value in state["selected_memory_indices"]]
    if len(candidate_indices) != 50_000:
        raise RuntimeError(f"Expected shared 50k candidate pool: {category_dir}")
    if len(selected_indices) != 10_000 or len(set(selected_indices)) != 10_000:
        raise RuntimeError(f"Invalid selected memory indices: {category_dir}")
    if not set(selected_indices).issubset(set(candidate_indices)):
        raise RuntimeError(f"Selected indices escape candidate pool: {category_dir}")

    resolved = _read_json(root / "resolved_config.json")
    runtime_hash = _runtime_signature(resolved)
    input_size = int(resolved["data"]["input_size"])
    if str(resolved["model"]["name"]) != "dinov2_vits14":
        raise RuntimeError("The audit requires the fixed DINOv2 ViT-S/14 backbone")
    patches_per_image = (input_size // 14) ** 2
    memory_images = len(manifest["memory"])
    counts = [0] * memory_images
    for index in selected_indices:
        image_index = index // patches_per_image
        if image_index < 0 or image_index >= memory_images:
            raise RuntimeError(f"Selected index cannot be mapped to an image: {index}")
        counts[image_index] += 1
    if protocol == "M_IBK10" and max(counts) - min(counts) > 1:
        raise RuntimeError("M_IBK10 violates its near-equal per-image quota")

    metric = _read_d0_row(category_dir / "metrics_summary.csv")
    if str(metric.get("memory_protocol")) != protocol:
        raise RuntimeError(f"Metric protocol mismatch: {category_dir}")
    row: dict[str, Any] = {
        "protocol": protocol,
        "category": category,
        "seed": seed,
        "manifest_hash": manifest["manifest_hash"],
        "candidate_hash": stable_hash(candidate_indices),
        "memory_algorithm": state["memory_algorithm"],
        "memory_build_seconds": float(state["memory_build_seconds"]),
        **balance_statistics(counts),
    }
    for metric_name in METRICS:
        if metric_name in metric and metric[metric_name] != "":
            value = float(metric[metric_name])
            row[metric_name] = value if math.isfinite(value) else float("nan")
    return (
        row,
        str(manifest["manifest_hash"]),
        stable_hash(candidate_indices),
        runtime_hash,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    categories = [str(value) for value in args.categories]
    for category in categories:
        expected_manifest = None
        expected_candidates = None
        expected_runtime = None
        for protocol in PROTOCOLS:
            row, manifest_hash, candidate_hash, runtime_hash = _load_category(
                _protocol_root(args, protocol), protocol, category, int(args.seed)
            )
            if expected_manifest is None:
                expected_manifest = manifest_hash
                expected_candidates = candidate_hash
                expected_runtime = runtime_hash
            elif (
                manifest_hash != expected_manifest
                or candidate_hash != expected_candidates
                or runtime_hash != expected_runtime
            ):
                raise RuntimeError(
                    "Protocols do not share manifest, candidate pool, and runtime "
                    f"settings for {category}"
                )
            rows.append(row)

    macro_rows: list[dict[str, Any]] = []
    for protocol in PROTOCOLS:
        selected = [row for row in rows if row["protocol"] == protocol]
        macro: dict[str, Any] = {
            "protocol": protocol,
            "seed": int(args.seed),
            "categories": len(selected),
        }
        for field in (
            *METRICS,
            "memory_build_seconds",
            "selected_per_image_cv",
            "selected_per_image_gini",
            "selected_max_image_fraction",
        ):
            values = [float(row[field]) for row in selected if field in row]
            if values:
                macro[field] = float(mean(values))
        macro_rows.append(macro)

    baseline = next(row for row in macro_rows if row["protocol"] == "M_K10")
    deltas: list[dict[str, Any]] = []
    for row in macro_rows:
        if row["protocol"] == "M_K10":
            continue
        delta: dict[str, Any] = {"protocol": row["protocol"], "baseline": "M_K10"}
        for field in METRICS:
            if field in row and field in baseline:
                delta[f"delta_{field}"] = float(row[field]) - float(baseline[field])
        deltas.append(delta)

    output_dir = Path(args.output_dir)
    _write_csv(output_dir / "category_results.csv", rows)
    _write_csv(output_dir / "macro_summary.csv", macro_rows)
    _write_csv(output_dir / "delta_vs_global_k10.csv", deltas)
    summary = {
        "status": "complete",
        "protocol": "memory_partition_audit_v1",
        "retrospective": True,
        "seed": int(args.seed),
        "categories": categories,
        "methods": list(PROTOCOLS),
        "shared_candidate_entries": 50_000,
        "final_memory_entries": 10_000,
        "macro_summary": macro_rows,
        "delta_vs_global_k10": deltas,
    }
    summary["summary_hash"] = stable_hash(
        {"rows": rows, "macro": macro_rows, "deltas": deltas}
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "comparison_complete.json").write_text(
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
    parser = argparse.ArgumentParser(
        description="Strict same-pool whole/partitioned/image-balanced K10 audit"
    )
    parser.add_argument("--global-root", required=True)
    parser.add_argument("--audit-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--categories", nargs="+", default=list(DEFAULT_CATEGORIES))
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
