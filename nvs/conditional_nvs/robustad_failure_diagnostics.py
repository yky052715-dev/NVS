"""Post-hoc failure attribution for the locked RobustAD D0/D2 experiment.

This module deliberately does not tune or alter the detector.  It replays the
locked source-only fit so that quantities absent from the aggregate experiment
CSV (score distributions, patch regions, and projection energy) can be
measured.  Existing experiment outputs are treated as the protocol anchor.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

from nvs.common import patch_scores_to_maps

from . import cli
from .metrics import average_relative_drop, pixel_aupr, safe_auroc
from .pipeline import ConditionalNVSPipeline, _cosine_topk
from .protocol import config_fingerprint, stable_hash
from .robustad_replay_verification import verify_replayed_metrics

from .robustad_category_protocol import (
    RobustADCategoryRecord,
    assert_source_only,
    category_protocol_payload,
    evaluation_groups,
    mask_scope,
    parse_category_manifest,
    parse_official_directory,
    source_training,
    validate_records,
)


METHODS = ("D0_NN", "D2_NVSGlobal")
POOLING_FRACTIONS = (0.01, 0.05, 0.10)


def _optional_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def shift_family(shift: str) -> str:
    """Map RobustAD shifts to the three predeclared attribution families."""

    value = str(shift).strip().lower()
    if value == "source":
        return "source"
    if value in {"lighting", "white_balancing", "shadow"}:
        return "photometric"
    if value in {"rotation", "position", "scale", "position_rotation"}:
        return "geometry"
    if value.startswith("background") or value in {
        "global_configuration",
        "configuration",
    }:
        return "background_global_configuration"
    raise ValueError(f"Unclassified RobustAD shift: {shift!r}")


def finite_metric(labels: np.ndarray, scores: np.ndarray, metric: str) -> float:
    labels = np.asarray(labels).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if labels.size == 0 or np.unique(labels).size < 2:
        return float("nan")
    if metric == "AUROC":
        return safe_auroc(labels, scores)
    if metric == "AUPR":
        return float(average_precision_score(labels, scores))
    raise KeyError(metric)


def score_distribution(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "count": 0,
            "mean": float("nan"),
            "p95": float("nan"),
            "p99": float("nan"),
        }
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
    }


def projection_energy_ratio(
    deviation: torch.Tensor,
    basis: torch.Tensor,
    epsilon: float = 1.0e-12,
) -> torch.Tensor:
    """Return ||U d||^2 / (||d||^2 + eps) for row-basis U."""

    values = deviation.float()
    local_basis = basis.float().to(values.device, non_blocking=True)
    projected_coordinates = values @ local_basis.T
    numerator = projected_coordinates.square().sum(dim=-1)
    denominator = values.square().sum(dim=-1) + float(epsilon)
    return numerator / denominator


def masks_to_patch_regions(masks: np.ndarray, grid_side: int) -> np.ndarray:
    """Use adaptive max pooling so small defects survive patch-grid mapping."""

    values = torch.as_tensor(np.asarray(masks), dtype=torch.float32)
    if values.ndim != 3:
        raise ValueError("masks must have shape [N,H,W]")
    pooled = F.adaptive_max_pool2d(
        values[:, None], (int(grid_side), int(grid_side))
    )
    return pooled[:, 0].reshape(values.shape[0], -1).numpy() > 0


def top_fraction_mean(patch_scores: np.ndarray, fraction: float) -> np.ndarray:
    values = np.asarray(patch_scores, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError("patch_scores must have shape [N,P]")
    count = max(1, int(math.ceil(values.shape[1] * float(fraction))))
    partitioned = np.partition(values, values.shape[1] - count, axis=1)
    return partitioned[:, -count:].mean(axis=1)


def pooling_diagnostics(
    labels: np.ndarray, patch_scores: np.ndarray
) -> list[dict[str, Any]]:
    """Fixed, diagnostic-only image aggregations for PiledBags."""

    values = np.asarray(patch_scores, dtype=np.float64)
    aggregations = {
        "max": values.max(axis=1),
        "mean": values.mean(axis=1),
        **{
            f"top_{int(fraction * 100)}pct_mean": top_fraction_mean(
                values, fraction
            )
            for fraction in POOLING_FRACTIONS
        },
    }
    rows = []
    labels = np.asarray(labels, dtype=np.int64)
    for aggregation, image_scores in aggregations.items():
        normal = image_scores[labels == 0]
        anomaly = image_scores[labels == 1]
        rows.append(
            {
                "aggregation": aggregation,
                "image_AUROC": finite_metric(labels, image_scores, "AUROC"),
                "image_AUPR": finite_metric(labels, image_scores, "AUPR"),
                "normal_mean": float(normal.mean()) if normal.size else float("nan"),
                "anomaly_mean": (
                    float(anomaly.mean()) if anomaly.size else float("nan")
                ),
                "anomaly_minus_normal": (
                    float(anomaly.mean() - normal.mean())
                    if normal.size and anomaly.size
                    else float("nan")
                ),
                "diagnostic_only": True,
            }
        )
    return rows


def add_per_shift_ard(rows: list[dict[str, Any]]) -> None:
    """Attach min(0, target-source) for every available AUROC/AUPR metric."""

    source_by_key: dict[tuple[str, str, str], float] = {}
    metrics = ("image_AUROC", "image_AUPR", "pixel_AUROC", "pixel_AUPR")
    for row in rows:
        if row["domain"] != "source":
            continue
        for metric in metrics:
            value = _optional_float(row.get(metric))
            if math.isfinite(value):
                source_by_key[(str(row["category"]), str(row["method"]), metric)] = value
    for row in rows:
        for metric in metrics:
            key = (str(row["category"]), str(row["method"]), metric)
            value = _optional_float(row.get(metric))
            source = source_by_key.get(key)
            row[f"ARD_{metric}"] = (
                min(0.0, value - source)
                if row["domain"] == "target"
                and source is not None
                and math.isfinite(value)
                else (0.0 if row["domain"] == "source" and source is not None else float("nan"))
            )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_baseline(
    baseline_dir: Path,
    seed: int,
    categories: Sequence[str],
    expected_config_fingerprint: str,
) -> dict[str, Any]:
    marker_path = baseline_dir / "robustad_complete.json"
    metrics_path = baseline_dir / "robustad_metrics.csv"
    if not marker_path.is_file() or not metrics_path.is_file():
        raise FileNotFoundError(
            f"Baseline output requires {marker_path} and {metrics_path}"
        )
    marker = _load_json(marker_path)
    if marker.get("status") != "complete":
        raise ValueError("Baseline RobustAD experiment is not complete")
    if int(marker.get("seed", -1)) != int(seed):
        raise ValueError("Baseline seed does not match diagnostic seed")
    if tuple(marker.get("categories", ())) != tuple(categories):
        raise ValueError("Baseline category list does not match diagnostic categories")
    if not set(METHODS).issubset(set(marker.get("methods", ()))):
        raise ValueError("Baseline must contain D0_NN and D2_NVSGlobal")
    resolved_config_path = baseline_dir / "resolved_config.json"
    if not resolved_config_path.is_file():
        raise FileNotFoundError(resolved_config_path)
    baseline_config = _load_json(resolved_config_path)
    if config_fingerprint(baseline_config) != str(expected_config_fingerprint):
        raise ValueError(
            "Diagnostic config does not exactly match the locked baseline config"
        )
    return marker


def _deviation_and_eta(
    features: torch.Tensor, pipeline: ConditionalNVSPipeline
) -> tuple[torch.Tensor, torch.Tensor]:
    pipeline._require_fitted()
    if pipeline.whitener is not None:
        raise ValueError("Failure attribution currently requires cosine core retrieval")
    assert pipeline.memory_result is not None
    assert pipeline.search_bank is not None
    assert pipeline.prototype_model is not None
    shape = features.shape
    query = pipeline._normalized(features).reshape(-1, shape[-1])
    _, indices = _cosine_topk(
        query,
        pipeline.search_bank,
        1,
        pipeline.query_chunk_size,
        pipeline.bank_chunk_size,
    )
    nearest = pipeline.memory_result.memory_bank[
        indices[:, 0].to(pipeline.memory_result.memory_bank.device)
    ]
    deviation = query - nearest
    eta = projection_energy_ratio(
        deviation, pipeline.prototype_model.global_delta_basis
    )
    return deviation.reshape(shape), eta.reshape(shape[:-1])


def _region_row(
    values: np.ndarray,
    region: np.ndarray,
    *,
    category: str,
    domain: str,
    shift: str,
    method: str,
    region_name: str,
) -> dict[str, Any]:
    stats = score_distribution(np.asarray(values)[np.asarray(region, dtype=bool)])
    return {
        "category": category,
        "domain": domain,
        "shift": shift,
        "shift_family": shift_family(shift),
        "method": method,
        "region": region_name,
        **stats,
    }


def _score_group(
    records: Sequence[RobustADCategoryRecord],
    pipeline: ConditionalNVSPipeline,
    config: dict[str, Any],
    model: Any,
    device: torch.device,
    *,
    category: str,
    domain: str,
    shift: str,
    scope: str,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    image_records = [record.as_image_record() for record in records]
    features, grid_side = cli._features(image_records, config, model, device)
    raw = pipeline.score_patch_features(features)
    normalized = pipeline.normalize_scores(raw)
    labels = np.asarray([record.label for record in records], dtype=np.int64)
    masks = patch_regions = None
    if scope == "pixel":
        masks, labels = cli._masks_and_labels(
            image_records, int(config["data"]["input_size"])
        )
        patch_regions = masks_to_patch_regions(masks, grid_side)

    metric_rows: list[dict[str, Any]] = []
    normal_rows: list[dict[str, Any]] = []
    region_rows: list[dict[str, Any]] = []
    piled_rows: list[dict[str, Any]] = []
    for method in METHODS:
        patch = normalized[method].detach().cpu().numpy()
        maps = patch_scores_to_maps(
            normalized[method],
            grid_side=grid_side,
            output_size=int(config["data"]["input_size"]),
        ).numpy()
        image_scores = maps.reshape(maps.shape[0], -1).max(axis=1)
        row: dict[str, Any] = {
            "category": category,
            "domain": domain,
            "shift": shift,
            "shift_family": shift_family(shift),
            "metric_scope": scope,
            "method": method,
            "images": len(records),
            "image_AUROC": finite_metric(labels, image_scores, "AUROC"),
            "image_AUPR": finite_metric(labels, image_scores, "AUPR"),
        }
        if scope == "pixel" and masks is not None:
            row["pixel_AUROC"] = safe_auroc(masks, maps)
            row["pixel_AUPR"] = pixel_aupr(masks, maps)
        metric_rows.append(row)

        normal_scores = image_scores[labels == 0]
        threshold = float(pipeline.calibrations[method].threshold)
        if domain == "target":
            normal_rows.append(
                {
                    "category": category,
                    "domain": domain,
                    "shift": shift,
                    "shift_family": shift_family(shift),
                    "method": method,
                    "score_definition": "normalized_image_max",
                    "threshold": threshold,
                    **score_distribution(normal_scores),
                    "false_positive_rate": (
                        float((normal_scores >= threshold).mean())
                        if normal_scores.size
                        else float("nan")
                    ),
                }
            )

        if scope == "pixel" and masks is not None:
            defect = masks.astype(bool)
            normal = ~defect
            region_rows.extend(
                [
                    _region_row(
                        maps,
                        normal,
                        category=category,
                        domain=domain,
                        shift=shift,
                        method=method,
                        region_name="normal_pixel",
                    ),
                    _region_row(
                        maps,
                        defect,
                        category=category,
                        domain=domain,
                        shift=shift,
                        method=method,
                        region_name="defect_pixel",
                    ),
                ]
            )

        if category == "PiledBags":
            for diagnostic in pooling_diagnostics(labels, patch):
                piled_rows.append(
                    {
                        "category": category,
                        "domain": domain,
                        "shift": shift,
                        "shift_family": shift_family(shift),
                        "method": method,
                        **diagnostic,
                    }
                )

    if scope == "pixel" and patch_regions is not None:
        _, eta = _deviation_and_eta(features, pipeline)
        eta_values = eta.detach().cpu().numpy()
        region_rows.extend(
            [
                _region_row(
                    eta_values,
                    ~patch_regions,
                    category=category,
                    domain=domain,
                    shift=shift,
                    method="D2_projection_eta",
                    region_name="normal_patch",
                ),
                _region_row(
                    eta_values,
                    patch_regions,
                    category=category,
                    domain=domain,
                    shift=shift,
                    method="D2_projection_eta",
                    region_name="defect_patch",
                ),
            ]
        )
    return metric_rows, normal_rows, region_rows, piled_rows


def _region_deltas(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    index = {
        (
            str(row["category"]),
            str(row["domain"]),
            str(row["shift"]),
            str(row["region"]),
            str(row["method"]),
        ): row
        for row in rows
    }
    output = []
    groups = sorted(
        {
            (
                str(row["category"]),
                str(row["domain"]),
                str(row["shift"]),
                str(row["region"]),
            )
            for row in rows
            if row["method"] in METHODS
        }
    )
    for category, domain, shift, region in groups:
        d0 = index.get((category, domain, shift, region, "D0_NN"))
        d2 = index.get((category, domain, shift, region, "D2_NVSGlobal"))
        if d0 is None or d2 is None:
            continue
        output.append(
            {
                "category": category,
                "domain": domain,
                "shift": shift,
                "shift_family": shift_family(shift),
                "region": region,
                "D0_mean": float(d0["mean"]),
                "D2_mean": float(d2["mean"]),
                "D2_minus_D0_mean": float(d2["mean"]) - float(d0["mean"]),
                "D0_p95": float(d0["p95"]),
                "D2_p95": float(d2["p95"]),
                "D2_minus_D0_p95": float(d2["p95"]) - float(d0["p95"]),
                "D0_p99": float(d0["p99"]),
                "D2_p99": float(d2["p99"]),
                "D2_minus_D0_p99": float(d2["p99"]) - float(d0["p99"]),
            }
        )
    return output


def _eta_comparisons(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    selected = [row for row in rows if row["method"] == "D2_projection_eta"]
    index = {
        (str(row["category"]), str(row["domain"]), str(row["shift"]), str(row["region"])): row
        for row in selected
    }
    groups = sorted(
        {(str(row["category"]), str(row["domain"]), str(row["shift"])) for row in selected}
    )
    output = []
    for category, domain, shift in groups:
        normal = index.get((category, domain, shift, "normal_patch"))
        defect = index.get((category, domain, shift, "defect_patch"))
        if normal is None or defect is None:
            continue
        output.append(
            {
                "category": category,
                "domain": domain,
                "shift": shift,
                "shift_family": shift_family(shift),
                "normal_patch_count": int(normal["count"]),
                "defect_patch_count": int(defect["count"]),
                "eta_normal_mean": float(normal["mean"]),
                "eta_defect_mean": float(defect["mean"]),
                "eta_defect_minus_normal": float(defect["mean"])
                - float(normal["mean"]),
                "eta_normal_p95": float(normal["p95"]),
                "eta_defect_p95": float(defect["p95"]),
            }
        )
    return output

def _family_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    metrics = ("image_AUROC", "image_AUPR", "pixel_AUROC", "pixel_AUPR")
    groups = sorted(
        {
            (str(row["shift_family"]), str(row["method"]))
            for row in rows
            if row["domain"] == "target"
        }
    )
    for family, method in groups:
        selected = [
            row
            for row in rows
            if row["domain"] == "target"
            and row["shift_family"] == family
            and row["method"] == method
        ]
        result: dict[str, Any] = {
            "shift_family": family,
            "method": method,
            "rows": len(selected),
        }
        for metric in metrics:
            values = [
                float(row[metric])
                for row in selected
                if metric in row and math.isfinite(float(row[metric]))
            ]
            ards = [
                float(row[f"ARD_{metric}"])
                for row in selected
                if math.isfinite(float(row.get(f"ARD_{metric}", float("nan"))))
            ]
            result[metric] = mean(values) if values else float("nan")
            result[f"ARD_{metric}"] = mean(ards) if ards else float("nan")
        output.append(result)
    return output


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = cli._load_config(args.config)
    cli._validate_core_config(config)
    seed = int(args.seed if args.seed is not None else config["experiment"]["seed"])
    config["experiment"]["seed"] = seed
    config.setdefault("data", {})["root"] = str(args.data_root)
    categories_arg = args.categories or config["data"].get("categories")
    records = (
        parse_category_manifest(
            args.manifest, data_root=args.data_root, require_files=True
        )
        if args.manifest
        else parse_official_directory(
            args.data_root, categories=categories_arg, require_files=True
        )
    )
    categories = validate_records(records, categories_arg)
    locked_config_fingerprint = config_fingerprint(config)
    baseline_dir = Path(args.baseline_output)
    baseline_marker = _validate_baseline(
        baseline_dir, seed, categories, locked_config_fingerprint
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    completion_path = output_dir / "diagnostic_complete.json"
    required_outputs = (
        "metrics_by_category_shift.csv",
        "baseline_replay_verification.csv",
        "target_normal_score_stats.csv",
        "pixel_region_score_stats.csv",
        "projection_energy_comparison.csv",
        "piledbags_pooling_diagnostics.csv",
    )
    if not args.force and completion_path.is_file():
        existing = _load_json(completion_path)
        valid = (
            existing.get("status") == "complete"
            and int(existing.get("seed", -1)) == seed
            and tuple(existing.get("categories", ())) == tuple(categories)
            and existing.get("baseline_protocol_hash")
            == baseline_marker.get("protocol_hash")
            and existing.get("diagnostic_config_fingerprint")
            == locked_config_fingerprint
            and all((output_dir / name).is_file() for name in required_outputs)
        )
        if valid:
            print(f"[diagnostic] valid completion: {output_dir}", flush=True)
            return existing

    device = torch.device(args.device)
    model = cli.load_dinov2(
        config["model"]["name"], device, config["model"].get("hub_dir")
    )
    metric_rows: list[dict[str, Any]] = []
    normal_rows: list[dict[str, Any]] = []
    region_rows: list[dict[str, Any]] = []
    piled_rows: list[dict[str, Any]] = []
    protocol_hashes: dict[str, str] = {}
    for category in categories:
        category_dir = output_dir / category / f"seed{seed}"
        fit_records = source_training(records, category)
        assert_source_only(fit_records)
        pipeline, _, _ = cli._fit_category(
            [record.as_image_record() for record in fit_records],
            config,
            model,
            device,
            seed,
            category_dir,
            category,
        )
        if tuple(method for method in METHODS if method in pipeline.method_names()) != METHODS:
            raise AssertionError("Diagnostic replay requires D0 and D2")
        payload = category_protocol_payload(records, category, seed)
        protocol_hashes[category] = str(payload["manifest_hash"])
        for (group_category, domain, shift), group in sorted(
            evaluation_groups(records, category).items()
        ):
            scope = mask_scope(group)
            print(
                f"[diagnostic:{category}] domain={domain} shift={shift} scope={scope}",
                flush=True,
            )
            group_outputs = _score_group(
                group,
                pipeline,
                config,
                model,
                device,
                category=group_category,
                domain=domain,
                shift=shift,
                scope=scope,
            )
            for target, values in zip(
                (metric_rows, normal_rows, region_rows, piled_rows), group_outputs
            ):
                target.extend(values)

    add_per_shift_ard(metric_rows)
    verification_rows = verify_replayed_metrics(
        metric_rows, _read_csv(baseline_dir / "robustad_metrics.csv")
    )
    family_rows = _family_summary(metric_rows)
    delta_rows = _region_deltas(region_rows)
    eta_rows = _eta_comparisons(region_rows)
    _write_csv(output_dir / "metrics_by_category_shift.csv", metric_rows)
    _write_csv(output_dir / "baseline_replay_verification.csv", verification_rows)
    _write_csv(output_dir / "metrics_by_shift_family.csv", family_rows)
    _write_csv(output_dir / "target_normal_score_stats.csv", normal_rows)
    _write_csv(output_dir / "pixel_region_score_stats.csv", region_rows)
    _write_csv(output_dir / "pixel_region_d2_minus_d0.csv", delta_rows)
    _write_csv(output_dir / "projection_energy_comparison.csv", eta_rows)
    _write_csv(output_dir / "piledbags_pooling_diagnostics.csv", piled_rows)

    summary = {
        "status": "complete",
        "protocol": "robustad_failure_attribution_v1",
        "diagnostic_only": True,
        "model_or_threshold_tuning": False,
        "seed": seed,
        "categories": list(categories),
        "methods": list(METHODS),
        "baseline_protocol_hash": baseline_marker["protocol_hash"],
        "baseline_summary_hash": baseline_marker["summary_hash"],
        "diagnostic_config_fingerprint": locked_config_fingerprint,
        "category_manifest_hashes": protocol_hashes,
        "shift_families": sorted(
            {row["shift_family"] for row in metric_rows if row["domain"] == "target"}
        ),
        "row_counts": {
            "metrics": len(metric_rows),
            "baseline_verification": len(verification_rows),
            "normal_score_stats": len(normal_rows),
            "region_stats": len(region_rows),
            "region_deltas": len(delta_rows),
            "projection_energy_comparisons": len(eta_rows),
            "piledbags_pooling": len(piled_rows),
        },
    }
    summary["output_hash"] = stable_hash(
        {
            "metrics": metric_rows,
            "normal": normal_rows,
            "regions": region_rows,
            "piledbags": piled_rows,
        }
    )
    cli._save_json(output_dir / "diagnostic_complete.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Frozen post-hoc attribution for locked RobustAD D0/D2 outputs"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--baseline-output", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--categories", nargs="+")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def main() -> None:
    result = run(parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
