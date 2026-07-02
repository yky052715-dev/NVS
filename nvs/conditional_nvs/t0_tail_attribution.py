"""T0 defect-tail attribution for the frozen RobustAD D0/D2 detector.

The experiment is deliberately diagnostic-only.  It reuses the exact D0
nearest neighbour for both methods, decomposes its deviation into NVS
projection and residual energies, and reports raw, shared-calibration, and
legacy-calibration views without changing any fitted detector parameter.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from nvs.common import patch_scores_to_maps

from . import cli
from .metrics import binary_f1, pixel_aupr, safe_auroc
from .pipeline import ConditionalNVSPipeline, _cosine_topk
from .protocol import CalibrationState, fit_calibration, stable_hash
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
from .robustad_failure_diagnostics import masks_to_patch_regions, shift_family


METHODS = ("D0_NN", "D2_NVSGlobal")
VIEWS = ("raw_energy", "shared_calibration", "legacy_independent")
EPSILON = 1.0e-12
CONSISTENCY_TOLERANCE = 1.0e-5


@dataclass
class ConsistencyAccumulator:
    """Global maxima plus per-batch evidence for fail-fast checks."""

    tolerance: float = CONSISTENCY_TOLERANCE
    batches: int = 0
    max_abs_d0_minus_e0: float = 0.0
    max_abs_d2_minus_sqrt_2e2: float = 0.0
    max_relative_energy_identity_error: float = 0.0
    batch_records: list[dict[str, Any]] = field(default_factory=list)

    def update(
        self,
        d0: torch.Tensor,
        d2: torch.Tensor,
        e0: torch.Tensor,
        e_projected: torch.Tensor,
        e2: torch.Tensor,
        *,
        context: str,
    ) -> dict[str, Any]:
        d0_error = float((d0 - e0).abs().max().item())
        d2_error = float(
            (d2 - torch.sqrt(torch.clamp(2.0 * e2, min=0.0))).abs().max().item()
        )
        identity_error = (e0 - e_projected - e2).abs() / torch.clamp(
            e0.abs(), min=EPSILON
        )
        relative_error = float(identity_error.max().item())
        values = {
            "batch": self.batches,
            "context": context,
            "max_abs_d0_minus_e0": d0_error,
            "max_abs_d2_minus_sqrt_2e2": d2_error,
            "max_relative_energy_identity_error": relative_error,
        }
        if (
            not all(math.isfinite(value) for value in (d0_error, d2_error, relative_error))
            or d0_error >= self.tolerance
            or d2_error >= self.tolerance
            or relative_error >= self.tolerance
        ):
            raise AssertionError(
                "T0 consistency check failed; aggregation is forbidden: "
                + json.dumps(values, ensure_ascii=False, sort_keys=True)
            )
        self.batch_records.append(values)
        self.batches += 1
        self.max_abs_d0_minus_e0 = max(self.max_abs_d0_minus_e0, d0_error)
        self.max_abs_d2_minus_sqrt_2e2 = max(
            self.max_abs_d2_minus_sqrt_2e2, d2_error
        )
        self.max_relative_energy_identity_error = max(
            self.max_relative_energy_identity_error, relative_error
        )
        return values


def decompose_patch_energies(
    features: torch.Tensor,
    pipeline: ConditionalNVSPipeline,
    *,
    image_batch_size: int = 16,
    checks: ConsistencyAccumulator | None = None,
    context: str = "evaluation",
) -> dict[str, torch.Tensor]:
    """Use one raw NN lookup to compute D0, projection, and D2 energies."""

    pipeline._require_fitted()
    if pipeline.whitener is not None:
        raise ValueError("T0 requires the frozen cosine D0/D2 protocol")
    if features.ndim != 3:
        raise ValueError("features must have shape [images, patches, channels]")
    assert pipeline.memory_result is not None
    assert pipeline.search_bank is not None
    assert pipeline.prototype_model is not None
    basis = pipeline.prototype_model.global_delta_basis.float().to(
        pipeline.compute_device, non_blocking=True
    )
    outputs: dict[str, list[torch.Tensor]] = {
        "e0": [],
        "eP": [],
        "e2": [],
        "eta": [],
        "R_retain": [],
        "D0_NN": [],
        "D2_NVSGlobal": [],
    }
    accumulator = checks if checks is not None else ConsistencyAccumulator()
    batch_size = max(1, int(image_batch_size))
    for start in range(0, int(features.shape[0]), batch_size):
        stop = min(int(features.shape[0]), start + batch_size)
        batch = pipeline._normalized(features[start:stop])
        shape = batch.shape
        query = batch.reshape(-1, shape[-1])
        similarity, indices = _cosine_topk(
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
        coeff = deviation @ basis.T
        projected = coeff @ basis
        residual = deviation - projected
        e0 = 0.5 * deviation.square().sum(dim=-1)
        e_projected = 0.5 * projected.square().sum(dim=-1)
        e2 = 0.5 * residual.square().sum(dim=-1)
        d0 = 1.0 - similarity[:, 0]
        d2 = torch.linalg.norm(residual, dim=-1)
        accumulator.update(
            d0,
            d2,
            e0,
            e_projected,
            e2,
            context=f"{context}[{start}:{stop}]",
        )
        denominator = e0 + EPSILON
        batch_outputs = {
            "e0": e0,
            "eP": e_projected,
            "e2": e2,
            "eta": e_projected / denominator,
            "R_retain": e2 / denominator,
            "D0_NN": d0,
            "D2_NVSGlobal": d2,
        }
        for key, value in batch_outputs.items():
            outputs[key].append(value.reshape(shape[:-1]).float())
    return {key: torch.cat(parts, dim=0) for key, parts in outputs.items()}


def _normalize(values: torch.Tensor, state: CalibrationState) -> torch.Tensor:
    return torch.clamp(
        (values.float() - float(state.median)) / float(state.mad), min=0.0
    )


def fit_calibration_views(
    calibration_energy: Mapping[str, torch.Tensor],
    *,
    image_quantile: float,
    mad_epsilon: float,
) -> dict[str, CalibrationState]:
    """Fit shared energy and frozen legacy calibrations on source normals."""

    e0 = calibration_energy["e0"].detach().cpu().numpy()
    d0 = calibration_energy["D0_NN"].detach().cpu().numpy()
    d2 = calibration_energy["D2_NVSGlobal"].detach().cpu().numpy()
    shared = fit_calibration(
        e0,
        image_quantile=image_quantile,
        mad_epsilon=mad_epsilon,
        scope="identity_calibration",
    )
    legacy_d0 = fit_calibration(
        d0,
        image_quantile=image_quantile,
        mad_epsilon=mad_epsilon,
        scope="identity_calibration",
    )
    legacy_d2 = fit_calibration(
        d2,
        image_quantile=image_quantile,
        mad_epsilon=mad_epsilon,
        scope="identity_calibration",
    )
    return {
        "shared_e0": shared,
        "legacy_D0_NN": legacy_d0,
        "legacy_D2_NVSGlobal": legacy_d2,
    }


def score_views(
    energy: Mapping[str, torch.Tensor],
    calibrations: Mapping[str, CalibrationState],
) -> dict[str, dict[str, torch.Tensor]]:
    """Return the three predeclared score views without fitting anything."""

    e0, e2 = energy["e0"], energy["e2"]
    shared = calibrations["shared_e0"]
    return {
        "raw_energy": {"D0_NN": e0, "D2_NVSGlobal": e2},
        "shared_calibration": {
            "D0_NN": _normalize(e0, shared),
            "D2_NVSGlobal": _normalize(e2, shared),
        },
        "legacy_independent": {
            "D0_NN": _normalize(
                energy["D0_NN"], calibrations["legacy_D0_NN"]
            ),
            "D2_NVSGlobal": _normalize(
                energy["D2_NVSGlobal"], calibrations["legacy_D2_NVSGlobal"]
            ),
        },
    }


def distribution(values: np.ndarray) -> dict[str, float | int]:
    finite = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {
            "count": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "p90": float("nan"),
            "p95": float("nan"),
            "p99": float("nan"),
        }
    return {
        "count": int(finite.size),
        "mean": float(finite.mean()),
        "median": float(np.median(finite)),
        "p90": float(np.quantile(finite, 0.90)),
        "p95": float(np.quantile(finite, 0.95)),
        "p99": float(np.quantile(finite, 0.99)),
    }


def macro_region_statistics(
    values: np.ndarray, regions: np.ndarray
) -> tuple[dict[str, float | int], list[dict[str, float | int]]]:
    """Compute per-image distributions first, then macro-average the summaries."""

    arrays = np.asarray(values)
    masks = np.asarray(regions, dtype=bool)
    if arrays.shape != masks.shape or arrays.ndim < 2:
        raise ValueError("values and regions must have matching [N,...] shapes")
    per_image = [
        distribution(arrays[index][masks[index]])
        for index in range(arrays.shape[0])
        if masks[index].any()
    ]
    fields = ("mean", "median", "p90", "p95", "p99")
    result: dict[str, float | int] = {
        "images": len(per_image),
        "count": int(sum(int(row["count"]) for row in per_image)),
    }
    for field in fields:
        valid = [
            float(row[field])
            for row in per_image
            if math.isfinite(float(row[field]))
        ]
        result[field] = float(np.mean(valid)) if valid else float("nan")
    return result, per_image


def _finite_mean(values: Sequence[float]) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(finite.mean()) if finite.size else float("nan")


def _image_metric_rows(
    records: Sequence[RobustADCategoryRecord],
    masks: np.ndarray,
    patch_regions: np.ndarray,
    level_values: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]],
    thresholds: Mapping[tuple[str, str], float],
    *,
    category: str,
    domain: str,
    shift: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    labels = np.asarray([record.label for record in records], dtype=np.int64)
    for view in VIEWS:
        for method in METHODS:
            patch = level_values["patch"][view][method]
            pixel = level_values["pixel"][view][method]
            threshold = thresholds.get((view, method), float("nan"))
            for index, record in enumerate(records):
                mask = masks[index].astype(bool)
                patch_mask = patch_regions[index].astype(bool)
                pixel_auroc_value = (
                    safe_auroc(mask, pixel[index])
                    if np.unique(mask).size == 2
                    else float("nan")
                )
                pixel_aupr_value = (
                    pixel_aupr(mask, pixel[index])
                    if np.unique(mask).size == 2
                    else float("nan")
                )
                calibrated = math.isfinite(threshold)
                image_score = float(np.max(pixel[index]))
                image_prediction = (
                    bool(image_score >= threshold) if calibrated else None
                )
                row: dict[str, Any] = {
                    "category": category,
                    "domain": domain,
                    "shift": shift,
                    "shift_family": shift_family(shift),
                    "image_id": str(record.path),
                    "label": int(labels[index]),
                    "view": view,
                    "method": method,
                    "threshold": threshold,
                    "image_score": image_score,
                    "image_prediction": image_prediction,
                    "normal_image_fp": (
                        float(image_prediction)
                        if calibrated and int(labels[index]) == 0
                        else float("nan")
                    ),
                    "pixel_AUROC": pixel_auroc_value,
                    "pixel_AUPR": pixel_aupr_value,
                    "pixel_F1_calibrated": (
                        binary_f1(mask, pixel[index] >= threshold)
                        if calibrated
                        else float("nan")
                    ),
                }
                for level, values, region in (
                    ("patch", patch[index], patch_mask),
                    ("pixel", pixel[index], mask),
                ):
                    normal_stats = distribution(values[~region])
                    defect_stats = distribution(values[region])
                    for region_name, stats in (
                        ("normal", normal_stats),
                        ("defect", defect_stats),
                    ):
                        for stat_name, stat_value in stats.items():
                            row[f"{level}_{region_name}_{stat_name}"] = stat_value
                    row[f"{level}_G99"] = (
                        float(defect_stats["p99"]) - float(normal_stats["p99"])
                        if int(defect_stats["count"]) > 0
                        and int(normal_stats["count"]) > 0
                        else float("nan")
                    )
                rows.append(row)
    return rows


def aggregate_calibration_metrics(
    image_rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    groups = sorted(
        {
            (
                str(row["category"]),
                str(row["domain"]),
                str(row["shift"]),
                str(row["view"]),
                str(row["method"]),
            )
            for row in image_rows
        }
    )
    output: list[dict[str, Any]] = []
    for category, domain, shift, view, method in groups:
        selected = [
            row
            for row in image_rows
            if (
                str(row["category"]),
                str(row["domain"]),
                str(row["shift"]),
                str(row["view"]),
                str(row["method"]),
            )
            == (category, domain, shift, view, method)
        ]
        labels = np.asarray([int(row["label"]) for row in selected], dtype=np.int64)
        image_scores = np.asarray(
            [float(row["image_score"]) for row in selected], dtype=np.float64
        )
        row: dict[str, Any] = {
            "category": category,
            "domain": domain,
            "shift": shift,
            "shift_family": shift_family(shift),
            "view": view,
            "method": method,
            "images": len(selected),
            "threshold": float(selected[0]["threshold"]),
            "image_AUROC": safe_auroc(labels, image_scores),
            "image_AUPR": (
                float(average_precision_score(labels, image_scores))
                if np.unique(labels).size == 2
                else float("nan")
            ),
        }
        for metric in (
            "pixel_AUROC",
            "pixel_AUPR",
            "pixel_F1_calibrated",
            "normal_image_fp",
        ):
            row[metric] = _finite_mean([float(item[metric]) for item in selected])
        for level in ("patch", "pixel"):
            for region in ("normal", "defect"):
                for stat in ("mean", "median", "p90", "p95", "p99"):
                    field = f"{level}_{region}_{stat}"
                    row[field] = _finite_mean(
                        [float(item[field]) for item in selected]
                    )
            row[f"{level}_G99"] = (
                float(row[f"{level}_defect_p99"])
                - float(row[f"{level}_normal_p99"])
            )
        output.append(row)
    return output


def tail_retention_rows(
    aggregate_rows: Sequence[Mapping[str, Any]], epsilon: float = EPSILON
) -> list[dict[str, Any]]:
    index = {
        (
            str(row["category"]),
            str(row["domain"]),
            str(row["shift"]),
            str(row["view"]),
            str(row["method"]),
        ): row
        for row in aggregate_rows
    }
    output: list[dict[str, Any]] = []
    groups = sorted(
        {
            (
                str(row["category"]),
                str(row["domain"]),
                str(row["shift"]),
                str(row["view"]),
            )
            for row in aggregate_rows
        }
    )
    for category, domain, shift, view in groups:
        d0 = index[(category, domain, shift, view, "D0_NN")]
        d2 = index[(category, domain, shift, view, "D2_NVSGlobal")]
        for level in ("patch", "pixel"):
            g0 = float(d0[f"{level}_G99"])
            g2 = float(d2[f"{level}_G99"])
            output.append(
                {
                    "category": category,
                    "domain": domain,
                    "shift": shift,
                    "shift_family": shift_family(shift),
                    "view": view,
                    "level": level,
                    "D0_normal_p99": float(d0[f"{level}_normal_p99"]),
                    "D0_defect_p99": float(d0[f"{level}_defect_p99"]),
                    "D2_normal_p99": float(d2[f"{level}_normal_p99"]),
                    "D2_defect_p99": float(d2[f"{level}_defect_p99"]),
                    "G99_D0": g0,
                    "G99_D2": g2,
                    "R_G99": g2 / (g0 + float(epsilon)),
                }
            )
    return output


def _metric_from_rows(rows: Sequence[Mapping[str, Any]], metric: str) -> float:
    if metric in {"image_AUROC", "image_AUPR"}:
        labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
        scores = np.asarray([float(row["image_score"]) for row in rows])
        if np.unique(labels).size < 2:
            return float("nan")
        if metric == "image_AUROC":
            return safe_auroc(labels, scores)
        return float(average_precision_score(labels, scores))
    if metric in {"patch_G99", "pixel_G99"}:
        level = metric.split("_", 1)[0]
        defect = _finite_mean(
            [float(row[f"{level}_defect_p99"]) for row in rows]
        )
        normal = _finite_mean(
            [float(row[f"{level}_normal_p99"]) for row in rows]
        )
        return defect - normal
    return _finite_mean([float(row[metric]) for row in rows])


def paired_bootstrap_delta(
    d0_rows: Sequence[Mapping[str, Any]],
    d2_rows: Sequence[Mapping[str, Any]],
    metric: str,
    *,
    repeats: int,
    seed: int,
) -> dict[str, float | int]:
    """Bootstrap images jointly so D0/D2 keep the exact same resamples."""

    d0_index = {str(row["image_id"]): row for row in d0_rows}
    d2_index = {str(row["image_id"]): row for row in d2_rows}
    image_ids = sorted(set(d0_index) & set(d2_index))
    if not image_ids:
        raise ValueError("paired bootstrap requires aligned images")
    aligned_d0 = [d0_index[key] for key in image_ids]
    aligned_d2 = [d2_index[key] for key in image_ids]
    estimate = _metric_from_rows(aligned_d2, metric) - _metric_from_rows(
        aligned_d0, metric
    )
    rng = np.random.default_rng(int(seed))
    draws: list[float] = []
    for _ in range(max(1, int(repeats))):
        indices = rng.integers(0, len(image_ids), size=len(image_ids))
        sampled_d0 = [aligned_d0[int(index)] for index in indices]
        sampled_d2 = [aligned_d2[int(index)] for index in indices]
        value = _metric_from_rows(sampled_d2, metric) - _metric_from_rows(
            sampled_d0, metric
        )
        if math.isfinite(value):
            draws.append(value)
    if not draws:
        low = high = float("nan")
    else:
        low, high = np.quantile(np.asarray(draws), [0.025, 0.975]).tolist()
    return {
        "images": len(image_ids),
        "bootstrap_repeats": int(repeats),
        "valid_repeats": len(draws),
        "delta": float(estimate),
        "ci_low": float(low),
        "ci_high": float(high),
    }


def bootstrap_summary_rows(
    image_rows: Sequence[Mapping[str, Any]], *, repeats: int, seed: int
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    group_keys = sorted(
        {
            (str(row["category"]), str(row["domain"]), str(row["shift"]))
            for row in image_rows
        }
    )
    metrics = (
        "image_AUROC",
        "image_AUPR",
        "pixel_AUROC",
        "pixel_AUPR",
        "pixel_F1_calibrated",
        "normal_image_fp",
        "patch_G99",
        "pixel_G99",
    )
    for category, domain, shift in group_keys:
        group = [
            row
            for row in image_rows
            if (str(row["category"]), str(row["domain"]), str(row["shift"]))
            == (category, domain, shift)
        ]
        for view in VIEWS:
            by_method = {
                method: [
                    row
                    for row in group
                    if row["view"] == view and row["method"] == method
                ]
                for method in METHODS
            }
            for metric in metrics:
                if view == "raw_energy" and metric in {
                    "pixel_F1_calibrated",
                    "normal_image_fp",
                }:
                    continue
                result = paired_bootstrap_delta(
                    by_method["D0_NN"],
                    by_method["D2_NVSGlobal"],
                    metric,
                    repeats=repeats,
                    seed=seed + len(output) * 997,
                )
                output.append(
                    {
                        "category": category,
                        "domain": domain,
                        "shift": shift,
                        "shift_family": shift_family(shift),
                        "comparison": "D2_minus_D0",
                        "view": view,
                        "metric": metric,
                        **result,
                    }
                )
        for method in METHODS:
            shared = [
                row
                for row in group
                if row["view"] == "shared_calibration" and row["method"] == method
            ]
            legacy = [
                row
                for row in group
                if row["view"] == "legacy_independent" and row["method"] == method
            ]
            for metric in metrics:
                result = paired_bootstrap_delta(
                    legacy,
                    shared,
                    metric,
                    repeats=repeats,
                    seed=seed + len(output) * 997,
                )
                output.append(
                    {
                        "category": category,
                        "domain": domain,
                        "shift": shift,
                        "shift_family": shift_family(shift),
                        "comparison": "shared_minus_legacy",
                        "view": method,
                        "metric": metric,
                        **result,
                    }
                )
    return output


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _raw_region_rows(
    energy: Mapping[str, torch.Tensor],
    maps: Mapping[str, np.ndarray],
    patch_regions: np.ndarray,
    masks: np.ndarray,
    *,
    category: str,
    domain: str,
    shift: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for quantity in ("e0", "eP", "e2", "eta", "R_retain"):
        patch = energy[quantity].detach().cpu().numpy()
        for level, values, regions in (
            ("patch", patch, patch_regions),
            ("pixel", maps[quantity], masks.astype(bool)),
        ):
            for region_name, region in (
                ("normal", ~regions),
                ("defect", regions),
            ):
                stats, _ = macro_region_statistics(values, region)
                rows.append(
                    {
                        "category": category,
                        "domain": domain,
                        "shift": shift,
                        "shift_family": shift_family(shift),
                        "level": level,
                        "quantity": quantity,
                        "region": region_name,
                        **stats,
                    }
                )
    return rows


def _thresholds(
    calibrations: Mapping[str, CalibrationState],
) -> dict[tuple[str, str], float]:
    return {
        ("shared_calibration", method): calibrations["shared_e0"].threshold
        for method in METHODS
    } | {
        ("legacy_independent", "D0_NN"): calibrations[
            "legacy_D0_NN"
        ].threshold,
        ("legacy_independent", "D2_NVSGlobal"): calibrations[
            "legacy_D2_NVSGlobal"
        ].threshold,
    }


def _filter_groups(
    groups: Mapping[tuple[str, str, str], list[RobustADCategoryRecord]],
    shifts: Sequence[str] | None,
) -> dict[tuple[str, str, str], list[RobustADCategoryRecord]]:
    if not shifts:
        return dict(groups)
    requested = {str(shift) for shift in shifts}
    selected = {
        key: records
        for key, records in groups.items()
        if str(key[2]) in requested
    }
    missing = requested - {key[2] for key in selected}
    if missing:
        raise ValueError(f"Requested shifts not found: {sorted(missing)}")
    return selected


def _validate_t0_config(config: Mapping[str, Any]) -> None:
    cli._validate_core_config(dict(config))
    if str(config["model"]["name"]) != "dinov2_vits14":
        raise ValueError("T0 is locked to DINOv2 ViT-S/14")
    if int(config["data"]["input_size"]) != 518:
        raise ValueError("T0 is locked to 518 input")
    if str(config["memory"]["protocol"]) != "M_K10":
        raise ValueError("T0 is locked to M_K10")
    if int(config["subspace"]["rank"]) != 8:
        raise ValueError("T0 is locked to global rank 8")
    if len(config.get("fit_transforms", ())) != 13:
        raise ValueError("T0 requires exactly 13 frozen NVS fit transforms")
    if str((config.get("whitening", {}) or {}).get("mode", "cosine")) != "cosine":
        raise ValueError("T0 requires cosine raw NN")
    if bool((config.get("augmem", {}) or {}).get("enabled", False)):
        raise ValueError("T0 forbids AugMem")
    if bool((config.get("fusion", {}) or {}).get("enabled", False)):
        raise ValueError("T0 forbids score fusion")


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = cli._load_config(args.config)
    if args.gpu_batch_size is not None:
        config["model"]["gpu_batch_size"] = int(args.gpu_batch_size)
    if args.num_workers is not None:
        config["data"]["num_workers"] = int(args.num_workers)
    if args.query_chunk_size is not None:
        config["memory"]["gpu_query_chunk_size"] = int(args.query_chunk_size)
    if args.bank_chunk_size is not None:
        config["memory"]["gpu_bank_chunk_size"] = int(args.bank_chunk_size)
    _validate_t0_config(config)
    seed = int(args.seed if args.seed is not None else config["experiment"]["seed"])
    config["experiment"]["seed"] = seed
    config["data"]["root"] = str(args.data_root)
    categories = list(args.categories or ("MetalParts", "PCB"))
    if set(categories) - {"MetalParts", "PCB"}:
        raise ValueError("T0 patch-tail attribution is restricted to MetalParts and PCB")
    records = (
        parse_category_manifest(
            args.manifest, data_root=args.data_root, require_files=True
        )
        if args.manifest
        else parse_official_directory(
            args.data_root, categories=categories, require_files=True
        )
    )
    categories = validate_records(records, categories)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checks = ConsistencyAccumulator()
    raw_rows: list[dict[str, Any]] = []
    image_rows: list[dict[str, Any]] = []
    protocols: dict[str, Any] = {}
    calibration_payload: dict[str, Any] = {}
    device = torch.device(args.device)
    model = cli.load_dinov2(
        config["model"]["name"], device, config["model"].get("hub_dir")
    )
    for category in categories:
        fit_records = source_training(records, category)
        assert_source_only(fit_records)
        category_dir = output_dir / "_state" / category / f"seed{seed}"
        pipeline, manifest, _ = cli._fit_category(
            [record.as_image_record() for record in fit_records],
            config,
            model,
            device,
            seed,
            category_dir,
            category,
        )
        if pipeline.whitener is not None:
            raise AssertionError("T0 unexpectedly fitted a non-cosine detector")
        split_paths = set(manifest["calibration"])
        calibration_records = [
            record.as_image_record()
            for record in fit_records
            if str(record.path) in split_paths
        ]
        if not calibration_records:
            raise AssertionError("Could not recover source calibration split")
        calibration_features, _ = cli._features(
            calibration_records, config, model, device
        )
        calibration_energy = decompose_patch_energies(
            calibration_features,
            pipeline,
            image_batch_size=int(args.score_image_batch_size),
            checks=checks,
            context=f"{category}/calibration",
        )
        calibrations = fit_calibration_views(
            calibration_energy,
            image_quantile=float(config["calibration"].get("image_quantile", 0.95)),
            mad_epsilon=float(config["calibration"].get("mad_epsilon", 1.0e-6)),
        )
        for method in METHODS:
            existing = pipeline.calibrations[method]
            fitted = calibrations[f"legacy_{method}"]
            if not np.allclose(
                [existing.median, existing.mad, existing.threshold],
                [fitted.median, fitted.mad, fitted.threshold],
                rtol=1.0e-7,
                atol=1.0e-8,
            ):
                raise AssertionError(
                    f"Legacy calibration replay mismatch for {category}/{method}"
                )
        calibration_payload[category] = {
            key: asdict(value) for key, value in calibrations.items()
        }
        protocols[category] = {
            **category_protocol_payload(records, category, seed),
            "split_manifest_hash": stable_hash(manifest),
        }
        groups = _filter_groups(evaluation_groups(records, category), args.shifts)
        for (group_category, domain, shift), group in sorted(groups.items()):
            if mask_scope(group) != "pixel":
                raise ValueError(
                    f"T0 requires GT masks for {group_category}/{domain}/{shift}"
                )
            print(
                f"[T0:{category}] domain={domain} shift={shift} images={len(group)}",
                flush=True,
            )
            image_records = [record.as_image_record() for record in group]
            features, grid_side = cli._features(
                image_records, config, model, device
            )
            energy = decompose_patch_energies(
                features,
                pipeline,
                image_batch_size=int(args.score_image_batch_size),
                checks=checks,
                context=f"{category}/{domain}/{shift}",
            )
            masks, _ = cli._masks_and_labels(
                image_records, int(config["data"]["input_size"])
            )
            patch_regions = masks_to_patch_regions(masks, grid_side)
            energy_maps = {
                key: patch_scores_to_maps(
                    value,
                    grid_side=grid_side,
                    output_size=int(config["data"]["input_size"]),
                ).numpy()
                for key, value in energy.items()
                if key in {"e0", "eP", "e2", "eta", "R_retain"}
            }
            raw_rows.extend(
                _raw_region_rows(
                    energy,
                    energy_maps,
                    patch_regions,
                    masks,
                    category=group_category,
                    domain=domain,
                    shift=shift,
                )
            )
            views = score_views(energy, calibrations)
            level_values: dict[str, dict[str, dict[str, np.ndarray]]] = {
                "patch": {},
                "pixel": {},
            }
            for view, method_scores in views.items():
                level_values["patch"][view] = {}
                level_values["pixel"][view] = {}
                for method, patch_score in method_scores.items():
                    level_values["patch"][view][method] = (
                        patch_score.detach().cpu().numpy()
                    )
                    level_values["pixel"][view][method] = patch_scores_to_maps(
                        patch_score,
                        grid_side=grid_side,
                        output_size=int(config["data"]["input_size"]),
                    ).numpy()
            image_rows.extend(
                _image_metric_rows(
                    group,
                    masks,
                    patch_regions,
                    level_values,
                    _thresholds(calibrations),
                    category=group_category,
                    domain=domain,
                    shift=shift,
                )
            )
    aggregate_rows = aggregate_calibration_metrics(image_rows)
    tail_rows = tail_retention_rows(aggregate_rows)
    bootstrap_rows = bootstrap_summary_rows(
        image_rows, repeats=int(args.bootstrap_repeats), seed=seed
    )
    _write_csv(output_dir / "raw_patch_energy_stats.csv", raw_rows)
    _write_csv(output_dir / "tail_retention_by_shift.csv", tail_rows)
    _write_csv(output_dir / "calibration_comparison.csv", aggregate_rows)
    _write_csv(output_dir / "paired_image_metrics.csv", image_rows)
    _write_csv(output_dir / "bootstrap_summary.csv", bootstrap_rows)
    consistency_payload = {
        "status": "passed",
        **asdict(checks),
        "strict_inequality": True,
    }
    _save_json(output_dir / "consistency_checks.json", consistency_payload)
    run_payload = {
        "status": "attribution_complete",
        "protocol": "T0_defect_high_score_tail_attribution_v1",
        "diagnostic_only": True,
        "robustad_role": "observed_diagnostic_not_final_validation",
        "detector_modified": False,
        "parameters_tuned": False,
        "soft_projection_added": False,
        "top_k_added": False,
        "seed": seed,
        "categories": categories,
        "shifts": list(args.shifts or []),
        "methods": list(METHODS),
        "views": list(VIEWS),
        "config": {
            "backbone": config["model"]["name"],
            "input_size": int(config["data"]["input_size"]),
            "memory_protocol": config["memory"]["protocol"],
            "nvs_rank": int(config["subspace"]["rank"]),
            "fit_transforms": 13,
            "gpu_batch_size": int(
                config["model"].get(
                    "gpu_batch_size", config["model"].get("batch_size", 1)
                )
            ),
            "num_workers": int(config["data"].get("num_workers", 0)),
            "gpu_query_chunk_size": int(
                config["memory"].get("gpu_query_chunk_size", 0)
            ),
            "gpu_bank_chunk_size": int(
                config["memory"].get("gpu_bank_chunk_size", 0)
            ),
        },
        "calibrations": calibration_payload,
        "category_protocols": protocols,
        "row_counts": {
            "raw_energy": len(raw_rows),
            "tail_retention": len(tail_rows),
            "calibration_comparison": len(aggregate_rows),
            "paired_images": len(image_rows),
            "bootstrap": len(bootstrap_rows),
        },
        "consistency": consistency_payload,
    }
    run_payload["output_hash"] = stable_hash(
        {
            "raw": raw_rows,
            "tail": tail_rows,
            "calibration": aggregate_rows,
            "bootstrap": bootstrap_rows,
        }
    )
    _save_json(output_dir / "t0_attribution_complete.json", run_payload)
    return run_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Frozen T0 D0/D2 defect-tail attribution"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--categories", nargs="+", default=["MetalParts", "PCB"])
    parser.add_argument("--shifts", nargs="+")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--score-image-batch-size", type=int, default=16)
    parser.add_argument("--gpu-batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--query-chunk-size", type=int)
    parser.add_argument("--bank-chunk-size", type=int)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def main() -> None:
    result = run(parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
