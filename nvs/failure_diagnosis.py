from __future__ import annotations

import argparse
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from nvs.common import (
    build_mvtec_records,
    load_config,
    load_dinov2,
    resize_pil,
    save_json,
    write_csv,
)
from nvs.detection import (
    _add_fusion_maps,
    _calibrate_methods,
    _fit_state,
    _normalize_base_maps,
    _score_records,
)
from nvs.metrics import _label_components, binary_f1, keep_topk_components, oracle_pixel_f1, safe_auroc


DEFAULT_METHODS = ["R0_nn_distance", "R2_nvs_residual", "P_topk3_r2"]


def _topk_method_name(k: int) -> str:
    return f"P_topk{int(k)}_r2"


def _method_outputs(
    raw_maps: dict[str, np.ndarray],
    calibrations: dict[str, dict[str, Any]],
    config: dict[str, Any],
    methods: list[str],
) -> dict[str, dict[str, Any]]:
    base_maps = _normalize_base_maps(raw_maps, calibrations)
    maps_by_method = _add_fusion_maps(base_maps, config)
    outputs: dict[str, dict[str, Any]] = {}
    for method in methods:
        if method not in calibrations:
            raise KeyError(f"Method {method!r} is unavailable. Available: {sorted(calibrations)}")
        cal = calibrations[method]
        threshold = float(cal["pixel_threshold"])
        if cal.get("kind") == "topk_component":
            source = str(cal["source_method"])
            maps = maps_by_method[source]
            pre_topk = maps >= threshold
            predictions = keep_topk_components(pre_topk, k=int(cal["topk_components"]))
        else:
            maps = maps_by_method[method]
            pre_topk = maps >= threshold
            predictions = pre_topk
        outputs[method] = {
            "maps": maps,
            "predictions": predictions.astype(bool),
            "pre_topk_predictions": pre_topk.astype(bool),
            "threshold": threshold,
            "calibration": cal,
        }
    return outputs


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _pixel_recall(mask: np.ndarray, prediction: np.ndarray) -> float:
    mask = np.asarray(mask, dtype=bool)
    prediction = np.asarray(prediction, dtype=bool)
    gt = float(mask.sum())
    if gt <= 0.0:
        return float("nan")
    return float(np.logical_and(mask, prediction).sum() / gt)


def _overseg(mask: np.ndarray, prediction: np.ndarray) -> float:
    mask = np.asarray(mask, dtype=bool)
    prediction = np.asarray(prediction, dtype=bool)
    gt = float(mask.sum())
    if gt <= 0.0:
        return float("nan")
    return float(prediction.sum() / gt)


def _score_stats(mask: np.ndarray, score_map: np.ndarray) -> dict[str, float]:
    mask = np.asarray(mask, dtype=bool)
    scores = np.asarray(score_map, dtype=np.float64)
    gt_scores = scores[mask]
    bg_scores = scores[~mask]
    if gt_scores.size == 0 or bg_scores.size == 0:
        return {
            "gt_score_mean": float("nan"),
            "gt_score_p95": float("nan"),
            "bg_score_mean": float("nan"),
            "bg_score_p95": float("nan"),
            "gt_bg_margin_mean": float("nan"),
            "gt_bg_margin_p95": float("nan"),
            "per_image_pixel_AUROC": float("nan"),
        }
    return {
        "gt_score_mean": float(gt_scores.mean()),
        "gt_score_p95": float(np.quantile(gt_scores, 0.95)),
        "bg_score_mean": float(bg_scores.mean()),
        "bg_score_p95": float(np.quantile(bg_scores, 0.95)),
        "gt_bg_margin_mean": float(gt_scores.mean() - bg_scores.mean()),
        "gt_bg_margin_p95": float(np.quantile(gt_scores, 0.95) - np.quantile(bg_scores, 0.95)),
        "per_image_pixel_AUROC": safe_auroc(mask.reshape(-1), scores.reshape(-1)),
    }


def _component_stats(binary: np.ndarray) -> dict[str, float]:
    binary = np.asarray(binary, dtype=bool)
    labels, count = _label_components(binary)
    total = float(binary.size)
    area = float(binary.sum())
    if count <= 0:
        return {
            "component_count": 0,
            "area_fraction": area / total,
            "largest_component_fraction": 0.0,
            "largest_component_area": 0.0,
            "largest_component_aspect_ratio": float("nan"),
        }

    sizes = np.bincount(labels.reshape(-1), minlength=count + 1).astype(np.float64)
    sizes[0] = 0.0
    largest = int(np.argmax(sizes))
    ys, xs = np.where(labels == largest)
    if ys.size == 0 or xs.size == 0:
        aspect = float("nan")
    else:
        height = float(ys.max() - ys.min() + 1)
        width = float(xs.max() - xs.min() + 1)
        aspect = float(max(height, width) / max(1.0, min(height, width)))
    return {
        "component_count": int(count),
        "area_fraction": area / total,
        "largest_component_fraction": float(sizes[largest] / total),
        "largest_component_area": float(sizes[largest]),
        "largest_component_aspect_ratio": aspect,
    }


def _image_metrics(mask: np.ndarray, score_map: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    oracle_f1, oracle_threshold = oracle_pixel_f1(mask[None].astype(bool), score_map[None].astype(np.float32))
    return {
        "f1": binary_f1(mask, prediction),
        "recall": _pixel_recall(mask, prediction),
        "overseg": _overseg(mask, prediction),
        "oracle_f1": oracle_f1,
        "oracle_threshold": oracle_threshold,
        **_score_stats(mask, score_map),
    }


def _diagnosis_hints(row: dict[str, Any]) -> list[str]:
    hints: list[str] = []
    r0_f1 = _safe_float(row["r0_f1"])
    r2_f1 = _safe_float(row["r2_f1"])
    p3_f1 = _safe_float(row["p3_f1"])
    r0_oracle = _safe_float(row["r0_oracle_f1"])
    r2_oracle = _safe_float(row["r2_oracle_f1"])
    r0_auroc = _safe_float(row["r0_per_image_pixel_AUROC"])
    r2_auroc = _safe_float(row["r2_per_image_pixel_AUROC"])
    r0_recall = _safe_float(row["r0_recall"])
    r2_recall = _safe_float(row["r2_recall"])
    p3_recall = _safe_float(row["p3_recall"])
    r0_overseg = _safe_float(row["r0_overseg"])
    r2_overseg = _safe_float(row["r2_overseg"])
    p3_overseg = _safe_float(row["p3_overseg"])
    r0_margin = _safe_float(row["r0_gt_bg_margin_mean"])
    r2_margin = _safe_float(row["r2_gt_bg_margin_mean"])

    if r2_oracle <= r0_oracle + 0.005 and r2_auroc <= r0_auroc + 0.002:
        hints.append("clean_score_no_gain")
    if r2_oracle - r2_f1 >= 0.10:
        hints.append("threshold_mismatch")
    if r2_margin < r0_margin - 0.05:
        hints.append("gt_bg_margin_degraded_by_nvs")
    if _safe_float(row["r2_gt_score_mean"]) < _safe_float(row["r0_gt_score_mean"]) - 0.10:
        hints.append("gt_score_suppressed_by_nvs")
    if r2_overseg > r0_overseg + 0.25:
        hints.append("score_diffusion")
    if r2_recall < r0_recall - 0.05:
        hints.append("recall_loss_after_nvs")
    if p3_recall < r2_recall - 0.03 or _safe_float(row["p3_deleted_gt_fraction"]) > 0.03:
        hints.append("topk_deleted_gt")
    if _safe_float(row["gt_area_fraction"]) > 0.02 and p3_recall < r2_recall - 0.02:
        hints.append("extended_defect_not_topk_friendly")
    if _safe_float(row["gt_component_count"]) > 3 and p3_recall < r2_recall - 0.02:
        hints.append("multi_component_defect_not_topk_friendly")
    if _safe_float(row["r2_component_count"]) > max(5.0, 3.0 * _safe_float(row["gt_component_count"])):
        hints.append("fragmented_prediction")
    if p3_overseg < r2_overseg - 0.10 and p3_recall >= r2_recall - 0.02:
        hints.append("topk_helped_without_recall_loss")
    if p3_f1 > r0_f1 + 0.02:
        hints.append("candidate_helped")
    if not hints:
        hints.append("mixed_or_unclear")
    return hints


def _load_resized_image(path: str, size: int) -> np.ndarray:
    with Image.open(path) as handle:
        image = resize_pil(handle, size, is_mask=False)
    return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _norm_heatmap(score_map: np.ndarray, vmax: float | None = None) -> np.ndarray:
    values = np.asarray(score_map, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros_like(values, dtype=np.float64)
    lo = float(np.quantile(finite, 0.01))
    hi = float(np.quantile(finite, 0.99)) if vmax is None else float(vmax)
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0)


def _plot_case(
    row: dict[str, Any],
    image: np.ndarray,
    mask: np.ndarray,
    r0_map: np.ndarray,
    r2_map: np.ndarray,
    r0_pred: np.ndarray,
    r2_pred: np.ndarray,
    p3_pred: np.ndarray,
    output_path: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - depends on runtime extras
        raise RuntimeError("matplotlib is required for visualizations") from exc

    deleted = np.logical_and(r2_pred, ~p3_pred)
    deleted_gt = np.logical_and(deleted, mask)
    diff = r2_map - r0_map
    vmax = max(float(np.quantile(r0_map, 0.99)), float(np.quantile(r2_map, 0.99)), 1.0)

    panels = [
        ("image", image, None),
        ("GT", mask, "gray"),
        (f"R0 map\nF1={row['r0_f1']:.3f}", _norm_heatmap(r0_map, vmax=vmax), "magma"),
        (f"R0 mask\nR={row['r0_recall']:.3f} O={row['r0_overseg']:.2f}", r0_pred, "gray"),
        (f"R2 map\nF1={row['r2_f1']:.3f}", _norm_heatmap(r2_map, vmax=vmax), "magma"),
        (f"R2 mask\nR={row['r2_recall']:.3f} O={row['r2_overseg']:.2f}", r2_pred, "gray"),
        (f"P3 mask\nF1={row['p3_f1']:.3f}", p3_pred, "gray"),
        ("R2-R0 diff", diff, "coolwarm"),
        ("P3 deleted", deleted, "gray"),
        ("deleted GT", deleted_gt, "gray"),
    ]
    fig, axes = plt.subplots(2, 5, figsize=(16, 6), constrained_layout=True)
    for ax, (title, data, cmap) in zip(axes.reshape(-1), panels):
        if cmap is None:
            ax.imshow(data)
        else:
            ax.imshow(data, cmap=cmap)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.suptitle(
        f"{row['category']} | {row['defect_type']} | {Path(str(row['image_path'])).name} | {row['diagnosis_hint']}",
        fontsize=10,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _case_row(
    category: str,
    index: int,
    scored: dict[str, Any],
    outputs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    mask = scored["masks"][index].astype(bool)
    r0_map = outputs["R0_nn_distance"]["maps"][index]
    r2_map = outputs["R2_nvs_residual"]["maps"][index]
    r0_pred = outputs["R0_nn_distance"]["predictions"][index]
    r2_pred = outputs["R2_nvs_residual"]["predictions"][index]
    p3_pred = outputs["P_topk3_r2"]["predictions"][index]
    p3_pre = outputs["P_topk3_r2"]["pre_topk_predictions"][index]

    r0 = _image_metrics(mask, r0_map, r0_pred)
    r2 = _image_metrics(mask, r2_map, r2_pred)
    p3 = _image_metrics(mask, r2_map, p3_pred)
    gt_components = _component_stats(mask)
    r2_components = _component_stats(r2_pred)
    p3_components = _component_stats(p3_pred)
    deleted = np.logical_and(p3_pre, ~p3_pred)
    deleted_gt = np.logical_and(deleted, mask)
    gt_count = max(1.0, float(mask.sum()))
    pre_area = max(1.0, float(p3_pre.sum()))

    row: dict[str, Any] = {
        "category": category,
        "image_path": str(scored["paths"][index]),
        "defect_type": str(scored["defect_types"][index]),
        "gt_area_fraction": gt_components["area_fraction"],
        "gt_component_count": gt_components["component_count"],
        "gt_largest_component_fraction": gt_components["largest_component_fraction"],
        "gt_largest_component_aspect_ratio": gt_components["largest_component_aspect_ratio"],
        "r0_threshold": outputs["R0_nn_distance"]["threshold"],
        "r2_threshold": outputs["R2_nvs_residual"]["threshold"],
        "p3_threshold": outputs["P_topk3_r2"]["threshold"],
        "p3_deleted_area_fraction": float(deleted.sum() / pre_area),
        "p3_deleted_gt_fraction": float(deleted_gt.sum() / gt_count),
        "r2_component_count": r2_components["component_count"],
        "r2_pred_area_fraction": r2_components["area_fraction"],
        "r2_largest_component_fraction": r2_components["largest_component_fraction"],
        "p3_component_count": p3_components["component_count"],
        "p3_pred_area_fraction": p3_components["area_fraction"],
        "p3_largest_component_fraction": p3_components["largest_component_fraction"],
    }
    for prefix, metrics in (("r0", r0), ("r2", r2), ("p3", p3)):
        for key, value in metrics.items():
            row[f"{prefix}_{key}"] = value
    row["r2_f1_delta_vs_r0"] = row["r2_f1"] - row["r0_f1"]
    row["p3_f1_delta_vs_r0"] = row["p3_f1"] - row["r0_f1"]
    row["r2_recall_delta_vs_r0"] = row["r2_recall"] - row["r0_recall"]
    row["p3_recall_delta_vs_r0"] = row["p3_recall"] - row["r0_recall"]
    row["r2_overseg_delta_vs_r0"] = row["r2_overseg"] - row["r0_overseg"]
    row["p3_overseg_delta_vs_r0"] = row["p3_overseg"] - row["r0_overseg"]
    row["r2_oracle_delta_vs_r0"] = row["r2_oracle_f1"] - row["r0_oracle_f1"]
    row["r2_calibration_gap"] = row["r2_oracle_f1"] - row["r2_f1"]
    row["p3_calibration_gap"] = row["p3_oracle_f1"] - row["p3_f1"]
    row["r2_gt_score_delta_vs_r0"] = row["r2_gt_score_mean"] - row["r0_gt_score_mean"]
    row["r2_bg_score_delta_vs_r0"] = row["r2_bg_score_mean"] - row["r0_bg_score_mean"]
    row["r2_margin_delta_vs_r0"] = row["r2_gt_bg_margin_mean"] - row["r0_gt_bg_margin_mean"]
    hints = _diagnosis_hints(row)
    row["diagnosis_hint"] = ";".join(hints)
    return row


def _normal_row(
    category: str,
    index: int,
    scored: dict[str, Any],
    outputs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "category": category,
        "image_path": str(scored["paths"][index]),
        "defect_type": str(scored["defect_types"][index]),
    }
    for method, prefix in (
        ("R0_nn_distance", "r0"),
        ("R2_nvs_residual", "r2"),
        ("P_topk3_r2", "p3"),
    ):
        pred = outputs[method]["predictions"][index]
        score_map = outputs[method]["maps"][index]
        row[f"{prefix}_is_positive"] = bool(pred.any())
        row[f"{prefix}_positive_area_fraction"] = float(pred.mean())
        row[f"{prefix}_image_score_max"] = float(np.max(score_map))
        row[f"{prefix}_threshold"] = float(outputs[method]["threshold"])
    row["r2_positive_area_delta_vs_r0"] = row["r2_positive_area_fraction"] - row["r0_positive_area_fraction"]
    row["p3_positive_area_delta_vs_r0"] = row["p3_positive_area_fraction"] - row["r0_positive_area_fraction"]
    return row


def _write_summary(case_rows: list[dict[str, Any]], normal_rows: list[dict[str, Any]], path: Path) -> None:
    hint_counter = Counter()
    category_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in case_rows:
        category_rows[str(row["category"])].append(row)
        hint_counter.update(str(row["diagnosis_hint"]).split(";"))

    lines = [
        "# NVS Validation10 failure diagnosis",
        "",
        "This report is diagnostic rather than confirmatory: hints are non-exclusive evidence tags, not a closed list of failure causes.",
        "",
        "## Category mean deltas",
        "",
        "| category | cases | P3 F1 Δ | R2 oracle Δ | P3 recall Δ | P3 OverSeg Δ | R2 margin Δ | P3 deleted GT |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for category in sorted(category_rows):
        rows = category_rows[category]
        lines.append(
            "| "
            + " | ".join(
                [
                    category,
                    str(len(rows)),
                    f"{np.nanmean([_safe_float(row['p3_f1_delta_vs_r0']) for row in rows]):.6f}",
                    f"{np.nanmean([_safe_float(row['r2_oracle_delta_vs_r0']) for row in rows]):.6f}",
                    f"{np.nanmean([_safe_float(row['p3_recall_delta_vs_r0']) for row in rows]):.6f}",
                    f"{np.nanmean([_safe_float(row['p3_overseg_delta_vs_r0']) for row in rows]):.6f}",
                    f"{np.nanmean([_safe_float(row['r2_margin_delta_vs_r0']) for row in rows]):.6f}",
                    f"{np.nanmean([_safe_float(row['p3_deleted_gt_fraction']) for row in rows]):.6f}",
                ]
            )
            + " |"
        )

    lines.extend(["", "## Hint counts", "", "| hint | count |", "|---|---:|"])
    for hint, count in hint_counter.most_common():
        lines.append(f"| {hint} | {count} |")

    worst = sorted(case_rows, key=lambda row: (_safe_float(row["p3_f1_delta_vs_r0"]), _safe_float(row["p3_recall_delta_vs_r0"])))[:20]
    lines.extend(
        [
            "",
            "## Worst P3-vs-R0 anomaly cases",
            "",
            "| category | image | defect | F1 R0→P3 | recall R0→P3 | OverSeg R0→P3 | hints |",
            "|---|---|---|---:|---:|---:|---|",
        ]
    )
    for row in worst:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["category"]),
                    Path(str(row["image_path"])).name,
                    str(row["defect_type"]),
                    f"{row['r0_f1']:.3f}→{row['p3_f1']:.3f}",
                    f"{row['r0_recall']:.3f}→{row['p3_recall']:.3f}",
                    f"{row['r0_overseg']:.3f}→{row['p3_overseg']:.3f}",
                    str(row["diagnosis_hint"]),
                ]
            )
            + " |"
        )

    if normal_rows:
        lines.extend(["", "## Normal-image false positive summary", ""])
        for prefix, name in (("r0", "R0"), ("r2", "R2"), ("p3", "P3")):
            fp = np.nanmean([1.0 if row[f"{prefix}_is_positive"] else 0.0 for row in normal_rows])
            area = np.nanmean([_safe_float(row[f"{prefix}_positive_area_fraction"]) for row in normal_rows])
            lines.append(f"- {name}: image FP={fp:.6f}, positive area={area:.6f}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _visualize_cases(
    rows: list[dict[str, Any]],
    scored_by_category: dict[str, dict[str, Any]],
    outputs_by_category: dict[str, dict[str, Any]],
    output_dir: Path,
    input_size: int,
    max_per_category: int,
) -> None:
    selected: list[dict[str, Any]] = []
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_category[str(row["category"])].append(row)
    for category, category_rows in by_category.items():
        ranked = sorted(
            category_rows,
            key=lambda row: (
                _safe_float(row["p3_f1_delta_vs_r0"]),
                _safe_float(row["p3_recall_delta_vs_r0"]),
                -_safe_float(row["p3_overseg_delta_vs_r0"]),
            ),
        )
        selected.extend(ranked[: int(max_per_category)])

    for row in selected:
        category = str(row["category"])
        scored = scored_by_category[category]
        outputs = outputs_by_category[category]
        index = int(row["case_index"])
        image = _load_resized_image(str(row["image_path"]), input_size)
        mask = scored["masks"][index].astype(bool)
        r0_map = outputs["R0_nn_distance"]["maps"][index]
        r2_map = outputs["R2_nvs_residual"]["maps"][index]
        r0_pred = outputs["R0_nn_distance"]["predictions"][index]
        r2_pred = outputs["R2_nvs_residual"]["predictions"][index]
        p3_pred = outputs["P_topk3_r2"]["predictions"][index]
        stem = f"{category}_{Path(str(row['image_path'])).stem}_{str(row['defect_type']).replace('/', '-')}.png"
        try:
            _plot_case(row, image, mask, r0_map, r2_map, r0_pred, r2_pred, p3_pred, output_dir / "visualizations" / category / stem)
        except RuntimeError as exc:
            print(f"[warn] skip visualization: {exc}")
            return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Open-ended diagnosis for NVS Validation10 failures")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--categories", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--seed", type=int, help="Override experiment.seed from config")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-visualizations-per-category", type=int, default=8)
    parser.add_argument("--no-visualizations", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config["data"]["categories"] = list(args.categories)
    if args.seed is not None:
        config["experiment"]["seed"] = int(args.seed)
    seed = int(config["experiment"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(config, output_dir / "resolved_config.json")

    device = torch.device(args.device)
    model = load_dinov2(config["model"]["name"], device, config["model"].get("hub_dir"))
    case_rows: list[dict[str, Any]] = []
    normal_rows: list[dict[str, Any]] = []
    scored_by_category: dict[str, dict[str, Any]] = {}
    outputs_by_category: dict[str, dict[str, Any]] = {}

    required_methods = list(dict.fromkeys(args.methods))
    if "R0_nn_distance" not in required_methods:
        required_methods.insert(0, "R0_nn_distance")
    if "R2_nvs_residual" not in required_methods:
        required_methods.insert(1, "R2_nvs_residual")
    if "P_topk3_r2" not in required_methods:
        required_methods.append("P_topk3_r2")

    for category in config["data"]["categories"]:
        print(f"[diagnose] {category}")
        train_records, test_records = build_mvtec_records(args.data_root, category)
        state = _fit_state(category, train_records, config, model, device)
        calibrations = _calibrate_methods(state, config, model, device)
        scored = _score_records(test_records, state, config, model, device, include_mask=True)
        outputs = _method_outputs(scored["maps"], calibrations, config, required_methods)
        scored_by_category[category] = scored
        outputs_by_category[category] = outputs
        labels = scored["labels"]
        for index in range(len(labels)):
            if int(labels[index]) == 1:
                row = _case_row(category, index, scored, outputs)
                row["case_index"] = int(index)
                case_rows.append(row)
            else:
                normal_rows.append(_normal_row(category, index, scored, outputs))
        write_csv(case_rows, output_dir / "failure_cases.csv")
        write_csv(normal_rows, output_dir / "normal_cases.csv")

    _write_summary(case_rows, normal_rows, output_dir / "failure_diagnosis_summary.md")
    if not args.no_visualizations:
        _visualize_cases(
            case_rows,
            scored_by_category,
            outputs_by_category,
            output_dir,
            input_size=int(config["data"]["input_size"]),
            max_per_category=int(args.max_visualizations_per_category),
        )
    save_json(
        {
            "status": "complete",
            "categories": list(config["data"]["categories"]),
            "completed_count": len(config["data"]["categories"]),
            "case_count": len(case_rows),
            "normal_count": len(normal_rows),
            "methods": required_methods,
        },
        output_dir / "experiment_complete.json",
    )


if __name__ == "__main__":
    main()
