from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from nvs.common import write_csv


METRICS = [
    "pixel_AUROC",
    "pixel_F1_calibrated",
    "pixel_F1_oracle",
    "localization_overseg_anomaly_macro",
    "localization_recall_anomaly_macro",
    "localization_small_defect_recall_macro",
    "localization_test_normal_image_positive_rate",
    "inference_ms_per_image",
]

POSITIVE_METRICS = {
    "pixel_AUROC",
    "pixel_F1_calibrated",
    "pixel_F1_oracle",
    "localization_recall_anomaly_macro",
    "localization_small_defect_recall_macro",
}
NEGATIVE_METRICS = {
    "localization_overseg_anomaly_macro",
    "localization_test_normal_image_positive_rate",
    "inference_ms_per_image",
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _categories(seed_root: Path) -> list[str]:
    complete = _load_json(seed_root / "experiment_complete.json")
    return list(complete["categories"])


def _mean_for_method(seed_root: Path, method: str, categories: list[str]) -> dict[str, float]:
    rows = [_load_json(seed_root / category / f"{method}.json") for category in categories]
    output = {"method": method, "categories": len(rows)}
    for metric in METRICS:
        output[metric] = float(np.nanmean([float(row[metric]) for row in rows]))
    return output


def _delta(candidate: dict[str, float], baseline: dict[str, float]) -> dict[str, float]:
    return {metric: float(candidate[metric] - baseline[metric]) for metric in METRICS}


def _seed_pass(delta: dict[str, float]) -> bool:
    return (
        delta["pixel_F1_calibrated"] > 0.0
        and delta["pixel_AUROC"] >= 0.0
        and delta["localization_overseg_anomaly_macro"] <= 0.0
        and delta["localization_test_normal_image_positive_rate"] <= 0.02
        and delta["localization_recall_anomaly_macro"] >= -0.02
    )


def _format_float(value: float) -> str:
    if not np.isfinite(value):
        return "nan"
    return f"{value:.6f}"


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        values = []
        for value in row:
            if isinstance(value, float):
                values.append(_format_float(value))
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def summarize(root: Path, seeds: list[int], baseline: str, candidate: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    seed_rows: list[dict[str, Any]] = []
    category_delta_rows: list[dict[str, Any]] = []
    for seed in seeds:
        seed_root = root / f"seed_{seed}"
        categories = _categories(seed_root)
        base_mean = _mean_for_method(seed_root, baseline, categories)
        cand_mean = _mean_for_method(seed_root, candidate, categories)
        deltas = _delta(cand_mean, base_mean)
        seed_row: dict[str, Any] = {"seed": int(seed), "pass": _seed_pass(deltas)}
        for metric in METRICS:
            seed_row[f"{metric}_baseline"] = base_mean[metric]
            seed_row[f"{metric}_candidate"] = cand_mean[metric]
            seed_row[f"{metric}_delta"] = deltas[metric]
        seed_rows.append(seed_row)

        for category in categories:
            base = _load_json(seed_root / category / f"{baseline}.json")
            cand = _load_json(seed_root / category / f"{candidate}.json")
            row: dict[str, Any] = {"seed": int(seed), "category": category}
            for metric in METRICS:
                row[f"{metric}_baseline"] = float(base[metric])
                row[f"{metric}_candidate"] = float(cand[metric])
                row[f"{metric}_delta"] = float(cand[metric] - base[metric])
            category_delta_rows.append(row)

    markdown = render_markdown(root, seeds, baseline, candidate, seed_rows, category_delta_rows)
    return seed_rows, category_delta_rows, markdown


def render_markdown(
    root: Path,
    seeds: list[int],
    baseline: str,
    candidate: str,
    seed_rows: list[dict[str, Any]],
    category_rows: list[dict[str, Any]],
) -> str:
    lines: list[str] = [
        "# NVS detection three-seed stability",
        "",
        f"root: `{root}`",
        f"baseline: `{baseline}`",
        f"candidate: `{candidate}`",
        f"seeds: `{', '.join(map(str, seeds))}`",
        "",
        "## Per-seed deltas (candidate - baseline)",
        "",
    ]
    headers = ["seed", "AUROC", "F1", "oracle F1", "OverSeg", "Recall", "small recall", "normal FP", "pass"]
    rows = []
    for row in seed_rows:
        rows.append([
            row["seed"],
            row["pixel_AUROC_delta"],
            row["pixel_F1_calibrated_delta"],
            row["pixel_F1_oracle_delta"],
            row["localization_overseg_anomaly_macro_delta"],
            row["localization_recall_anomaly_macro_delta"],
            row["localization_small_defect_recall_macro_delta"],
            row["localization_test_normal_image_positive_rate_delta"],
            row["pass"],
        ])
    lines.extend(_markdown_table(headers, rows))

    lines.extend(["", "## Aggregate deltas", ""])
    aggregate_rows = []
    for metric in METRICS:
        values = np.asarray([float(row[f"{metric}_delta"]) for row in seed_rows], dtype=np.float64)
        aggregate_rows.append([metric, float(np.nanmean(values)), float(np.nanstd(values)), float(np.nanmin(values)), float(np.nanmax(values))])
    lines.extend(_markdown_table(["metric", "mean", "std", "min", "max"], aggregate_rows))

    categories = sorted({row["category"] for row in category_rows})
    lines.extend(["", "## Per-category mean deltas", ""])
    category_summary_rows = []
    for category in categories:
        rows_for_category = [row for row in category_rows if row["category"] == category]
        f1_values = [float(row["pixel_F1_calibrated_delta"]) for row in rows_for_category]
        overseg_values = [float(row["localization_overseg_anomaly_macro_delta"]) for row in rows_for_category]
        recall_values = [float(row["localization_recall_anomaly_macro_delta"]) for row in rows_for_category]
        normal_fp_values = [float(row["localization_test_normal_image_positive_rate_delta"]) for row in rows_for_category]
        category_summary_rows.append([
            category,
            float(np.nanmean(f1_values)),
            float(np.nanmean(overseg_values)),
            float(np.nanmean(recall_values)),
            float(np.nanmean(normal_fp_values)),
            f"{sum(value > 0.0 for value in f1_values)}/{len(f1_values)}",
            f"{sum(value < 0.0 for value in overseg_values)}/{len(overseg_values)}",
        ])
    lines.extend(_markdown_table(["category", "F1", "OverSeg", "Recall", "Normal FP", "F1 improved", "OverSeg improved"], category_summary_rows))

    all_pass = all(bool(row["pass"]) for row in seed_rows)
    lines.extend([
        "",
        f"All seeds pass: {all_pass}",
        "",
        "## Pass rule",
        "",
        "A seed passes when candidate satisfies all of:",
        "",
        "```text",
        "F1 delta > 0",
        "AUROC delta >= 0",
        "OverSeg delta <= 0",
        "Normal FP delta <= +0.02",
        "Recall delta >= -0.02",
        "```",
        "",
    ])
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize NVS detection three-seed stability")
    parser.add_argument("--root", default="outputs/nvs/detection_stability_dev5")
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--baseline", default="R0_nn_distance")
    parser.add_argument("--candidate", default="R2_nvs_residual")
    parser.add_argument("--output-prefix", default="detection_stability")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    seed_rows, category_rows, markdown = summarize(root, args.seeds, args.baseline, args.candidate)
    write_csv(seed_rows, root / f"{args.output_prefix}_seed_summary.csv")
    write_csv(category_rows, root / f"{args.output_prefix}_category_delta_summary.csv")
    (root / f"{args.output_prefix}_summary.md").write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()