"""Aggregate RobustAD failure-attribution and AugMem-K10 outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable, Mapping, Sequence

from .protocol import stable_hash
from .robustad_failure_diagnostics import add_per_shift_ard, shift_family


METRICS = ("image_AUROC", "image_AUPR", "pixel_AUROC", "pixel_AUPR")


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


def _mean_std(values: Iterable[Any]) -> tuple[float, float, int]:
    finite = []
    for value in values:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(parsed):
            finite.append(parsed)
    return (
        mean(finite) if finite else float("nan"),
        stdev(finite) if len(finite) > 1 else 0.0 if finite else float("nan"),
        len(finite),
    )


def aggregate_rows(
    rows: Sequence[Mapping[str, Any]], keys: Sequence[str]
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(str(row[key]) for key in keys)].append(row)
    output = []
    for values, selected in sorted(groups.items()):
        result: dict[str, Any] = dict(zip(keys, values))
        result["paired_rows"] = len(selected)
        for metric in METRICS:
            metric_mean, metric_std, count = _mean_std(
                row.get(metric) for row in selected
            )
            ard_mean, ard_std, ard_count = _mean_std(
                row.get(f"ARD_{metric}") for row in selected
            )
            result[f"{metric}_mean"] = metric_mean
            result[f"{metric}_std"] = metric_std
            result[f"{metric}_count"] = count
            result[f"ARD_{metric}_mean"] = ard_mean
            result[f"ARD_{metric}_std"] = ard_std
            result[f"ARD_{metric}_count"] = ard_count
        output.append(result)
    return output


def aggregate_fields(
    rows: Sequence[Mapping[str, Any]],
    keys: Sequence[str],
    fields: Sequence[str],
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(str(row[key]) for key in keys)].append(row)
    output = []
    for values, selected in sorted(groups.items()):
        result: dict[str, Any] = dict(zip(keys, values))
        result["seeds"] = len({int(row["seed"]) for row in selected})
        for field in fields:
            avg, std, count = _mean_std(row.get(field) for row in selected)
            result[f"{field}_mean"] = avg
            result[f"{field}_std"] = std
            result[f"{field}_count"] = count
        output.append(result)
    return output

def paired_d2_minus_augmem(
    rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    index = {
        (
            int(row["seed"]),
            str(row["category"]),
            str(row["domain"]),
            str(row["shift"]),
            str(row["method"]),
        ): row
        for row in rows
    }
    groups: dict[tuple[str, str, str, str], list[dict[str, float]]] = defaultdict(list)
    identities = sorted(
        {
            (
                int(row["seed"]),
                str(row["category"]),
                str(row["domain"]),
                str(row["shift"]),
            )
            for row in rows
        }
    )
    for seed, category, domain, shift in identities:
        d2 = index.get((seed, category, domain, shift, "D2_NVSGlobal"))
        aug = index.get((seed, category, domain, shift, "AugMem_K10"))
        if d2 is None or aug is None:
            continue
        delta = {}
        for metric in METRICS:
            d2_value = _mean_std([d2.get(metric)])[0]
            aug_value = _mean_std([aug.get(metric)])[0]
            delta[metric] = (
                d2_value - aug_value
                if math.isfinite(d2_value) and math.isfinite(aug_value)
                else float("nan")
            )
        groups[(category, domain, shift, shift_family(shift))].append(delta)

    output = []
    for (category, domain, shift, family), deltas in sorted(groups.items()):
        row: dict[str, Any] = {
            "category": category,
            "domain": domain,
            "shift": shift,
            "shift_family": family,
            "seeds": len(deltas),
        }
        for metric in METRICS:
            avg, std, count = _mean_std(delta[metric] for delta in deltas)
            row[f"D2_minus_AugMem_{metric}_mean"] = avg
            row[f"D2_minus_AugMem_{metric}_std"] = std
            row[f"D2_minus_AugMem_{metric}_count"] = count
        output.append(row)
    return output


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_base = Path(args.output_base)
    rows: list[dict[str, Any]] = []
    normal_rows: list[dict[str, Any]] = []
    region_delta_rows: list[dict[str, Any]] = []
    eta_rows: list[dict[str, Any]] = []
    piled_rows: list[dict[str, Any]] = []
    input_hashes: dict[str, str] = {}
    for seed in args.seeds:
        diagnostic_dir = output_base / f"robustad_failure_diagnostics_seed{seed}"
        diagnostic_marker = _load_json(
            diagnostic_dir / "diagnostic_complete.json"
        )
        if diagnostic_marker.get("status") != "complete":
            raise ValueError(f"Diagnostic seed {seed} is incomplete")
        diagnostic_rows = _read_csv(
            diagnostic_dir / "metrics_by_category_shift.csv"
        )
        for row in diagnostic_rows:
            row["seed"] = int(seed)
        rows.extend(diagnostic_rows)
        for filename, target in (
            ("target_normal_score_stats.csv", normal_rows),
            ("pixel_region_d2_minus_d0.csv", region_delta_rows),
            ("projection_energy_comparison.csv", eta_rows),
            ("piledbags_pooling_diagnostics.csv", piled_rows),
        ):
            for row in _read_csv(diagnostic_dir / filename):
                row["seed"] = int(seed)
                target.append(row)
        input_hashes[f"diagnostic_seed{seed}"] = str(
            diagnostic_marker["output_hash"]
        )

        augmem_dir = output_base / f"robustad_augmem_k10_seed{seed}"
        augmem_marker = _load_json(augmem_dir / "robustad_complete.json")
        if augmem_marker.get("status") != "complete":
            raise ValueError(f"AugMem seed {seed} is incomplete")
        augmem_rows: list[dict[str, Any]] = []
        for row in _read_csv(augmem_dir / "robustad_metrics.csv"):
            normalized: dict[str, Any] = dict(row)
            normalized["seed"] = int(seed)
            normalized["shift_family"] = shift_family(str(row["shift"]))
            augmem_rows.append(normalized)
        add_per_shift_ard(augmem_rows)
        rows.extend(augmem_rows)
        input_hashes[f"augmem_seed{seed}"] = str(augmem_marker["summary_hash"])

    by_shift = aggregate_rows(
        rows, ("category", "domain", "shift", "shift_family", "method")
    )
    by_family = aggregate_rows(
        [row for row in rows if str(row["domain"]) == "target"],
        ("shift_family", "method"),
    )
    paired = paired_d2_minus_augmem(rows)
    normal_summary = aggregate_fields(
        normal_rows,
        ("category", "shift", "shift_family", "method"),
        ("mean", "p95", "p99", "false_positive_rate"),
    )
    region_delta_summary = aggregate_fields(
        region_delta_rows,
        ("category", "domain", "shift", "shift_family", "region"),
        ("D2_minus_D0_mean", "D2_minus_D0_p95", "D2_minus_D0_p99"),
    )
    eta_summary = aggregate_fields(
        eta_rows,
        ("category", "domain", "shift", "shift_family"),
        ("eta_normal_mean", "eta_defect_mean", "eta_defect_minus_normal"),
    )
    piled_summary = aggregate_fields(
        piled_rows,
        ("domain", "shift", "shift_family", "method", "aggregation"),
        ("image_AUROC", "image_AUPR", "normal_mean", "anomaly_mean", "anomaly_minus_normal"),
    )
    output_dir = Path(args.output_dir)
    _write_csv(output_dir / "three_seed_by_category_shift.csv", by_shift)
    _write_csv(output_dir / "three_seed_by_shift_family.csv", by_family)
    _write_csv(output_dir / "three_seed_d2_minus_augmem.csv", paired)
    _write_csv(output_dir / "three_seed_target_normal_scores.csv", normal_summary)
    _write_csv(output_dir / "three_seed_pixel_region_deltas.csv", region_delta_summary)
    _write_csv(output_dir / "three_seed_projection_energy.csv", eta_summary)
    _write_csv(output_dir / "three_seed_piledbags_pooling.csv", piled_summary)
    summary = {
        "status": "complete",
        "protocol": "robustad_failure_attribution_summary_v1",
        "seeds": [int(seed) for seed in args.seeds],
        "methods": sorted({str(row["method"]) for row in rows}),
        "input_hashes": input_hashes,
        "row_counts": {
            "raw": len(rows),
            "by_shift": len(by_shift),
            "by_family": len(by_family),
            "paired_d2_minus_augmem": len(paired),
            "target_normal_scores": len(normal_summary),
            "pixel_region_deltas": len(region_delta_summary),
            "projection_energy": len(eta_summary),
            "piledbags_pooling": len(piled_summary),
        },
    }
    summary["summary_hash"] = stable_hash(
        {"by_shift": by_shift, "by_family": by_family, "paired": paired,
         "normal": normal_summary, "regions": region_delta_summary,
         "eta": eta_summary, "piledbags": piled_summary}
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary_complete.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate D0/D2 attribution and AugMem-K10 RobustAD outputs"
    )
    parser.add_argument("--output-base", default="outputs/conditional_nvs")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument(
        "--output-dir",
        default="outputs/conditional_nvs/robustad_failure_attribution_summary",
    )
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
