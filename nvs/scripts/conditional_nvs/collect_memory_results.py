from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect D0 memory-ablation outputs into lock-protocol rows"
    )
    parser.add_argument("roots", nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-categories", type=int)
    parser.add_argument("--require-protocols", nargs="*", default=[])
    parser.add_argument("--require-seeds", nargs="*", type=int, default=[])
    args = parser.parse_args()
    grouped: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    seen = set()
    for root in args.roots:
        for path in Path(root).rglob("category_metrics.csv"):
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    if row.get("method") != "D0_NN":
                        continue
                    identity = (
                        row["memory_protocol"],
                        int(row["seed"]),
                        row["category"],
                    )
                    if identity in seen:
                        continue
                    seen.add(identity)
                    grouped[(identity[0], identity[1])].append(row)
    if args.expected_categories is not None:
        if int(args.expected_categories) <= 0:
            raise ValueError("--expected-categories must be positive")
        required_protocols = [str(value) for value in args.require_protocols]
        required_seeds = [int(value) for value in args.require_seeds]
        if bool(required_protocols) != bool(required_seeds):
            raise ValueError(
                "--require-protocols and --require-seeds must be provided together"
            )
        required_pairs = (
            [(protocol, seed) for protocol in required_protocols for seed in required_seeds]
            if required_protocols
            else sorted(grouped)
        )
        for protocol, seed in required_pairs:
            rows = grouped.get((protocol, seed), [])
            categories = {str(row["category"]) for row in rows}
            if len(categories) != int(args.expected_categories):
                raise RuntimeError(
                    f"Incomplete memory result {protocol}/seed{seed}: "
                    f"{len(categories)}/{int(args.expected_categories)} categories"
                )
        if required_protocols:
            allowed_protocols = set(required_protocols)
            allowed_seeds = set(required_seeds)
            grouped = defaultdict(
                list,
                {
                    key: rows
                    for key, rows in grouped.items()
                    if key[0] in allowed_protocols and key[1] in allowed_seeds
                },
            )

    output_rows = []
    for (protocol, seed), rows in sorted(grouped.items()):
        output_rows.append(
            {
                "memory_protocol": protocol,
                "seed": seed,
                "categories": len(rows),
                "pixel_AUROC": float(
                    np.nanmean([float(row["pixel_AUROC"]) for row in rows])
                ),
                "normal_FP": float(
                    np.nanmean(
                        [
                            float(
                                row[
                                    "localization_test_normal_image_positive_rate"
                                ]
                            )
                            for row in rows
                        ]
                    )
                ),
                "capacity": int(round(np.mean([float(row["memory_entries"]) for row in rows]))),
                "inference_ms": float(
                    np.nanmean(
                        [float(row["inference_ms_per_image"]) for row in rows]
                    )
                ),
            }
        )
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]) if output_rows else [])
        if output_rows:
            writer.writeheader()
            writer.writerows(output_rows)


if __name__ == "__main__":
    main()
