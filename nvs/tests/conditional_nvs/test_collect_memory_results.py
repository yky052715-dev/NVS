from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


FIELDS = (
    "method",
    "memory_protocol",
    "seed",
    "category",
    "pixel_AUROC",
    "localization_test_normal_image_positive_rate",
    "memory_entries",
    "inference_ms_per_image",
)


def _write_result(
    root: Path, protocol: str, seed: int, category: str, auroc: float
) -> None:
    output = root / protocol / category / f"seed{seed}"
    output.mkdir(parents=True)
    row = {
        "method": "D0_NN",
        "memory_protocol": protocol,
        "seed": seed,
        "category": category,
        "pixel_AUROC": auroc,
        "localization_test_normal_image_positive_rate": 0.1,
        "memory_entries": 30_000,
        "inference_ms_per_image": 1.0,
    }
    with (output / "metrics_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerow(row)
    (output / "experiment_complete.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "category": category,
                "seed": seed,
            }
        ),
        encoding="utf-8",
    )


def test_collector_preserves_multiple_seeds_in_one_protocol_root(tmp_path) -> None:
    root = tmp_path / "confirmation"
    for seed in (43, 44):
        for category in ("bottle", "grid"):
            _write_result(root, "M_R30", seed, category, 0.98 + seed / 100_000)

    output = tmp_path / "paired.csv"
    script = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "conditional_nvs"
        / "collect_memory_results.py"
    )
    subprocess.run(
        [
            sys.executable,
            str(script),
            str(root),
            "--output",
            str(output),
            "--expected-categories",
            "2",
            "--require-protocols",
            "M_R30",
            "--require-seeds",
            "43",
            "44",
        ],
        check=True,
    )

    with output.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [(row["memory_protocol"], int(row["seed"])) for row in rows] == [
        ("M_R30", 43),
        ("M_R30", 44),
    ]
