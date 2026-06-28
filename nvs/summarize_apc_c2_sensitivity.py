from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from nvs.apc_c2_sensitivity import _sensitivity_rows
from nvs.common import write_csv


TEXT_KEYS = {"category", "shift_group", "condition", "method"}


def _read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    output: list[dict[str, Any]] = []
    for row in rows:
        converted: dict[str, Any] = {}
        for key, value in row.items():
            if key in TEXT_KEYS:
                converted[key] = value
            elif value in {"", None}:
                converted[key] = None
            else:
                converted[key] = float(value)
        output.append(converted)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate correctly conditioned C2 quantile ranges"
    )
    parser.add_argument("--root", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    rows = _read_rows(root / "category_condition_metrics.csv")
    summary = _sensitivity_rows(rows)
    write_csv(summary, root / "quantile_sensitivity_ranges.csv")
    print(
        f"wrote {len(summary)} condition-specific rows to "
        f"{root / 'quantile_sensitivity_ranges.csv'}"
    )


if __name__ == "__main__":
    main()
