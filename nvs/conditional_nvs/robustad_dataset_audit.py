from __future__ import annotations

import argparse
import json
from pathlib import Path

from .robustad_category_protocol import (
    evaluation_groups,
    mask_scope,
    parse_category_manifest,
    parse_official_directory,
    source_training,
    validate_records,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit local RobustAD source/target protocol without fitting"
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--categories", nargs="+")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.manifest:
        records = parse_category_manifest(
            args.manifest, data_root=args.data_root, require_files=True
        )
    else:
        records = parse_official_directory(
            args.data_root, categories=args.categories, require_files=True
        )
    categories = validate_records(records, args.categories)
    rows = []
    for category in categories:
        training = source_training(records, category)
        print(f"{category}: source_normal_train={len(training)}", flush=True)
        for (_, domain, shift), group in sorted(
            evaluation_groups(records, category).items()
        ):
            scope = mask_scope(group)
            normal = sum(int(record.label) == 0 for record in group)
            anomaly = sum(int(record.label) == 1 for record in group)
            row = {
                "category": category,
                "domain": domain,
                "shift": shift,
                "metric_scope": scope,
                "images": len(group),
                "normal": normal,
                "anomaly": anomaly,
            }
            rows.append(row)
            print(
                f"  {domain:6s} {shift:24s} scope={scope:5s} "
                f"normal={normal} anomaly={anomaly}",
                flush=True,
            )
    payload = {
        "status": "complete",
        "layout": "explicit_manifest" if args.manifest else "official_imagefolder",
        "categories": categories,
        "rows": rows,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(
        f"audit_status=complete categories={len(categories)} "
        f"evaluation_domains={len(rows)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
