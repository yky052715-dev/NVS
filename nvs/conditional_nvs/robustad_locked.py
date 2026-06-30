"""Stable CLI entry point for the locked per-category RobustAD protocol."""

from __future__ import annotations

import json

from .robustad_category_protocol import (
    parse_args,
    parse_category_manifest,
    run_locked_protocol,
)


def main() -> None:
    args = parse_args()
    if args.manifest and not args.categories:
        records = parse_category_manifest(
            args.manifest, data_root=args.data_root, require_files=True
        )
        args.categories = sorted({record.category for record in records})
    result = run_locked_protocol(args)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
