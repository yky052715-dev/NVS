from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence

import torch

from . import cli
from .datasets import ImageRecord
from .metrics import average_relative_drop
from .protocol import (
    completion_is_valid,
    config_fingerprint,
    protocol_metadata,
    split_three_way,
    stable_hash,
    write_completion,
)


@dataclass(frozen=True)
class RobustADCategoryRecord:
    path: Path
    label: int
    category: str
    domain: str
    role: str
    shift: str
    mask_path: Path | None = None

    def as_image_record(self) -> ImageRecord:
        return ImageRecord(
            path=self.path,
            label=int(self.label),
            defect_type=self.shift,
            mask_path=self.mask_path,
        )


OFFICIAL_LAYOUT: dict[str, dict[str, Any]] = {
    "MetalParts": {
        "prefix": "metal_parts_data_dir",
        "targets": {
            1: "lighting",
            2: "position",
            3: "rotation",
            4: "scale",
            5: "background_1",
            6: "background_2",
        },
    },
    "PCB": {
        "prefix": "pcb_data_dir",
        "targets": {
            1: "lighting",
            2: "white_balancing",
            3: "rotation",
            4: "position",
            5: "shadow",
        },
    },
    "PiledBags": {
        "prefix": "piled_bags_data_dir",
        "targets": {
            1: "lighting",
            2: "background_box_color",
            3: "position_rotation",
            4: "scale",
            5: "shadow",
        },
    },
}


def _resolve(root: Path, value: Any) -> Path | None:
    if value in (None, "", "null"):
        return None
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def _load_manifest_rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))
    if suffix == ".jsonl":
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return list(payload["records"] if isinstance(payload, dict) else payload)
    raise ValueError("RobustAD manifest must be .json, .jsonl, or .csv")


def parse_category_manifest(
    manifest_path: str | Path,
    data_root: str | Path | None = None,
    require_files: bool = True,
) -> list[RobustADCategoryRecord]:
    manifest = Path(manifest_path)
    root = Path(data_root) if data_root is not None else manifest.parent
    records = []
    for row in _load_manifest_rows(manifest):
        category = str(
            row.get("category", row.get("product", row.get("object_class", "")))
        ).strip()
        domain = str(row.get("domain", "")).strip().lower()
        role = str(row.get("role", row.get("split", ""))).strip().lower()
        shift = str(row.get("shift", row.get("subset", ""))).strip()
        path = _resolve(root, row.get("path"))
        mask_path = _resolve(root, row.get("mask_path"))
        if not category:
            raise ValueError("Every RobustAD record requires category/product")
        if domain not in {"source", "target"}:
            raise ValueError(f"Invalid RobustAD domain: {domain!r}")
        if role not in {"train", "calibration", "test", "evaluation"}:
            raise ValueError(f"Invalid RobustAD role: {role!r}")
        if not shift:
            raise ValueError("Every RobustAD record requires shift/subset")
        if path is None:
            raise ValueError("Every RobustAD record requires path")
        if require_files and not path.is_file():
            raise FileNotFoundError(path)
        if require_files and mask_path is not None and not mask_path.is_file():
            raise FileNotFoundError(mask_path)
        records.append(
            RobustADCategoryRecord(
                path=path,
                label=int(row.get("label", 0)),
                category=category,
                domain=domain,
                role=role,
                shift=shift,
                mask_path=mask_path,
            )
        )
    validate_records(records)
    return records


def _resolve_official_mask(
    root: Path, split_dir: Path, value: Any
) -> Path | None:
    if value in (None, "", "null"):
        return None
    path = Path(str(value))
    candidates = (
        path if path.is_absolute() else root / path,
        split_dir / path,
        split_dir / "masks" / path.name,
    )
    return next((candidate for candidate in candidates if candidate.is_file()), candidates[0])


def _parse_official_split(
    root: Path,
    category: str,
    split_dir: Path,
    domain: str,
    role: str,
    shift: str,
    require_files: bool,
) -> list[RobustADCategoryRecord]:
    metadata = split_dir / "metadata.jsonl"
    if not metadata.is_file():
        raise FileNotFoundError(metadata)
    records = []
    for line in metadata.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not row.get("file_name"):
            raise ValueError(f"{metadata} contains a row without file_name")
        path = split_dir / str(row["file_name"])
        mask_path = _resolve_official_mask(root, split_dir, row.get("mask"))
        if require_files and not path.is_file():
            raise FileNotFoundError(path)
        if require_files and mask_path is not None and not mask_path.is_file():
            raise FileNotFoundError(mask_path)
        records.append(
            RobustADCategoryRecord(
                path=path,
                label=int(row.get("label", 0)),
                category=category,
                domain=domain,
                role=role,
                shift=shift,
                mask_path=mask_path,
            )
        )
    return records


def parse_official_directory(
    data_root: str | Path,
    categories: Sequence[str] | None = None,
    require_files: bool = True,
) -> list[RobustADCategoryRecord]:
    """Parse the official AmazonScience/RobustAD imagefolder release.

    ``train`` supplies source-domain normal training images. ``test0`` is the
    held source-domain evaluation set. Later ``testN`` directories are target
    shifts. Source anomalies shipped in ``train`` are intentionally ignored in
    this unsupervised protocol.
    """

    root = Path(data_root)
    selected = (
        list(OFFICIAL_LAYOUT)
        if categories is None
        else [str(category) for category in categories]
    )
    unknown = sorted(set(selected) - set(OFFICIAL_LAYOUT))
    if unknown:
        raise ValueError(f"Unknown official RobustAD categories: {unknown}")
    records: list[RobustADCategoryRecord] = []
    for category in selected:
        layout = OFFICIAL_LAYOUT[category]
        prefix = str(layout["prefix"])
        category_root = root / category
        train = _parse_official_split(
            root,
            category,
            category_root / f"{prefix}_train",
            "source",
            "train",
            "source_train",
            require_files,
        )
        records.extend(record for record in train if int(record.label) == 0)
        records.extend(
            _parse_official_split(
                root,
                category,
                category_root / f"{prefix}_test0",
                "source",
                "evaluation",
                "source",
                require_files,
            )
        )
        for index, shift in sorted(dict(layout["targets"]).items()):
            records.extend(
                _parse_official_split(
                    root,
                    category,
                    category_root / f"{prefix}_test{index}",
                    "target",
                    "evaluation",
                    str(shift),
                    require_files,
                )
            )
    validate_records(records, selected)
    return records


def source_training(
    records: Sequence[RobustADCategoryRecord], category: str
) -> list[RobustADCategoryRecord]:
    selected = [
        record
        for record in records
        if record.category == str(category)
        and record.domain == "source"
        and record.role == "train"
        and int(record.label) == 0
    ]
    assert_source_only(selected)
    return selected


def assert_source_only(records: Iterable[RobustADCategoryRecord]) -> None:
    violations = [
        str(record.path)
        for record in records
        if record.domain != "source"
        or record.role not in {"train", "calibration"}
        or int(record.label) != 0
    ]
    if violations:
        raise AssertionError(
            "Only source-domain normal images may enter fit/calibration: "
            + ", ".join(violations[:3])
        )


def evaluation_groups(
    records: Sequence[RobustADCategoryRecord],
    category: str | None = None,
) -> dict[tuple[str, str, str], list[RobustADCategoryRecord]]:
    groups: dict[tuple[str, str, str], list[RobustADCategoryRecord]] = {}
    for record in records:
        if record.role not in {"test", "evaluation"}:
            continue
        if category is not None and record.category != str(category):
            continue
        key = (record.category, record.domain, record.shift)
        groups.setdefault(key, []).append(record)
    return groups


def mask_scope(records: Sequence[RobustADCategoryRecord]) -> str:
    anomalies = [record for record in records if int(record.label) == 1]
    if not anomalies:
        raise ValueError("Evaluation group requires anomalous samples")
    present = [record.mask_path is not None for record in anomalies]
    if any(present) and not all(present):
        raise ValueError("Evaluation group has partial anomaly masks")
    return "pixel" if all(present) else "image"


def validate_records(
    records: Sequence[RobustADCategoryRecord],
    categories: Sequence[str] | None = None,
) -> list[str]:
    if not records:
        raise ValueError("RobustAD records are empty")
    selected = (
        sorted({record.category for record in records})
        if categories is None
        else [str(category) for category in categories]
    )
    missing = sorted(set(selected) - {record.category for record in records})
    if missing:
        raise ValueError(f"RobustAD categories missing from records: {missing}")
    seen: set[tuple[str, str, str, str, str]] = set()
    for record in records:
        key = (
            record.category,
            record.domain,
            record.role,
            record.shift,
            str(record.path),
        )
        if key in seen:
            raise ValueError(f"Duplicate RobustAD record: {key}")
        seen.add(key)
        if int(record.label) not in {0, 1}:
            raise ValueError(f"RobustAD label must be 0/1: {record.label}")
        if record.domain == "target" and record.role in {"train", "calibration"}:
            raise ValueError("Target-domain records cannot be train/calibration")
        if record.role == "calibration" and (
            record.domain != "source" or int(record.label) != 0
        ):
            raise ValueError("Calibration records must be source-domain normal")
    groups = evaluation_groups(records)
    for category in selected:
        training = source_training(records, category)
        if len(training) < 3:
            raise ValueError(f"{category} requires at least three source normals")
        category_groups = {
            key: group for key, group in groups.items() if key[0] == category
        }
        source_keys = [key for key in category_groups if key[1] == "source"]
        target_keys = [key for key in category_groups if key[1] == "target"]
        if len(source_keys) != 1:
            raise ValueError(
                f"{category} requires exactly one source evaluation domain"
            )
        if not target_keys:
            raise ValueError(f"{category} requires at least one target domain")
        scopes = set()
        for key, group in category_groups.items():
            if {int(record.label) for record in group} != {0, 1}:
                raise ValueError(f"{key} evaluation requires labels 0 and 1")
            scopes.add(mask_scope(group))
        if len(scopes) != 1:
            raise ValueError(f"{category} mixes image-only and pixel domains")
    return selected


def category_protocol_payload(
    records: Sequence[RobustADCategoryRecord],
    category: str,
    seed: int,
) -> dict[str, Any]:
    training = source_training(records, category)
    split = split_three_way(
        training,
        split_seed=int(seed),
        nvs_split_seed=int(seed),
        calibration_fraction=0.20,
        nvs_fit_fraction_of_remainder=0.30,
        key=lambda record: str(record.path),
    )
    evaluation = [
        {
            "path": str(record.path),
            "mask_path": None if record.mask_path is None else str(record.mask_path),
            "label": int(record.label),
            "domain": record.domain,
            "shift": record.shift,
        }
        for key, group in sorted(evaluation_groups(records, category).items())
        for record in sorted(group, key=lambda item: str(item.path))
    ]
    payload = {
        "category": category,
        "seed": int(seed),
        "source_split": split.manifest(key=lambda record: str(record.path)),
        "evaluation": evaluation,
        "fit_isolation": "source_normal_only",
        "target_usage": "final_evaluation_only",
    }
    payload["manifest_hash"] = stable_hash(payload)
    return payload


def robustad_ard_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    categories = sorted({str(row["category"]) for row in rows})
    methods = sorted({str(row["method"]) for row in rows})
    for category in categories:
        for method in methods:
            selected = [
                row
                for row in rows
                if str(row["category"]) == category
                and str(row["method"]) == method
            ]
            for metric in ("image_AUROC", "pixel_AUROC"):
                source = [
                    float(row[metric])
                    for row in selected
                    if str(row["domain"]) == "source"
                    and metric in row
                    and math.isfinite(float(row[metric]))
                ]
                targets = [
                    float(row[metric])
                    for row in selected
                    if str(row["domain"]) == "target"
                    and metric in row
                    and math.isfinite(float(row[metric]))
                ]
                if not targets:
                    continue
                if len(source) != 1:
                    raise ValueError(
                        f"{category}/{method}/{metric} requires one source row"
                    )
                output.append(
                    {
                        "category": category,
                        "method": method,
                        "metric": metric,
                        "source": source[0],
                        "target_macro": mean(targets),
                        "target_count": len(targets),
                        "ARD": average_relative_drop(source[0], targets),
                    }
                )
    return output


def robustad_macro_summary(
    rows: Sequence[Mapping[str, Any]],
    ard_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    methods = sorted({str(row["method"]) for row in rows})
    summary: dict[str, Any] = {}
    for method in methods:
        summary[method] = {}
        for metric in ("image_AUROC", "pixel_AUROC"):
            selected = [
                row
                for row in ard_rows
                if str(row["method"]) == method and str(row["metric"]) == metric
            ]
            if not selected:
                continue
            summary[method][metric] = {
                "source_category_macro": mean(float(row["source"]) for row in selected),
                "target_category_macro": mean(
                    float(row["target_macro"]) for row in selected
                ),
                "ARD_category_macro": mean(float(row["ARD"]) for row in selected),
                "category_count": len(selected),
            }
    return summary


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def run_locked_protocol(args: argparse.Namespace) -> dict[str, Any]:
    config = cli._load_config(args.config)
    cli._validate_core_config(config)
    seed = int(args.seed if args.seed is not None else config["experiment"]["seed"])
    output_dir = Path(args.output_dir or config["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    config.setdefault("data", {})["root"] = str(args.data_root)
    configured_categories = args.categories or config["data"].get("categories")
    if args.manifest:
        records = parse_category_manifest(
            args.manifest, data_root=args.data_root, require_files=True
        )
    else:
        records = parse_official_directory(
            args.data_root,
            categories=configured_categories,
            require_files=True,
        )
    categories = validate_records(records, configured_categories)
    config["experiment"]["seed"] = seed
    config["data"]["categories"] = list(categories)
    config["data"]["manifest"] = None if args.manifest is None else str(args.manifest)
    config["data"]["layout"] = (
        "explicit_manifest" if args.manifest else "official_imagefolder"
    )
    cli._save_json(output_dir / "resolved_config.json", config)

    device = torch.device(args.device)
    model = cli.load_dinov2(
        config["model"]["name"], device, config["model"].get("hub_dir")
    )
    methods = cli._configured_methods(config)
    all_rows: list[dict[str, Any]] = []
    child_protocol_hashes = {}
    for category in categories:
        category_dir = output_dir / category / f"seed{seed}"
        protocol_payload = category_protocol_payload(records, category, seed)
        cli._save_json(
            category_dir / "robustad_category_protocol.json", protocol_payload
        )
        metadata = protocol_metadata(
            category,
            seed,
            {"manifest_hash": protocol_payload["manifest_hash"]},
            config,
            methods,
        )
        marker = category_dir / "experiment_complete.json"
        metrics_path = category_dir / "robustad_category_metrics.csv"
        if (
            not args.force
            and completion_is_valid(marker, metadata)
            and metrics_path.is_file()
        ):
            print(
                f"[{category}] valid completion marker; loading saved CSV",
                flush=True,
            )
            all_rows.extend(_read_csv(metrics_path))
            child_protocol_hashes[category] = metadata["protocol_hash"]
            continue

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
        category_rows = []
        for (group_category, domain, shift), group in sorted(
            evaluation_groups(records, category).items()
        ):
            scope = mask_scope(group)
            print(
                f"[{category}] evaluating domain={domain} shift={shift} "
                f"scope={scope}",
                flush=True,
            )
            evaluated = cli._evaluate_records(
                [record.as_image_record() for record in group],
                pipeline,
                config,
                model,
                device,
                pixel_metrics=scope == "pixel",
            )
            for row in evaluated:
                row.update(
                    category=group_category,
                    domain=domain,
                    shift=shift,
                    metric_scope=scope,
                    seed=seed,
                    memory_protocol=config["memory"]["protocol"],
                )
                if scope == "pixel":
                    row["normal_image_fp_rate"] = row.get(
                        "localization_test_normal_image_positive_rate"
                    )
            category_rows.extend(evaluated)
        cli._write_csv(metrics_path, category_rows)
        write_completion(marker, metadata)
        child_protocol_hashes[category] = metadata["protocol_hash"]
        all_rows.extend(category_rows)

    cli._write_csv(output_dir / "robustad_metrics.csv", all_rows)
    ard_rows = robustad_ard_rows(all_rows)
    cli._write_csv(output_dir / "robustad_ard.csv", ard_rows)
    cli._save_json(output_dir / "robustad_ard.json", ard_rows)
    summary = robustad_macro_summary(all_rows, ard_rows)
    cli._save_json(output_dir / "robustad_summary.json", summary)
    root_payload = {
        "status": "complete",
        "protocol": "category_locked_source_to_target_v1",
        "seed": seed,
        "categories": list(categories),
        "methods": list(methods),
        "config_fingerprint": config_fingerprint(config),
        "child_protocol_hashes": child_protocol_hashes,
        "input_manifest_hash": stable_hash(
            {
                category: category_protocol_payload(records, category, seed)[
                    "manifest_hash"
                ]
                for category in categories
            }
        ),
        "summary_hash": stable_hash(summary),
    }
    root_payload["protocol_hash"] = stable_hash(root_payload)
    cli._save_json(output_dir / "robustad_complete.json", root_payload)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Locked per-category source-to-target RobustAD protocol"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--categories", nargs="+")
    parser.add_argument("--output-dir")
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    result = run_locked_protocol(parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
