from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    label: int
    defect_type: str
    mask_path: Path | None = None


@dataclass(frozen=True)
class RobustADRecord:
    path: Path
    label: int
    domain: str
    role: str
    subset: str
    mask_path: Path | None = None

    @property
    def has_mask(self) -> bool:
        return self.mask_path is not None

    def as_image_record(self) -> ImageRecord:
        return ImageRecord(
            path=self.path,
            label=int(self.label),
            defect_type=self.subset,
            mask_path=self.mask_path,
        )


def _list_images(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMG_EXTS
    )


def build_mvtec_dataset(
    root: str | Path, category: str
) -> tuple[list[ImageRecord], list[ImageRecord]]:
    category_root = Path(root) / category
    train_root = category_root / "train" / "good"
    test_root = category_root / "test"
    ground_truth = category_root / "ground_truth"
    if not train_root.is_dir() or not test_root.is_dir():
        raise FileNotFoundError(f"Invalid MVTec category directory: {category_root}")
    train = [ImageRecord(path, 0, "good") for path in _list_images(train_root)]
    test: list[ImageRecord] = []
    for defect_dir in sorted(path for path in test_root.iterdir() if path.is_dir()):
        for path in _list_images(defect_dir):
            if defect_dir.name == "good":
                test.append(ImageRecord(path, 0, "good"))
            else:
                mask = ground_truth / defect_dir.name / f"{path.stem}_mask.png"
                if not mask.is_file():
                    raise FileNotFoundError(mask)
                test.append(ImageRecord(path, 1, defect_dir.name, mask))
    return train, test


def _resolve(root: Path, value: Any) -> Path | None:
    if value in (None, "", "null"):
        return None
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def parse_robustad_manifest(
    manifest_path: str | Path,
    data_root: str | Path | None = None,
    require_files: bool = True,
) -> list[RobustADRecord]:
    manifest = Path(manifest_path)
    root = Path(data_root) if data_root is not None else manifest.parent
    suffix = manifest.suffix.lower()
    if suffix == ".csv":
        with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    elif suffix == ".jsonl":
        rows = [
            json.loads(line)
            for line in manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    elif suffix == ".json":
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        rows = payload["records"] if isinstance(payload, dict) else payload
    else:
        raise ValueError("RobustAD manifest must be .json, .jsonl, or .csv")
    output = []
    for row in rows:
        domain = str(row.get("domain", "")).lower()
        role = str(row.get("role", row.get("split", ""))).lower()
        if domain not in {"source", "target"}:
            raise ValueError(f"Invalid RobustAD domain: {domain!r}")
        if role not in {"train", "calibration", "test", "evaluation"}:
            raise ValueError(f"Invalid RobustAD role: {role!r}")
        path = _resolve(root, row.get("path"))
        mask_path = _resolve(root, row.get("mask_path"))
        if path is None:
            raise ValueError("Every RobustAD record requires path")
        if require_files and not path.is_file():
            raise FileNotFoundError(path)
        if require_files and mask_path is not None and not mask_path.is_file():
            raise FileNotFoundError(mask_path)
        output.append(
            RobustADRecord(
                path=path,
                label=int(row.get("label", 0)),
                domain=domain,
                role=role,
                subset=str(row.get("subset", row.get("shift", "default"))),
                mask_path=mask_path,
            )
        )
    return output


def source_normal_training(records: Sequence[RobustADRecord]) -> list[RobustADRecord]:
    selected = [
        record
        for record in records
        if record.domain == "source"
        and record.role == "train"
        and int(record.label) == 0
    ]
    if not selected:
        raise ValueError("No source-domain normal training images found")
    assert_fit_records_source_only(selected)
    return selected


def target_evaluation(records: Sequence[RobustADRecord]) -> list[RobustADRecord]:
    return [
        record
        for record in records
        if record.domain == "target" and record.role in {"test", "evaluation"}
    ]


def assert_fit_records_source_only(records: Iterable[RobustADRecord]) -> None:
    violations = [
        str(record.path)
        for record in records
        if record.domain != "source" or int(record.label) != 0
    ]
    if violations:
        raise AssertionError(
            "Only source-domain normal images may enter fit/calibration: "
            + ", ".join(violations[:3])
        )


def group_robustad_evaluation(
    records: Sequence[RobustADRecord],
) -> dict[str, list[RobustADRecord]]:
    groups: dict[str, list[RobustADRecord]] = {}
    for record in records:
        if record.role in {"test", "evaluation"}:
            groups.setdefault(f"{record.domain}:{record.subset}", []).append(record)
    return groups


def mask_capability(records: Sequence[RobustADRecord]) -> str:
    anomalous = [record for record in records if int(record.label) == 1]
    return (
        "pixel"
        if anomalous and all(record.mask_path is not None for record in anomalous)
        else "image"
    )
