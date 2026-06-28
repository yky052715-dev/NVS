from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from nvs.common import ImageRecord


def _path_key(path: str | Path) -> str:
    return str(Path(path).expanduser().resolve(strict=False))


def _value_key(value: Any) -> str:
    return f"{float(value):.12g}"


@dataclass(frozen=True)
class PerturbationCacheEntry:
    category: str
    transform_name: str
    transform_value: str
    source: Path
    output: Path
    split: str
    defect_type: str


class PerturbationCache:
    """In-memory index over a cache_perturbed_mvtec manifest."""

    def __init__(self, manifest_path: str | Path) -> None:
        self.manifest_path = Path(manifest_path).expanduser().resolve(strict=False)
        if not self.manifest_path.is_file():
            raise FileNotFoundError(
                f"Perturbation cache manifest not found: {self.manifest_path}"
            )

        self._entries: dict[tuple[str, str, str, str], PerturbationCacheEntry] = {}
        with self.manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {
                "category",
                "transform_name",
                "transform_value",
                "source",
                "output",
                "split",
                "defect_type",
            }
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(
                    f"Perturbation manifest is missing columns: {sorted(missing)}"
                )
            for row_number, row in enumerate(reader, start=2):
                source = Path(row["source"]).expanduser().resolve(strict=False)
                output = Path(row["output"]).expanduser().resolve(strict=False)
                entry = PerturbationCacheEntry(
                    category=str(row["category"]),
                    transform_name=str(row["transform_name"]).lower(),
                    transform_value=_value_key(row["transform_value"]),
                    source=source,
                    output=output,
                    split=str(row["split"]),
                    defect_type=str(row["defect_type"]),
                )
                key = (
                    entry.category,
                    entry.transform_name,
                    entry.transform_value,
                    _path_key(entry.source),
                )
                if key in self._entries:
                    raise ValueError(
                        f"Duplicate perturbation cache entry at manifest row {row_number}: "
                        f"{key}"
                    )
                self._entries[key] = entry

        if not self._entries:
            raise ValueError(f"Perturbation manifest is empty: {self.manifest_path}")

    def __len__(self) -> int:
        return len(self._entries)

    def records_for(
        self,
        records: Iterable[ImageRecord],
        *,
        category: str,
        transform_spec: dict[str, Any],
    ) -> list[ImageRecord]:
        name = str(transform_spec.get("name", "identity")).lower()
        if name == "identity":
            return list(records)
        value = _value_key(transform_spec.get("value", 0))

        cached: list[ImageRecord] = []
        missing: list[str] = []
        for record in records:
            key = (str(category), name, value, _path_key(record.path))
            entry = self._entries.get(key)
            if entry is None:
                missing.append(str(record.path))
                continue
            if not entry.output.is_file():
                raise FileNotFoundError(
                    f"Cached perturbation file is missing: {entry.output}"
                )
            if entry.defect_type and entry.defect_type != str(record.defect_type):
                raise ValueError(
                    "Cached defect type does not match source record: "
                    f"{entry.defect_type!r} != {record.defect_type!r} for {record.path}"
                )
            cached.append(
                ImageRecord(
                    path=entry.output,
                    label=int(record.label),
                    defect_type=str(record.defect_type),
                    mask_path=record.mask_path,
                )
            )

        if missing:
            preview = ", ".join(missing[:3])
            raise KeyError(
                f"Manifest {self.manifest_path} has no {name}={value} cache entry for "
                f"{len(missing)}/{len(cached) + len(missing)} {category} records. "
                f"Examples: {preview}"
            )
        return cached
