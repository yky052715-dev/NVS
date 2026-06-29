from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np

from .metrics import average_relative_drop, safe_auroc
from .protocol import CalibrationState


def frozen_stress_predictions(
    raw_maps: np.ndarray, calibration: CalibrationState
) -> tuple[np.ndarray, np.ndarray]:
    normalized = calibration.normalize(raw_maps)
    return normalized, normalized >= calibration.threshold


def aggregate_ard(
    rows: Sequence[Mapping[str, Any]],
    metric: str = "pixel_AUROC",
    identity_name: str = "identity_0",
) -> list[dict[str, Any]]:
    optional_group_fields = tuple(
        field
        for field in ("category", "seed")
        if rows and all(field in row for row in rows)
    )
    group_fields = (*optional_group_fields, "method")
    groups: dict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        key = tuple(str(row[field]) for field in group_fields)
        groups[key].append(row)
    output = []
    for key, selected in sorted(groups.items()):
        group = dict(zip(group_fields, key))
        identity = [
            float(row[metric])
            for row in selected
            if str(row["transform"]) == identity_name
        ]
        if len(identity) != 1:
            label = ", ".join(f"{field}={value}" for field, value in group.items())
            raise ValueError(f"{label} requires exactly one identity row")
        targets = [
            float(row[metric])
            for row in selected
            if str(row["transform"]) != identity_name
        ]
        output.append(
            {
                **group,
                f"source_{metric}": identity[0],
                f"ARD_{metric}": average_relative_drop(identity[0], targets),
                "target_count": len(targets),
            }
        )
    return output


def image_metrics(labels: np.ndarray, score_maps: np.ndarray) -> dict[str, float]:
    scores = np.asarray(score_maps).reshape(score_maps.shape[0], -1).max(axis=1)
    return {"image_AUROC": safe_auroc(labels, scores)}


def robustad_metric_scope(has_pixel_masks: bool) -> tuple[str, ...]:
    if has_pixel_masks:
        return (
            "image_AUROC",
            "ARD_image_AUROC",
            "pixel_AUROC",
            "pixel_AUPRO",
            "ARD_pixel_AUROC",
        )
    return ("image_AUROC", "ARD_image_AUROC")
