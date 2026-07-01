"""Verification helpers for frozen RobustAD diagnostic replay."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


KEYS = ("category", "domain", "shift", "method")
METRICS = ("image_AUROC", "pixel_AUROC", "pixel_AUPR")


def verify_replayed_metrics(
    replayed: Sequence[Mapping[str, Any]],
    baseline: Sequence[Mapping[str, Any]],
    tolerance: float = 1.0e-6,
) -> list[dict[str, Any]]:
    """Compare rank metrics already present in the completed baseline CSV."""

    baseline_index = {
        tuple(str(row[key]) for key in KEYS): row for row in baseline
    }
    output = []
    for row in replayed:
        key = tuple(str(row[name]) for name in KEYS)
        if key not in baseline_index:
            raise ValueError(f"Replay row is absent from baseline: {key}")
        reference = baseline_index[key]
        for metric in METRICS:
            if metric not in row or metric not in reference:
                continue
            actual = float(row[metric])
            expected = float(reference[metric])
            if not (math.isfinite(actual) and math.isfinite(expected)):
                continue
            difference = actual - expected
            output.append(
                {
                    **dict(zip(KEYS, key)),
                    "metric": metric,
                    "baseline": expected,
                    "replayed": actual,
                    "absolute_difference": abs(difference),
                    "within_tolerance": abs(difference) <= float(tolerance),
                    "tolerance": float(tolerance),
                }
            )
    expected_keys = {
        key
        for key, row in baseline_index.items()
        if str(row.get("method")) in {"D0_NN", "D2_NVSGlobal"}
    }
    replayed_keys = {tuple(str(row[name]) for name in KEYS) for row in replayed}
    missing = expected_keys - replayed_keys
    if missing:
        raise ValueError(f"Baseline rows are absent from replay: {sorted(missing)}")
    failures = [row for row in output if not row["within_tolerance"]]
    if failures:
        worst = max(failures, key=lambda row: row["absolute_difference"])
        raise ValueError(
            "Diagnostic replay does not reproduce baseline metrics; "
            f"worst={worst}"
        )
    return output
