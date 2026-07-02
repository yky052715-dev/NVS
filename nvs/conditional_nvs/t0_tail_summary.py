"""Summarize frozen T0 outputs into the predeclared attribution decisions."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .protocol import stable_hash


REQUIRED_OUTPUTS = (
    "raw_patch_energy_stats.csv",
    "tail_retention_by_shift.csv",
    "calibration_comparison.csv",
    "paired_image_metrics.csv",
    "bootstrap_summary.csv",
    "consistency_checks.json",
    "t0_attribution_complete.json",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    keys = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _finite(row: Mapping[str, Any] | None, field: str) -> float:
    if row is None:
        return float("nan")
    try:
        return float(row[field])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def _negative(row: Mapping[str, Any] | None) -> bool:
    return _finite(row, "ci_high") < 0.0


def _positive(row: Mapping[str, Any] | None) -> bool:
    return _finite(row, "ci_low") > 0.0


def attribution_decisions(
    bootstrap_rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Apply only CI-sign rules; no post-hoc effect threshold is introduced."""

    index = {
        (
            str(row["category"]),
            str(row["domain"]),
            str(row["shift"]),
            str(row["comparison"]),
            str(row["view"]),
            str(row["metric"]),
        ): row
        for row in bootstrap_rows
    }
    groups = sorted(
        {
            (str(row["category"]), str(row["domain"]), str(row["shift"]))
            for row in bootstrap_rows
        }
    )
    output: list[dict[str, Any]] = []
    for category, domain, shift in groups:
        def get(comparison: str, view: str, metric: str) -> Mapping[str, Any] | None:
            return index.get(
                (category, domain, shift, comparison, view, metric)
            )

        raw_aupr = get("D2_minus_D0", "raw_energy", "pixel_AUPR")
        raw_gap = get("D2_minus_D0", "raw_energy", "pixel_G99")
        legacy_f1 = get(
            "D2_minus_D0", "legacy_independent", "pixel_F1_calibrated"
        )
        legacy_fp = get(
            "D2_minus_D0", "legacy_independent", "normal_image_fp"
        )
        legacy_gap = get(
            "D2_minus_D0", "legacy_independent", "pixel_G99"
        )
        recovery_f1 = get(
            "shared_minus_legacy", "D2_NVSGlobal", "pixel_F1_calibrated"
        )
        recovery_fp = get(
            "shared_minus_legacy", "D2_NVSGlobal", "normal_image_fp"
        )
        recovery_gap = get(
            "shared_minus_legacy", "D2_NVSGlobal", "pixel_G99"
        )
        projection_evidence = _negative(raw_aupr) or _negative(raw_gap)
        legacy_worse = (
            _negative(legacy_f1)
            or _positive(legacy_fp)
            or _negative(legacy_gap)
        )
        shared_recovers = (
            _positive(recovery_f1)
            or _negative(recovery_fp)
            or _positive(recovery_gap)
        )
        calibration_evidence = legacy_worse and shared_recovers
        if projection_evidence and calibration_evidence:
            decision = "mixed"
            next_step = "soft_projection_eligible"
        elif projection_evidence:
            decision = "projection_dominant"
            next_step = "soft_projection_eligible"
        elif calibration_evidence:
            decision = "calibration_dominant"
            next_step = "do_not_add_soft_projection"
        else:
            decision = "neither"
            next_step = "enter_T2_check_mstar"
        output.append(
            {
                "category": category,
                "domain": domain,
                "shift": shift,
                "decision": decision,
                "projection_evidence": projection_evidence,
                "legacy_worse": legacy_worse,
                "shared_recovers": shared_recovers,
                "calibration_evidence": calibration_evidence,
                "next_step": next_step,
                "raw_pixel_AUPR_delta": _finite(raw_aupr, "delta"),
                "raw_pixel_AUPR_ci_low": _finite(raw_aupr, "ci_low"),
                "raw_pixel_AUPR_ci_high": _finite(raw_aupr, "ci_high"),
                "raw_pixel_G99_delta": _finite(raw_gap, "delta"),
                "raw_pixel_G99_ci_low": _finite(raw_gap, "ci_low"),
                "raw_pixel_G99_ci_high": _finite(raw_gap, "ci_high"),
                "legacy_F1_delta": _finite(legacy_f1, "delta"),
                "legacy_FP_delta": _finite(legacy_fp, "delta"),
                "shared_vs_legacy_D2_F1_delta": _finite(
                    recovery_f1, "delta"
                ),
                "shared_vs_legacy_D2_FP_delta": _finite(
                    recovery_fp, "delta"
                ),
            }
        )
    return output


def summarize(output_dir: str | Path) -> dict[str, Any]:
    root = Path(output_dir)
    missing = [name for name in REQUIRED_OUTPUTS if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete T0 output, missing: {missing}")
    checks = _load_json(root / "consistency_checks.json")
    if checks.get("status") != "passed":
        raise AssertionError("T0 consistency checks did not pass")
    tolerance = float(checks["tolerance"])
    check_fields = (
        "max_abs_d0_minus_e0",
        "max_abs_d2_minus_sqrt_2e2",
        "max_relative_energy_identity_error",
    )
    if any(
        not math.isfinite(float(checks[field]))
        or float(checks[field]) >= tolerance
        for field in check_fields
    ):
        raise AssertionError("T0 consistency maxima violate strict tolerance")
    run = _load_json(root / "t0_attribution_complete.json")
    rows = _read_csv(root / "bootstrap_summary.csv")
    decisions = attribution_decisions(rows)
    _write_csv(root / "attribution_by_shift.csv", decisions)
    counts: dict[str, int] = {}
    for row in decisions:
        key = str(row["decision"])
        counts[key] = counts.get(key, 0) + 1
    eligible = any(
        row["decision"] in {"projection_dominant", "mixed"}
        for row in decisions
    )
    payload = {
        "status": "complete",
        "protocol": run["protocol"],
        "diagnostic_only": True,
        "seed": run["seed"],
        "categories": run["categories"],
        "shifts": run["shifts"],
        "consistency": checks,
        "decision_counts": counts,
        "decisions": decisions,
        "soft_projection_eligible": eligible,
        "soft_projection_implemented_in_T0": False,
        "rule": (
            "soft projection is eligible only for projection_dominant or mixed; "
            "neither enters T2 m* diagnosis"
        ),
        "source_output_hash": run["output_hash"],
    }
    payload["summary_hash"] = stable_hash(payload)
    (root / "t0_complete.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize frozen T0 attribution")
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    result = summarize(parse_args().output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
