from __future__ import annotations

import argparse
import hashlib
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from nvs.apc import decompose_deviation, spatial_consistency
from nvs.common import (
    build_mvtec_records,
    extract_features,
    load_config,
    load_dinov2,
    nearest_with_indices,
    save_json,
    write_csv,
)
from nvs.detection import _fit_state, _loader, _transformed_loader
from nvs.metrics import safe_auroc
from nvs.transforms import is_spatially_aligned, transform_name


def _seed(base: int, *parts: str) -> int:
    digest = hashlib.sha256("::".join(parts).encode()).digest()
    return int(base) + int.from_bytes(digest[:4], "little")


def _stats(
    features: torch.Tensor,
    state: dict[str, Any],
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    distances, indices = nearest_with_indices(
        features,
        state["memory_bank"],
        query_chunk_size=int(config["memory"]["query_chunk_size"]),
        bank_chunk_size=int(config["memory"]["bank_chunk_size"]),
        device=device,
    )
    b, p, d = features.shape
    nearest = state["memory_bank"][indices.reshape(-1)].reshape(b, p, d).to(
        device,
        non_blocking=True,
    )
    basis = state["nvs_basis"].to(device, non_blocking=True)
    delta = features.to(device, non_blocking=True).float() - nearest.float()
    decomposition = decompose_deviation(delta, basis)
    return {
        "r0_nn_distance": distances.to(device, non_blocking=True).float(),
        "parallel_norm": decomposition["parallel_norm"],
        "perpendicular_norm": decomposition["perpendicular_norm"],
        "total_norm": decomposition["total_norm"],
        "rho": decomposition["rho"],
        "consistency": spatial_consistency(decomposition["coefficients"]),
    }


def _defect_patch_mask(
    masks: torch.Tensor,
    grid_side: int,
    threshold: float,
) -> torch.Tensor:
    pooled = F.interpolate(
        masks[:, None].float(),
        size=(int(grid_side), int(grid_side)),
        mode="area",
    )[:, 0]
    return pooled.reshape(pooled.shape[0], -1) >= float(threshold)


def _sample_rows(
    category: str,
    group: str,
    transform: str,
    paths: list[str],
    stats: dict[str, torch.Tensor],
    budget: int,
    seed: int,
    eligible: torch.Tensor | None = None,
    small_flags: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    _, patches = stats["rho"].shape
    flat_count = stats["rho"].numel()
    candidates = (
        np.arange(flat_count)
        if eligible is None
        else np.flatnonzero(eligible.reshape(-1).cpu().numpy())
    )
    if candidates.size == 0:
        return []
    rng = np.random.default_rng(seed)
    chosen = rng.choice(
        candidates,
        size=min(int(budget), candidates.size),
        replace=False,
    )
    arrays = {key: value.reshape(-1).cpu().numpy() for key, value in stats.items()}
    rows: list[dict[str, Any]] = []
    for flat_index in np.sort(chosen).tolist():
        image_index = int(flat_index // patches)
        rows.append(
            {
                "category": category,
                "group": group,
                "transform": transform,
                "image_path": paths[image_index],
                "patch_index": int(flat_index % patches),
                "image_small_defect": (
                    bool(small_flags[image_index])
                    if small_flags is not None
                    else False
                ),
                **{
                    key: float(values[flat_index])
                    for key, values in arrays.items()
                },
            }
        )
    return rows


def _save_maps(
    path: Path,
    paths: list[str],
    stats: dict[str, torch.Tensor],
    grid_side: int,
    defect_mask: torch.Tensor | None = None,
    small_flags: np.ndarray | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"paths": np.asarray(paths, dtype=str)}
    for key, value in stats.items():
        payload[key] = (
            value.reshape(value.shape[0], grid_side, grid_side)
            .cpu()
            .numpy()
            .astype(np.float32)
        )
    if defect_mask is not None:
        payload["defect_patch_mask"] = (
            defect_mask.reshape(defect_mask.shape[0], grid_side, grid_side)
            .cpu()
            .numpy()
            .astype(np.uint8)
        )
    if small_flags is not None:
        payload["image_small_defect"] = small_flags.astype(np.uint8)
    np.savez_compressed(path, **payload)


def _normal_rows(
    category: str,
    group: str,
    records,
    specs: list[dict[str, Any] | None],
    state: dict[str, Any],
    config: dict[str, Any],
    model,
    device: torch.device,
    output_dir: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    budget = int(config["apc_diagnostics"]["max_patch_rows_per_group"])
    per_spec = max(1, budget // max(1, len(specs)))
    for spec in specs:
        if spec is None:
            name = "identity"
            loader = _loader(records, config)
        else:
            if not is_spatially_aligned(spec):
                raise ValueError(f"Transform must be spatially aligned: {spec}")
            name = transform_name(spec)
            loader = _transformed_loader(records, config, spec)
        features, paths, grid_side = extract_features(model, loader, device)
        stats = _stats(features, state, config, device)
        rows += _sample_rows(
            category,
            group,
            name,
            paths,
            stats,
            per_spec,
            _seed(int(config["experiment"]["seed"]), category, group, name),
        )
        if bool(config["apc_diagnostics"].get("export_maps", True)):
            _save_maps(
                output_dir / "maps" / category / f"{group}__{name}.npz",
                paths,
                stats,
                grid_side,
            )
    return rows


@torch.inference_mode()
def _defect_rows(
    category: str,
    records,
    state: dict[str, Any],
    config: dict[str, Any],
    model,
    device: torch.device,
    output_dir: Path,
) -> list[dict[str, Any]]:
    from dinov2_mvtec_nn import extract_patch_tokens

    records = [record for record in records if int(record.label) == 1]
    chunks: list[dict[str, Any]] = []
    for batch_index, batch in enumerate(
        tqdm(_loader(records, config, include_mask=True), desc=f"APC defects {category}")
    ):
        features, grid_side = extract_patch_tokens(model, batch["image"], device)
        stats = _stats(features, state, config, device)
        defect_mask = _defect_patch_mask(
            batch["mask"],
            grid_side,
            float(config["apc_diagnostics"]["defect_mask_threshold"]),
        )
        area = batch["mask"].float().flatten(1).mean(dim=1).numpy()
        small = area <= float(config["metrics"]["small_defect_area_fraction"])
        chunks.append(
            {
                "index": batch_index,
                "paths": list(batch["path"]),
                "grid_side": grid_side,
                "stats": stats,
                "mask": defect_mask,
                "small": small,
            }
        )
        if bool(config["apc_diagnostics"].get("export_maps", True)):
            _save_maps(
                output_dir
                / "maps"
                / category
                / f"real_defect__batch_{batch_index:04d}.npz",
                list(batch["path"]),
                stats,
                grid_side,
                defect_mask,
                small,
            )

    budget = int(config["apc_diagnostics"]["max_patch_rows_per_group"])
    rows: list[dict[str, Any]] = []
    for group in ("real_defect", "small_defect"):
        per_chunk = max(1, budget // max(1, len(chunks)))
        for chunk in chunks:
            eligible = chunk["mask"]
            if group == "small_defect":
                eligible = eligible & torch.from_numpy(chunk["small"])[:, None]
            rows += _sample_rows(
                category,
                group,
                "identity",
                chunk["paths"],
                chunk["stats"],
                per_chunk,
                _seed(
                    int(config["experiment"]["seed"]),
                    category,
                    group,
                    str(chunk["index"]),
                ),
                eligible,
                chunk["small"],
            )
    return rows


def _describe(values: list[float]) -> dict[str, float]:
    x = np.asarray(values, dtype=np.float64)
    if not x.size:
        return {
            key: float("nan")
            for key in ("mean", "std", "q25", "median", "q75", "p90")
        }
    return {
        "mean": float(x.mean()),
        "std": float(x.std()),
        "q25": float(np.quantile(x, 0.25)),
        "median": float(np.median(x)),
        "q75": float(np.quantile(x, 0.75)),
        "p90": float(np.quantile(x, 0.90)),
    }


def _distribution_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["category"], row["group"])].append(row)
    output: list[dict[str, Any]] = []
    for (category, group), selected in sorted(grouped.items()):
        result: dict[str, Any] = {
            "category": category,
            "group": group,
            "patches": len(selected),
        }
        for metric in (
            "rho",
            "consistency",
            "parallel_norm",
            "perpendicular_norm",
            "total_norm",
            "r0_nn_distance",
        ):
            for key, value in _describe(
                [float(row[metric]) for row in selected]
            ).items():
                result[f"{metric}_{key}"] = value
        output.append(result)
    return output


def _auc(
    negative: list[dict[str, Any]],
    positive: list[dict[str, Any]],
    score,
) -> float:
    if not negative or not positive:
        return float("nan")
    labels = [0] * len(negative) + [1] * len(positive)
    scores = [score(row) for row in negative + positive]
    return safe_auroc(labels, scores)


def _separability(
    category: str,
    rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["group"]].append(row)
    identity = groups["identity_normal"]
    transformed = groups["transformed_normal_known"]
    unseen = groups["transformed_normal_unseen_strength"]
    defect = groups["real_defect"]
    small = groups["small_defect"]
    transformed_rho = _describe([row["rho"] for row in transformed])
    small_rho = _describe([row["rho"] for row in small])
    joint = lambda row: 1.0 - 0.5 * (
        float(row["rho"]) + float(row["consistency"])
    )
    small_joint_auc = _auc(transformed, small, joint)
    margin = transformed_rho["median"] - small_rho["median"]
    evaluable = bool(small)
    passed = None if not evaluable else bool(
        np.isfinite(small_joint_auc)
        and small_joint_auc
        >= float(config["apc_diagnostics"]["separability_auc_min"])
        and np.isfinite(margin)
        and margin
        >= float(config["apc_diagnostics"]["rho_median_margin_min"])
    )
    return {
        "category": category,
        "identity_patches": len(identity),
        "transformed_known_patches": len(transformed),
        "transformed_unseen_patches": len(unseen),
        "real_defect_patches": len(defect),
        "small_defect_patches": len(small),
        "identity_vs_transformed_rho_auc": _auc(
            identity, transformed, lambda row: row["rho"]
        ),
        "transformed_vs_defect_one_minus_rho_auc": _auc(
            transformed, defect, lambda row: 1.0 - row["rho"]
        ),
        "transformed_vs_defect_joint_auc": _auc(
            transformed, defect, joint
        ),
        "transformed_vs_small_one_minus_rho_auc": _auc(
            transformed, small, lambda row: 1.0 - row["rho"]
        ),
        "transformed_vs_small_one_minus_consistency_auc": _auc(
            transformed, small, lambda row: 1.0 - row["consistency"]
        ),
        "transformed_vs_small_joint_auc": small_joint_auc,
        "transformed_rho_median": transformed_rho["median"],
        "small_defect_rho_median": small_rho["median"],
        "rho_median_margin": margin,
        "small_defect_evaluable": evaluable,
        "rho_small_defect_hypothesis_pass": passed,
    }


def _mean_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    evaluable = [row for row in rows if row["small_defect_evaluable"]]
    result: dict[str, Any] = {
        "category": "mean",
        "small_defect_evaluable": bool(evaluable),
        "rho_small_defect_hypothesis_pass": bool(
            evaluable
            and all(row["rho_small_defect_hypothesis_pass"] for row in evaluable)
        ),
    }
    for key in rows[0]:
        if key in {
            "category",
            "small_defect_evaluable",
            "rho_small_defect_hypothesis_pass",
        }:
            continue
        values = np.asarray([float(row[key]) for row in rows])
        result[key] = (
            float(np.nanmean(values))
            if np.isfinite(values).any()
            else float("nan")
        )
    return result


def _fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, bool):
        return str(value)
    value = float(value)
    return f"{value:.6f}" if np.isfinite(value) else "nan"


def _write_report(rows: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# APC-NVS rho/c separability diagnosis",
        "",
        "| category | I→T rho AUC | T→defect rho AUC | T→defect joint AUC | T→small rho AUC | T→small c AUC | T→small joint AUC | T rho median | small rho median | rho margin | pass |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        keys = (
            "identity_vs_transformed_rho_auc",
            "transformed_vs_defect_one_minus_rho_auc",
            "transformed_vs_defect_joint_auc",
            "transformed_vs_small_one_minus_rho_auc",
            "transformed_vs_small_one_minus_consistency_auc",
            "transformed_vs_small_joint_auc",
            "transformed_rho_median",
            "small_defect_rho_median",
            "rho_median_margin",
            "rho_small_defect_hypothesis_pass",
        )
        lines.append(
            "| "
            + " | ".join([row["category"]] + [_fmt(row[key]) for key in keys])
            + " |"
        )
    lines += [
        "",
        "## Decision",
        "",
        f"All-category hypothesis pass: `{rows[-1]['rho_small_defect_hypothesis_pass']}`",
        "",
        "`I→T rho AUC` uses rho; defect AUCs use `1-rho`, `1-c`, or `1-(rho+c)/2`. "
        "The pass flag requires the configured joint AUC and rho median margin in every evaluable category. Categories without small-defect patches are reported as NA. "
        "Failure means per-category distributions and exported maps must be inspected before C2.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot(rows: list[dict[str, Any]], output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[warn] plots skipped: {exc}")
        return
    for category in sorted({row["category"] for row in rows}):
        selected = [row for row in rows if row["category"] == category]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        for group in (
            "identity_normal",
            "transformed_normal_known",
            "transformed_normal_unseen_strength",
            "real_defect",
            "small_defect",
        ):
            group_rows = [row for row in selected if row["group"] == group]
            if not group_rows:
                continue
            for axis, metric in zip(axes, ("rho", "consistency")):
                x = np.sort([row[metric] for row in group_rows])
                axis.plot(x, np.linspace(0, 1, len(x)), label=group)
                axis.set_xlabel(metric)
                axis.set_ylabel("ECDF")
                axis.set_xlim(0, 1)
        axes[0].set_title(f"{category}: rho")
        axes[1].set_title(f"{category}: consistency")
        axes[1].legend(fontsize=7)
        path = output_dir / "plots" / f"{category}_rho_consistency.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=160)
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="APC-NVS rho/c diagnosis")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--categories", nargs="+")
    parser.add_argument("--output-dir")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.categories:
        config["data"]["categories"] = args.categories
    if args.output_dir:
        config["experiment"]["output_dir"] = args.output_dir
    seed = int(config["experiment"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(args.device)
    model = load_dinov2(
        config["model"]["name"],
        device,
        config["model"].get("hub_dir"),
    )
    output_dir = Path(config["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(config, output_dir / "resolved_config.json")

    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    unseen = config["apc_diagnostics"]["unseen_strength_transforms"]
    for category in config["data"]["categories"]:
        print(f"[APC-separability] {category}")
        train_records, test_records = build_mvtec_records(args.data_root, category)
        state = _fit_state(category, train_records, config, model, device)
        records = state["calibration_records"]
        category_rows = _normal_rows(
            category,
            "identity_normal",
            records,
            [None],
            state,
            config,
            model,
            device,
            output_dir,
        )
        category_rows += _normal_rows(
            category,
            "transformed_normal_known",
            records,
            list(config["transforms"]),
            state,
            config,
            model,
            device,
            output_dir,
        )
        category_rows += _normal_rows(
            category,
            "transformed_normal_unseen_strength",
            records,
            list(unseen),
            state,
            config,
            model,
            device,
            output_dir,
        )
        category_rows += _defect_rows(
            category,
            test_records,
            state,
            config,
            model,
            device,
            output_dir,
        )
        all_rows += category_rows
        summaries.append(_separability(category, category_rows, config))
        current = summaries + [_mean_row(summaries)]
        write_csv(all_rows, output_dir / "patch_statistics.csv")
        write_csv(_distribution_summary(all_rows), output_dir / "distribution_summary.csv")
        write_csv(current, output_dir / "separability_summary.csv")
        _write_report(current, output_dir / "apc_separability_summary.md")

    final = summaries + [_mean_row(summaries)]
    if not args.no_plots:
        _plot(all_rows, output_dir)
    passed = bool(final[-1]["rho_small_defect_hypothesis_pass"])
    save_json(
        {
            "status": "complete",
            "categories": list(config["data"]["categories"]),
            "completed_count": len(config["data"]["categories"]),
            "patch_rows": len(all_rows),
            "all_category_hypothesis_pass": passed,
            "next_step": (
                "implement_c2" if passed else "inspect_distributions_before_c2"
            ),
        },
        output_dir / "experiment_complete.json",
    )


if __name__ == "__main__":
    main()
