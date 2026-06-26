from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def safe_auroc(labels, scores) -> float:
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    if labels.size == 0 or np.unique(labels).size < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def summarize_values(values) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "p90": float("nan"),
            "p95": float("nan"),
            "max": float("nan"),
        }
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(np.max(array)),
    }


def threshold_image_max(calibration_maps: np.ndarray, image_quantile: float) -> float:
    flattened = calibration_maps.reshape(calibration_maps.shape[0], -1)
    return float(np.quantile(flattened.max(axis=1), float(image_quantile)))


def image_score_from_map(maps: np.ndarray, method: str, topk_fraction: float) -> np.ndarray:
    flattened = maps.reshape(maps.shape[0], -1)
    if method == "max":
        return flattened.max(axis=1)
    if method == "topk_mean":
        count = max(1, int(round(flattened.shape[1] * float(topk_fraction))))
        partition = np.partition(flattened, flattened.shape[1] - count, axis=1)
        return partition[:, -count:].mean(axis=1)
    raise ValueError(f"Unsupported image score method: {method}")


def binary_f1(mask: np.ndarray, prediction: np.ndarray) -> float:
    mask = np.asarray(mask, dtype=bool)
    prediction = np.asarray(prediction, dtype=bool)
    tp = float(np.logical_and(mask, prediction).sum())
    fp = float(np.logical_and(~mask, prediction).sum())
    fn = float(np.logical_and(mask, ~prediction).sum())
    denominator = 2.0 * tp + fp + fn
    if denominator <= 0.0:
        return float("nan")
    return float((2.0 * tp) / denominator)


def oracle_pixel_f1(masks: np.ndarray, maps: np.ndarray, max_thresholds: int = 512) -> tuple[float, float]:
    masks = np.asarray(masks, dtype=bool)
    maps = np.asarray(maps, dtype=np.float64)
    flat_scores = maps.reshape(-1)
    if flat_scores.size == 0:
        return float("nan"), float("nan")
    if flat_scores.size > int(max_thresholds):
        thresholds = np.quantile(
            flat_scores,
            np.linspace(0.0, 1.0, int(max_thresholds), dtype=np.float64),
        )
        thresholds = np.unique(thresholds)
    else:
        thresholds = np.unique(flat_scores)
    best_f1 = -1.0
    best_threshold = float(thresholds[0])
    for threshold in thresholds:
        value = binary_f1(masks, maps >= float(threshold))
        if np.isfinite(value) and value > best_f1:
            best_f1 = float(value)
            best_threshold = float(threshold)
    if best_f1 < 0.0:
        return float("nan"), float("nan")
    return best_f1, best_threshold


def _label_components(binary: np.ndarray) -> tuple[np.ndarray, int]:
    binary = np.asarray(binary, dtype=bool)
    try:
        from scipy import ndimage

        structure = np.asarray([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
        labels, count = ndimage.label(binary, structure=structure)
        return labels.astype(np.int32, copy=False), int(count)
    except Exception:
        labels = np.zeros(binary.shape, dtype=np.int32)
        current = 0
        height, width = binary.shape
        for y in range(height):
            for x in range(width):
                if not binary[y, x] or labels[y, x] != 0:
                    continue
                current += 1
                stack = [(y, x)]
                labels[y, x] = current
                while stack:
                    cy, cx = stack.pop()
                    for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                        if 0 <= ny < height and 0 <= nx < width and binary[ny, nx] and labels[ny, nx] == 0:
                            labels[ny, nx] = current
                            stack.append((ny, nx))
        return labels, current


def keep_topk_components(predictions: np.ndarray, k: int) -> np.ndarray:
    predictions = np.asarray(predictions, dtype=bool)
    if int(k) <= 0:
        raise ValueError("k must be positive")
    if predictions.ndim == 2:
        predictions = predictions[None]
        squeeze = True
    elif predictions.ndim == 3:
        squeeze = False
    else:
        raise ValueError("predictions must have shape [H, W] or [N, H, W]")

    output = np.zeros_like(predictions, dtype=bool)
    for index, binary in enumerate(predictions):
        labels, count = _label_components(binary)
        if count == 0:
            continue
        sizes = np.bincount(labels.reshape(-1), minlength=count + 1)
        sizes[0] = 0
        keep = np.argsort(sizes)[-int(k) :]
        keep = keep[sizes[keep] > 0]
        if keep.size > 0:
            output[index] = np.isin(labels, keep)
    return output[0] if squeeze else output


def localization_metrics_from_prediction(
    masks: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray,
    small_defect_area_fraction: float = 0.01,
) -> dict[str, float]:
    masks = np.asarray(masks, dtype=bool)
    predictions = np.asarray(predictions, dtype=bool)
    labels = np.asarray(labels, dtype=np.int64)
    flat_masks = masks.reshape(masks.shape[0], -1)
    flat_predictions = predictions.reshape(predictions.shape[0], -1)
    anomaly_indices = np.flatnonzero(labels == 1)
    normal_indices = np.flatnonzero(labels == 0)

    recalls: list[float] = []
    overseg: list[float] = []
    image_f1: list[float] = []
    small_f1: list[float] = []
    small_recall: list[float] = []
    for index in anomaly_indices:
        gt = flat_masks[index]
        pred = flat_predictions[index]
        gt_count = float(gt.sum())
        pred_count = float(pred.sum())
        if gt_count <= 0.0:
            continue
        tp = float(np.logical_and(gt, pred).sum())
        recall = tp / gt_count
        recalls.append(recall)
        overseg.append(pred_count / gt_count)
        image_f1.append(binary_f1(gt, pred))
        if gt_count / float(gt.size) <= float(small_defect_area_fraction):
            small_recall.append(recall)
            small_f1.append(binary_f1(gt, pred))

    normal_positive_rate = float("nan")
    if normal_indices.size > 0:
        normal_positive_rate = float(flat_predictions[normal_indices].any(axis=1).mean())

    return {
        "localization_overseg_anomaly_macro": float(np.nanmean(overseg)) if overseg else float("nan"),
        "localization_recall_anomaly_macro": float(np.nanmean(recalls)) if recalls else float("nan"),
        "localization_anomaly_image_f1_macro": float(np.nanmean(image_f1)) if image_f1 else float("nan"),
        "localization_small_defect_f1_macro": float(np.nanmean(small_f1)) if small_f1 else float("nan"),
        "localization_small_defect_recall_macro": float(np.nanmean(small_recall)) if small_recall else float("nan"),
        "localization_test_normal_image_positive_rate": normal_positive_rate,
    }


def localization_metrics(
    masks: np.ndarray,
    maps: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    small_defect_area_fraction: float = 0.01,
) -> dict[str, float]:
    maps = np.asarray(maps, dtype=np.float64)
    return localization_metrics_from_prediction(
        masks,
        maps >= float(threshold),
        labels,
        small_defect_area_fraction=small_defect_area_fraction,
    )