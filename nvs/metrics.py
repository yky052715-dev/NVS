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

