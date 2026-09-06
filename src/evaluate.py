"""Threshold-free and operating-point metrics for anomaly detection.

`accuracy` is deliberately absent: it depends on the threshold and is misleading
under the class imbalance of the MVTec test splits.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.ndimage import label
from sklearn.metrics import precision_recall_curve, roc_auc_score

DEFAULT_MAX_FPR = 0.3
DEFAULT_PRO_THRESHOLDS = 200


@dataclass(frozen=True)
class Metrics:
    image_auroc: float
    pixel_auroc: float
    au_pro: float
    f1_max: float
    recall_at_threshold: float
    fpr_at_threshold: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def image_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels).ravel()
    if len(np.unique(labels)) < 2:
        raise ValueError("image AUROC needs both normal and anomalous images")
    return float(roc_auc_score(labels, np.asarray(scores).ravel()))


def pixel_auroc(masks: np.ndarray, maps: np.ndarray) -> float:
    truth = (np.asarray(masks).ravel() > 0.5).astype(np.uint8)
    if len(np.unique(truth)) < 2:
        raise ValueError("pixel AUROC needs both defective and clean pixels")
    return float(roc_auc_score(truth, np.asarray(maps).ravel()))


def f1_max(labels: np.ndarray, scores: np.ndarray) -> float:
    """Best F1 over every threshold.

    Picking that threshold uses the test labels, so this is an oracle upper bound.
    Report it as such, never as achieved performance.
    """
    precision, recall, _ = precision_recall_curve(
        np.asarray(labels).ravel(), np.asarray(scores).ravel()
    )
    denominator = precision + recall
    f1 = np.divide(
        2 * precision * recall,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    return float(f1.max())


def recall_at(labels: np.ndarray, scores: np.ndarray, threshold: float) -> float:
    labels = np.asarray(labels).ravel()
    predicted = np.asarray(scores).ravel() >= threshold
    anomalous = labels == 1
    if not anomalous.any():
        raise ValueError("recall needs at least one anomalous image")
    return float(predicted[anomalous].mean())


def fpr_at(labels: np.ndarray, scores: np.ndarray, threshold: float) -> float:
    labels = np.asarray(labels).ravel()
    predicted = np.asarray(scores).ravel() >= threshold
    normal = labels == 0
    if not normal.any():
        raise ValueError("false positive rate needs at least one normal image")
    return float(predicted[normal].mean())


def au_pro(
    masks: np.ndarray,
    maps: np.ndarray,
    max_fpr: float = DEFAULT_MAX_FPR,
    num_thresholds: int = DEFAULT_PRO_THRESHOLDS,
) -> float:
    """Area under the per-region-overlap curve, integrated up to `max_fpr`.

    Every connected defect region counts once however large it is, so a handful of
    big defects cannot hide the small ones the way plain pixel AUROC lets them.
    Normalized by `max_fpr`, so 1.0 is perfect.
    """
    masks = np.asarray(masks)
    maps = np.asarray(maps)
    if masks.shape != maps.shape:
        raise ValueError(f"masks {masks.shape} and maps {maps.shape} must have the same shape")

    binary = masks > 0.5
    if not binary.any():
        raise ValueError("AU-PRO needs at least one defect region")

    # Scores inside each connected defect region, sorted, so the overlap at a given
    # threshold is a binary search rather than a full pass over the region.
    region_scores: list[np.ndarray] = []
    for index in range(len(binary)):
        labelled, count = label(binary[index])
        for region_id in range(1, count + 1):
            region_scores.append(np.sort(maps[index][labelled == region_id]))

    clean_scores = np.sort(maps[~binary])
    if clean_scores.size == 0:
        raise ValueError("AU-PRO needs clean pixels to measure a false positive rate")

    thresholds = _pro_thresholds(maps, clean_scores, max_fpr, num_thresholds)
    false_positives = _tail_fraction(clean_scores, thresholds)
    overlap = np.mean([_tail_fraction(scores, thresholds) for scores in region_scores], axis=0)
    return _normalized_area(false_positives, overlap, max_fpr)


def _tail_fraction(sorted_values: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Fraction of `sorted_values` at or above each threshold."""
    return 1.0 - np.searchsorted(sorted_values, thresholds, side="left") / sorted_values.size


def _pro_thresholds(
    maps: np.ndarray, clean_scores: np.ndarray, max_fpr: float, num_thresholds: int
) -> np.ndarray:
    """Thresholds dense in the FPR range we integrate over, plus the fpr == 0 end.

    Quantiles of the clean pixels place most thresholds where they matter. The linear
    span keeps the curve usable when those scores are degenerate, which is exactly
    what happens with a perfect predictor.
    """
    targets = np.linspace(0.0, max_fpr, num_thresholds)
    positions = np.clip((clean_scores.size * (1.0 - targets)).astype(int), 0, clean_scores.size - 1)
    span = np.linspace(float(maps.min()), float(maps.max()), num_thresholds)
    above_everything = np.nextafter(float(maps.max()), np.inf)
    return np.unique(np.concatenate([clean_scores[positions], span, [above_everything]]))


def _normalized_area(false_positives: np.ndarray, overlap: np.ndarray, max_fpr: float) -> float:
    # One point per false positive rate, keeping the best overlap reached there.
    order = np.lexsort((-overlap, false_positives))
    x, y = false_positives[order], overlap[order]
    first = np.concatenate([[True], np.diff(x) > 0])
    x, y = x[first], y[first]

    inside = x <= max_fpr
    area_x, area_y = x[inside], y[inside]
    if area_x.size == 0 or area_x[0] > 0.0:
        area_x = np.concatenate([[0.0], area_x])
        area_y = np.concatenate([[0.0], area_y])
    if area_x[-1] < max_fpr:
        area_x = np.append(area_x, max_fpr)
        area_y = np.append(area_y, np.interp(max_fpr, x, y))

    return float(np.trapezoid(area_y, area_x) / max_fpr)


def evaluate(
    labels: np.ndarray,
    scores: np.ndarray,
    masks: np.ndarray,
    maps: np.ndarray,
    threshold: float,
) -> Metrics:
    """All label-based metrics of section 10.

    Latency and memory-bank size are system measurements and come from the benchmark
    path instead.
    """
    return Metrics(
        image_auroc=image_auroc(labels, scores),
        pixel_auroc=pixel_auroc(masks, maps),
        au_pro=au_pro(masks, maps),
        f1_max=f1_max(labels, scores),
        recall_at_threshold=recall_at(labels, scores, threshold),
        fpr_at_threshold=fpr_at(labels, scores, threshold),
    )
