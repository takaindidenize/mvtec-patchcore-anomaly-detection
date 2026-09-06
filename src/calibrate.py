"""Threshold calibration from held-out normal images.

The training data contains no anomalies, so a threshold cannot be chosen by
maximizing F1 on labelled data without touching the test set. Instead we score
`val_normal` — normal images the model never saw while fitting — and cut at the
quantile that leaves `target_fpr` of them above the line. The threshold then has a
statement attached to it: on normal parts this configuration raises a false alarm
roughly `target_fpr` of the time.
"""

from __future__ import annotations

import numpy as np


def calibrate_threshold(val_scores: np.ndarray, target_fpr: float) -> float:
    scores = np.asarray(val_scores, dtype=np.float64).ravel()
    if scores.size == 0:
        raise ValueError("calibration needs at least one validation score")
    if not 0.0 < target_fpr < 1.0:
        raise ValueError(f"target_fpr must be in (0, 1), got {target_fpr}")
    return float(np.quantile(scores, 1.0 - target_fpr))
