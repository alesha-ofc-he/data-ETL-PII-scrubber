"""Offline classification and fixed-budget ranking metrics."""

import math

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


def classification_metrics(
    labels: NDArray[np.int64],
    scores: NDArray[np.float64],
    *,
    top_fraction: float = 0.10,
) -> dict[str, float]:
    """Evaluate scores without choosing a threshold from the same labels."""
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.ndim != 1 or scores.ndim != 1 or len(labels) != len(scores) or len(labels) == 0:
        raise ValueError("Labels and scores must be non-empty aligned vectors")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Both binary labels are required for these metrics")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Scores must be finite probabilities in [0, 1]")
    if not 0 < top_fraction <= 1:
        raise ValueError("top_fraction must be in (0, 1]")

    count = max(1, math.ceil(len(scores) * top_fraction))
    boundary_score = np.partition(scores, len(scores) - count)[len(scores) - count]
    above_boundary = scores > boundary_score
    tied_at_boundary = scores == boundary_score
    remaining_slots = count - int(above_boundary.sum())
    expected_positives = float(labels[above_boundary].sum()) + remaining_slots * float(
        labels[tied_at_boundary].mean()
    )
    positives = int(labels.sum())
    precision_at_k = expected_positives / count
    recall_at_k = expected_positives / positives
    prevalence = positives / len(labels)

    return {
        "average_precision": float(average_precision_score(labels, scores)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "brier": float(brier_score_loss(labels, scores)),
        "log_loss": float(log_loss(labels, scores, labels=[0, 1])),
        "precision_at_10pct": float(precision_at_k),
        "recall_at_10pct": float(recall_at_k),
        "lift_at_10pct": float(precision_at_k / prevalence),
    }
