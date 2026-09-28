import numpy as np
import pytest

from retention_lab.metrics import classification_metrics


def test_fixed_budget_metrics_match_manual_count() -> None:
    labels = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 0], dtype=np.int64)
    scores = np.array([0.9, 0.8, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    metrics = classification_metrics(labels, scores)
    assert metrics["precision_at_10pct"] == 1.0
    assert metrics["recall_at_10pct"] == 0.5
    assert metrics["lift_at_10pct"] == 5.0


def test_scores_outside_probability_range_are_rejected() -> None:
    with pytest.raises(ValueError, match="probabilities"):
        classification_metrics(np.array([0, 1]), np.array([0.2, 1.1]))


def test_tied_scores_have_order_independent_expected_precision() -> None:
    labels = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int64)
    scores = np.full(10, 0.1)
    metrics = classification_metrics(labels, scores)
    assert metrics["precision_at_10pct"] == pytest.approx(0.1)
    assert metrics["recall_at_10pct"] == pytest.approx(0.1)
    assert metrics["lift_at_10pct"] == pytest.approx(1.0)
    assert classification_metrics(labels[::-1].copy(), scores) == pytest.approx(metrics)
