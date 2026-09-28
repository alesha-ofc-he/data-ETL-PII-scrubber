import numpy as np
import pytest

from retention_lab.shap_audit import aggregate_to_original_features


def test_one_hot_shap_values_are_summed_with_sign_preserved() -> None:
    values = np.array([[0.2, 0.4, -0.1], [-0.3, 0.1, 0.2]], dtype=np.float64)
    aggregated = aggregate_to_original_features(
        values,
        ["numeric__age", "categorical__tariff_plan_1", "categorical__tariff_plan_2"],
        ["age", "tariff_plan"],
    )
    np.testing.assert_allclose(aggregated, [[0.2, 0.3], [-0.3, 0.3]])
    np.testing.assert_allclose(aggregated.sum(axis=1), values.sum(axis=1), atol=1e-15)


def test_unmapped_encoded_feature_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown encoded feature"):
        aggregate_to_original_features(np.array([[0.1]]), ["categorical__other_1"], ["other"])
