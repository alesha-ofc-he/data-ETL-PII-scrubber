import pandas as pd
import pytest

from retention_lab.data import SOURCE_COLUMNS, model_features
from retention_lab.models import MODEL_NAMES, ModelFactory


def test_factory_returns_fresh_pipelines_that_accept_valid_features() -> None:
    records = []
    for index in range(12):
        record = {name: 1 for name in SOURCE_COLUMNS.values()}
        record["seconds_of_use"] = 100 + index
        record["tariff_plan"] = 1 + index % 2
        record["churn"] = index % 2
        records.append(record)
    frame = pd.DataFrame(records)
    features = model_features(frame)

    for name in MODEL_NAMES:
        first = ModelFactory.build(name, seed=7)
        second = ModelFactory.build(name, seed=7)
        assert first is not second
        first.fit(features, frame["churn"])
        probabilities = first.predict_proba(features)
        assert probabilities.shape == (12, 2)
        assert (probabilities >= 0).all()
        assert (probabilities <= 1).all()


def test_factory_rejects_unknown_model() -> None:
    with pytest.raises(ValueError, match="Unknown model"):
        ModelFactory.build("from_config_import", seed=7)


def test_factory_can_exclude_a_questionable_feature() -> None:
    pipeline = ModelFactory.build(
        "random_forest", seed=7, excluded_features=frozenset({"complains"})
    )
    numeric_columns = pipeline.named_steps["preprocess"].transformers[0][2]
    assert "complains" not in numeric_columns
    with pytest.raises(ValueError, match="Unknown excluded features"):
        ModelFactory.build("random_forest", seed=7, excluded_features=frozenset({"churn"}))
