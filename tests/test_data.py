import pandas as pd
import pytest

from retention_lab.data import (
    SOURCE_COLUMNS,
    DatasetError,
    _normalize_source_table,
    model_features,
    validate_snapshot,
)


def valid_frame() -> pd.DataFrame:
    values = {name: [1, 2] for name in SOURCE_COLUMNS.values()}
    values["complains"] = [0, 1]
    values["tariff_plan"] = [1, 2]
    values["status"] = [1, 2]
    values["age_group"] = [1, 2]
    values["churn"] = [0, 1]
    return pd.DataFrame(values)


def test_validate_snapshot_excludes_outcome_proxies_from_features() -> None:
    frame = validate_snapshot(valid_frame())
    features = model_features(frame)
    assert "churn" not in features.columns
    assert "status" not in features.columns
    assert "customer_value" not in features.columns


def test_validate_snapshot_rejects_schema_change() -> None:
    frame = valid_frame().drop(columns="call_failure")
    with pytest.raises(DatasetError, match="Unexpected columns"):
        validate_snapshot(frame)


def test_validate_snapshot_rejects_nonfinite_number() -> None:
    frame = valid_frame()
    frame["seconds_of_use"] = frame["seconds_of_use"].astype(float)
    frame.loc[0, "seconds_of_use"] = float("inf")
    with pytest.raises(DatasetError, match="Non-finite"):
        validate_snapshot(frame)


def test_source_headers_accept_extra_whitespace_without_hiding_collisions() -> None:
    frame = valid_frame()
    features = frame.drop(columns="churn").rename(
        columns={canonical: source for source, canonical in SOURCE_COLUMNS.items()}
    )
    features = features.rename(columns={"Call Failure": "Call  Failure"})
    target = frame[["churn"]].rename(columns={"churn": "Churn"})

    normalized = _normalize_source_table(features, target)
    assert "call_failure" in normalized.columns
    assert "Call  Failure" not in normalized.columns

    collision = features.copy()
    collision["Call Failure"] = collision["Call  Failure"]
    with pytest.raises(DatasetError, match="collide"):
        _normalize_source_table(collision, target)
