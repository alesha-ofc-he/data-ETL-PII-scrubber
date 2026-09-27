import pandas as pd

from retention_lab.data import SOURCE_COLUMNS
from retention_lab.splitting import make_split_plan, plan_summary


def test_identical_features_never_cross_splits() -> None:
    records = []
    for number in range(100):
        record = {name: 1 for name in SOURCE_COLUMNS.values()}
        record["seconds_of_use"] = number
        record["complains"] = number % 2
        record["tariff_plan"] = 1 + number % 2
        record["status"] = 1
        record["age_group"] = 1 + number % 5
        record["churn"] = int(number % 5 == 0)
        records.extend([record.copy(), record.copy()])

    frame = pd.DataFrame(records)
    first = make_split_plan(frame, seed=42)
    second = make_split_plan(frame, seed=42)
    assert first["split"].tolist() == second["split"].tolist()
    assert first.groupby("duplicate_group")["split"].nunique().max() == 1
    assert first["row_id"].is_unique
    assert set(first["split"]) == {"train", "calibration", "validation", "test"}
    assert plan_summary(first)["split_sha256"] == plan_summary(second)["split_sha256"]
