"""Leakage-aware development splits for a non-temporal source dataset."""

import json
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from retention_lab.data import TARGET, DatasetError, model_features

SPLIT_NAMES = ("train", "calibration", "validation", "test")


def duplicate_groups(frame: pd.DataFrame) -> pd.Series:
    """Group identical eligible features without looking at labels."""
    features = model_features(frame)
    keys = pd.util.hash_pandas_object(features, index=False)
    return pd.Series(keys.to_numpy(), index=frame.index, name="duplicate_group")


def make_split_plan(frame: pd.DataFrame, *, seed: int) -> pd.DataFrame:
    """Create deterministic 60/10/15/15 group-disjoint stratified splits."""
    if len(frame) < 100:
        raise DatasetError("At least 100 rows required for four research splits")
    groups = duplicate_groups(frame)
    if groups.nunique() < 20:
        raise DatasetError("At least 20 distinct feature groups required")
    splitter = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=seed)
    fold_number = np.full(len(frame), -1, dtype=np.int8)
    features = model_features(frame)
    for number, (_, fold_indices) in enumerate(
        splitter.split(features, frame[TARGET], groups=groups)
    ):
        fold_number[fold_indices] = number
    if (fold_number < 0).any():
        raise DatasetError("A row was not assigned to a fold")
    names = np.empty(len(frame), dtype=object)
    for first, last, name in (
        (0, 12, "train"),
        (12, 14, "calibration"),
        (14, 17, "validation"),
        (17, 20, "test"),
    ):
        names[(fold_number >= first) & (fold_number < last)] = name
    plan = pd.DataFrame(
        {
            "row_id": np.arange(len(frame), dtype=np.int64),
            "duplicate_group": groups.to_numpy(),
            "fold": fold_number,
            "split": names,
            TARGET: frame[TARGET].to_numpy(),
        }
    )
    for name in SPLIT_NAMES:
        subset = plan.loc[plan["split"] == name]
        if subset.empty or subset[TARGET].nunique() != 2:
            raise DatasetError(f"Split {name} lacks a binary outcome; change seed or source")
    if plan.groupby("duplicate_group")["split"].nunique().max() != 1:
        raise DatasetError("Identical features crossed split boundaries")
    return plan


def plan_summary(plan: pd.DataFrame) -> dict[str, object]:
    """Public metadata only; row-level assignments stay under ignored data/."""
    rows = plan.sort_values("row_id")["split"].tolist()
    digest = sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()
    counts = {
        name: {
            "rows": int((segment := plan.loc[plan["split"] == name]).shape[0]),
            "positives": int(segment[TARGET].sum()),
            "prevalence": float(segment[TARGET].mean()),
            "groups": int(segment["duplicate_group"].nunique()),
        }
        for name in SPLIT_NAMES
    }
    return {"split_sha256": digest, "counts": counts}


def verify_saved_plan(plan: pd.DataFrame, path: Path) -> None:
    """Refuse to train when the cached partition differs from the recomputed plan."""
    if not path.exists():
        raise DatasetError("Saved split plan missing; run `retention-lab split-data` first")
    saved = pd.read_parquet(path)
    for column in ("row_id", "duplicate_group", "fold", "split", TARGET):
        if column not in saved or not saved[column].equals(plan[column]):
            raise DatasetError(f"Saved split plan differs in {column}; regenerate and audit it")
