"""Fetch and validate the original UCI Iranian Churn snapshot."""

import json
import os
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from ucimlrepo import fetch_ucirepo

from retention_lab.config import ResearchConfig

SOURCE_URL = "https://archive.ics.uci.edu/dataset/563/iranian%2Bchurn%2Bdataset"
SOURCE_COLUMNS = {
    "Call Failure": "call_failure",
    "Complains": "complains",
    "Subscription Length": "subscription_length",
    "Charge Amount": "charge_amount",
    "Seconds of Use": "seconds_of_use",
    "Frequency of use": "frequency_of_use",
    "Frequency of SMS": "frequency_of_sms",
    "Distinct Called Numbers": "distinct_called_numbers",
    "Age Group": "age_group",
    "Tariff Plan": "tariff_plan",
    "Status": "status",
    "Age": "age",
    "Customer Value": "customer_value",
}
EXCLUDED_FEATURES = frozenset({"status", "customer_value"})
TARGET = "churn"


class DatasetManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_id: int
    source_url: str
    license_name: str
    fetched_at_utc: datetime
    sha256: str
    row_count: int = Field(ge=1)
    source_columns: list[str]
    columns: list[str]
    positive_count: int = Field(ge=0)
    excluded_features: list[str]


class DatasetError(ValueError):
    """An input snapshot is unusable for the declared research design."""


def _normalize_source_table(features: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(features, pd.DataFrame) or not isinstance(targets, pd.DataFrame):
        raise DatasetError("UCI did not return pandas DataFrames")
    if len(targets.columns) != 1 or targets.columns[0].strip().lower() != TARGET:
        raise DatasetError("UCI target column changed: expected Churn")
    if len(features) != len(targets):
        raise DatasetError("Feature and target row counts differ")

    source_names = [" ".join(str(name).split()) for name in features.columns]
    if len(source_names) != len(set(source_names)):
        raise DatasetError("Source columns collide after whitespace normalization")
    unknown = set(source_names) - SOURCE_COLUMNS.keys()
    missing = SOURCE_COLUMNS.keys() - set(source_names)
    if unknown or missing:
        raise DatasetError(
            f"UCI schema changed: unknown={sorted(unknown)}, missing={sorted(missing)}"
        )
    normalized = features.copy()
    normalized.columns = source_names
    normalized = normalized.rename(columns=SOURCE_COLUMNS).reset_index(drop=True)
    normalized[TARGET] = targets.iloc[:, 0].reset_index(drop=True)
    return validate_snapshot(normalized)


def validate_snapshot(frame: pd.DataFrame, *, expected_min_rows: int = 1) -> pd.DataFrame:
    """Return a validated copy; never mutate a caller-owned DataFrame."""
    expected = set(SOURCE_COLUMNS.values()) | {TARGET}
    unknown = set(frame.columns) - expected
    missing = expected - set(frame.columns)
    if unknown or missing:
        raise DatasetError(
            f"Unexpected columns: unknown={sorted(unknown)}, missing={sorted(missing)}"
        )
    if len(frame) < expected_min_rows:
        raise DatasetError(f"Snapshot has {len(frame)} rows; expected at least {expected_min_rows}")
    if frame.columns.duplicated().any():
        raise DatasetError("Duplicate column names")

    clean = frame.copy()
    for column in clean.columns:
        clean[column] = pd.to_numeric(clean[column], errors="raise")
        if clean[column].isna().any():
            raise DatasetError(f"Missing values in {column}")
        if not np.isfinite(clean[column].to_numpy(dtype=float)).all():
            raise DatasetError(f"Non-finite values in {column}")
    if not clean[TARGET].isin((0, 1)).all() or clean[TARGET].nunique() != 2:
        raise DatasetError("Churn must contain both binary classes 0 and 1")
    for column, allowed in {
        "complains": {0, 1},
        "tariff_plan": {1, 2},
        "status": {1, 2},
        "age_group": {1, 2, 3, 4, 5},
    }.items():
        if not clean[column].isin(allowed).all():
            raise DatasetError(f"Unexpected value in {column}")
    for column in ("call_failure", "subscription_length", "seconds_of_use", "frequency_of_use"):
        if (clean[column] < 0).any():
            raise DatasetError(f"Negative values in {column}")
    clean[TARGET] = clean[TARGET].astype("int8")
    return clean


def model_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Select only features plausibly available before the churn outcome."""
    columns = [name for name in SOURCE_COLUMNS.values() if name not in EXCLUDED_FEATURES]
    return frame.loc[:, columns].copy()


def snapshot_paths(data_dir: Path) -> tuple[Path, Path]:
    raw_dir = data_dir / "raw"
    return raw_dir / "iranian_churn.parquet", raw_dir / "dataset_manifest.json"


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def fetch_snapshot(config: ResearchConfig) -> DatasetManifest:
    """Fetch once, validate, and save a local snapshot for network-free runs."""
    if config.dataset_id != 563:
        raise DatasetError("Only vetted UCI dataset 563 is supported")
    dataset = fetch_ucirepo(id=config.dataset_id)
    frame = _normalize_source_table(dataset.data.features, dataset.data.targets)
    frame = validate_snapshot(frame, expected_min_rows=config.expected_min_rows)
    snapshot, manifest_path = snapshot_paths(config.data_dir)
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    temporary = snapshot.with_name(f".{snapshot.name}.{uuid4().hex}.tmp")
    try:
        frame.to_parquet(temporary, index=False)
        digest = sha256(temporary.read_bytes()).hexdigest()
        os.replace(temporary, snapshot)
    finally:
        temporary.unlink(missing_ok=True)
    manifest = DatasetManifest(
        dataset_id=config.dataset_id,
        source_url=SOURCE_URL,
        license_name="CC BY 4.0",
        fetched_at_utc=datetime.now(UTC),
        sha256=digest,
        row_count=len(frame),
        source_columns=[str(name) for name in dataset.data.features.columns],
        columns=frame.columns.tolist(),
        positive_count=int(frame[TARGET].sum()),
        excluded_features=sorted(EXCLUDED_FEATURES),
    )
    _atomic_json(manifest_path, manifest.model_dump(mode="json"))
    return manifest


def load_snapshot(config: ResearchConfig) -> tuple[pd.DataFrame, DatasetManifest]:
    snapshot, manifest_path = snapshot_paths(config.data_dir)
    if not snapshot.exists() or not manifest_path.exists():
        raise DatasetError("Snapshot missing; run `retention-lab fetch-data` first")
    manifest = DatasetManifest.model_validate_json(manifest_path.read_text())
    if sha256(snapshot.read_bytes()).hexdigest() != manifest.sha256:
        raise DatasetError("Snapshot SHA-256 does not match manifest")
    frame = validate_snapshot(pd.read_parquet(snapshot), expected_min_rows=config.expected_min_rows)
    if len(frame) != manifest.row_count or frame.columns.tolist() != manifest.columns:
        raise DatasetError("Snapshot content differs from manifest")
    return frame, manifest
