"""Validated configuration for the data research stage."""

import tomllib
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class ResearchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data_dir: Path = Path("data")
    seed: int = Field(default=42, ge=0, le=2**32 - 1)
    dataset_id: int = Field(default=563, ge=1)
    expected_min_rows: int = Field(default=1000, ge=1)
    tracking_db: Path = Path("mlflow.db")
    artifact_dir: Path = Path("mlruns")
    report_dir: Path = Path("reports")


def read_config(path: Path) -> ResearchConfig:
    """Load a local TOML config with no implicit environment overrides."""
    with path.open("rb") as config_file:
        values = tomllib.load(config_file)
    return ResearchConfig.model_validate(values)
