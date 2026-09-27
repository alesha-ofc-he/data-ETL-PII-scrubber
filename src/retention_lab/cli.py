"""Small command line interface for repeatable data research."""

import json
from pathlib import Path
from typing import Annotated

import typer

from retention_lab.config import read_config
from retention_lab.data import DatasetError, fetch_snapshot, load_snapshot
from retention_lab.splitting import make_split_plan, plan_summary

app = typer.Typer(help="Retention ML research commands", no_args_is_help=True)


@app.command("fetch-data")
def fetch_data(
    config: Annotated[Path, typer.Option()] = Path("configs/full.toml"),
) -> None:
    """Download and validate the UCI source once."""
    try:
        manifest = fetch_snapshot(read_config(config))
    except (DatasetError, OSError, ValueError) as error:
        typer.echo(f"Data fetch failed: {error}", err=True)
        raise typer.Exit(code=1) from error
    typer.echo(manifest.model_dump_json(indent=2))


@app.command("validate-data")
def validate_data(
    config: Annotated[Path, typer.Option()] = Path("configs/full.toml"),
) -> None:
    """Check a cached snapshot and report source facts."""
    try:
        frame, manifest = load_snapshot(read_config(config))
    except (DatasetError, OSError, ValueError) as error:
        typer.echo(f"Validation failed: {error}", err=True)
        raise typer.Exit(code=1) from error
    typer.echo(
        json.dumps(
            {
                "source": manifest.source_url,
                "sha256": manifest.sha256,
                "rows": len(frame),
                "positive_count": int(frame["churn"].sum()),
                "excluded_features": manifest.excluded_features,
            },
            indent=2,
        )
    )


@app.command("split-data")
def split_data(
    config: Annotated[Path, typer.Option()] = Path("configs/full.toml"),
) -> None:
    """Cache a deterministic plan without exposing row-level data in Git."""
    try:
        settings = read_config(config)
        frame, _ = load_snapshot(settings)
        plan = make_split_plan(frame, seed=settings.seed)
        destination = settings.data_dir / "processed" / "split_plan.parquet"
        destination.parent.mkdir(parents=True, exist_ok=True)
        plan.to_parquet(destination, index=False)
    except (DatasetError, OSError, ValueError) as error:
        typer.echo(f"Split failed: {error}", err=True)
        raise typer.Exit(code=1) from error
    typer.echo(json.dumps(plan_summary(plan), indent=2))
