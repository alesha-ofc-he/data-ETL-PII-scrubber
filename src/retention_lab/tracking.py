"""Local MLflow setup for comparable, named experiment runs."""

from pathlib import Path

import mlflow


class MLflowTracker:
    def __init__(self, *, database: Path, artifact_dir: Path) -> None:
        self.database = database.resolve()
        self.artifact_dir = artifact_dir.resolve()

    def configure(self, experiment_name: str) -> None:
        self.database.parent.mkdir(parents=True, exist_ok=True)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(f"sqlite:///{self.database}")
        if mlflow.get_experiment_by_name(experiment_name) is None:
            mlflow.create_experiment(
                experiment_name,
                artifact_location=self.artifact_dir.as_uri(),
            )
        mlflow.set_experiment(experiment_name)
