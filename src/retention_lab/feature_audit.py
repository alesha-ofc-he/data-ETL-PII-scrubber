"""Held-out permutation audit of the current Random Forest candidate."""

import mlflow
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score

from retention_lab.config import ResearchConfig
from retention_lab.data import TARGET, load_snapshot, model_features
from retention_lab.models import ModelFactory
from retention_lab.splitting import make_split_plan, plan_summary, verify_saved_plan
from retention_lab.tracking import MLflowTracker, git_identity


def run_permutation_audit(config: ResearchConfig, *, repeats: int = 10) -> pd.DataFrame:
    """Measure AP loss when each original input column is shuffled on validation."""
    if not 1 <= repeats <= 100:
        raise ValueError("repeats must be between 1 and 100")
    frame, manifest = load_snapshot(config)
    plan = make_split_plan(frame, seed=config.seed)
    verify_saved_plan(plan, config.data_dir / "processed" / "split_plan.parquet")
    train_rows = plan.loc[plan["split"].eq("train"), "row_id"].to_numpy()
    validation_rows = plan.loc[plan["split"].eq("validation"), "row_id"].to_numpy()
    x_train = model_features(frame.iloc[train_rows]).reset_index(drop=True)
    x_validation = model_features(frame.iloc[validation_rows]).reset_index(drop=True)
    y_train = frame.iloc[train_rows][TARGET].to_numpy()
    y_validation = frame.iloc[validation_rows][TARGET].to_numpy()

    estimator = ModelFactory.build("random_forest", seed=config.seed)
    estimator.fit(x_train, y_train)
    validation_scores = estimator.predict_proba(x_validation)[:, 1]
    baseline_ap = float(average_precision_score(y_validation, validation_scores))
    importance = permutation_importance(
        estimator,
        x_validation,
        y_validation,
        scoring="average_precision",
        n_repeats=repeats,
        random_state=config.seed,
        n_jobs=1,
    )
    report = pd.DataFrame(
        {
            "feature": x_validation.columns,
            "mean_ap_drop": importance.importances_mean,
            "std_across_permutations": importance.importances_std,
            "baseline_ap": baseline_ap,
            "validation_rows": len(x_validation),
            "repeats": repeats,
        }
    ).sort_values("mean_ap_drop", ascending=False, kind="stable")
    report = report.reset_index(drop=True)
    destination = config.report_dir / "permutation_importance.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(destination, index=False)

    tracker = MLflowTracker(database=config.tracking_db, artifact_dir=config.artifact_dir)
    tracker.configure("retention-offline")
    git_sha, git_dirty = git_identity()
    with mlflow.start_run(run_name="random_forest_permutation_audit"):
        mlflow.set_tags(
            {
                "study_phase": "feature_audit",
                "dataset_sha256": manifest.sha256,
                "split_sha256": str(plan_summary(plan)["split_sha256"]),
                "git_sha": git_sha,
                "git_dirty": str(git_dirty).lower(),
            }
        )
        mlflow.log_params(
            {"model_name": "random_forest", "scoring": "average_precision", "repeats": repeats}
        )
        mlflow.log_metric("validation/average_precision", baseline_ap)
        mlflow.log_artifact(str(destination))
    return report
