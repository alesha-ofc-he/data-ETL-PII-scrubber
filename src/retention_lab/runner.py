"""Baseline experiment: one split, shared folds, tracked candidate models."""

import json
from dataclasses import dataclass
from time import perf_counter

import mlflow
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from retention_lab.config import ResearchConfig
from retention_lab.data import TARGET, DatasetError, load_snapshot, model_features
from retention_lab.metrics import classification_metrics
from retention_lab.models import MODEL_NAMES, ModelFactory
from retention_lab.splitting import make_split_plan, plan_summary, verify_saved_plan
from retention_lab.tracking import MLflowTracker, git_identity


@dataclass(frozen=True)
class CandidateResult:
    name: str
    run_id: str
    cv_average_precision_mean: float
    cv_average_precision_std: float
    validation_average_precision: float
    validation_brier: float
    validation_precision_at_10pct: float
    fit_seconds: float


def run_baselines(config: ResearchConfig) -> list[CandidateResult]:
    """Fit only on train, compare on validation, leave calibration/test untouched."""
    frame, manifest = load_snapshot(config)
    plan = make_split_plan(frame, seed=config.seed)
    verify_saved_plan(plan, config.data_dir / "processed" / "split_plan.parquet")
    train_rows = plan.loc[plan["split"].eq("train"), "row_id"].to_numpy()
    validation_rows = plan.loc[plan["split"].eq("validation"), "row_id"].to_numpy()
    x_train = model_features(frame.iloc[train_rows]).reset_index(drop=True)
    x_validation = model_features(frame.iloc[validation_rows]).reset_index(drop=True)
    y_train = frame.iloc[train_rows][TARGET].to_numpy(dtype=np.int64)
    y_validation = frame.iloc[validation_rows][TARGET].to_numpy(dtype=np.int64)
    groups = plan.iloc[train_rows]["duplicate_group"].to_numpy()
    folds = list(
        StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=config.seed).split(
            x_train, y_train, groups
        )
    )

    tracker = MLflowTracker(database=config.tracking_db, artifact_dir=config.artifact_dir)
    tracker.configure("retention-offline")
    git_sha, git_dirty = git_identity()
    results: list[CandidateResult] = []
    with mlflow.start_run(run_name="baseline-comparison") as parent:
        mlflow.set_tags(
            {
                "dataset_sha256": manifest.sha256,
                "split_sha256": str(plan_summary(plan)["split_sha256"]),
                "git_sha": git_sha,
                "git_dirty": str(git_dirty).lower(),
                "study_phase": "baselines",
            }
        )
        mlflow.log_param("seed", config.seed)
        mlflow.log_param("cv_folds", len(folds))
        mlflow.log_text(json.dumps(plan_summary(plan), indent=2), "split_summary.json")

        for name in MODEL_NAMES:
            with mlflow.start_run(run_name=name, nested=True) as child:
                mlflow.log_param("model_name", name)
                mlflow.log_param("seed", config.seed)
                fold_ap: list[float] = []
                for fold_number, (fold_train, fold_valid) in enumerate(folds):
                    if set(groups[fold_train]) & set(groups[fold_valid]):
                        raise DatasetError("Duplicate group crossed a CV fold")
                    estimator = ModelFactory.build(name, seed=config.seed + fold_number)
                    estimator.fit(x_train.iloc[fold_train], y_train[fold_train])
                    scores = estimator.predict_proba(x_train.iloc[fold_valid])[:, 1]
                    metrics = classification_metrics(y_train[fold_valid], scores)
                    fold_ap.append(metrics["average_precision"])
                    mlflow.log_metric(
                        "cv/average_precision", metrics["average_precision"], step=fold_number
                    )

                estimator = ModelFactory.build(name, seed=config.seed)
                started = perf_counter()
                estimator.fit(x_train, y_train)
                fit_seconds = perf_counter() - started
                validation_scores = estimator.predict_proba(x_validation)[:, 1]
                validation = classification_metrics(y_validation, validation_scores)
                cv_mean = float(np.mean(fold_ap))
                cv_std = float(np.std(fold_ap, ddof=1))
                mlflow.log_metrics(
                    {
                        "cv/average_precision_mean": cv_mean,
                        "cv/average_precision_std": cv_std,
                        "fit_seconds": fit_seconds,
                        **{f"validation/{key}": value for key, value in validation.items()},
                    }
                )
                results.append(
                    CandidateResult(
                        name=name,
                        run_id=child.info.run_id,
                        cv_average_precision_mean=cv_mean,
                        cv_average_precision_std=cv_std,
                        validation_average_precision=validation["average_precision"],
                        validation_brier=validation["brier"],
                        validation_precision_at_10pct=validation["precision_at_10pct"],
                        fit_seconds=fit_seconds,
                    )
                )
        mlflow.set_tag("candidate_count", str(len(results)))
        mlflow.set_tag("parent_run_id", parent.info.run_id)

    destination = config.report_dir / "candidate_comparison.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([result.__dict__ for result in results]).to_csv(destination, index=False)
    return results


def run_complaint_ablation(config: ResearchConfig) -> list[CandidateResult]:
    """Compare one model with/without complaints on leakage-safe shared rows."""
    frame, manifest = load_snapshot(config)
    plan = make_split_plan(frame, seed=config.seed)
    verify_saved_plan(plan, config.data_dir / "processed" / "split_plan.parquet")
    train_rows = plan.loc[plan["split"].eq("train"), "row_id"].to_numpy()
    validation_rows = plan.loc[plan["split"].eq("validation"), "row_id"].to_numpy()

    # Omitting a feature can make formerly distinct rows identical. Build CV groups
    # from the reduced feature set and omit validation rows duplicated in train.
    reduced = model_features(frame).drop(columns=["complains"])
    reduced_groups = pd.util.hash_pandas_object(reduced, index=False)
    train_groups = reduced_groups.iloc[train_rows].to_numpy()
    validation_groups = reduced_groups.iloc[validation_rows].to_numpy()
    overlaps_train = np.isin(validation_groups, train_groups)
    validation_rows = validation_rows[~overlaps_train]
    if len(validation_rows) == 0:
        raise DatasetError("No non-duplicate validation rows remain for ablation")

    x_train = model_features(frame.iloc[train_rows]).reset_index(drop=True)
    x_validation = model_features(frame.iloc[validation_rows]).reset_index(drop=True)
    y_train = frame.iloc[train_rows][TARGET].to_numpy(dtype=np.int64)
    y_validation = frame.iloc[validation_rows][TARGET].to_numpy(dtype=np.int64)
    folds = list(
        StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=config.seed).split(
            x_train, y_train, train_groups
        )
    )

    tracker = MLflowTracker(database=config.tracking_db, artifact_dir=config.artifact_dir)
    tracker.configure("retention-offline")
    git_sha, git_dirty = git_identity()
    results: list[CandidateResult] = []
    with mlflow.start_run(run_name="complaints-ablation"):
        mlflow.set_tags(
            {
                "dataset_sha256": manifest.sha256,
                "split_sha256": str(plan_summary(plan)["split_sha256"]),
                "git_sha": git_sha,
                "git_dirty": str(git_dirty).lower(),
                "study_phase": "feature_ablation",
            }
        )
        mlflow.log_params(
            {
                "seed": config.seed,
                "validation_rows_compared": len(validation_rows),
                "validation_rows_excluded_for_reduced_feature_duplicates": int(
                    overlaps_train.sum()
                ),
            }
        )
        for variant, excluded_features in (
            ("all_features", frozenset()),
            ("without_complaints", frozenset({"complains"})),
        ):
            with mlflow.start_run(run_name=f"random_forest_{variant}", nested=True) as child:
                mlflow.log_param("excluded_features", ",".join(sorted(excluded_features)))
                fold_ap: list[float] = []
                for fold_number, (fold_train, fold_valid) in enumerate(folds):
                    if set(train_groups[fold_train]) & set(train_groups[fold_valid]):
                        raise DatasetError("Reduced-feature duplicate group crossed a CV fold")
                    estimator = ModelFactory.build(
                        "random_forest",
                        seed=config.seed + fold_number,
                        excluded_features=excluded_features,
                    )
                    estimator.fit(x_train.iloc[fold_train], y_train[fold_train])
                    scores = estimator.predict_proba(x_train.iloc[fold_valid])[:, 1]
                    fold_ap.append(
                        classification_metrics(y_train[fold_valid], scores)["average_precision"]
                    )
                estimator = ModelFactory.build(
                    "random_forest", seed=config.seed, excluded_features=excluded_features
                )
                started = perf_counter()
                estimator.fit(x_train, y_train)
                fit_seconds = perf_counter() - started
                scores = estimator.predict_proba(x_validation)[:, 1]
                validation = classification_metrics(y_validation, scores)
                cv_mean = float(np.mean(fold_ap))
                cv_std = float(np.std(fold_ap, ddof=1))
                mlflow.log_metrics(
                    {
                        "cv/average_precision_mean": cv_mean,
                        "cv/average_precision_std": cv_std,
                        "validation/average_precision": validation["average_precision"],
                        "validation/brier": validation["brier"],
                        "validation/precision_at_10pct": validation["precision_at_10pct"],
                        "fit_seconds": fit_seconds,
                    }
                )
                results.append(
                    CandidateResult(
                        name=variant,
                        run_id=child.info.run_id,
                        cv_average_precision_mean=cv_mean,
                        cv_average_precision_std=cv_std,
                        validation_average_precision=validation["average_precision"],
                        validation_brier=validation["brier"],
                        validation_precision_at_10pct=validation["precision_at_10pct"],
                        fit_seconds=fit_seconds,
                    )
                )

    destination = config.report_dir / "complaints_ablation.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([result.__dict__ for result in results]).assign(
        validation_rows=len(y_validation),
        excluded_validation_rows=int(overlaps_train.sum()),
    ).to_csv(destination, index=False)
    return results
