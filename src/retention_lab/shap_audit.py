"""Verified Tree SHAP explanations for the uncalibrated forest candidate."""

import mlflow
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from retention_lab.config import ResearchConfig
from retention_lab.data import TARGET, DatasetError, load_snapshot, model_features
from retention_lab.models import ModelFactory
from retention_lab.splitting import make_split_plan, plan_summary, verify_saved_plan
from retention_lab.tracking import MLflowTracker, git_identity


def aggregate_to_original_features(
    values: NDArray[np.float64], encoded_names: list[str], original_names: list[str]
) -> NDArray[np.float64]:
    """Sum encoded-category contributions before taking absolute values."""
    if values.ndim != 2 or values.shape[1] != len(encoded_names):
        raise ValueError("SHAP array and encoded names do not align")
    source_names: list[str] = []
    for encoded_name in encoded_names:
        if encoded_name.startswith("numeric__"):
            source_name = encoded_name.removeprefix("numeric__")
        elif encoded_name.startswith("categorical__tariff_plan_"):
            source_name = "tariff_plan"
        else:
            raise ValueError(f"Unknown encoded feature: {encoded_name}")
        if source_name not in original_names:
            raise ValueError(f"Encoded feature has no original column: {encoded_name}")
        source_names.append(source_name)
    if set(source_names) != set(original_names):
        raise ValueError("At least one original feature is missing from encoded columns")
    return np.column_stack(
        [
            values[:, [i for i, name in enumerate(source_names) if name == raw]].sum(axis=1)
            for raw in original_names
        ]
    )


def run_shap_audit(
    config: ResearchConfig, *, background_rows: int = 64, explained_rows: int = 100
) -> dict[str, object]:
    """Explain raw forest P(churn=1), verify additivity, and export aggregate evidence."""
    if not 1 <= background_rows <= 100 or not 1 <= explained_rows <= 200:
        raise ValueError("background_rows must be 1..100 and explained_rows 1..200")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    frame, manifest = load_snapshot(config)
    plan = make_split_plan(frame, seed=config.seed)
    verify_saved_plan(plan, config.data_dir / "processed" / "split_plan.parquet")
    train_rows = plan.loc[plan["split"].eq("train"), "row_id"].to_numpy()
    validation_rows = plan.loc[plan["split"].eq("validation"), "row_id"].to_numpy()
    x_train = model_features(frame.iloc[train_rows]).reset_index(drop=True)
    x_validation = model_features(frame.iloc[validation_rows]).reset_index(drop=True)
    y_train = frame.iloc[train_rows][TARGET].to_numpy()
    estimator = ModelFactory.build("random_forest", seed=config.seed)
    estimator.fit(x_train, y_train)

    rng = np.random.default_rng(config.seed)
    background_indices = np.sort(
        rng.choice(len(x_train), size=min(background_rows, len(x_train)), replace=False)
    )
    explanation_indices = np.sort(
        rng.choice(len(x_validation), size=min(explained_rows, len(x_validation)), replace=False)
    )
    x_sample = x_validation.iloc[explanation_indices]
    preprocessor = estimator.named_steps["preprocess"]
    forest = estimator.named_steps["model"]
    encoded_background = np.asarray(
        preprocessor.transform(x_train.iloc[background_indices]), dtype=np.float64
    )
    encoded_sample = np.asarray(preprocessor.transform(x_sample), dtype=np.float64)
    encoded_names = list(preprocessor.get_feature_names_out())
    original_names = list(x_train.columns)
    explainer = shap.TreeExplainer(
        forest,
        data=encoded_background,
        feature_perturbation="interventional",
        model_output="probability",
    )
    raw_values = np.asarray(explainer.shap_values(encoded_sample), dtype=np.float64)
    classes = list(forest.classes_)
    if 1 not in classes or raw_values.shape != (len(x_sample), len(encoded_names), len(classes)):
        raise DatasetError(f"Unexpected SHAP output shape: {raw_values.shape}")
    positive_index = classes.index(1)
    values = raw_values[:, :, positive_index]
    expected_values = np.asarray(explainer.expected_value, dtype=np.float64)
    if expected_values.shape != (len(classes),):
        raise DatasetError(f"Unexpected SHAP base-value shape: {expected_values.shape}")
    base_value = float(expected_values[positive_index])
    original_values = aggregate_to_original_features(values, encoded_names, original_names)
    predictions = estimator.predict_proba(x_sample)[:, positive_index]
    forest_predictions = forest.predict_proba(encoded_sample)[:, positive_index]
    prediction_error = float(np.max(np.abs(predictions - forest_predictions)))
    additivity_error = float(np.max(np.abs(base_value + original_values.sum(axis=1) - predictions)))
    tolerance = 1e-4
    if prediction_error > tolerance or additivity_error > tolerance:
        raise DatasetError(
            f"SHAP reconstruction failed: prediction={prediction_error:.6g}, "
            f"additivity={additivity_error:.6g}"
        )

    summary = pd.DataFrame(
        {
            "feature": original_names,
            "mean_abs_probability_contribution": np.abs(original_values).mean(axis=0),
            "mean_signed_probability_contribution": original_values.mean(axis=0),
            "explained_validation_rows": len(x_sample),
            "background_train_rows": len(background_indices),
            "max_abs_additivity_error": additivity_error,
        }
    ).sort_values("mean_abs_probability_contribution", ascending=False, kind="stable")
    summary = summary.reset_index(drop=True)
    destination = config.report_dir / "rf_shap_summary.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(destination, index=False)

    figure_path = config.report_dir / "figures" / "rf_shap_global.png"
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    ordered = summary.iloc[::-1]
    fig, axis = plt.subplots(figsize=(8.5, 5.5))
    colors = ["#bc6c25" if name == "complains" else "#315f8a" for name in ordered["feature"]]
    axis.barh(ordered["feature"], ordered["mean_abs_probability_contribution"], color=colors)
    axis.set_xlabel("Mean absolute SHAP contribution to P(churn=1)")
    axis.set_title("Random Forest feature contributions")
    axis.grid(axis="x", color="#e5e7eb", linewidth=0.8)
    axis.set_axisbelow(True)
    fig.text(
        0.12,
        0.01,
        f"{len(x_sample)} validation rows · {len(background_indices)} train background rows",
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(figure_path, dpi=160)
    plt.close(fig)

    tracker = MLflowTracker(database=config.tracking_db, artifact_dir=config.artifact_dir)
    tracker.configure("retention-offline")
    git_sha, git_dirty = git_identity()
    with mlflow.start_run(run_name="random_forest_tree_shap"):
        mlflow.set_tags(
            {
                "study_phase": "feature_audit",
                "explanation_target": "uncalibrated_random_forest_probability_class_1",
                "dataset_sha256": manifest.sha256,
                "split_sha256": str(plan_summary(plan)["split_sha256"]),
                "git_sha": git_sha,
                "git_dirty": str(git_dirty).lower(),
            }
        )
        mlflow.log_params(
            {
                "background_rows": len(background_indices),
                "explained_rows": len(x_sample),
                "feature_perturbation": "interventional",
                "model_output": "probability",
            }
        )
        mlflow.log_metrics(
            {
                "shap/base_probability": base_value,
                "shap/max_abs_additivity_error": additivity_error,
                "shap/max_abs_preprocessing_error": prediction_error,
            }
        )
        mlflow.log_artifact(str(destination))
        mlflow.log_artifact(str(figure_path))
    return {
        "base_probability": base_value,
        "explained_validation_rows": len(x_sample),
        "background_train_rows": len(background_indices),
        "max_abs_additivity_error": additivity_error,
        "max_abs_preprocessing_error": prediction_error,
        "top_features": summary.head(5).to_dict(orient="records"),
        "summary_path": str(destination),
        "figure_path": str(figure_path),
    }
