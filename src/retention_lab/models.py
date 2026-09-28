"""Unfitted model factories with preprocessing inside each CV fold."""

from collections.abc import Callable

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from retention_lab.data import EXCLUDED_FEATURES, SOURCE_COLUMNS

MODEL_NAMES = ("dummy", "logistic", "random_forest")
CATEGORICAL = ("tariff_plan",)
NUMERIC = tuple(
    name
    for name in SOURCE_COLUMNS.values()
    if name not in EXCLUDED_FEATURES and name not in CATEGORICAL
)


def _preprocessor(*, scale_numeric: bool, excluded_features: frozenset[str]) -> ColumnTransformer:
    numeric_steps: list[tuple[str, object]] = [("impute", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scale", StandardScaler()))
    categorical_steps: list[tuple[str, object]] = [
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(numeric_steps),
                [name for name in NUMERIC if name not in excluded_features],
            ),
            (
                "categorical",
                Pipeline(categorical_steps),
                [name for name in CATEGORICAL if name not in excluded_features],
            ),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )


class ModelFactory:
    """Allowlisted construction; each call produces a fresh sklearn pipeline."""

    @staticmethod
    def build(name: str, *, seed: int, excluded_features: frozenset[str] = frozenset()) -> Pipeline:
        unknown = excluded_features - set(NUMERIC) - set(CATEGORICAL)
        if unknown:
            raise ValueError(f"Unknown excluded features: {sorted(unknown)}")
        if len(excluded_features) == len(NUMERIC) + len(CATEGORICAL):
            raise ValueError("At least one feature is required")
        builders: dict[str, Callable[[], Pipeline]] = {
            "dummy": lambda: Pipeline([("model", DummyClassifier(strategy="prior"))]),
            "logistic": lambda: Pipeline(
                [
                    (
                        "preprocess",
                        _preprocessor(scale_numeric=True, excluded_features=excluded_features),
                    ),
                    ("model", LogisticRegression(C=1.0, max_iter=1000, random_state=seed)),
                ]
            ),
            "random_forest": lambda: Pipeline(
                [
                    (
                        "preprocess",
                        _preprocessor(scale_numeric=False, excluded_features=excluded_features),
                    ),
                    (
                        "model",
                        RandomForestClassifier(
                            n_estimators=200,
                            max_depth=8,
                            min_samples_leaf=3,
                            n_jobs=2,
                            random_state=seed,
                        ),
                    ),
                ]
            ),
        }
        try:
            return builders[name]()
        except KeyError as error:
            raise ValueError(f"Unknown model {name!r}; choose from {MODEL_NAMES}") from error
