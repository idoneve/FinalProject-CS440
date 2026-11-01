"""Train scikit-learn baselines for SpecMine's engineered features.

The released dataset has no AI-authorship labels. By default, this module
trains against :func:`specmine.features.derive_proxy_label`; resulting scores
measure prediction of that heuristic, not authorship accuracy.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from specmine.data import load_data, load_feature_data
from specmine.evaluate import evaluate_model
from specmine.features import build_features, derive_proxy_label

LOGGER = logging.getLogger(__name__)
ID_COLUMN = "file_url_sha16"
REPOSITORY_COLUMN = "repo_name"
LABEL_COLUMN = "label"
DEFAULT_MODELS = ("logistic_regression", "random_forest")


def prepare_training_data(
    data: pd.DataFrame, feature_data: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Build features, proxy labels, and repository groups for ``data``.

    ``feature_data`` may be supplied by callers that have already loaded
    ``spec_content_features``. Otherwise, matching rows are loaded using the
    IDs in ``data``.
    """
    if data is None or data.empty:
        raise ValueError("Training data must be a non-empty DataFrame")
    missing = [
        column
        for column in (ID_COLUMN, REPOSITORY_COLUMN)
        if column not in data.columns
    ]
    if missing:
        raise KeyError(f"Training data is missing required columns: {missing}")

    if feature_data is None:
        LOGGER.info("Loading precomputed features for %d curated specs", len(data))
        feature_data = load_feature_data(data)
    if feature_data is None or feature_data.empty:
        raise ValueError("No precomputed feature rows were found for training")

    LOGGER.info("Engineering features for %d specs", len(feature_data))
    features = build_features(feature_data)
    repository_lookup = data[[ID_COLUMN, REPOSITORY_COLUMN]].drop_duplicates()
    if repository_lookup[ID_COLUMN].duplicated().any():
        raise ValueError("Training data maps a spec ID to multiple repositories")

    combined = features.merge(
        repository_lookup,
        on=ID_COLUMN,
        how="inner",
        validate="one_to_one",
    )
    if len(combined) < len(features):
        LOGGER.warning(
            "Dropped %d feature rows without matching curated repository metadata",
            len(features) - len(combined),
        )
    if combined.empty:
        raise ValueError("No feature rows matched the curated repository metadata")

    labels = derive_proxy_label(combined)
    groups = combined[REPOSITORY_COLUMN].copy()
    if groups.isna().any():
        raise ValueError("Every training spec must have a repository name")
    features = combined.drop(columns=REPOSITORY_COLUMN)
    LOGGER.info(
        "Prepared %d specs from %d repositories; proxy-label counts: %s",
        len(features),
        groups.nunique(),
        labels.value_counts().sort_index().to_dict(),
    )
    return features, labels, groups


def _build_estimators(seed: int) -> dict[str, Pipeline]:
    """Return the built-in estimator pipelines; extend here to add baselines."""
    return {
        "logistic_regression": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=200,
                        class_weight="balanced",
                        random_state=seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def _numeric_features(features: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    columns = [
        column
        for column in features.columns
        if column != ID_COLUMN and pd.api.types.is_numeric_dtype(features[column])
    ]
    if not columns:
        raise ValueError("Training features contain no numeric model columns")
    numeric = features.loc[:, columns].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.replace([np.inf, -np.inf], np.nan).astype(float)
    if not numeric.notna().to_numpy().any():
        raise ValueError("Training features contain no usable numeric values")
    return numeric, columns


def train_models(
    features: pd.DataFrame,
    labels: Sequence[Any] | pd.Series | None = None,
    repository_groups: Sequence[Any] | pd.Series | None = None,
    *,
    model_names: Sequence[str] = DEFAULT_MODELS,
    test_size: float = 0.2,
    seed: int = 42,
    model_dir: str | Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Fit selected baselines and score a repository-disjoint holdout.

    When ``labels`` is omitted, the existing proxy-label heuristic is applied
    to ``features``. ``repository_groups`` must identify the source repository
    for every row so no repository can appear in both partitions.
    """
    if features is None or not isinstance(features, pd.DataFrame) or features.empty:
        raise ValueError("features must be a non-empty pandas DataFrame")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    if repository_groups is None:
        raise ValueError("repository_groups are required for leakage-safe splitting")

    y = derive_proxy_label(features) if labels is None else pd.Series(labels)
    if len(y) != len(features):
        raise ValueError("labels must have the same number of rows as features")
    y = pd.to_numeric(y, errors="coerce")
    if y.isna().any() or not y.isin([0, 1]).all():
        raise ValueError("labels must contain only binary values 0 and 1")
    y = y.astype("int8").reset_index(drop=True)

    groups = pd.Series(repository_groups).reset_index(drop=True)
    if len(groups) != len(features):
        raise ValueError(
            "repository_groups must have the same number of rows as features"
        )
    if groups.isna().any():
        raise ValueError("repository_groups must not contain missing values")
    if groups.nunique() < 2:
        raise ValueError("At least two repositories are required for a holdout split")

    selected_names = tuple(model_names)
    estimators = _build_estimators(seed)
    unknown = sorted(set(selected_names) - set(estimators))
    if unknown:
        raise ValueError(
            f"Unknown model name(s) {unknown}; choose from {sorted(estimators)}"
        )
    if not selected_names:
        raise ValueError("Select at least one model to train")

    x, feature_columns = _numeric_features(features)
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=seed,
    )
    try:
        train_indices, test_indices = next(
            splitter.split(x, y, groups=groups.to_numpy())
        )
    except ValueError as exc:
        raise ValueError(
            "Unable to create a repository-disjoint holdout; adjust test_size "
            "or provide more repositories"
        ) from exc
    train_labels = y.iloc[train_indices]
    if train_labels.nunique() < 2:
        raise ValueError(
            "The repository-disjoint training split contains only one label class; "
            "use more curated repositories or choose a different seed/test_size"
        )

    x_train = x.iloc[train_indices]
    x_test = x.iloc[test_indices]
    y_train = train_labels
    y_test = y.iloc[test_indices]
    train_groups = groups.iloc[train_indices]
    test_groups = groups.iloc[test_indices]
    overlap = set(train_groups.unique()) & set(test_groups.unique())
    if overlap:
        raise RuntimeError("Repository leakage detected in the holdout split")

    LOGGER.info(
        "Split: %d training specs/%d repos, %d holdout specs/%d repos; "
        "features: %d; labels: train=%s holdout=%s",
        len(train_indices),
        train_groups.nunique(),
        len(test_indices),
        test_groups.nunique(),
        len(feature_columns),
        y_train.value_counts().sort_index().to_dict(),
        y_test.value_counts().sort_index().to_dict(),
    )
    LOGGER.warning(
        "Reported metrics evaluate the proxy-label heuristic, not AI-authorship accuracy"
    )

    destination = Path(model_dir) if model_dir is not None else None
    if destination is not None:
        destination.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict[str, Any]] = {}
    for name in selected_names:
        model = estimators[name]
        LOGGER.info("Training %s", name)
        model.fit(x_train, y_train)
        evaluation_frame = x_test.copy()
        if ID_COLUMN in features.columns:
            evaluation_frame.insert(
                0,
                ID_COLUMN,
                features.iloc[test_indices][ID_COLUMN].to_numpy(),
            )
        evaluation_frame[LABEL_COLUMN] = y_test.to_numpy()
        scored = evaluate_model(
            model,
            evaluation_frame,
        )
        model_path = None
        if destination is not None:
            model_path = destination / f"{name}.joblib"
            joblib.dump(model, model_path)
            LOGGER.info("Saved %s to %s", name, model_path)

        results[name] = {
            "model": model,
            "metrics": scored["metrics"],
            "predictions": scored["predictions"],
            "feature_columns": feature_columns,
            "n_train": len(train_indices),
            "n_test": len(test_indices),
            "n_train_repositories": int(train_groups.nunique()),
            "n_test_repositories": int(test_groups.nunique()),
            "train_repositories": sorted(train_groups.unique(), key=str),
            "test_repositories": sorted(test_groups.unique(), key=str),
            "model_path": model_path,
        }
        LOGGER.info("%s holdout proxy metrics: %s", name, scored["metrics"])
    return results


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and compare SpecMine scikit-learn baselines."
    )
    parser.add_argument("--sample-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=DEFAULT_MODELS,
        default=list(DEFAULT_MODELS),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Load curated data, train baselines, and return a CLI status code."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
        stream=sys.stdout,
    )
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.sample_size < 1:
        parser.error("--sample-size must be a positive integer")
    if not 0 < args.test_size < 1:
        parser.error("--test-size must be between 0 and 1")

    try:
        LOGGER.info("Loading curated SpecMine sample (up to %d specs)", args.sample_size)
        data = load_data(
            curated_only=True,
            sample_size=args.sample_size,
            seed=args.seed,
            data_dir=args.data_dir,
        )
        if data is None:
            raise RuntimeError("Curated SpecMine data loading failed")
        features, labels, groups = prepare_training_data(data)
        results = train_models(
            features,
            labels,
            groups,
            model_names=args.models,
            test_size=args.test_size,
            seed=args.seed,
            model_dir=args.model_dir,
        )
        for name, result in results.items():
            LOGGER.info(
                "%s complete: holdout=%d specs across %d repos; metrics=%s",
                name,
                result["n_test"],
                result["n_test_repositories"],
                result["metrics"],
            )
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        LOGGER.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
