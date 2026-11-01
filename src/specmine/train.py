"""Train scikit-learn baselines for SpecMine's engineered features.

Training requires a labels CSV keyed by SpecMine's ``file_url_sha16`` IDs.
Scores measure agreement with the supplied labels; they are evidence of
authorship accuracy only when those labels are verified ground truth.
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
from specmine.features import build_features
from specmine.labels import LABEL_COLUMN, SPEC_ID_COLUMN, read_labels_csv

LOGGER = logging.getLogger(__name__)
ID_COLUMN = "file_url_sha16"
REPOSITORY_COLUMN = "repo_name"
DEFAULT_MODELS = ("logistic_regression", "random_forest")


def prepare_training_data(
    data: pd.DataFrame,
    labels: pd.DataFrame,
    feature_data: pd.DataFrame | None = None,
    *,
    sample_size: int | None = None,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Build features, supplied labels, and repository groups for labeled data.

    ``feature_data`` may be supplied by callers that have already loaded
    ``spec_content_features``. Otherwise, matching rows are loaded using the
    labeled IDs in ``data``. Blank labels are excluded, allowing ambiguous
    examples to be left unlabeled.
    """
    if data is None or data.empty:
        raise ValueError("Training data must be a non-empty DataFrame")
    if labels is None or not isinstance(labels, pd.DataFrame) or labels.empty:
        raise ValueError("A non-empty, validated labels DataFrame is required")
    if sample_size is not None and sample_size < 1:
        raise ValueError("sample_size must be a positive integer")
    missing = [
        column
        for column in (ID_COLUMN, REPOSITORY_COLUMN)
        if column not in data.columns
    ]
    if missing:
        raise KeyError(f"Training data is missing required columns: {missing}")
    missing_labels = [
        column for column in (SPEC_ID_COLUMN, LABEL_COLUMN) if column not in labels
    ]
    if missing_labels:
        raise KeyError(
            f"Labels DataFrame is missing required columns: {missing_labels}"
        )

    repository_lookup = data[[ID_COLUMN, REPOSITORY_COLUMN]].drop_duplicates()
    if repository_lookup[ID_COLUMN].duplicated().any():
        raise ValueError("Training data maps a spec ID to multiple repositories")
    if repository_lookup[ID_COLUMN].isna().any():
        raise ValueError("Training data contains missing spec IDs")
    if repository_lookup[REPOSITORY_COLUMN].isna().any():
        raise ValueError("Every training spec must have a repository name")
    repository_lookup[ID_COLUMN] = repository_lookup[ID_COLUMN].astype("string")

    label_rows = labels[[SPEC_ID_COLUMN, LABEL_COLUMN]].copy()
    if label_rows[SPEC_ID_COLUMN].isna().any():
        raise ValueError("Labels CSV spec_id values must not be missing")
    label_rows[SPEC_ID_COLUMN] = label_rows[SPEC_ID_COLUMN].astype("string")
    if label_rows[SPEC_ID_COLUMN].duplicated().any():
        raise ValueError("Labels CSV must contain at most one row per spec_id")
    numeric_labels = pd.to_numeric(label_rows[LABEL_COLUMN], errors="coerce")
    invalid_labels = label_rows[LABEL_COLUMN].notna() & numeric_labels.isna()
    if invalid_labels.any() or not numeric_labels.dropna().isin([0, 1]).all():
        raise ValueError("Labels CSV label values must be 0, 1, or empty")
    label_rows[LABEL_COLUMN] = numeric_labels

    labeled_rows = label_rows.loc[label_rows[LABEL_COLUMN].notna()].copy()
    known_ids = set(repository_lookup[ID_COLUMN])
    unmatched_ids = sorted(set(labeled_rows[SPEC_ID_COLUMN]) - known_ids)
    if unmatched_ids:
        raise ValueError(
            f"{len(unmatched_ids)} labeled spec_id value(s) do not match the "
            "SpecMine dataset; check the CSV IDs"
        )
    if labeled_rows.empty:
        raise ValueError("Labels CSV contains no labeled examples; use 0, 1, or blank")

    training_rows = repository_lookup.merge(
        labeled_rows,
        how="inner",
        left_on=ID_COLUMN,
        right_on=SPEC_ID_COLUMN,
        validate="one_to_one",
    ).drop(columns=SPEC_ID_COLUMN)
    n_ambiguous = int(label_rows[LABEL_COLUMN].isna().sum())
    if n_ambiguous:
        LOGGER.info("Excluding %d blank/ambiguous labels", n_ambiguous)
    if sample_size is not None and len(training_rows) > sample_size:
        training_rows = training_rows.sample(
            n=sample_size, random_state=seed
        ).reset_index(drop=True)
    if training_rows.empty:
        raise ValueError("No labeled examples matched the SpecMine dataset")

    if feature_data is None:
        LOGGER.info(
            "Loading precomputed features for %d labeled specs", len(training_rows)
        )
        feature_data = load_feature_data(training_rows[[ID_COLUMN]])
    if feature_data is None or feature_data.empty:
        raise ValueError("No precomputed feature rows were found for labeled specs")

    LOGGER.info("Engineering features for %d labeled specs", len(feature_data))
    features = build_features(feature_data)

    combined = features.merge(
        training_rows[[ID_COLUMN, REPOSITORY_COLUMN, LABEL_COLUMN]],
        on=ID_COLUMN,
        how="inner",
        validate="one_to_one",
    )
    if len(combined) < len(training_rows):
        raise ValueError(
            f"Precomputed features are missing {len(training_rows) - len(combined)} "
            "labeled spec(s)"
        )
    if combined.empty:
        raise ValueError("No labeled specs matched the precomputed feature data")

    y = combined[LABEL_COLUMN].astype("int8").reset_index(drop=True)
    groups = combined[REPOSITORY_COLUMN].copy()
    features = combined.drop(columns=[REPOSITORY_COLUMN, LABEL_COLUMN])
    LOGGER.info(
        "Prepared %d labeled specs from %d repositories; supplied-label counts: %s",
        len(features),
        groups.nunique(),
        y.value_counts().sort_index().to_dict(),
    )
    return features, y, groups


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
    labels: Sequence[Any] | pd.Series,
    repository_groups: Sequence[Any] | pd.Series,
    *,
    model_names: Sequence[str] = DEFAULT_MODELS,
    test_size: float = 0.2,
    seed: int = 42,
    model_dir: str | Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Fit selected baselines and score a repository-disjoint holdout.

    ``labels`` must be supplied from a labels CSV or equivalent labeling
    process. ``repository_groups`` identifies source repositories so no
    repository can appear in both partitions.
    """
    if features is None or not isinstance(features, pd.DataFrame) or features.empty:
        raise ValueError("features must be a non-empty pandas DataFrame")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    if labels is None:
        raise ValueError(
            "labels are required; see README.md under 'Evaluating a Classifier' "
            "for the labels CSV format"
        )
    if repository_groups is None:
        raise ValueError("repository_groups are required for leakage-safe splitting")

    y = pd.Series(labels)
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
            "provide labels for both classes across more repositories or choose "
            "a different seed/test_size"
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
        "Metrics evaluate only the supplied labels. They are not evidence of "
        "AI-authorship accuracy unless those labels are verified ground truth."
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
        LOGGER.info("%s holdout supplied-label metrics: %s", name, scored["metrics"])
    return results


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and compare SpecMine scikit-learn baselines."
    )
    parser.add_argument(
        "--labels-csv",
        type=Path,
        help=(
            "Required CSV with spec_id,label columns; see README.md under "
            "'Evaluating a Classifier'"
        ),
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        help="Optional maximum number of matched labeled specs to train on",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=DEFAULT_MODELS,
        default=list(DEFAULT_MODELS),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Load SpecMine metadata, train baselines, and return a CLI status code."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
        stream=sys.stdout,
    )
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.labels_csv is None:
        parser.error(
            "--labels-csv is required; see README.md under 'Evaluating a Classifier' "
            "for the required spec_id,label CSV format"
        )
    if args.sample_size is not None and args.sample_size < 1:
        parser.error("--sample-size must be a positive integer")
    if not 0 < args.test_size < 1:
        parser.error("--test-size must be between 0 and 1")

    try:
        labels_frame = read_labels_csv(args.labels_csv)
        LOGGER.info("Loading SpecMine metadata for the supplied label IDs")
        data = load_data(seed=args.seed)
        if data is None:
            raise RuntimeError("SpecMine data loading failed")
        features, labels, groups = prepare_training_data(
            data,
            labels_frame,
            sample_size=args.sample_size,
            seed=args.seed,
        )
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
                "%s complete: holdout=%d specs across %d repos; "
                "supplied-label metrics=%s",
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
