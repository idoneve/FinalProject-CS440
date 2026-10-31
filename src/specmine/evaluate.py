"""Evaluate a SpecMine classifier.

Example:
    python -m specmine.evaluate --model-path models/specmine.joblib \\
        --labels-csv labels.csv --split test --output-json metrics.json
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from specmine.data import load_data, load_feature_data
from specmine.features import build_features

LOGGER = logging.getLogger(__name__)
ID_COLUMN = "file_url_sha16"
LABEL_COLUMN = "label"


def _validate_threshold(value: float) -> float:
    """Return a probability threshold in the inclusive [0, 1] range."""
    threshold = float(value)
    if not 0 <= threshold <= 1:
        raise argparse.ArgumentTypeError("threshold must be between 0 and 1")
    return threshold


def _validate_val_frac(value: float) -> float:
    """Return a validation fraction strictly between zero and one."""
    fraction = float(value)
    if not 0 < fraction < 1:
        raise argparse.ArgumentTypeError("val-frac must be between 0 and 1")
    return fraction


def _load_model(model_path: Path) -> Any:
    """Load a joblib estimator and validate its inference interface."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model file does not exist: {model_path}")
    if not model_path.is_file():
        raise ValueError(
            "Expected a joblib-serialized sklearn model file; "
            "Hugging Face model directories are not supported."
        )
    try:
        model = joblib.load(model_path)
    except (
        OSError,
        EOFError,
        ValueError,
        TypeError,
        ImportError,
        AttributeError,
        pickle.UnpicklingError,
    ) as exc:
        raise RuntimeError(f"Could not load model from {model_path}: {exc}") from exc
    if not callable(getattr(model, "predict_proba", None)):
        raise TypeError("Loaded model must expose a callable predict_proba method")
    return model


def _expected_columns(model: Any) -> list[str]:
    """Get named model inputs so inference cannot silently use the wrong columns."""
    columns = getattr(model, "feature_names_in_", None)
    if columns is None:
        raise ValueError(
            "Model does not expose feature_names_in_; fit the pipeline with a "
            "DataFrame so its expected feature columns can be checked."
        )
    return [str(column) for column in columns]


def _expected_positive_probabilities(model: Any, features: pd.DataFrame) -> np.ndarray:
    """Return probabilities for class 1 using the model's declared class order."""
    classes = getattr(model, "classes_", None)
    if classes is None:
        raise ValueError("Model must expose classes_ to identify positive class 1")
    if set(classes) != {0, 1}:
        raise ValueError("Model classes_ must contain exactly binary classes 0 and 1")
    positive_indices = [index for index, value in enumerate(classes) if value == 1]

    probabilities = np.asarray(model.predict_proba(features), dtype=float)
    if (
        probabilities.ndim != 2
        or probabilities.shape[0] != len(features)
        or probabilities.shape[1] != len(classes)
    ):
        raise ValueError("predict_proba returned an invalid probability array shape")
    positive_index = positive_indices[0]
    if positive_index >= probabilities.shape[1]:
        raise ValueError("predict_proba output does not match model classes_")
    positive_probabilities = probabilities[:, positive_index]
    if not np.isfinite(positive_probabilities).all():
        raise ValueError("predict_proba returned non-finite probabilities")
    if ((positive_probabilities < 0) | (positive_probabilities > 1)).any():
        raise ValueError("predict_proba returned probabilities outside [0, 1]")
    return positive_probabilities


def _expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute equal-width, 10-bin expected calibration error."""
    bin_edges = np.linspace(0.0, 1.0, 11)
    bin_ids = np.minimum(np.digitize(y_prob, bin_edges[1:-1]), 9)
    error = 0.0
    for bin_id in range(10):
        in_bin = bin_ids == bin_id
        if in_bin.any():
            error += float(in_bin.mean()) * abs(
                float(y_prob[in_bin].mean()) - float(y_true[in_bin].mean())
            )
    return error


def _prediction_rows(
    df: pd.DataFrame, y_prob: np.ndarray, y_pred: np.ndarray
) -> list[dict[str, Any]]:
    """Build portable per-example prediction records."""
    rows: list[dict[str, Any]] = []
    for index, (_, row) in enumerate(df.iterrows()):
        record: dict[str, Any] = {
            "id": row.get(ID_COLUMN),
            "y_true": int(row[LABEL_COLUMN]) if LABEL_COLUMN in df else None,
            "y_pred": int(y_pred[index]),
            "y_prob": float(y_prob[index]),
        }
        for column in ("repo_name", "file_path"):
            if column in df:
                record[column] = row[column]
        rows.append(record)
    return rows


def evaluate_model(
    model: Any, df: pd.DataFrame, threshold: float = 0.5
) -> dict[str, Any]:
    """Predict probabilities and compute metrics when ``df`` includes labels.

    The model must expose ``feature_names_in_`` and ``predict_proba``. Only its
    expected feature columns are passed to it; optional metadata is preserved
    in the returned per-example rows.
    """
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    if df is None or df.empty:
        raise ValueError("Cannot evaluate an empty data split")

    expected = _expected_columns(model)
    missing = [column for column in expected if column not in df.columns]
    if missing:
        raise ValueError(f"Evaluation data is missing model feature columns: {missing}")
    features = df.loc[:, expected]
    y_prob = _expected_positive_probabilities(model, features)
    y_pred = (y_prob >= threshold).astype("int8")

    metrics: dict[str, Any] | None = None
    if LABEL_COLUMN in df:
        if df[LABEL_COLUMN].isna().any():
            raise ValueError("Evaluation labels must not contain missing values")
        y_true_numeric = pd.to_numeric(df[LABEL_COLUMN], errors="coerce")
        if y_true_numeric.isna().any() or not y_true_numeric.isin([0, 1]).all():
            raise ValueError("Evaluation labels must contain only 0 and 1")
        y_true = y_true_numeric.to_numpy(dtype="int8")
        has_both_classes = np.unique(y_true).size == 2
        matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_true, y_prob))
            if has_both_classes
            else None,
            "pr_auc": (
                float(average_precision_score(y_true, y_prob))
                if has_both_classes
                else None
            ),
            "brier_score": float(brier_score_loss(y_true, y_prob)),
            "ece": _expected_calibration_error(y_true, y_prob),
            "confusion_matrix": matrix.astype(int).tolist(),
            "support": {
                "0": int(np.count_nonzero(y_true == 0)),
                "1": int(np.count_nonzero(y_true == 1)),
            },
        }

    return {
        "n_examples": len(df),
        "metrics": metrics,
        "predictions": _prediction_rows(df, y_prob, y_pred),
    }


def _read_labels(labels_path: Path) -> pd.DataFrame:
    """Read and validate the user's external ground-truth labels CSV."""
    if not labels_path.is_file():
        raise FileNotFoundError(f"Labels CSV does not exist: {labels_path}")
    labels = pd.read_csv(labels_path)
    required = {"spec_id", LABEL_COLUMN}
    missing = sorted(required - set(labels.columns))
    if missing:
        raise ValueError(f"Labels CSV is missing required columns: {missing}")
    if labels["spec_id"].isna().any():
        raise ValueError("Labels CSV spec_id values must not be missing")
    labels["spec_id"] = labels["spec_id"].astype("string")
    if labels["spec_id"].duplicated().any():
        raise ValueError("Labels CSV must contain at most one row per spec_id")

    numeric = pd.to_numeric(labels[LABEL_COLUMN], errors="coerce")
    invalid_text = labels[LABEL_COLUMN].notna() & numeric.isna()
    if invalid_text.any() or not numeric.dropna().isin([0, 1]).all():
        raise ValueError("Labels CSV label values must be 0, 1, or empty")
    labels[LABEL_COLUMN] = numeric
    return labels


def _attach_labels(df: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Left-join labels on spec_id == file_url_sha16 and retain match status."""
    if ID_COLUMN not in df.columns:
        raise ValueError(f"Loaded SpecMine data is missing identifier {ID_COLUMN!r}")
    if df[ID_COLUMN].isna().any():
        raise ValueError(f"Loaded SpecMine data contains missing {ID_COLUMN} values")
    if df[ID_COLUMN].duplicated().any():
        raise ValueError(f"Loaded SpecMine data contains duplicate {ID_COLUMN} values")
    result = df.copy()
    result["_spec_id_join"] = result[ID_COLUMN].astype("string")
    result = result.merge(
        labels,
        how="left",
        left_on="_spec_id_join",
        right_on="spec_id",
        validate="one_to_one",
        indicator="_label_merge",
    )
    result["_label_matched"] = result["_label_merge"].eq("both")
    return result.drop(columns=["_spec_id_join", "spec_id", "_label_merge"])


def _load_features(df: pd.DataFrame) -> pd.DataFrame:
    """Join the cached SpecMine features and apply the shared feature builder."""
    if ID_COLUMN not in df.columns:
        raise ValueError(f"Loaded SpecMine data is missing identifier {ID_COLUMN!r}")
    if df[ID_COLUMN].duplicated().any():
        raise ValueError(f"Loaded SpecMine data contains duplicate {ID_COLUMN} values")
    raw_features = load_feature_data(df[[ID_COLUMN]])
    engineered = build_features(raw_features)
    missing_ids = df.loc[~df[ID_COLUMN].isin(engineered[ID_COLUMN]), ID_COLUMN]
    if not missing_ids.empty:
        raise ValueError(
            f"SpecMine feature data is missing {len(missing_ids)} evaluation row(s)"
        )
    return df.merge(
        engineered,
        on=ID_COLUMN,
        how="left",
        validate="one_to_one",
        suffixes=("", "_feature"),
    )


def _validation_split(
    train_df: pd.DataFrame, val_frac: float, seed: Optional[int]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Carve a deterministic validation subset from train."""
    if train_df.empty or len(train_df) < 2:
        raise ValueError("Cannot create a validation split from fewer than 2 rows")
    stratify = None
    if LABEL_COLUMN in train_df:
        stratify = train_df[LABEL_COLUMN].fillna(-1).to_numpy()
    try:
        remaining_indices, val_indices = train_test_split(
            train_df.index.to_numpy(),
            test_size=val_frac,
            random_state=seed,
            stratify=stratify,
        )
    except ValueError as exc:
        if stratify is not None:
            raise ValueError(
                "Cannot create a stratified validation split; ensure each label "
                "class (including missing labels) has enough examples"
            ) from exc
        raise ValueError(f"Cannot create validation split: {exc}") from exc
    return (
        train_df.loc[remaining_indices].reset_index(drop=True),
        train_df.loc[val_indices].reset_index(drop=True),
    )


def _load_selected_data(
    args: argparse.Namespace, labels: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """Load the selected leak-safe partition, optionally deriving validation."""
    if args.split == "val" and not args.make_val:
        raise ValueError("--split val requires --make-val")
    if args.make_val and args.split not in ("train", "val"):
        raise ValueError(
            "--make-val can be used only with --split train or --split val"
        )

    load_options = {
        "curated_only": args.curated_only,
        "sample_size": args.sample_size,
        "seed": args.seed,
        "data_dir": args.data_dir,
    }
    if args.split == "all":
        data = load_data(**load_options)
        if data is None:
            raise RuntimeError("SpecMine data loading failed")
        return _attach_labels(data, labels) if labels is not None else data

    splits = load_data(split=True, **load_options)
    if splits is None:
        raise RuntimeError("SpecMine data loading failed")
    train_df, test_df = splits

    if labels is not None:
        train_df = _attach_labels(train_df, labels)
    if args.make_val:
        train_df, val_df = _validation_split(train_df, args.val_frac, args.seed)
    else:
        val_df = None

    if args.split == "train":
        return train_df
    if args.split == "test":
        return _attach_labels(test_df, labels) if labels is not None else test_df
    if val_df is None:
        raise ValueError("Validation data is unavailable; pass --make-val")
    return val_df


def _evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    """Load inputs, run inference, and assemble report metadata."""
    model = _load_model(args.model_path)
    labels = _read_labels(args.labels_csv) if args.labels_csv else None
    data = _load_selected_data(args, labels)
    if data.empty:
        raise ValueError(f"The selected {args.split!r} split is empty")
    data = _load_features(data)

    n_unlabeled = 0
    n_unmatched = 0
    if args.labels_csv:
        if LABEL_COLUMN not in data:
            raise ValueError("Labels were requested but could not be joined to data")
        n_unmatched = int((~data["_label_matched"]).sum())
        n_unlabeled = int((data["_label_matched"] & data[LABEL_COLUMN].isna()).sum())
        data = data.loc[data[LABEL_COLUMN].notna()].drop(columns="_label_matched")
        if data.empty:
            raise ValueError(
                "No labeled examples matched the selected split "
                f"(n_unlabeled={n_unlabeled}, n_unmatched={n_unmatched})"
            )
    else:
        n_unlabeled = len(data)
        LOGGER.warning(
            "No --labels-csv provided; emitting predictions without accuracy metrics"
        )

    result = evaluate_model(model, data, threshold=args.threshold)
    result.update(
        {
            "split": args.split,
            "threshold": args.threshold,
            "n_labeled": result["n_examples"] if args.labels_csv else 0,
            "n_unlabeled": n_unlabeled,
            "n_unmatched": n_unmatched,
        }
    )
    return result


def _write_outputs(result: Dict[str, Any], args: argparse.Namespace) -> None:
    """Write optional JSON summary and per-example CSV."""
    if args.output_json:
        with args.output_json.open("w", encoding="utf-8") as output_file:
            json.dump(result, output_file, indent=2, allow_nan=False)
            output_file.write("\n")
    if args.output_csv:
        pd.DataFrame(result["predictions"]).to_csv(args.output_csv, index=False)


def _print_summary(result: dict[str, Any]) -> None:
    """Log a concise human-readable evaluation summary."""
    LOGGER.info(
        "Split: %s | examples: %d | labeled: %d | unlabeled: %d | unmatched: %d",
        result["split"],
        result["n_examples"],
        result["n_labeled"],
        result["n_unlabeled"],
        result["n_unmatched"],
    )
    if result["metrics"] is None:
        LOGGER.warning("No supervised metrics computed because labels are unavailable")
        return
    for metric, value in result["metrics"].items():
        LOGGER.info("%s: %s", metric, value)


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description="Evaluate a SpecMine classifier.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--sample-size", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--curated-only", action="store_true")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--labels-csv", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument(
        "--split", choices=("train", "test", "all", "val"), default="test"
    )
    parser.add_argument("--threshold", type=_validate_threshold, default=0.5)
    parser.add_argument("--make-val", action="store_true")
    parser.add_argument("--val-frac", type=_validate_val_frac, default=0.15)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the evaluator CLI and return a process status code."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
        stream=sys.stdout,
    )
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.sample_size is not None and args.sample_size < 1:
        parser.error("--sample-size must be a positive integer")
    try:
        result = _evaluate(args)
        _print_summary(result)
        _write_outputs(result, args)
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        LOGGER.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
