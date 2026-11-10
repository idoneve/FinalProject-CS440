"""Shared parsing and validation for SpecMine label CSV files."""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

LABEL_COLUMN = "label"
CONFIDENCE_COLUMN = "confidence"
SPEC_ID_COLUMN = "spec_id"
REASONING_COLUMN = "reasoning"


def read_labels_csv(labels_path: str | Path) -> pd.DataFrame:
    """Read labels keyed by spec ID, with 0–100 confidence scores."""
    labels_path = Path(labels_path)
    if not labels_path.is_file():
        raise FileNotFoundError(f"Labels CSV does not exist: {labels_path}")
    labels = pd.read_csv(labels_path, dtype="string", skipinitialspace=True)
    labels.columns = labels.columns.str.strip()
    labels = labels.drop(columns=[REASONING_COLUMN], errors="ignore")
    required = {SPEC_ID_COLUMN, LABEL_COLUMN, CONFIDENCE_COLUMN}
    missing = sorted(required - set(labels.columns))
    if missing:
        raise ValueError(f"Labels CSV is missing required columns: {missing}")
    if labels[SPEC_ID_COLUMN].isna().any():
        raise ValueError("Labels CSV spec_id values must not be missing")
    labels[SPEC_ID_COLUMN] = labels[SPEC_ID_COLUMN].astype("string")

    numeric = pd.to_numeric(labels[LABEL_COLUMN], errors="coerce")
    invalid_text = labels[LABEL_COLUMN].notna() & numeric.isna()
    if invalid_text.any() or not numeric.dropna().isin([0, 1]).all():
        raise ValueError("Labels CSV label values must be 0, 1, or empty")
    labels[LABEL_COLUMN] = numeric

    confidence = pd.to_numeric(labels[CONFIDENCE_COLUMN], errors="coerce")
    invalid_confidence_text = (
        labels[CONFIDENCE_COLUMN].notna() & confidence.isna()
    )
    if (
        invalid_confidence_text.any()
        or not confidence.dropna().between(0, 100).all()
    ):
        raise ValueError(
            "Labels CSV confidence values must be between 0 and 100, or empty"
        )
    missing_confidence = labels[LABEL_COLUMN].notna() & confidence.isna()
    if missing_confidence.any():
        raise ValueError(
            "Labels CSV confidence values are required for nonblank labels"
        )
    labels[CONFIDENCE_COLUMN] = confidence

    duplicate_ids = labels.loc[
        labels[SPEC_ID_COLUMN].duplicated(keep=False), SPEC_ID_COLUMN
    ].drop_duplicates()
    if not duplicate_ids.empty:
        conflicting_ids = (
            labels.groupby(SPEC_ID_COLUMN, sort=False)[LABEL_COLUMN]
            .nunique(dropna=False)
            .loc[lambda counts: counts > 1]
            .index
        )
        conflict_list = sorted(str(spec_id) for spec_id in conflicting_ids)
        if conflict_list:
            warnings.warn(
                "Conflicting labels for duplicate spec_id value(s) "
                f"{conflict_list}; treating them as blank/ambiguous.",
                UserWarning,
                stacklevel=2,
            )
        labels = labels.drop_duplicates(subset=SPEC_ID_COLUMN, keep="first").copy()
        if conflict_list:
            labels.loc[
                labels[SPEC_ID_COLUMN].isin(conflicting_ids), LABEL_COLUMN
            ] = pd.NA
            labels.loc[
                labels[SPEC_ID_COLUMN].isin(conflicting_ids), CONFIDENCE_COLUMN
            ] = pd.NA

    return labels
