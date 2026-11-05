"""Shared parsing and validation for SpecMine label CSV files."""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

LABEL_COLUMN = "label"
SPEC_ID_COLUMN = "spec_id"


def read_labels_csv(labels_path: str | Path) -> pd.DataFrame:
    """Read labels keyed by spec ID, preserving string IDs and allowing blanks."""
    labels_path = Path(labels_path)
    if not labels_path.is_file():
        raise FileNotFoundError(f"Labels CSV does not exist: {labels_path}")
    labels = pd.read_csv(labels_path, dtype={SPEC_ID_COLUMN: "string"})
    required = {SPEC_ID_COLUMN, LABEL_COLUMN}
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

    return labels
