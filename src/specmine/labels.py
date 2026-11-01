"""Shared parsing and validation for SpecMine label CSV files."""

from __future__ import annotations

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
    if labels[SPEC_ID_COLUMN].duplicated().any():
        raise ValueError("Labels CSV must contain at most one row per spec_id")

    numeric = pd.to_numeric(labels[LABEL_COLUMN], errors="coerce")
    invalid_text = labels[LABEL_COLUMN].notna() & numeric.isna()
    if invalid_text.any() or not numeric.dropna().isin([0, 1]).all():
        raise ValueError("Labels CSV label values must be 0, 1, or empty")
    labels[LABEL_COLUMN] = numeric
    return labels
