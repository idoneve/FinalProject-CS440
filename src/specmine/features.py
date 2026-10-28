"""
The functions in this module consume the precomputed ``spec_content_features``
rows from the SpecMine feature table. They do not parse Markdown or accept the
combined ``spec_files`` rows returned directly by
:func:`specmine.data.load_data`; extract or load ``spec_content_features`` first,
then call :func:`build_features` (optionally supply rows from
``spec_file_commits``). They do not infer a classifier target or train a model.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

ID_COLUMN = "file_url_sha16"
PLACEHOLDER_DENSITY_THRESHOLD = 2.0
LOW_COMMIT_THRESHOLD = 1

_REQUIRED_COLUMNS = (
    ID_COLUMN,
    "n_lines",
    "n_words",
    "n_headings",
    "n_code_fences",
    "n_list_items",
    "n_checkboxes",
    "n_checked",
    "is_tiny",
    "has_ears",
    "has_gherkin",
    "has_user_story",
)
_OPTIONAL_PLACEHOLDER_COLUMNS = (
    "has_unfilled_placeholder",
    "has_lorem",
    "n_todo",
    "n_needs_clarification",
)

__all__ = ["build_features", "derive_proxy_label"]


def _missing_columns(df: pd.DataFrame, columns: Iterable[str]) -> list[str]:
    """Return requested columns that are absent from ``df``."""
    return [column for column in columns if column not in df.columns]


def _require_schema(df: pd.DataFrame) -> None:
    """Fail with an actionable schema message for required feature fields."""
    missing = _missing_columns(df, _REQUIRED_COLUMNS)
    if missing:
        raise KeyError(
            "SpecMine feature schema is missing required columns: "
            f"{missing}. Check spec_content_features in DATA_DICTIONARY.md."
        )


def _clean_and_impute(df: pd.DataFrame) -> pd.DataFrame:
    """Fill present count fields and normalize present boolean fields."""
    cleaned = df.copy()

    count_columns = [column for column in cleaned.columns if column.startswith("n_")]
    cleaned[count_columns] = cleaned[count_columns].apply(
        pd.to_numeric, errors="coerce"
    )
    cleaned[count_columns] = cleaned[count_columns].fillna(0)

    boolean_columns = [
        column for column in cleaned.columns if column.startswith("has_")
    ]
    boolean_columns.append("is_tiny")
    for column in set(boolean_columns):
        cleaned[column] = (
            pd.to_numeric(cleaned[column], errors="coerce")
            .fillna(0)
            .clip(lower=0, upper=1)
            .astype("int8")
        )

    cleaned[ID_COLUMN] = cleaned[ID_COLUMN].replace("", np.nan)
    return cleaned


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide values without producing infinities for zero denominators."""
    denominator = pd.to_numeric(denominator, errors="coerce").clip(lower=0)
    numerator = pd.to_numeric(numerator, errors="coerce").fillna(0)
    return pd.Series(
        np.divide(
            numerator.to_numpy(dtype=float),
            denominator.to_numpy(dtype=float).clip(min=1),
        ),
        index=numerator.index,
        dtype=float,
    )


def _add_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add structurally safe size and completion ratios."""
    result = df.copy()
    result["words_per_line"] = _safe_ratio(result["n_words"], result["n_lines"])
    result["list_items_per_heading"] = _safe_ratio(
        result["n_list_items"], result["n_headings"]
    )
    result["code_fences_per_word"] = _safe_ratio(
        result["n_code_fences"], result["n_words"]
    )
    result["checkbox_ratio"] = _safe_ratio(
        result["n_checkboxes"], result["n_list_items"]
    )
    result["checked_completion_ratio"] = _safe_ratio(
        result["n_checked"], result["n_checkboxes"]
    )

    available: list[pd.Series] = []
    for column in _OPTIONAL_PLACEHOLDER_COLUMNS:
        if column not in result:
            LOGGER.warning(
                "SpecMine placeholder indicator %s is missing; skipping it", column
            )
            continue
        if column.startswith("has_"):
            available.append(pd.to_numeric(result[column], errors="coerce").fillna(0))
        else:
            available.append(pd.to_numeric(result[column], errors="coerce").fillna(0))

    placeholder_sum = (
        sum(available, start=pd.Series(0.0, index=result.index))
        if available
        else pd.Series(0.0, index=result.index)
    )
    result["placeholder_density"] = (
        placeholder_sum
        / np.maximum(pd.to_numeric(result["n_words"], errors="coerce").fillna(0), 1)
        * 1000
    )
    return result


def _add_template_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Add combined signals for structured AI-template conventions."""
    result = df.copy()
    template_columns = ["has_gherkin", "has_ears", "has_user_story"]
    result["is_heavy_template"] = (
        result[template_columns].sum(axis=1).ge(2).astype("int8")
    )
    result["tiny_but_structured"] = (
        result["is_tiny"].eq(1) & result["n_headings"].gt(2)
    ).astype("int8")
    return result


def _add_volatility_features(
    df: pd.DataFrame, commits_df: pd.DataFrame
) -> pd.DataFrame:
    """Aggregate commit history per spec and left-join it onto feature rows."""
    missing_join = _missing_columns(commits_df, [ID_COLUMN])
    if missing_join:
        raise KeyError(
            f"Commit DataFrame is missing required join column {missing_join}; "
            "check spec_file_commits in DATA_DICTIONARY.md."
        )

    commit_columns = [
        column for column in ("commit_sha", "commit_hash") if column in commits_df
    ]
    if not commit_columns:
        raise KeyError(
            "Commit DataFrame needs commit_sha or commit_hash for counting; "
            "check spec_file_commits in DATA_DICTIONARY.md."
        )

    commit_column = commit_columns[0]
    valid_commits = commits_df.loc[
        commits_df[ID_COLUMN].notna()
        & commits_df[ID_COLUMN].astype(str).str.strip().ne("")
        & commits_df[commit_column].notna()
    ].copy()
    valid_commits = valid_commits.drop_duplicates(subset=[ID_COLUMN, commit_column])

    aggregation: dict[str, tuple[str, str]] = {
        "n_commits": (commit_column, "size"),
    }
    has_lines_changed = "lines_changed" in valid_commits
    if has_lines_changed:
        valid_commits["lines_changed"] = pd.to_numeric(
            valid_commits["lines_changed"], errors="coerce"
        )
        aggregation["total_lines_changed"] = ("lines_changed", "sum")

    volatility = valid_commits.groupby(ID_COLUMN).agg(**aggregation).reset_index()
    if not has_lines_changed:
        LOGGER.warning(
            "Commit history lacks lines_changed; total_lines_changed and "
            "curation_ratio will be unavailable"
        )
        volatility["total_lines_changed"] = np.nan
        volatility["curation_ratio"] = np.nan
    else:
        all_values_missing = (
            valid_commits.groupby(ID_COLUMN)["lines_changed"]
            .apply(lambda values: values.isna().all())
        )
        missing_ids = all_values_missing[all_values_missing].index
        volatility.loc[
            volatility[ID_COLUMN].isin(missing_ids), "total_lines_changed"
        ] = np.nan
        volatility["curation_ratio"] = volatility["n_commits"] / (
            volatility["total_lines_changed"] + 1
        )

    result = df.merge(volatility, on=ID_COLUMN, how="left", validate="1:1")
    result["n_commits"] = result["n_commits"].astype("Int64")
    result["total_lines_changed"] = pd.to_numeric(
        result["total_lines_changed"], errors="coerce"
    )
    result["curation_ratio"] = pd.to_numeric(
        result["curation_ratio"], errors="coerce"
    )
    return result


def build_features(
    df: pd.DataFrame, commits_df: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Build a feature matrix from SpecMine feature rows."""
    if df is None:
        raise TypeError("df must be a pandas DataFrame")
    _require_schema(df)

    result = _clean_and_impute(df)
    result = _add_ratio_features(result)
    result = _add_template_signals(result)

    if commits_df is None:
        LOGGER.info("Commit history was not provided; volatility features skipped")
    else:
        result = _add_volatility_features(result, commits_df)

    return result


def derive_proxy_label(df: pd.DataFrame) -> pd.Series:
    """This heuristic labels rows as 1 when they combine a tiny structured layout,
    unusually dense placeholder signals, or a heavy template with few/no commits.
    It is intentionally conservative and is for rapid experimentation only.
    Real labeling should use commit-history or PR-text evidence instead of this
    proxy. Thresholds are module-level constants so they can be tuned.
    """
    missing = _missing_columns(
        df,
        [
            "placeholder_density",
            "is_heavy_template",
            "is_tiny",
            "tiny_but_structured",
        ],
    )
    if missing:
        raise KeyError(
            f"Proxy-label features are missing: {missing}. Call build_features first."
        )

    placeholders = df["placeholder_density"] > PLACEHOLDER_DENSITY_THRESHOLD
    tiny_structured = df["is_tiny"].eq(1) & df["tiny_but_structured"].eq(1)
    if "n_commits" in df:
        known_history = df["n_commits"].notna()
        low_commits = known_history & df["n_commits"].le(LOW_COMMIT_THRESHOLD)
    else:
        LOGGER.info(
            "n_commits is unavailable; skipping the commit-history proxy branch"
        )
        known_history = pd.Series(False, index=df.index)
        low_commits = pd.Series(False, index=df.index)

    label = (
        tiny_structured | placeholders | (df["is_heavy_template"].eq(1) & low_commits)
    )
    return label.fillna(False).astype("int8")
