"""Repository-aware sampling helpers for manual SpecMine labeling."""

from __future__ import annotations

from numbers import Integral, Real

import pandas as pd

SPEC_ID_COLUMN = "file_url_sha16"
REPO_ID_COLUMN = "repo_id"


def _normalize_id(value) -> str | None:
    if pd.isna(value):
        return None
    if isinstance(value, Integral):
        return str(int(value))
    if isinstance(value, Real) and value.is_integer():
        return str(int(value))
    normalized = str(value).strip()
    if normalized.endswith(".0") and normalized[:-2].isdigit():
        normalized = normalized[:-2]
    return normalized or None


def _spec_repository_map(specs: pd.DataFrame) -> dict[str, str]:
    required = {SPEC_ID_COLUMN, REPO_ID_COLUMN}
    missing = required - set(specs.columns)
    if missing:
        raise KeyError(f"SpecMine metadata is missing columns: {sorted(missing)}")

    pairs = specs[[SPEC_ID_COLUMN, REPO_ID_COLUMN]].copy()
    pairs[SPEC_ID_COLUMN] = pairs[SPEC_ID_COLUMN].map(_normalize_id)
    pairs[REPO_ID_COLUMN] = pairs[REPO_ID_COLUMN].map(_normalize_id)
    pairs = pairs.dropna().drop_duplicates()
    ambiguous = pairs.groupby(SPEC_ID_COLUMN)[REPO_ID_COLUMN].nunique()
    ambiguous_ids = ambiguous[ambiguous > 1].index.tolist()
    if ambiguous_ids:
        raise ValueError(
            "Spec IDs map to multiple repositories: "
            f"{sorted(ambiguous_ids)[:5]}"
        )
    return pairs.set_index(SPEC_ID_COLUMN)[REPO_ID_COLUMN].to_dict()


def repositories_for_labels(
    labels: pd.DataFrame, specs: pd.DataFrame
) -> set[str]:
    """Resolve all label rows, including blank labels, to repository IDs."""
    if "spec_id" not in labels.columns:
        raise KeyError("Labels are missing the spec_id column")
    spec_repositories = _spec_repository_map(specs)
    resolved = set()

    for row in labels.itertuples(index=False):
        values = row._asdict()
        spec_id = _normalize_id(values.get("spec_id"))
        declared_repo_id = _normalize_id(values.get(REPO_ID_COLUMN))
        mapped_repo_id = spec_repositories.get(spec_id)
        if (
            declared_repo_id is not None
            and mapped_repo_id is not None
            and declared_repo_id != mapped_repo_id
        ):
            raise ValueError(
                f"Label repo_id does not match SpecMine metadata for spec_id {spec_id}"
            )
        repo_id = declared_repo_id or mapped_repo_id
        if repo_id is None:
            raise ValueError(
                f"Cannot resolve spec_id {spec_id!r} to a SpecMine repo_id"
            )
        resolved.add(repo_id)
    return resolved


def select_review_spec(
    specs: pd.DataFrame,
    real_labels: pd.DataFrame,
    fake_labels: pd.DataFrame,
    seed: int | None = None,
    label_metadata: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Choose one spec from one unlabeled repository, rejecting label leakage."""
    required = {SPEC_ID_COLUMN, REPO_ID_COLUMN, "repo_name"}
    missing = required - set(specs.columns)
    if missing:
        raise KeyError(f"SpecMine metadata is missing columns: {sorted(missing)}")

    label_metadata = specs if label_metadata is None else label_metadata
    real_repositories = repositories_for_labels(real_labels, label_metadata)
    fake_repositories = repositories_for_labels(fake_labels, label_metadata)
    overlap = real_repositories & fake_repositories
    if overlap:
        raise ValueError(
            "Repository leakage between real_labels.csv and fake_labels.csv: "
            f"{sorted(overlap)[:10]}"
        )

    candidates = specs.copy()
    candidates[REPO_ID_COLUMN] = candidates[REPO_ID_COLUMN].map(_normalize_id)
    candidates[SPEC_ID_COLUMN] = candidates[SPEC_ID_COLUMN].map(_normalize_id)
    candidates = candidates.dropna(subset=[REPO_ID_COLUMN, SPEC_ID_COLUMN])
    excluded = real_repositories | fake_repositories
    candidates = candidates.loc[~candidates[REPO_ID_COLUMN].isin(excluded)]
    if candidates.empty:
        raise ValueError("No unlabeled repositories are available to review")

    repositories = sorted(candidates[REPO_ID_COLUMN].unique())
    selected_repo_id = pd.Series(repositories).sample(
        n=1, random_state=seed
    ).iloc[0]
    repo_specs = candidates.loc[
        candidates[REPO_ID_COLUMN].eq(selected_repo_id)
    ].drop_duplicates(subset=[SPEC_ID_COLUMN])
    return repo_specs.sample(n=1, random_state=seed).reset_index(drop=True)


def match_review_features(
    review_spec: pd.DataFrame, features: pd.DataFrame
) -> pd.DataFrame:
    """Return exactly one feature row for the one selected spec."""
    if len(review_spec) != 1:
        raise ValueError("Review selection must contain exactly one spec")
    if SPEC_ID_COLUMN not in features.columns:
        raise KeyError(f"Feature data is missing {SPEC_ID_COLUMN}")

    spec_id = _normalize_id(review_spec.iloc[0][SPEC_ID_COLUMN])
    matching = features.loc[
        features[SPEC_ID_COLUMN].map(_normalize_id).eq(spec_id)
    ].copy()
    if len(matching) != 1:
        raise ValueError(
            f"Expected exactly one feature row for spec_id {spec_id}; "
            f"found {len(matching)}"
        )
    return matching.reset_index(drop=True)
