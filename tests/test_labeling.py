from pathlib import Path

import pandas as pd
import pytest

from specmine.labeling import match_review_features, select_review_spec
from specmine.labels import read_labels_csv


@pytest.fixture
def spec_metadata():
    return pd.DataFrame(
        {
            "repo_id": [101, 101, 202, 303, 303],
            "repo_name": [
                "owner/alpha",
                "owner/alpha",
                "owner/beta",
                "owner/gamma",
                "owner/gamma",
            ],
            "file_url_sha16": ["a1", "a2", "b1", "c1", "c2"],
            "file_path": [
                "specs/a1.md",
                "specs/a2.md",
                "specs/b1.md",
                "specs/c1.md",
                "specs/c2.md",
            ],
        }
    )


def _labels(spec_ids, repo_ids=None):
    labels = pd.DataFrame({"spec_id": spec_ids})
    if repo_ids is not None:
        labels["repo_id"] = repo_ids
    labels["label"] = pd.NA
    labels["confidence"] = pd.NA
    return labels


def test_rejects_same_repository_in_real_and_fake_labels(spec_metadata):
    real = _labels(["a1"])
    fake = _labels(["a2"])

    with pytest.raises(ValueError, match="Repository leakage"):
        select_review_spec(spec_metadata, real, fake, seed=3)


def test_excludes_all_previously_labeled_repositories_and_selects_one_spec(
    spec_metadata,
):
    real = _labels(["a1"])
    fake = _labels(["b1"])

    selected = select_review_spec(spec_metadata, real, fake, seed=3)

    assert len(selected) == 1
    assert selected.iloc[0]["repo_id"] == "303"
    assert selected.iloc[0]["file_url_sha16"] in {"c1", "c2"}


def test_repository_id_column_is_checked_against_spec_metadata(spec_metadata):
    real = _labels(["a1"], ["202"])
    fake = _labels([])

    with pytest.raises(ValueError, match="does not match SpecMine metadata"):
        select_review_spec(spec_metadata, real, fake, seed=3)


def test_csv_repo_ids_exclude_repositories_by_numeric_identity(spec_metadata, tmp_path):
    labels_path = tmp_path / "real_labels.csv"
    labels_path.write_text(
        "repo_id,spec_id,label,confidence\n101,a1,,\n",
        encoding="utf-8",
    )
    real = read_labels_csv(labels_path)
    fake = _labels([])

    selected = select_review_spec(spec_metadata, real, fake, seed=3)

    assert selected.iloc[0]["repo_id"] != "101"


def test_feature_match_returns_only_the_selected_spec_row(spec_metadata):
    selected = spec_metadata.loc[spec_metadata["file_url_sha16"].eq("c2")]
    features = pd.DataFrame(
        {
            "file_url_sha16": ["c1", "c2", "a1"],
            "n_lines": [10, 20, 30],
        }
    )

    matched = match_review_features(selected, features)

    assert len(matched) == 1
    assert matched.to_dict("records") == [{"file_url_sha16": "c2", "n_lines": 20}]


@pytest.mark.parametrize(
    "feature_ids, expected_count",
    [([], 0), (["c2", "c2"], 2)],
)
def test_feature_match_rejects_missing_or_duplicated_feature_rows(
    spec_metadata, feature_ids, expected_count
):
    selected = spec_metadata.loc[spec_metadata["file_url_sha16"].eq("c2")]
    features = pd.DataFrame(
        {"file_url_sha16": feature_ids, "n_lines": [1] * expected_count}
    )

    with pytest.raises(ValueError, match="Expected exactly one feature row"):
        match_review_features(selected, features)


def test_real_label_template_keeps_training_columns_and_repo_identity():
    labels_path = Path(__file__).resolve().parents[1] / "examples" / "real_labels.csv"
    labels = read_labels_csv(labels_path)

    assert labels.empty
    assert {"repo_id", "spec_id", "label", "confidence"} <= set(labels.columns)


def test_fake_label_fixture_has_repo_ids_and_preserves_duplicate_rows():
    labels_path = Path(__file__).resolve().parents[1] / "examples" / "fake_labels.csv"
    labels = pd.read_csv(labels_path, dtype={"spec_id": "string"})
    duplicate_id = "c8fa12389f12500c"

    assert labels.columns.tolist() == ["repo_id", "spec_id", "label", "confidence"]
    assert labels["repo_id"].notna().all()
    assert labels["spec_id"].notna().all()
    duplicate_repo_ids = (
        labels.loc[labels["spec_id"].eq(duplicate_id), "repo_id"]
        .astype(str)
        .tolist()
    )
    assert duplicate_repo_ids == ["1183842687", "1183842687"]
