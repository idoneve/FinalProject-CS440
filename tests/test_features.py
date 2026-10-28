import numpy as np
import pandas as pd
from specmine.features import build_features, derive_proxy_label


def feature_frame():
    return pd.DataFrame(
        {
            "file_url_sha16": ["abc1", "def2", "ghi3"],
            "n_lines": [0, 10, 5],
            "n_words": [0, 100, 20],
            "n_headings": [0, 1, 3],
            "n_code_fences": [0, 2, 0],
            "n_list_items": [0, 0, 4],
            "n_checkboxes": [0, 0, 0],
            "n_checked": [0, 0, 0],
            "n_todo": [None, 0, 0],
            "n_needs_clarification": [0, 0, 5],
            "is_tiny": [1, 0, 1],
            "has_ears": [1, 0, 1],
            "has_gherkin": [1, 0, 1],
            "has_user_story": [0, 1, 1],
            "has_unfilled_placeholder": [1, 0, 0],
            "has_lorem": [0, 0, 0],
        }
    )


def test_build_features_preserves_rows_and_id():
    features = build_features(feature_frame())

    assert len(features) == 3
    assert features["file_url_sha16"].tolist() == ["abc1", "def2", "ghi3"]


def test_ratio_features_are_safe_for_zero_divisors():
    features = build_features(feature_frame())
    ratio_columns = [
        "words_per_line",
        "list_items_per_heading",
        "code_fences_per_word",
        "checkbox_ratio",
        "checked_completion_ratio",
        "placeholder_density",
    ]

    for column in ratio_columns:
        assert np.isfinite(features[column].to_numpy()).all()


def test_template_signals_are_added():
    features = build_features(feature_frame())

    assert features["is_heavy_template"].tolist() == [1, 0, 1]
    assert features["tiny_but_structured"].tolist() == [0, 0, 1]


def test_proxy_label_contains_only_binary_values():
    features = build_features(feature_frame())
    labels = derive_proxy_label(features)

    assert labels.dtype.kind in "iu"
    assert set(labels.unique()).issubset({0, 1})
    assert labels.tolist() == [1, 0, 1]


def test_commits_add_volatility_features():
    commits = pd.DataFrame(
        {
            "file_url_sha16": ["abc1", "abc1", "def2"],
            "commit_hash": ["a1", "a2", "d1"],
            "lines_changed": [10, 20, 3],
        }
    )

    features = build_features(feature_frame(), commits_df=commits)

    assert features["n_commits"].iloc[:2].tolist() == [2, 1]
    assert pd.isna(features["n_commits"].iloc[2])
    assert features["total_lines_changed"].iloc[:2].tolist() == [30, 3]
    assert pd.isna(features["total_lines_changed"].iloc[2])
    assert features["curation_ratio"].iloc[:2].tolist() == [2 / 31, 1 / 4]
    assert pd.isna(features["curation_ratio"].iloc[2])


def test_none_commits_does_not_crash_and_skips_volatility():
    features = build_features(feature_frame(), commits_df=None)

    assert "n_commits" not in features
    assert "total_lines_changed" not in features
    assert "curation_ratio" not in features


def test_duplicate_commits_and_missing_ids_do_not_inflate_counts():
    commits = pd.DataFrame(
        {
            "file_url_sha16": ["abc1", "abc1", None, ""],
            "commit_sha": ["a1", "a1", "missing", "missing"],
        }
    )

    features = build_features(feature_frame(), commits_df=commits)

    assert features["n_commits"].iloc[0] == 1
    assert features["n_commits"].iloc[1:].isna().all()


def test_missing_lines_changed_is_not_substituted_with_zero():
    commits = pd.DataFrame(
        {
            "file_url_sha16": ["abc1"],
            "commit_sha": ["a1"],
        }
    )

    features = build_features(feature_frame(), commits_df=commits)

    assert features["n_commits"].iloc[0] == 1
    assert pd.isna(features["total_lines_changed"]).all()
    assert pd.isna(features["curation_ratio"]).all()


def test_unknown_commit_history_does_not_trigger_heavy_template_label():
    frame = feature_frame().iloc[[1]].copy()
    frame["is_tiny"] = 0
    frame["has_ears"] = 1
    frame["has_gherkin"] = 1
    frame["has_user_story"] = 1
    frame["n_todo"] = 0
    frame["n_needs_clarification"] = 0
    frame["has_unfilled_placeholder"] = 0
    frame["has_lorem"] = 0
    commits = pd.DataFrame(
        {"file_url_sha16": ["another-id"], "commit_sha": ["a1"]}
    )

    features = build_features(frame, commits_df=commits)

    assert derive_proxy_label(features).tolist() == [0]
