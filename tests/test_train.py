from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from specmine import train


def training_frames():
    rows = []
    labels = []
    repositories = []
    for repo_index in range(20):
        label = repo_index % 2
        for document_index in range(2):
            rows.append(
                {
                    "file_url_sha16": f"{repo_index:02d}-{document_index}",
                    "n_lines": 2 + label * 8 + document_index,
                    "n_words": 20 + label * 100 + document_index * 3,
                    "n_headings": 1 + label * 2,
                    "placeholder_density": float(label * 4),
                    "is_tiny": 0,
                    "is_heavy_template": label,
                    "tiny_but_structured": 0,
                }
            )
            labels.append(label)
            repositories.append(f"owner/repo-{repo_index:02d}")
    return pd.DataFrame(rows), pd.Series(labels), pd.Series(repositories)


def test_train_models_fits_both_baselines_and_keeps_repositories_disjoint(
    tmp_path, caplog
):
    features, labels, repositories = training_frames()
    output_dir = tmp_path / "models"

    results = train.train_models(
        features,
        labels,
        repositories,
        seed=17,
        model_dir=output_dir,
    )

    assert set(results) == {"logistic_regression", "random_forest"}
    assert "Metrics evaluate only the supplied labels" in caplog.text
    assert "not evidence of" in caplog.text
    for name, result in results.items():
        assert result["n_train"] + result["n_test"] == len(features)
        assert result["n_train_repositories"] + result["n_test_repositories"] == 20
        assert set(result["train_repositories"]).isdisjoint(
            result["test_repositories"]
        )
        assert len(result["test_repositories"]) == result["n_test_repositories"]
        assert result["metrics"]["accuracy"] >= 0
        assert len(result["predictions"]) == result["n_test"]
        assert all(row["id"] is not None for row in result["predictions"])
        assert result["model_path"] == output_dir / f"{name}.joblib"
        loaded = joblib.load(result["model_path"])
        holdout_ids = [row["id"] for row in result["predictions"]]
        holdout_repos = {int(spec_id.split("-")[0]) for spec_id in holdout_ids}
        assert len(holdout_repos) == result["n_test_repositories"]
        assert holdout_repos == {
            int(repository.rsplit("-", 1)[1])
            for repository in result["test_repositories"]
        }
        model_features = features.loc[:, result["feature_columns"]]
        assert np.array_equal(
            loaded.predict(model_features),
            result["model"].predict(model_features),
        )


def test_train_models_is_deterministic_for_fixed_seed():
    features, labels, repositories = training_frames()

    first = train.train_models(
        features,
        labels,
        repositories,
        model_names=["logistic_regression"],
        seed=11,
    )
    second = train.train_models(
        features,
        labels,
        repositories,
        model_names=["logistic_regression"],
        seed=11,
    )

    assert first["logistic_regression"]["predictions"] == second[
        "logistic_regression"
    ]["predictions"]


def test_train_models_requires_explicit_labels():
    features, _, repositories = training_frames()

    with pytest.raises(ValueError, match="labels are required"):
        train.train_models(
            features,
            labels=None,
            repository_groups=repositories,
            model_names=["random_forest"],
        )


def test_prepare_training_data_joins_features_and_supplied_labels():
    data = pd.DataFrame(
        {
            "file_url_sha16": ["spec-a", "spec-b"],
            "repo_name": ["owner/one", "owner/two"],
        }
    )
    feature_rows = pd.DataFrame(
        {
            "file_url_sha16": ["spec-a", "spec-b"],
            "n_lines": [2, 20],
            "n_words": [5, 100],
            "n_headings": [3, 1],
            "n_code_fences": [0, 1],
            "n_list_items": [0, 1],
            "n_checkboxes": [0, 0],
            "n_checked": [0, 0],
            "is_tiny": [1, 0],
            "has_ears": [0, 0],
            "has_gherkin": [0, 0],
            "has_user_story": [0, 0],
            "has_unfilled_placeholder": [0, 0],
            "has_lorem": [0, 0],
            "n_todo": [0, 0],
            "n_needs_clarification": [0, 0],
        }
    )

    labels_frame = pd.DataFrame(
        {"spec_id": ["spec-a", "spec-b"], "label": [1, 0]}
    )
    features, labels, groups = train.prepare_training_data(
        data, labels_frame, feature_rows
    )

    assert features["file_url_sha16"].tolist() == ["spec-a", "spec-b"]
    assert labels.tolist() == [1, 0]
    assert groups.tolist() == ["owner/one", "owner/two"]
    assert "repo_name" not in features


def test_prepare_training_data_excludes_blank_labels():
    data = pd.DataFrame(
        {
            "file_url_sha16": ["spec-a", "spec-b"],
            "repo_name": ["owner/one", "owner/two"],
        }
    )
    feature_rows = pd.DataFrame(
        {
            "file_url_sha16": ["spec-a", "spec-b"],
            "n_lines": [2, 20],
            "n_words": [5, 100],
            "n_headings": [3, 1],
            "n_code_fences": [0, 1],
            "n_list_items": [0, 1],
            "n_checkboxes": [0, 0],
            "n_checked": [0, 0],
            "is_tiny": [1, 0],
            "has_ears": [0, 0],
            "has_gherkin": [0, 0],
            "has_user_story": [0, 0],
        }
    )
    labels_frame = pd.DataFrame(
        {"spec_id": ["spec-a", "spec-b"], "label": [1, None]}
    )

    features, labels, groups = train.prepare_training_data(
        data, labels_frame, feature_rows
    )

    assert features["file_url_sha16"].tolist() == ["spec-a"]
    assert labels.tolist() == [1]
    assert groups.tolist() == ["owner/one"]


def test_prepare_training_data_applies_sample_limit_after_label_matching():
    data = pd.DataFrame(
        {
            "file_url_sha16": ["a", "b", "c", "d"],
            "repo_name": ["r-a", "r-b", "r-c", "r-d"],
        }
    )
    feature_rows = pd.DataFrame(
        {
            "file_url_sha16": ["a", "b", "c", "d"],
            "n_lines": [2, 20, 3, 30],
            "n_words": [5, 100, 10, 200],
            "n_headings": [3, 1, 2, 1],
            "n_code_fences": [0, 1, 0, 1],
            "n_list_items": [0, 1, 0, 1],
            "n_checkboxes": [0, 0, 0, 0],
            "n_checked": [0, 0, 0, 0],
            "is_tiny": [1, 0, 1, 0],
            "has_ears": [0, 0, 0, 0],
            "has_gherkin": [0, 0, 0, 0],
            "has_user_story": [0, 0, 0, 0],
        }
    )
    labels_frame = pd.DataFrame({"spec_id": ["a", "b"], "label": [0, 1]})

    features, labels, groups = train.prepare_training_data(
        data,
        labels_frame,
        feature_rows,
        sample_size=2,
        seed=0,
    )

    assert features["file_url_sha16"].tolist() == ["a", "b"]
    assert labels.tolist() == [0, 1]
    assert groups.tolist() == ["r-a", "r-b"]


def test_prepare_training_data_rejects_unknown_labeled_ids():
    data = pd.DataFrame(
        {"file_url_sha16": ["spec-a"], "repo_name": ["owner/one"]}
    )
    labels_frame = pd.DataFrame({"spec_id": ["not-in-dataset"], "label": [1]})

    with pytest.raises(ValueError, match="do not match the SpecMine dataset"):
        train.prepare_training_data(data, labels_frame, feature_data=pd.DataFrame())


def test_synthetic_labels_csv_is_a_parseable_smoke_test_fixture():
    labels_path = (
        Path(__file__).resolve().parents[1] / "examples" / "synthetic_labels.csv"
    )

    labels = train.read_labels_csv(labels_path)

    assert list(labels.columns) == ["spec_id", "label"]
    assert len(labels) >= 8
    assert set(labels["label"]) == {0, 1}


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"test_size": 0}, "test_size"),
        ({"repository_groups": None}, "repository_groups"),
        ({"model_names": ["unsupported"]}, "Unknown model"),
    ],
)
def test_train_models_rejects_invalid_options(kwargs, message):
    features, labels, repositories = training_frames()
    options = {
        "labels": labels,
        "repository_groups": repositories,
        "model_names": ["logistic_regression"],
    }
    options.update(kwargs)

    with pytest.raises(ValueError, match=message):
        train.train_models(features, **options)


def test_train_models_rejects_one_class_training_data():
    features, _, repositories = training_frames()

    with pytest.raises(ValueError, match="only one label class"):
        train.train_models(
            features,
            labels=pd.Series(np.zeros(len(features), dtype=int)),
            repository_groups=repositories,
        )


def test_train_models_requires_multiple_repositories():
    features, labels, _ = training_frames()

    with pytest.raises(ValueError, match="At least two repositories"):
        train.train_models(
            features,
            labels,
            repository_groups=["owner/only"] * len(features),
        )


def test_main_trains_selected_models_and_saves_them(tmp_path, monkeypatch):
    features, labels, repositories = training_frames()
    labels_path = tmp_path / "labels.csv"
    labels_path.write_text("spec_id,label\nspec-a,0\n", encoding="utf-8")
    monkeypatch.setattr(
        train,
        "read_labels_csv",
        lambda path: pd.DataFrame({"spec_id": ["spec-a"], "label": [0]}),
    )
    monkeypatch.setattr(train, "load_data", lambda **kwargs: pd.DataFrame({"ok": [1]}))
    monkeypatch.setattr(
        train,
        "prepare_training_data",
        lambda data, labels_frame, **kwargs: (features, labels, repositories),
    )
    model_dir = tmp_path / "cli-models"

    status = train.main(
        [
            "--labels-csv",
            str(labels_path),
            "--sample-size",
            "40",
            "--models",
            "logistic_regression",
            "--model-dir",
            str(model_dir),
        ]
    )

    assert status == 0
    assert (model_dir / "logistic_regression.joblib").is_file()


def test_main_requires_labels_before_loading_data(capsys):
    with pytest.raises(SystemExit) as exc_info:
        train.main([])

    assert exc_info.value.code == 2
    error = capsys.readouterr().err
    assert "--labels-csv is required" in error
    assert "Evaluating a Classifier" in error
