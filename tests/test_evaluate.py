import argparse
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from specmine import evaluate


class DummyModel:
    feature_names_in_ = np.array(["score"])
    classes_ = np.array([0, 1])

    def predict_proba(self, features):
        positive = features["score"].to_numpy(dtype=float)
        return np.column_stack((1 - positive, positive))


class FeatureDummyModel:
    feature_names_in_ = np.array(["n_lines"])
    classes_ = np.array([0, 1])

    def predict_proba(self, features):
        positive = np.where(features["n_lines"].to_numpy(dtype=float) > 0, 0.9, 0.1)
        return np.column_stack((1 - positive, positive))


class MarkdownDummyModel:
    feature_names_in_ = np.array([evaluate.MARKDOWN_COLUMN])
    classes_ = np.array([0, 1])

    def predict_proba(self, features):
        positive = np.where(features[evaluate.MARKDOWN_COLUMN].str.contains("AI"), 0.9, 0.1)
        return np.column_stack((1 - positive, positive))


def labeled_frame():
    return pd.DataFrame(
        {
            "file_url_sha16": ["a", "b", "c", "d"],
            "score": [0.2, 0.8, 0.7, 0.1],
            "label": [0, 1, 1, 0],
            "repo_name": ["r1", "r2", "r3", "r4"],
            "file_path": ["a.md", "b.md", "c.md", "d.md"],
        }
    )


def test_evaluate_model_computes_metrics_and_thresholded_predictions():
    result = evaluate.evaluate_model(DummyModel(), labeled_frame(), threshold=0.5)

    assert result["n_examples"] == 4
    assert result["metrics"]["accuracy"] == 1.0
    assert result["metrics"]["roc_auc"] == 1.0
    assert result["metrics"]["pr_auc"] == 1.0
    assert result["metrics"]["confusion_matrix"] == [[2, 0], [0, 2]]
    assert result["metrics"]["support"] == {"0": 2, "1": 2}
    assert [row["y_pred"] for row in result["predictions"]] == [0, 1, 1, 0]
    assert result["predictions"][0]["id"] == "a"


def test_uniform_guesser_samples_labels_but_reports_half_probability(tmp_path):
    inputs = pd.DataFrame(index=range(2000))
    labels = np.tile([0, 1], 1000)
    model = Pipeline(
        [("classifier", DummyClassifier(strategy="uniform", random_state=17))]
    ).fit(inputs, labels)
    model_path = tmp_path / "uniform.joblib"
    evaluate.joblib.dump(model, model_path)
    loaded = evaluate._load_model(model_path)

    probabilities, predictions = evaluate.predict_model(loaded, inputs)
    repeated_probabilities, repeated_predictions = evaluate.predict_model(
        loaded, inputs, threshold=0.9
    )

    np.testing.assert_array_equal(probabilities, np.full(len(inputs), 0.5))
    np.testing.assert_array_equal(probabilities, repeated_probabilities)
    np.testing.assert_array_equal(predictions, repeated_predictions)
    assert 0.47 < predictions.mean() < 0.53
    assert set(predictions) == {0, 1}
    scored = evaluate.evaluate_model(loaded, pd.DataFrame({"label": labels}))
    assert [row["y_pred"] for row in scored["predictions"]] == predictions.tolist()
    assert scored["metrics"]["roc_auc"] == 0.5


def test_evaluate_model_handles_single_class_labels():
    frame = labeled_frame().iloc[[0, 3]].copy()
    result = evaluate.evaluate_model(DummyModel(), frame)

    assert result["metrics"]["roc_auc"] is None
    assert result["metrics"]["pr_auc"] is None
    assert result["metrics"]["support"] == {"0": 2, "1": 0}


def test_evaluate_model_without_labels_returns_predictions_only():
    frame = labeled_frame().drop(columns="label")
    result = evaluate.evaluate_model(DummyModel(), frame)

    assert result["metrics"] is None
    assert len(result["predictions"]) == len(frame)
    assert result["predictions"][0]["y_true"] is None


def test_evaluate_model_rejects_missing_model_features():
    frame = pd.DataFrame({"other": [0.5]})

    with pytest.raises(ValueError, match="missing model feature columns.*score"):
        evaluate.evaluate_model(DummyModel(), frame)


def test_evaluate_model_rejects_empty_split():
    with pytest.raises(ValueError, match="empty data split"):
        evaluate.evaluate_model(DummyModel(), pd.DataFrame())


def test_load_model_reads_joblib_and_checks_prediction_interface(tmp_path):
    model_path = tmp_path / "model.joblib"
    evaluate.joblib.dump(DummyModel(), model_path)

    loaded = evaluate._load_model(model_path)

    assert loaded.predict_proba(pd.DataFrame({"score": [0.7]}))[0, 1] == 0.7


def test_load_model_rejects_objects_without_predict_proba(tmp_path):
    model_path = tmp_path / "model.joblib"
    evaluate.joblib.dump(object(), model_path)

    with pytest.raises(TypeError, match="predict_proba"):
        evaluate._load_model(model_path)


def test_read_labels_preserves_leading_zero_ids(tmp_path):
    labels_path = tmp_path / "labels.csv"
    labels_path.write_text(
        "spec_id,label,confidence\n0000000000000042,1,90\n", encoding="utf-8"
    )

    labels = evaluate._read_labels(labels_path)

    assert labels.loc[0, "spec_id"] == "0000000000000042"
    assert labels.loc[0, "label"] == 1


def test_evaluate_model_rejects_invalid_threshold():
    with pytest.raises(ValueError, match="threshold"):
        evaluate.evaluate_model(DummyModel(), labeled_frame(), threshold=1.1)


def test_text_evaluation_loads_only_labeled_markdown_and_reports_missing(monkeypatch):
    data = pd.DataFrame(
        {
            "file_url_sha16": ["a", "b", "c"],
            "repo_id": [101, 102, 103],
            "repo_name": ["r1", "r2", "r3"],
            "file_url": ["url-a", "url-b", "url-c"],
        }
    )
    labels = pd.DataFrame(
        {
            "repo_id": [201, 202, 203],
            "spec_id": ["a", "b", "c"],
            "label": [0, 1, np.nan],
            "confidence": [90, 80, 70],
        }
    )
    captured = {}
    monkeypatch.setattr(evaluate, "_load_model", lambda path: MarkdownDummyModel())
    monkeypatch.setattr(evaluate, "_read_labels", lambda path: labels)
    monkeypatch.setattr(
        evaluate,
        "_load_selected_data",
        lambda args, selected_labels: evaluate._attach_labels(data, selected_labels),
    )

    def unexpected_features(*args, **kwargs):
        raise AssertionError("text-only models do not need numeric feature loading")

    def fake_markdown(metadata, **kwargs):
        captured["ids"] = metadata["file_url_sha16"].tolist()
        captured["repo_ids"] = metadata["repo_id"].tolist()
        captured.update(kwargs)
        return pd.DataFrame(
            {"file_url_sha16": ["a"], evaluate.MARKDOWN_COLUMN: ["Human text"]}
        )

    monkeypatch.setattr(evaluate, "_load_features", unexpected_features)
    monkeypatch.setattr(evaluate, "load_spec_markdown", fake_markdown)
    args = argparse.Namespace(
        model_path="model.joblib",
        labels_csv="labels.csv",
        sample_size=None,
        split="test",
        local_data_only=True,
        data_dir=None,
        threshold=0.5,
    )

    result = evaluate._evaluate(args)

    assert captured["ids"] == ["a", "b"]
    assert captured["repo_ids"] == [101, 102]
    assert captured["local_data_only"] is True
    assert result["n_examples"] == 1
    assert result["n_unavailable_markdown"] == 1
    assert result["n_unlabeled"] == 1
    assert result["predictions"][0]["id"] == "a"


def test_text_evaluation_requires_bounded_unlabeled_sample(monkeypatch):
    monkeypatch.setattr(evaluate, "_load_model", lambda path: MarkdownDummyModel())
    args = argparse.Namespace(model_path="model.joblib", labels_csv=None, sample_size=None)
    with pytest.raises(ValueError, match="requires --sample-size"):
        evaluate._evaluate(args)


def test_main_joins_labels_and_writes_json_and_csv(tmp_path, monkeypatch):
    eval_frame = pd.DataFrame(
        {
            "file_url_sha16": ["a", "b", "c"],
            "repo_name": ["r1", "r2", "r3"],
            "file_path": ["a.md", "b.md", "c.md"],
        }
    )
    train_frame = eval_frame.iloc[:0].copy()
    feature_frame = pd.DataFrame(
        {
            "file_url_sha16": ["a", "b", "c"],
            "n_lines": [1, 0, 1],
            "n_words": [10, 5, 10],
            "n_headings": [2, 1, 2],
            "n_code_fences": [0, 0, 0],
            "n_list_items": [1, 1, 1],
            "n_checkboxes": [0, 0, 0],
            "n_checked": [0, 0, 0],
            "is_tiny": [0, 0, 0],
            "has_ears": [0, 0, 0],
            "has_gherkin": [0, 0, 0],
            "has_user_story": [0, 0, 0],
        }
    )
    labels_path = tmp_path / "labels.csv"
    pd.DataFrame(
        {
            "spec_id": ["a", "b", "outside"],
            "label": [1, None, 0],
            "confidence": [90, None, 80],
        }
    ).to_csv(labels_path, index=False)
    model_path = tmp_path / "model.joblib"
    model_path.touch()
    output_json = tmp_path / "result.json"
    output_csv = tmp_path / "predictions.csv"
    seen_options = {}
    requested_feature_ids = []

    def fake_load_data(**kwargs):
        seen_options.update(kwargs)
        return train_frame, eval_frame

    def fake_load_features(ids, **kwargs):
        requested_feature_ids.extend(ids["file_url_sha16"].tolist())
        return feature_frame.loc[
            feature_frame["file_url_sha16"].eq("a")
            & feature_frame["file_url_sha16"].isin(ids["file_url_sha16"])
        ].copy()

    monkeypatch.setattr(evaluate, "load_data", fake_load_data)
    monkeypatch.setattr(evaluate, "load_feature_data", fake_load_features)
    monkeypatch.setattr(evaluate.joblib, "load", lambda path: FeatureDummyModel())

    status = evaluate.main(
        [
            "--model-path",
            str(model_path),
            "--labels-csv",
            str(labels_path),
            "--split",
            "test",
            "--data-dir",
            str(tmp_path / "cache"),
            "--local-data-only",
            "--output-json",
            str(output_json),
            "--output-csv",
            str(output_csv),
        ]
    )

    result = json.loads(output_json.read_text(encoding="utf-8"))
    predictions = pd.read_csv(output_csv)
    assert status == 0
    assert seen_options["data_dir"] == tmp_path / "cache"
    assert seen_options["local_data_only"] is True
    assert result["n_examples"] == 1
    assert result["n_labeled"] == 1
    assert result["n_unlabeled"] == 1
    assert result["n_unmatched"] == 1
    assert predictions["id"].tolist() == ["a"]
    assert requested_feature_ids == ["a"]


def test_main_rejects_missing_model_file(tmp_path):
    status = evaluate.main(["--model-path", str(tmp_path / "missing.joblib")])

    assert status == 1


def test_main_without_labels_writes_predictions_only(tmp_path, monkeypatch):
    data = pd.DataFrame(
        {
            "file_url_sha16": ["a", "b"],
            "repo_name": ["r1", "r2"],
            "file_path": ["a.md", "b.md"],
        }
    )
    feature_frame = pd.DataFrame(
        {
            "file_url_sha16": ["a", "b"],
            "n_lines": [1, 0],
            "n_words": [10, 5],
            "n_headings": [2, 1],
            "n_code_fences": [0, 0],
            "n_list_items": [1, 1],
            "n_checkboxes": [0, 0],
            "n_checked": [0, 0],
            "is_tiny": [0, 0],
            "has_ears": [0, 0],
            "has_gherkin": [0, 0],
            "has_user_story": [0, 0],
        }
    )
    model_path = tmp_path / "model.joblib"
    model_path.touch()
    output_json = tmp_path / "result.json"
    monkeypatch.setattr(evaluate, "load_data", lambda **kwargs: (data.iloc[:0], data))
    monkeypatch.setattr(
        evaluate,
        "load_feature_data",
        lambda ids, **kwargs: feature_frame.loc[
            feature_frame["file_url_sha16"].isin(ids["file_url_sha16"])
        ].copy(),
    )
    monkeypatch.setattr(evaluate.joblib, "load", lambda path: FeatureDummyModel())

    status = evaluate.main(
        [
            "--model-path",
            str(model_path),
            "--output-json",
            str(output_json),
        ]
    )

    result = json.loads(output_json.read_text(encoding="utf-8"))
    assert status == 0
    assert result["metrics"] is None
    assert result["n_labeled"] == 0
    assert result["n_unlabeled"] == 2
    assert len(result["predictions"]) == 2


def test_write_outputs_escapes_spreadsheet_formulas_in_csv_metadata(tmp_path):
    output_csv = tmp_path / "predictions.csv"
    result = {
        "predictions": [
            {
                "id": '=HYPERLINK("https://example.com")',
                "repo_name": "\t@SUM(1,1)",
                "file_path": "-1+2",
                "y_pred": 1,
            },
            {
                "id": "safe-id",
                "repo_name": "owner/repo",
                "file_path": "spec.md",
                "y_pred": 0,
            },
        ]
    }

    evaluate._write_outputs(
        result,
        argparse.Namespace(output_json=None, output_csv=output_csv),
    )
    predictions = pd.read_csv(output_csv, dtype=str)

    assert predictions["id"].tolist() == [
        "'" + '=HYPERLINK("https://example.com")',
        "safe-id",
    ]
    assert predictions["repo_name"].tolist() == ["'\t@SUM(1,1)", "owner/repo"]
    assert predictions["file_path"].tolist() == ["'-1+2", "spec.md"]


def test_validation_split_is_deterministic_and_stratified():
    frame = pd.DataFrame(
        {
            "index_value": range(20),
            "label": [0, 1] * 10,
        }
    )

    train_a, val_a = evaluate._validation_split(frame, 0.25, 7)
    train_b, val_b = evaluate._validation_split(frame, 0.25, 7)

    assert val_a["index_value"].tolist() == val_b["index_value"].tolist()
    counts = val_a["label"].value_counts().to_dict()
    assert set(counts) == {0, 1}
    assert sorted(counts.values()) == [2, 3]
    assert len(train_a) + len(val_a) == len(frame)
    assert len(train_b) + len(val_b) == len(frame)
