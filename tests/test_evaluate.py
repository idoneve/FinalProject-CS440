import json

import numpy as np
import pandas as pd
import pytest

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


def test_evaluate_model_rejects_invalid_threshold():
    with pytest.raises(ValueError, match="threshold"):
        evaluate.evaluate_model(DummyModel(), labeled_frame(), threshold=1.1)


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
        {"spec_id": ["a", "b", "outside"], "label": [1, None, 0]}
    ).to_csv(labels_path, index=False)
    model_path = tmp_path / "model.joblib"
    model_path.touch()
    output_json = tmp_path / "result.json"
    output_csv = tmp_path / "predictions.csv"
    seen_options = {}

    def fake_load_data(**kwargs):
        seen_options.update(kwargs)
        return train_frame, eval_frame

    monkeypatch.setattr(evaluate, "load_data", fake_load_data)
    monkeypatch.setattr(
        evaluate,
        "load_feature_data",
        lambda ids: feature_frame.loc[
            feature_frame["file_url_sha16"].isin(ids["file_url_sha16"])
        ].copy(),
    )
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
    assert result["n_examples"] == 1
    assert result["n_labeled"] == 1
    assert result["n_unlabeled"] == 1
    assert result["n_unmatched"] == 1
    assert predictions["id"].tolist() == ["a"]


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
        lambda ids: feature_frame.loc[
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
