import json
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
import pytest

import specmine.data as data
import specmine.train as train

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def notebook_data(tmp_path, monkeypatch):
    rows = []
    feature_rows = []
    label_rows = []
    for index in range(80):
        labeled = index < 40
        spec_id = f"s{index:02d}" if labeled else f"u{index - 40:02d}"
        label = index % 2
        rows.append(
            {
                "file_url_sha16": spec_id,
                "repo_name": f"owner/repo-{index:02d}",
                "file_url": f"https://github.com/owner/repo-{index}/blob/abc/spec.md",
                "file_path": "spec.md",
                "file_sha": f"{index:040x}",
            }
        )
        feature_rows.append(
            {
                "file_url_sha16": spec_id,
                "n_lines": 2 + label * 8,
                "n_words": 20 + label * 100,
                "n_headings": 1 + label * 2,
                "n_code_fences": label,
                "n_list_items": 1 + label,
                "n_checkboxes": 0,
                "n_checked": 0,
                "is_tiny": 0,
                "has_ears": label,
                "has_gherkin": label,
                "has_user_story": 0,
            }
        )
        if labeled:
            label_rows.append({"spec_id": spec_id, "label": label, "confidence": 80})

    metadata = pd.DataFrame(rows)
    feature_data = pd.DataFrame(feature_rows)
    labels_path = tmp_path / "labels.csv"
    pd.DataFrame(label_rows).to_csv(labels_path, index=False)
    missing_unlabeled = set(
        metadata.iloc[40:]
        .sample(frac=1, random_state=42)
        .head(3)["file_url_sha16"]
    )
    missing_markdown = {"s03", "s07"} | missing_unlabeled
    requests = {"features": [], "markdown": []}

    def fake_load_data(**kwargs):
        assert kwargs["local_data_only"] is True
        return metadata.copy()

    def fake_load_features(spec_ids, **kwargs):
        assert kwargs["local_data_only"] is True
        ids = spec_ids["file_url_sha16"].astype(str).tolist()
        requests["features"].append(ids)
        return feature_data.loc[feature_data["file_url_sha16"].isin(ids)].copy()

    def fake_load_markdown(selected, **kwargs):
        assert kwargs["local_data_only"] is True
        ids = selected["file_url_sha16"].astype(str).tolist()
        requests["markdown"].append(ids)
        records = [
            {
                "file_url_sha16": spec_id,
                data.MARKDOWN_COLUMN: f"# Specification {spec_id}\nContent for {spec_id}\n",
            }
            for spec_id in ids
            if spec_id not in missing_markdown
        ]
        return pd.DataFrame(records, columns=["file_url_sha16", data.MARKDOWN_COLUMN])

    monkeypatch.setattr(data, "load_data", fake_load_data)
    monkeypatch.setattr(data, "load_feature_data", fake_load_features)
    monkeypatch.setattr(data, "load_spec_markdown", fake_load_markdown)
    monkeypatch.setattr(train, "load_feature_data", fake_load_features)
    monkeypatch.setattr(train, "load_spec_markdown", fake_load_markdown)
    monkeypatch.chdir(ROOT)
    return labels_path, requests, missing_unlabeled


def notebook_sources(name):
    notebook = json.loads((ROOT / "notebooks" / name).read_text(encoding="utf-8"))
    return [
        "".join(cell.get("source", [])) if cell["cell_type"] == "code" else None
        for cell in notebook["cells"]
    ]


def test_data_init_downloads_missing_markdown_by_default(monkeypatch, tmp_path):
    notebook = json.loads(
        (ROOT / "notebooks" / "data_init.ipynb").read_text(encoding="utf-8")
    )
    cells = {
        cell["id"]: "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    }
    metadata = pd.DataFrame(
        {
            "repo_id": [123],
            "repo_name": ["owner/repo"],
            "file_url_sha16": ["0123456789abcdef"],
            "file_url": ["https://github.com/owner/repo/blob/main/spec.md"],
            "file_path": ["spec.md"],
        }
    )
    spec_path = tmp_path / "spec.md"
    spec_path.write_text("# Test specification\n", encoding="utf-8")
    sample = metadata.assign(local_path=str(spec_path))
    feature_rows = pd.DataFrame(
        {"file_url_sha16": ["0123456789abcdef"], "n_lines": [1]}
    )
    calls = {}

    def fake_load_data(**kwargs):
        assert kwargs["download_raw_specs"] is False
        return metadata.copy()

    def fake_download_random_spec_sample(*args, **kwargs):
        calls["local_data_only"] = kwargs["local_data_only"]
        return sample.copy()

    def fake_load_feature_data(selected_ids, **kwargs):
        assert selected_ids["file_url_sha16"].tolist() == ["0123456789abcdef"]
        return feature_rows.copy()

    monkeypatch.setattr(data, "load_data", fake_load_data)
    monkeypatch.setattr(
        data, "download_random_spec_sample", fake_download_random_spec_sample
    )
    monkeypatch.setattr(data, "load_feature_data", fake_load_feature_data)
    monkeypatch.chdir(ROOT)

    namespace = {}
    exec(cells["data-init-setup"], namespace)
    namespace["DATA_DIR"] = tmp_path
    exec(cells["download-sample"], namespace)
    exec(cells["select-review-repository"], namespace)
    exec(cells["show-review-spec"], namespace)

    assert calls["local_data_only"] is False
    assert (tmp_path / "spec_sample.csv").is_file()
    assert namespace["review_spec"]["file_url_sha16"].tolist() == [
        "0123456789abcdef"
    ]
    assert namespace["review_features"]["n_lines"].tolist() == [1]
    assert namespace["spec_markdown"] == "# Test specification\n"


@pytest.mark.parametrize(
    "model_name",
    [
        "logistic_regression",
        "random_forest",
        "naive_bayes",
        "uniform_guess",
        "markdown_plus_feature",
    ],
)
def test_prediction_notebook_runs_selected_models_offline(
    model_name, notebook_data, tmp_path
):
    labels_path, requests, missing_unlabeled = notebook_data
    cells = notebook_sources("predict.ipynb")
    namespace = {"display": lambda *args, **kwargs: None}
    exec(cells[1], namespace)
    namespace.update(
        {
            "MODEL_NAME": model_name,
            "TEXT_FEATURE": "n_lines",
            "PREDICTION_SAMPLE_SIZE": 12,
            "LABELS_CSV": labels_path,
            "MODEL_DIR": tmp_path / "models",
            "PREDICTIONS_CSV": tmp_path / "predictions.csv",
            "DATA_DIR": tmp_path,
            "LOCAL_DATA_ONLY": True,
        }
    )
    for index in (3, 5, 7):
        exec(cells[index], namespace)

    output_paths = list(tmp_path.glob("predictions_*.csv"))
    assert len(output_paths) == 1
    predictions = pd.read_csv(output_paths[0])
    assert len(predictions) == 12
    if model_name in train.TEXT_MODELS:
        assert set(predictions["spec_id"]).isdisjoint(missing_unlabeled)
    assert predictions["spec_id"].str.startswith("u").all()
    assert predictions["probability_ai_assisted"].between(0, 1).all()

    model = joblib.load(tmp_path / "models" / f"{model_name}_all_labeled.joblib")
    if model_name == "uniform_guess":
        assert predictions["probability_ai_assisted"].eq(0.5).all()
        assert set(predictions["predicted_label"]) == {0, 1}
    if model_name == "naive_bayes":
        assert list(model.feature_names_in_) == [data.MARKDOWN_COLUMN]
    if model_name == "markdown_plus_feature":
        assert list(model.feature_names_in_) == [data.MARKDOWN_COLUMN, "n_lines"]

    unlabeled_feature_requests = [
        ids for ids in requests["features"] if any(value.startswith("u") for value in ids)
    ]
    assert bool(unlabeled_feature_requests) == (
        model_name in {"logistic_regression", "random_forest", "markdown_plus_feature"}
    )
    assert bool(requests["markdown"]) == (model_name in train.TEXT_MODELS)
    if model_name in train.TEXT_MODELS:
        assert any(
            missing_unlabeled.intersection(ids) for ids in requests["markdown"]
        )


def test_results_notebook_uses_common_available_holdout_offline(
    notebook_data, tmp_path, monkeypatch
):
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    labels_path, _, _ = notebook_data
    cells = notebook_sources("results.ipynb")
    namespace = {}
    with matplotlib.rc_context():
        exec(cells[1], namespace)
        namespace.update(
            {
                "LABELS_CSV": labels_path,
                "FIGURE_DIR": tmp_path,
                "DATA_DIR": tmp_path,
                "LOCAL_DATA_ONLY": True,
                "BOOTSTRAP_REPLICATES": 10,
                "display": lambda *args, **kwargs: None,
            }
        )
        monkeypatch.setattr(Figure, "savefig", lambda *args, **kwargs: None)
        monkeypatch.setattr(plt, "show", lambda: plt.close("all"))
        for cell in cells[2:]:
            if cell is not None:
                exec(cell, namespace)

    results = namespace["training_results"]
    assert set(results) == set(train.AVAILABLE_MODELS)
    assert len(namespace["excluded"]) == 2
    assert len(namespace["training_features"]) == 38
    assert set(namespace["summary_table"].index) == set(train.AVAILABLE_MODELS)
    holdout_ids = {
        name: [row["id"] for row in result["predictions"]]
        for name, result in results.items()
    }
    assert all(ids == holdout_ids["logistic_regression"] for ids in holdout_ids.values())
    guesses = namespace["prediction_frames"]["uniform_guess"]
    np.testing.assert_array_equal(guesses["y_prob"], np.full(len(guesses), 0.5))
