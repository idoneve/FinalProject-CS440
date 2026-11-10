import builtins
import csv
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def labeler(monkeypatch):
    module_name = "ai_labeler_under_test"
    spec = importlib.util.spec_from_file_location(
        module_name, ROOT / "src" / "ai-labeler.py"
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


def test_labeler_default_data_directory(labeler):
    from specmine.data import DEFAULT_LOCAL_DATA_DIR

    assert labeler.DEFAULT_DATA_DIR == ROOT / "data" == DEFAULT_LOCAL_DATA_DIR


class FakeAPIError(Exception):
    def __init__(self, code):
        super().__init__("private-fake-error-details")
        self.code = code


class FakeTransportError(Exception):
    pass


class FakeClient:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []
        self.models = self
        self.closed = False

    def generate_content(self, *, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        response = self.responses[contents]
        if isinstance(response, Exception):
            raise response
        if not isinstance(response, str):
            response = json.dumps(response)
        return SimpleNamespace(text=response)

    def close(self):
        self.closed = True


def install_fake_client(monkeypatch, labeler, responses):
    client = FakeClient(responses)
    monkeypatch.setattr(
        labeler,
        "create_gemini_analyzer",
        lambda model: labeler.GeminiAnalyzer(
            client, model, FakeAPIError, (FakeTransportError,)
        ),
    )
    return client


def write_spec(data_dir, repo_name, repo_id, spec_id, content):
    path = data_dir / f"{repo_name}-{repo_id}-{spec_id}.md"
    path.write_bytes(content.encode("utf-8"))
    return path


def write_results(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["repo_id", "spec_id", "label", "confidence", "reasoning"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def read_results(path):
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
    return reader.fieldnames, rows


def test_analyzes_each_file_and_resumes_per_spec(labeler, monkeypatch, tmp_path):
    first_id = "0000000000000001"
    second_id = "0000000000000002"
    third_id = "0000000000000003"
    first_content = "# Already analyzed\n"
    second_content = "# Alpha\r\n- A requirement\r\n"
    third_content = "# Beta\nA different specification.\n"
    first = write_spec(
        tmp_path, "owner-name_repo-with-dashes", "100", first_id, first_content
    )
    second = write_spec(
        tmp_path, "owner-name_repo-with-dashes", "100", second_id, second_content
    )
    third = write_spec(tmp_path, "other_owner_repo", "200", third_id, third_content)
    output = tmp_path / "ai_analysis_results.csv"
    write_results(
        output,
        [
            {
                "repo_id": "100",
                "spec_id": first_id,
                "label": 0,
                "confidence": 80,
                "reasoning": "Existing result",
            }
        ],
    )
    output.write_bytes(output.read_bytes().rstrip(b"\n"))
    reasoning = 'A heading, then "requirements".\nSpecific detail.'
    client = install_fake_client(
        monkeypatch,
        labeler,
        {
            second_content: {"label": 1, "confidence": 77, "reasoning": reasoning},
            third_content: {"label": 0, "confidence": 24, "reasoning": "Beta details"},
        },
    )

    status = labeler.main(
        ["--data-dir", str(tmp_path), "--model", "test-model", "--delay-seconds", "0"]
    )

    assert status == 0
    assert client.closed
    assert {call["contents"] for call in client.calls} == {
        second_content,
        third_content,
    }
    assert all(call["model"] == "test-model" for call in client.calls)
    assert all(
        call["config"]["response_mime_type"] == "application/json"
        and call["config"]["response_json_schema"]["required"]
        == ["label", "confidence", "reasoning"]
        for call in client.calls
    )
    assert first.read_bytes() == first_content.encode()
    assert second.read_bytes() == second_content.encode()
    assert third.read_bytes() == third_content.encode()

    header, rows = read_results(output)
    assert header == ["repo_id", "spec_id", "label", "confidence", "reasoning"]
    by_id = {row["spec_id"]: row for row in rows}
    assert set(by_id) == {first_id, second_id, third_id}
    assert by_id[second_id] == {
        "repo_id": "100",
        "spec_id": second_id,
        "label": "1",
        "confidence": "77",
        "reasoning": reasoning,
    }
    assert by_id[third_id]["repo_id"] == "200"
    original_output = output.read_bytes()

    def unexpected_client(model):
        raise AssertionError("completed specs must not contact Gemini")

    monkeypatch.setattr(labeler, "create_gemini_analyzer", unexpected_client)
    assert labeler.main(["--data-dir", str(tmp_path), "--delay-seconds", "0"]) == 0
    assert output.read_bytes() == original_output


def test_uppercase_spec_ids_are_normalized_for_selection_and_resume(
    labeler, monkeypatch, tmp_path
):
    spec_id = "ABCDEF0123456789"
    write_spec(tmp_path, "owner_repo", "100", spec_id, "# Already analyzed\n")
    output = tmp_path / "ai_analysis_results.csv"
    write_results(
        output,
        [
            {
                "repo_id": "100",
                "spec_id": spec_id,
                "label": 0,
                "confidence": 80,
                "reasoning": "Existing result",
            }
        ],
    )
    client = install_fake_client(monkeypatch, labeler, {})

    status = labeler.main(
        [
            "--data-dir",
            str(tmp_path),
            "--spec-id",
            spec_id.lower(),
            "--delay-seconds",
            "0",
        ]
    )

    assert status == 0
    assert client.calls == []
    _, rows = read_results(output)
    assert len(rows) == 1
    assert rows[0]["spec_id"] == spec_id


def test_real_sdk_serializes_selected_files_without_network(
    labeler, monkeypatch, tmp_path
):
    genai = pytest.importorskip("google.genai")
    httpx = pytest.importorskip("httpx")
    from google.genai import errors

    first_id = "0123456789abcdef"
    second_id = "fedcba9876543210"
    third_id = "0000000000000003"
    first_content = "# Synthetic alpha\r\nAn alpha requirement.\r\n"
    second_content = "# Synthetic beta\nA beta requirement.\n"
    write_spec(tmp_path, "owner_alpha", "101", first_id, first_content)
    write_spec(tmp_path, "owner_beta", "202", second_id, second_content)
    write_spec(tmp_path, "owner_gamma", "303", third_id, "# Not selected\n")
    requests = []

    def handle(request):
        payload = json.loads(request.content)
        requests.append(payload)
        content = payload["contents"][0]["parts"][0]["text"]
        response = {
            "label": 1 if "alpha" in content else 0,
            "confidence": 63,
            "reasoning": "The synthetic alpha heading." if "alpha" in content
            else "The synthetic beta heading.",
        }
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {
                            "parts": [{"text": json.dumps(response)}],
                            "role": "model",
                        },
                        "finishReason": "STOP",
                    }
                ]
            },
        )

    client = genai.Client(
        api_key="offline-test-key",
        vertexai=False,
        http_options={"client_args": {"transport": httpx.MockTransport(handle)}},
    )
    monkeypatch.setattr(
        labeler,
        "create_gemini_analyzer",
        lambda model: labeler.GeminiAnalyzer(
            client, model, errors.APIError, (httpx.RequestError,)
        ),
    )
    status = labeler.main(
        [
            "--data-dir",
            str(tmp_path),
            "--spec-id",
            first_id,
            "--spec-id",
            second_id,
            "--delay-seconds",
            "0",
        ]
    )

    assert status == 0
    assert {
        payload["contents"][0]["parts"][0]["text"] for payload in requests
    } == {first_content, second_content}
    assert all(
        payload["generationConfig"]["responseMimeType"] == "application/json"
        and payload["generationConfig"]["responseJsonSchema"]["required"]
        == ["label", "confidence", "reasoning"]
        for payload in requests
    )
    _, rows = read_results(tmp_path / "ai_analysis_results.csv")
    assert {(row["repo_id"], row["spec_id"], row["label"]) for row in rows} == {
        ("101", first_id, "1"),
        ("202", second_id, "0"),
    }


@pytest.mark.parametrize(
    "response",
    [
        None,
        "",
        "not JSON",
        "[]",
        "{}",
        '{"label": true, "confidence": 70, "reasoning": "text"}',
        '{"label": 2, "confidence": 70, "reasoning": "text"}',
        '{"label": 1, "confidence": -1, "reasoning": "text"}',
        '{"label": 1, "confidence": 101, "reasoning": "text"}',
        '{"label": 1, "confidence": 50.0, "reasoning": "text"}',
        '{"label": 1, "confidence": true, "reasoning": "text"}',
        '{"label": 1, "confidence": 50, "reasoning": "  "}',
    ],
)
def test_rejects_invalid_gemini_responses(labeler, response):
    with pytest.raises(labeler.AnalysisError):
        labeler.validate_analysis(response)


@pytest.mark.parametrize("label, confidence", [(0, 0), (1, 100)])
def test_accepts_valid_binary_response_boundaries(labeler, label, confidence):
    result = labeler.validate_analysis(
        json.dumps({"label": label, "confidence": confidence, "reasoning": "  detail  "})
    )
    assert result == {"label": label, "confidence": confidence, "reasoning": "detail"}


def test_escapes_formula_reasoning_and_preserves_repository_id(
    labeler, monkeypatch, tmp_path
):
    spec_id = "0000000000000001"
    content = "# Formula safety\n"
    write_spec(tmp_path, "owner_repo-with-dash", "00101.0", spec_id, content)
    reason = '=HYPERLINK("https://example.invalid","reason")'
    install_fake_client(
        monkeypatch,
        labeler,
        {content: {"label": 0, "confidence": 40, "reasoning": reason}},
    )

    assert labeler.main(["--data-dir", str(tmp_path), "--delay-seconds", "0"]) == 0
    _, rows = read_results(tmp_path / "ai_analysis_results.csv")
    assert rows[0]["repo_id"] == "00101"
    assert rows[0]["spec_id"] == spec_id
    assert rows[0]["reasoning"] == "'" + reason


def test_delays_between_requests_not_after_the_last(
    labeler, monkeypatch, tmp_path
):
    first = "# First\n"
    second = "# Second\n"
    write_spec(tmp_path, "owner_one", "1", "0000000000000001", first)
    write_spec(tmp_path, "owner_two", "2", "0000000000000002", second)
    install_fake_client(
        monkeypatch,
        labeler,
        {
            first: {"label": 0, "confidence": 20, "reasoning": "First"},
            second: {"label": 1, "confidence": 80, "reasoning": "Second"},
        },
    )
    delays = []
    monkeypatch.setattr(labeler.time, "sleep", delays.append)

    assert labeler.main(["--data-dir", str(tmp_path), "--delay-seconds", "2.5"]) == 0
    assert delays == [2.5]


def test_reports_failures_and_continues_with_other_files(
    labeler, monkeypatch, tmp_path, capsys
):
    bad_utf8 = write_spec(tmp_path, "a_repo", "101", "0000000000000001", "placeholder")
    bad_utf8.write_bytes(b"\xff")
    write_spec(tmp_path, "b_repo", "102", "0000000000000002", "  \n")
    write_spec(tmp_path, "c_repo", "103", "0000000000000003", "# API failure\n")
    write_spec(tmp_path, "d_repo", "104", "0000000000000004", "# Bad response\n")
    write_spec(tmp_path, "e_repo", "105", "0000000000000005", "# Network failure\n")
    write_spec(tmp_path, "f_repo", "106", "0000000000000006", "# Good response\n")
    (tmp_path / "not-a-cached-spec.md").write_text("# invalid filename\n")
    client = install_fake_client(
        monkeypatch,
        labeler,
        {
            "# API failure\n": FakeAPIError(429),
            "# Bad response\n": {"label": 2, "confidence": 50, "reasoning": "bad"},
            "# Network failure\n": FakeTransportError("private-fake-error-details"),
            "# Good response\n": {"label": 1, "confidence": 50, "reasoning": "good"},
        },
    )

    status = labeler.main(["--data-dir", str(tmp_path), "--delay-seconds", "0"])
    captured = capsys.readouterr()

    assert status == 1
    assert len(client.calls) == 4
    assert client.closed
    assert "Invalid cached spec filename" in captured.err
    assert "UnicodeDecodeError" in captured.err
    assert "Spec file is empty" in captured.err
    assert "HTTP 429" in captured.err
    assert "label must be 0 or 1" in captured.err
    assert "Gemini network request failed (FakeTransportError)" in captured.err
    assert "private-fake-error-details" not in captured.err
    _, rows = read_results(tmp_path / "ai_analysis_results.csv")
    assert [row["spec_id"] for row in rows] == ["0000000000000006"]


def test_duplicate_ids_and_repository_conflicts_do_not_hide_other_specs(
    labeler, monkeypatch, tmp_path, capsys
):
    duplicate_id = "0000000000000001"
    conflict_id = "0000000000000002"
    good_id = "0000000000000003"
    write_spec(tmp_path, "owner_one", "100", duplicate_id, "# First copy\n")
    write_spec(tmp_path, "owner_two", "200", duplicate_id, "# Second copy\n")
    write_spec(tmp_path, "owner_three", "300", conflict_id, "# Conflicting repo\n")
    write_spec(tmp_path, "owner_four", "400", good_id, "# Good file\n")
    output = tmp_path / "ai_analysis_results.csv"
    write_results(
        output,
        [
            {
                "repo_id": "999",
                "spec_id": conflict_id,
                "label": 0,
                "confidence": 30,
                "reasoning": "Prior result",
            }
        ],
    )
    client = install_fake_client(
        monkeypatch,
        labeler,
        {"# Good file\n": {"label": 1, "confidence": 75, "reasoning": "Good"}},
    )

    status = labeler.main(["--data-dir", str(tmp_path), "--delay-seconds", "0"])
    captured = capsys.readouterr()

    assert status == 1
    assert len(client.calls) == 1
    assert "Multiple cached files" in captured.err
    assert "existing results identify repository 999" in captured.err
    _, rows = read_results(output)
    assert [row["spec_id"] for row in rows] == [conflict_id, good_id]


@pytest.mark.parametrize(
    "contents, error",
    [
        ("wrong,header\n", "Invalid results CSV header"),
        (
            "repo_id,spec_id,label,confidence,reasoning\n"
            "1,0000000000000001,0,20,first\n"
            "1,0000000000000001,1,80,second\n",
            "Duplicate spec ID",
        ),
        (
            "repo_id,spec_id,label,confidence,reasoning\n"
            "1,0000000000000001,0,101,reason\n",
            "Invalid confidence",
        ),
        (
            "repo_id,spec_id,label,confidence,reasoning\n"
            '1,0000000000000001,0,20,"unfinished\n',
            "Cannot read results CSV",
        ),
    ],
)
def test_invalid_existing_csv_is_reported_without_changes(
    labeler, monkeypatch, tmp_path, capsys, contents, error
):
    write_spec(tmp_path, "owner_repo", "1", "0000000000000001", "# Spec\n")
    output = tmp_path / "ai_analysis_results.csv"
    output.write_text(contents, encoding="utf-8")
    original = output.read_bytes()

    def unexpected_client(model):
        raise AssertionError("invalid existing results must be rejected first")

    monkeypatch.setattr(labeler, "create_gemini_analyzer", unexpected_client)
    assert labeler.main(["--data-dir", str(tmp_path)]) == 1
    assert error in capsys.readouterr().err
    assert output.read_bytes() == original


def test_missing_key_and_missing_sdk_are_reported_without_exposing_key(
    labeler, monkeypatch, tmp_path, capsys
):
    write_spec(tmp_path, "owner_repo", "1", "0000000000000001", "# Spec\n")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert labeler.main(["--data-dir", str(tmp_path)]) == 1
    assert "Set GEMINI_API_KEY" in capsys.readouterr().err
    assert not (tmp_path / "ai_analysis_results.csv").exists()

    fake_key = "private-fake-test-key"
    monkeypatch.setenv("GEMINI_API_KEY", fake_key)
    original_import = builtins.__import__

    def missing_google(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "google" and "genai" in fromlist:
            raise ImportError("simulated missing SDK")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", missing_google)
    assert labeler.main(["--data-dir", str(tmp_path)]) == 1
    captured = capsys.readouterr()
    assert "Gemini SDK is missing" in captured.err
    assert fake_key not in captured.err
    assert not (tmp_path / "ai_analysis_results.csv").exists()


def test_authentication_failure_stops_and_resume_retries_unfinished_specs(
    labeler, monkeypatch, tmp_path, capsys
):
    first_id, second_id = "0000000000000001", "0000000000000002"
    write_spec(tmp_path, "a_repo", "1", first_id, "# First\n")
    write_spec(tmp_path, "b_repo", "2", second_id, "# Second\n")
    client = install_fake_client(monkeypatch, labeler, {"# First\n": FakeAPIError(401)})

    assert labeler.main(["--data-dir", str(tmp_path), "--delay-seconds", "0"]) == 1
    captured = capsys.readouterr()
    assert "HTTP 401" in captured.err
    assert "1 remaining spec(s) were not attempted" in captured.err
    assert len(client.calls) == 1
    assert client.closed
    _, rows = read_results(tmp_path / "ai_analysis_results.csv")
    assert rows == []

    client = install_fake_client(
        monkeypatch,
        labeler,
        {
            "# First\n": {"label": 0, "confidence": 35, "reasoning": "First"},
            "# Second\n": {"label": 1, "confidence": 60, "reasoning": "Second"},
        },
    )
    assert labeler.main(["--data-dir", str(tmp_path), "--delay-seconds", "0"]) == 0
    assert len(client.calls) == 2
    _, rows = read_results(tmp_path / "ai_analysis_results.csv")
    assert {row["spec_id"] for row in rows} == {first_id, second_id}


def test_missing_requested_id_and_symlink_are_reported(labeler, tmp_path, capsys):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    outside = tmp_path / "outside.md"
    outside.write_text("# Do not transmit\n", encoding="utf-8")
    link = data_dir / "owner_repo-1-0000000000000001.md"
    link.symlink_to(outside)

    status = labeler.main(
        ["--data-dir", str(data_dir), "--spec-id", "0000000000000001"]
    )
    captured = capsys.readouterr()
    assert status == 1
    assert "symbolic link" in captured.err
    assert "No cached Markdown file found" in captured.err
    assert not (data_dir / "ai_analysis_results.csv").exists()
