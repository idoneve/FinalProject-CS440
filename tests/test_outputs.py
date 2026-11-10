from pathlib import Path

import pandas as pd
import pytest

from specmine.outputs import FigureRun, write_csv_atomic


def test_atomic_csv_preserves_existing_manifest_on_write_failure(tmp_path, monkeypatch):
    destination = tmp_path / "spec_sample.csv"
    destination.write_text("original\n", encoding="utf-8")

    def broken_write(self, stream, **kwargs):
        stream.write("partial")
        raise OSError("interrupted")

    monkeypatch.setattr(pd.DataFrame, "to_csv", broken_write)
    with pytest.raises(OSError, match="interrupted"):
        write_csv_atomic(pd.DataFrame({"id": [1]}), destination)
    assert destination.read_text(encoding="utf-8") == "original\n"
    assert list(tmp_path.iterdir()) == [destination]


def test_atomic_csv_replaces_complete_manifest(tmp_path):
    destination = tmp_path / "spec_sample.csv"
    destination.write_text("old\n", encoding="utf-8")
    write_csv_atomic(pd.DataFrame({"id": [1, 2]}), destination)
    assert pd.read_csv(destination)["id"].tolist() == [1, 2]


class FakeFigure:
    def __init__(self, content, fail_extension=None):
        self.content = content
        self.fail_extension = fail_extension

    def savefig(self, path, **kwargs):
        path = Path(path)
        path.write_text(self.content + path.suffix, encoding="utf-8")
        if path.suffix == self.fail_extension:
            raise OSError("save interrupted")


def test_figure_runs_publish_complete_separate_sets(tmp_path):
    first, second = FigureRun(tmp_path), FigureRun(tmp_path)
    first.save(FakeFigure("first"), "plot")
    second.save(FakeFigure("second"), "plot")
    assert not list(tmp_path.glob("run-*"))
    second_paths = second.publish()
    first_paths = first.publish()
    assert first.destination != second.destination
    assert all(path.read_text().startswith("first") for path in first_paths)
    assert all(path.read_text().startswith("second") for path in second_paths)


def test_figure_failure_cannot_publish_partial_run(tmp_path):
    run = FigureRun(tmp_path)
    with pytest.raises(OSError, match="interrupted"):
        run.save(FakeFigure("partial", ".pdf"), "plot")
    with pytest.raises(RuntimeError, match="failed"):
        run.publish()
    assert not list(tmp_path.glob("run-*"))


def test_missing_staged_format_prevents_publication(tmp_path):
    run = FigureRun(tmp_path)
    run.save(FakeFigure("complete"), "plot")
    (run.staging / "plot.pdf").unlink()
    with pytest.raises(RuntimeError, match="missing or empty"):
        run.publish()
    assert not list(tmp_path.glob("run-*"))
