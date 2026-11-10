import pytest

from specmine.migrate_cache import copy_tree


def test_copy_cache_is_idempotent_and_preserves_destinations(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    (source / "data").mkdir(parents=True)
    (source / "data" / "spec_files.parquet").write_bytes(b"complete parquet")
    assert copy_tree(source, destination) == 1
    assert (destination / "data" / "spec_files.parquet").read_bytes() == b"complete parquet"
    assert copy_tree(source, destination) == 0


def test_interrupted_copy_never_publishes_partial_file(tmp_path, monkeypatch):
    from specmine import migrate_cache

    source = tmp_path / "source"
    source.mkdir()
    (source / "spec_files.parquet").write_bytes(b"complete parquet")
    destination = tmp_path / "destination"
    original = migrate_cache.shutil.copyfileobj

    def interrupted(src, dst):
        dst.write(b"partial")
        raise OSError("interrupted")

    monkeypatch.setattr(migrate_cache.shutil, "copyfileobj", interrupted)
    with pytest.raises(OSError, match="interrupted"):
        copy_tree(source, destination)
    assert not (destination / "spec_files.parquet").exists()
    assert list(destination.iterdir()) == []
    monkeypatch.setattr(migrate_cache.shutil, "copyfileobj", original)
    assert copy_tree(source, destination) == 1
    assert (destination / "spec_files.parquet").read_bytes() == b"complete parquet"


def test_existing_partial_cache_is_reported_not_silently_skipped(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "destination"
    source.mkdir()
    destination.mkdir()
    (source / "spec_files.parquet").write_bytes(b"complete parquet")
    (destination / "spec_files.parquet").write_bytes(b"partial")
    with pytest.raises(FileExistsError, match="Destination differs"):
        copy_tree(source, destination)
    assert (destination / "spec_files.parquet").read_bytes() == b"partial"
