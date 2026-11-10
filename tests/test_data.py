import os
from pathlib import Path

import pandas as pd
import pytest

import specmine.data as sm


@pytest.fixture
def fake_parquet_df():
    rows = []
    shared_sha = "a" * 64
    specs = [
        (
            "alice/repoA",
            "spec.md",
            "https://github.com/alice/repoA/blob/aaa/spec.md",
            "s1" * 32,
        ),
        (
            "alice/repoA",
            "plan.md",
            "https://github.com/alice/repoA/blob/aaa/plan.md",
            shared_sha,
        ),
        (
            "bob/repoB",
            "spec.md",
            "https://github.com/bob/repoB/blob/bbb/spec.md",
            shared_sha,
        ),
        (
            "carol/repoC",
            "design.md",
            "https://github.com/carol/repoC/blob/ccc/design.md",
            "s3" * 32,
        ),
        (
            "carol/repoC",
            "spec.md",
            "https://github.com/carol/repoC/blob/ccc/spec.md",
            "s4" * 32,
        ),
        (
            "dave/repoD",
            "spec.md",
            "https://github.com/dave/repoD/blob/ddd/spec.md",
            "s5" * 32,
        ),
    ]
    repo_ids = {
        "alice/repoA": 101,
        "bob/repoB": 102,
        "carol/repoC": 103,
        "dave/repoD": 104,
    }
    for index, (repo, path, url, content_sha) in enumerate(specs, start=1):
        rows.append(
            {
                "repo_id": repo_ids[repo],
                "repo_name": repo,
                "file_url_sha16": f"spec-{index:02}",
                "file_path": path,
                "file_url": url,
                "file_sha": content_sha[:16],
                "file_content_sha": content_sha,
                "file_last_commit_sha": content_sha[:40],
                "file_name": path,
                "spec_tool": "openspec",
                "spec_role": "spec",
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def fake_raw_response(monkeypatch):
    class _Resp:
        def __init__(self, content, ctype="text/plain"):
            self.content = content
            self.text = content.decode()
            self.headers = {"Content-Type": ctype}
            self.status_code = 200

        def raise_for_status(self):
            pass

    calls = []

    def fake_get(url, timeout=None, allow_redirects=True):
        calls.append(url)
        if "sample_repos.txt" in url:
            body = "alice/repoA\ncarol/repoC\n"
            return _Resp(body.encode(), "text/plain")
        return _Resp(b"# spec\nhello\n", "text/plain")

    monkeypatch.setattr(sm.requests, "get", fake_get)
    return calls


@pytest.fixture
def patched_parquet(monkeypatch, fake_parquet_df, tmp_path):
    monkeypatch.setattr(sm, "DATA_DIR", tmp_path)
    monkeypatch.setattr(sm, "DEFAULT_LOCAL_DATA_DIR", tmp_path)
    curated_df = fake_parquet_df.loc[
        fake_parquet_df["repo_name"].isin({"alice/repoA", "carol/repoC"})
    ].copy()

    def fake_download(filename, data_dir):
        path = Path(data_dir) / "data" / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        return path

    def fake_read_parquet(path, engine=None):
        if Path(path).name == sm.CURATED_METADATA_FILENAME:
            return curated_df.copy()
        return fake_parquet_df.copy()

    monkeypatch.setattr(sm, "_download_dataset_parquet", fake_download)
    monkeypatch.setattr(sm.pd, "read_parquet", fake_read_parquet)


class TestLoadFeatureData:
    @pytest.fixture(autouse=True)
    def clear_feature_table_cache(self):
        sm._load_feature_table.cache_clear()
        yield
        sm._load_feature_table.cache_clear()

    def test_loads_and_filters_feature_table(self, monkeypatch, tmp_path):
        feature_df = pd.DataFrame(
            {
                "file_url_sha16": ["a", "b", "c"],
                "n_lines": [1, 2, 3],
            }
        )
        parquet_path = tmp_path / "data" / "spec_content_features.parquet"
        parquet_path.parent.mkdir()
        parquet_path.touch()

        def fake_read_parquet(url, engine=None):
            assert Path(url) == parquet_path
            assert engine == "fastparquet"
            return feature_df.copy()

        monkeypatch.setattr(sm.pd, "read_parquet", fake_read_parquet)

        result = sm.load_feature_data(
            pd.DataFrame({"file_url_sha16": ["b", "c"]}),
            local_data_only=True,
            data_dir=tmp_path,
        )

        assert result["file_url_sha16"].tolist() == ["b", "c"]
        assert result[["file_url_sha16", "n_lines"]].to_dict("records") == [
            {"file_url_sha16": "b", "n_lines": 2},
            {"file_url_sha16": "c", "n_lines": 3},
        ]

    def test_loads_full_feature_table_when_filter_is_omitted_and_caches(
        self, monkeypatch, tmp_path
    ):
        feature_df = pd.DataFrame(
            {
                "file_url_sha16": ["a", "b"],
                "n_lines": [1, 2],
            }
        )
        parquet_path = tmp_path / "data" / "spec_content_features.parquet"
        parquet_path.parent.mkdir()
        parquet_path.touch()
        read_calls = []

        def fake_read_parquet(url, engine=None):
            read_calls.append((url, engine))
            return feature_df.copy()

        monkeypatch.setattr(sm.pd, "read_parquet", fake_read_parquet)

        first = sm.load_feature_data(data_dir=tmp_path)
        first.loc[0, "n_lines"] = 99
        second = sm.load_feature_data(None, data_dir=tmp_path)

        assert second.to_dict("records") == [
            {"file_url_sha16": "a", "n_lines": 1},
            {"file_url_sha16": "b", "n_lines": 2},
        ]
        assert read_calls == [(parquet_path, "fastparquet")]

    def test_downloads_missing_feature_parquet_to_cache(
        self, monkeypatch, tmp_path
    ):
        feature_df = pd.DataFrame(
            {"file_url_sha16": ["a"], "n_lines": [1]}
        )
        downloads = []

        def fake_download(filename, data_dir):
            downloads.append(filename)
            path = data_dir / "data" / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
            return path

        monkeypatch.setattr(sm, "_download_dataset_parquet", fake_download)
        monkeypatch.setattr(
            sm.pd,
            "read_parquet",
            lambda path, engine=None: feature_df.copy(),
        )

        result = sm.load_feature_data(data_dir=tmp_path)

        assert result.equals(feature_df)
        assert downloads == ["spec_content_features.parquet"]
        assert (tmp_path / "data" / "spec_content_features.parquet").is_file()

    def test_loads_feature_table_from_local_cache(self, monkeypatch, tmp_path):
        feature_df = pd.DataFrame(
            {"file_url_sha16": ["a"], "n_lines": [1]}
        )
        parquet_path = tmp_path / "data" / "spec_content_features.parquet"
        parquet_path.parent.mkdir()
        parquet_path.touch()
        read_calls = []

        def fake_read_parquet(path, engine=None):
            read_calls.append((Path(path), engine))
            return feature_df.copy()

        monkeypatch.setattr(sm.pd, "read_parquet", fake_read_parquet)

        result = sm.load_feature_data(local_data_only=True, data_dir=tmp_path)

        assert result.equals(feature_df)
        assert read_calls == [(parquet_path, "fastparquet")]


class TestBlobToRaw:
    def test_converts_standard_blob_url(self):
        blob = "https://github.com/owner/repo/blob/abc123/path/to/spec.md"
        assert sm._blob_to_raw(blob) == (
            "https://raw.githubusercontent.com/owner/repo/abc123/path/to/spec.md"
        )

    def test_passthrough_when_not_blob(self):
        raw = "https://raw.githubusercontent.com/o/r/abc/spec.md"
        assert sm._blob_to_raw(raw) == raw

    def test_rejects_non_github_url(self):
        url = "https://example.com/spec.md"
        with pytest.raises(ValueError, match="approved GitHub host"):
            sm._blob_to_raw(url)

    @pytest.mark.parametrize(
        "url",
        [
            "http://github.com/o/r/blob/sha/spec.md",
            "https://github.com.evil.example/o/r/blob/sha/spec.md",
            "https://user@github.com/o/r/blob/sha/spec.md",
            "https://github.com:443/o/r/blob/sha/spec.md",
        ],
    )
    def test_rejects_unexpected_scheme_or_host_details(self, url):
        with pytest.raises(ValueError):
            sm._blob_to_raw(url)


class TestSafeFilename:
    def test_includes_repo_name_repo_id_and_spec_id(self):
        name = sm._safe_local_filename(
            "owner/repo", 123, "0123456789abcdef"
        )
        assert name == "owner_repo-123-0123456789abcdef.md"

    def test_sanitizes_weird_chars(self):
        name = sm._safe_local_filename("o/r", 123, "a b/c:d")
        assert all(c.isalnum() or c in "._-" for c in name)
        assert name.endswith(".md")

    def test_no_collision_across_specs_in_same_repo(self):
        a = sm._safe_local_filename("o/r", 123, "aaaa1111")
        b = sm._safe_local_filename("o/r", 123, "bbbb2222")
        assert a != b

    def test_repository_id_distinguishes_repositories_with_same_name(self):
        a = sm._safe_local_filename("o/r", 123, "same-spec")
        b = sm._safe_local_filename("o/r", 456, "same-spec")
        assert a != b

    def test_spec_local_path_uses_repository_and_spec_metadata(self, tmp_path):
        first = pd.Series(
            {
                "repo_name": "o/r",
                "repo_id": 123,
                "file_url_sha16": "first-spec",
                "file_url": "https://github.com/o/r/blob/main/first.md",
            }
        )
        second = first.copy()
        second["file_url_sha16"] = "second-spec"

        assert sm._spec_local_path(first, tmp_path) != sm._spec_local_path(
            second, tmp_path
        )

class TestDownloadSpecFiles:
    def test_downloads_every_repo_to_requested_directory(
        self, fake_parquet_df, fake_raw_response, tmp_path
    ):
        data_dir = tmp_path / "all-specs"

        result = sm.download_spec_files(fake_parquet_df, data_dir=data_dir)

        assert len(result) == len(fake_parquet_df)
        assert set(result["repo_name"]) == set(fake_parquet_df["repo_name"])
        assert all(Path(path).parent == data_dir for path in result["local_path"])
        assert all(Path(path).is_file() for path in result["local_path"])
        assert len(fake_raw_response) == len(fake_parquet_df)

    def test_reuses_existing_files(self, fake_parquet_df, fake_raw_response, tmp_path):
        data_dir = tmp_path / "all-specs"

        first = sm.download_spec_files(fake_parquet_df, data_dir=data_dir)
        request_count = len(fake_raw_response)
        second = sm.download_spec_files(fake_parquet_df, data_dir=data_dir)

        assert second["local_path"].tolist() == first["local_path"].tolist()
        assert len(fake_raw_response) == request_count

    def test_returns_paths_for_successful_rows_and_reports_failures(
        self, fake_parquet_df, monkeypatch, tmp_path
    ):
        attempted = []

        def fake_download(url, dest):
            attempted.append(url)
            if len(attempted) == 1:
                raise OSError("network unavailable")
            dest.write_text("# spec\n")
            return True

        monkeypatch.setattr(sm, "_download_file", fake_download)

        result = sm.download_spec_files(fake_parquet_df, data_dir=tmp_path)

        assert len(attempted) == len(fake_parquet_df)
        assert len(result) == len(fake_parquet_df) - 1
        assert "local_path" in result.columns
        assert result["file_url"].tolist() == fake_parquet_df["file_url"].iloc[1:].tolist()
        assert all(Path(path).is_file() for path in result["local_path"])


class TestLoadSpecMarkdown:
    def test_downloads_and_reuses_complete_markdown(
        self, fake_parquet_df, fake_raw_response, tmp_path
    ):
        metadata = fake_parquet_df.iloc[:2].copy()
        metadata["file_url_sha16"] = ["first", "second"]

        first = sm.load_spec_markdown(metadata, data_dir=tmp_path)
        second = sm.load_spec_markdown(metadata, data_dir=tmp_path)

        assert first["file_url_sha16"].tolist() == ["first", "second"]
        assert first[sm.MARKDOWN_COLUMN].tolist() == ["# spec\nhello\n"] * 2
        pd.testing.assert_frame_equal(first, second)
        assert len(fake_raw_response) == 2

    def test_reports_failed_downloads_and_keeps_available_specs(
        self, fake_parquet_df, monkeypatch, tmp_path, caplog, capsys
    ):
        metadata = fake_parquet_df.iloc[:2].copy()
        metadata["file_url_sha16"] = ["available", "unavailable"]

        def fake_download(url, dest):
            if url.endswith("/plan.md"):
                raise OSError("file unavailable")
            dest.write_text("# available\n", encoding="utf-8")
            return True

        monkeypatch.setattr(sm, "_download_file", fake_download)
        result = sm.load_spec_markdown(metadata, data_dir=tmp_path)

        assert result["file_url_sha16"].tolist() == ["available"]
        assert "Markdown unavailable for 1 spec(s): unavailable" in caplog.text
        assert "Skipped 1 unavailable files" in capsys.readouterr().out

    def test_local_only_skips_missing_and_invalid_files_without_network(
        self, fake_parquet_df, monkeypatch, tmp_path, caplog
    ):
        metadata = fake_parquet_df.iloc[:3].copy()
        metadata["file_url_sha16"] = ["cached", "invalid", "missing"]
        sm._spec_local_path(metadata.iloc[0], tmp_path).write_text(
            "# cached\n", encoding="utf-8"
        )
        sm._spec_local_path(metadata.iloc[1], tmp_path).write_bytes(b"\xff")

        def unexpected_request(*args, **kwargs):
            raise AssertionError("local-only Markdown loading must not use the network")

        monkeypatch.setattr(sm.requests, "get", unexpected_request)
        result = sm.load_spec_markdown(metadata, local_data_only=True, data_dir=tmp_path)

        assert result.to_dict("records") == [
            {"file_url_sha16": "cached", sm.MARKDOWN_COLUMN: "# cached\n"}
        ]
        assert "Cannot read Markdown for spec invalid" in caplog.text
        assert "invalid, missing" in caplog.text

    def test_rejects_duplicate_spec_ids(self, fake_parquet_df, tmp_path):
        metadata = fake_parquet_df.iloc[:2].copy()
        metadata["file_url_sha16"] = ["same", "same"]
        with pytest.raises(ValueError, match="at most one row per spec ID"):
            sm.load_spec_markdown(metadata, local_data_only=True, data_dir=tmp_path)


class TestDownloadRandomSpecSample:
    @pytest.fixture
    def metadata(self):
        return pd.DataFrame(
            {
                "repo_id": [10, 10, 20, 20, 30, 40, 50, 60],
                "repo_name": [
                    "owner/a",
                    "owner/a",
                    "owner/b",
                    "owner/b",
                    "owner/c",
                    "owner/d",
                    "owner/e",
                    "owner/f",
                ],
                "file_url_sha16": [
                    "a1",
                    "a2",
                    "b1",
                    "b2",
                    "c1",
                    "d1",
                    "e1",
                    "f1",
                ],
                "file_url": [f"https://github.com/o/r/blob/s/{i}.md" for i in range(8)],
            }
        )

    def test_selects_requested_count_with_distinct_specs_and_repositories(
        self, metadata, monkeypatch, tmp_path
    ):
        monkeypatch.setattr(
            sm,
            "download_spec_files",
            lambda df, data_dir: df.assign(local_path="cached.md"),
        )

        result = sm.download_random_spec_sample(
            metadata, 4, seed=7, data_dir=tmp_path
        )

        assert len(result) == 4
        assert result["file_url_sha16"].is_unique
        assert result["repo_id"].is_unique
        assert result["local_path"].eq("cached.md").all()

    def test_sample_is_reproducible_for_a_fixed_seed(
        self, metadata, monkeypatch, tmp_path
    ):
        monkeypatch.setattr(
            sm,
            "download_spec_files",
            lambda df, data_dir: df.assign(local_path="cached.md"),
        )

        first = sm.download_random_spec_sample(
            metadata, 4, seed=7, data_dir=tmp_path
        )
        second = sm.download_random_spec_sample(
            metadata, 4, seed=7, data_dir=tmp_path
        )

        assert first["file_url_sha16"].tolist() == second["file_url_sha16"].tolist()

    def test_replaces_failed_downloads_with_later_distinct_repositories(
        self, metadata, monkeypatch, tmp_path
    ):
        attempted_batches = []

        def fake_download(df, data_dir):
            attempted_batches.append(df.copy())
            batch = df.copy()
            if len(attempted_batches) == 1:
                batch = batch.iloc[1:]
            return batch.assign(local_path="cached.md")

        monkeypatch.setattr(sm, "download_spec_files", fake_download)

        result = sm.download_random_spec_sample(
            metadata, 3, seed=7, data_dir=tmp_path
        )

        assert len(attempted_batches) == 2
        assert len(result) == 3
        assert result["repo_id"].is_unique
        assert result["file_url_sha16"].is_unique
        assert not set(attempted_batches[0]["repo_id"]) & set(
            attempted_batches[1]["repo_id"]
        )

    def test_replaces_empty_download_with_later_candidate(
        self, monkeypatch, tmp_path
    ):
        metadata = pd.DataFrame(
            {
                "repo_id": [1, 2],
                "repo_name": ["o/first", "o/second"],
                "file_url_sha16": ["first", "second"],
                "file_url": [
                    "https://github.com/o/first/blob/main/spec.md",
                    "https://github.com/o/second/blob/main/spec.md",
                ],
                "file_path": ["spec.md", "spec.md"],
                "file_sha": ["sha1", "sha2"],
            }
        )
        requested_urls = []

        class Response:
            headers = {"Content-Type": "text/plain"}

            def __init__(self, content):
                self.content = content

            def raise_for_status(self):
                pass

        def fake_response(url, *, timeout):
            requested_urls.append(url)
            content = b"" if len(requested_urls) == 1 else b"# spec\n"
            return Response(content)

        monkeypatch.setattr(sm, "_get_github_response", fake_response)

        result = sm.download_random_spec_sample(metadata, 1, seed=0, data_dir=tmp_path)

        assert len(requested_urls) == 2
        assert len(result) == 1
        assert Path(result.iloc[0]["local_path"]).stat().st_size > 0

    def test_collapses_duplicate_spec_ids_before_sampling(
        self, metadata, monkeypatch, tmp_path
    ):
        duplicate = metadata.iloc[[0]].copy()
        duplicate["file_url"] = "https://github.com/o/r/blob/other/duplicate.md"
        metadata = pd.concat([metadata, duplicate], ignore_index=True)
        monkeypatch.setattr(
            sm,
            "download_spec_files",
            lambda df, data_dir: df.assign(local_path="cached.md"),
        )

        result = sm.download_random_spec_sample(
            metadata, 5, seed=11, data_dir=tmp_path
        )

        assert len(result) == 5
        assert result["file_url_sha16"].is_unique
        assert result["repo_id"].is_unique

    def test_rejects_spec_id_mapped_to_multiple_repositories(
        self, metadata, monkeypatch, tmp_path
    ):
        ambiguous = metadata.iloc[[0]].copy()
        ambiguous["repo_id"] = 999
        ambiguous["repo_name"] = "owner/other"
        metadata = pd.concat([metadata, ambiguous], ignore_index=True)

        with pytest.raises(ValueError, match="multiple repositories"):
            sm.download_random_spec_sample(metadata, 2, seed=1, data_dir=tmp_path)

    def test_rejects_request_larger_than_unique_repository_count(
        self, metadata, tmp_path
    ):
        with pytest.raises(ValueError, match="only 6 eligible repositories"):
            sm.download_random_spec_sample(metadata, 7, seed=1, data_dir=tmp_path)

    def test_local_only_uses_cached_specs_without_downloading(
        self, metadata, monkeypatch, tmp_path
    ):
        metadata["file_path"] = [f"{spec_id}.md" for spec_id in metadata["file_url_sha16"]]
        metadata["file_sha"] = metadata["file_url_sha16"]
        cached_rows = metadata.iloc[[1, 2]]
        for _, row in cached_rows.iterrows():
            path = sm._spec_local_path(row, tmp_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# cached spec\n")

        def unexpected_download(*args, **kwargs):
            raise AssertionError("local-only sampling must not download specs")

        monkeypatch.setattr(sm, "download_spec_files", unexpected_download)

        result = sm.download_random_spec_sample(
            metadata,
            5,
            seed=7,
            data_dir=tmp_path,
            local_data_only=True,
        )

        assert len(result) == len(cached_rows)
        assert result["repo_id"].is_unique
        assert result["file_url_sha16"].is_unique
        assert all(Path(path).is_file() for path in result["local_path"])


class TestFetchSampleRepoNames:
    def test_parses_lines_and_skips_blanks_and_comments(self, monkeypatch):
        class _Resp:
            status_code = 200
            text = "# comment\n\nowner1/repo1\nowner2/repo2\n  \n"
            headers = {}

            def raise_for_status(self):
                pass

        monkeypatch.setattr(sm.requests, "get", lambda *a, **k: _Resp())
        names = sm._fetch_sample_repo_names()
        assert names == ["owner1/repo1", "owner2/repo2"]


class TestSplitTrainTest:
    def test_no_repo_appears_in_both_splits(self, fake_parquet_df):
        train, test = sm.split_train_test(fake_parquet_df, test_size=0.5)
        assert set(train["repo_name"]) & set(test["repo_name"]) == set()

    def test_every_row_is_assigned_exactly_once(self, fake_parquet_df):
        train, test = sm.split_train_test(fake_parquet_df, test_size=0.5)
        assert len(train) + len(test) == len(fake_parquet_df)
        assigned_urls = train["file_url"].tolist() + test["file_url"].tolist()
        assert sorted(assigned_urls) == sorted(fake_parquet_df["file_url"].tolist())

    def test_content_dedup_prevents_cross_split_leak(self, fake_parquet_df):
        train, test = sm.split_train_test(fake_parquet_df, test_size=0.5)
        shared_sha = "a" * 64
        in_train = (train["file_content_sha"] == shared_sha).any()
        in_test = (test["file_content_sha"] == shared_sha).any()
        assert not (in_train and in_test), "shared content leaked across splits"

    def test_deterministic_across_calls(self, fake_parquet_df):
        t1, s1 = sm.split_train_test(fake_parquet_df, test_size=0.5)
        t2, s2 = sm.split_train_test(fake_parquet_df, test_size=0.5)
        assert set(t1["repo_name"]) == set(t2["repo_name"])
        assert set(s1["repo_name"]) == set(s2["repo_name"])

    def test_repo_assignment_stable_when_subsetting(self, fake_parquet_df):
        _full_train, full_test = sm.split_train_test(fake_parquet_df, test_size=0.5)
        test_repos_full = set(full_test["repo_name"])

        subset = fake_parquet_df[
            fake_parquet_df["repo_name"].isin(["bob/repoB", "carol/repoC"])
        ].reset_index(drop=True)
        sub_train, sub_test = sm.split_train_test(subset, test_size=0.5)

        for repo in sub_test["repo_name"]:
            assert repo in test_repos_full, f"{repo} moved out of test in subset"
        for repo in sub_train["repo_name"]:
            assert repo not in test_repos_full, f"{repo} moved into train in subset"

    def test_content_linked_repositories_stay_in_one_split(self, fake_parquet_df):
        train, test = sm.split_train_test(fake_parquet_df, test_size=0.5)
        train_repos = set(train["repo_name"])
        test_repos = set(test["repo_name"])
        assert ("alice/repoA" in train_repos) == ("bob/repoB" in train_repos)
        assert ("alice/repoA" in test_repos) == ("bob/repoB" in test_repos)

    def test_empty_input(self):
        empty = pd.DataFrame(columns=["repo_name", "file_content_sha"])
        train, test = sm.split_train_test(empty)
        assert train.empty and test.empty

    @pytest.mark.parametrize("test_size", [-0.1, 1.1])
    def test_rejects_invalid_test_size(self, fake_parquet_df, test_size):
        with pytest.raises(ValueError, match="test_size"):
            sm.split_train_test(fake_parquet_df, test_size=test_size)

    def test_null_repo_joins_content_group(self, fake_parquet_df):
        repo_row = fake_parquet_df.iloc[[1]].copy()
        missing_repo_row = repo_row.copy()
        missing_repo_row["repo_name"] = pd.NA
        df = pd.concat([repo_row, missing_repo_row], ignore_index=True)

        train, test = sm.split_train_test(df, test_size=1.0)

        assert train.empty
        assert len(test) == 2
        assert test["file_content_sha"].nunique() == 1


class TestAssertNoLeakage:
    def test_passes_on_clean_split(self, fake_parquet_df):
        train, test = sm.split_train_test(fake_parquet_df, test_size=0.5)
        sm.assert_no_leakage(train, test)

    def test_raises_on_repo_leak(self, fake_parquet_df):
        train = fake_parquet_df.copy()
        test = fake_parquet_df.copy()
        with pytest.raises(AssertionError, match="Repo leakage"):
            sm.assert_no_leakage(train, test)

    def test_raises_on_content_leak(self, fake_parquet_df):
        train = fake_parquet_df.iloc[[1]].copy()
        test = fake_parquet_df.iloc[[2]].copy()
        assert set(train["repo_name"]).isdisjoint(test["repo_name"])
        with pytest.raises(AssertionError, match="Content leakage"):
            sm.assert_no_leakage(train, test)


class TestLoadDataFull:
    def test_returns_parquet_df(self, patched_parquet):
        df = sm.load_data()
        assert df is not None and len(df) == 6

    def test_sample_size_limits_rows(self, patched_parquet):
        df = sm.load_data(sample_size=3)
        assert len(df) == 3

    def test_seed_selects_expected_sample(self, patched_parquet, fake_parquet_df):
        a = sm.load_data(sample_size=3, seed=0)
        b = sm.load_data(sample_size=3, seed=0)
        expected = fake_parquet_df.sample(n=3, random_state=0).reset_index(drop=True)
        assert a["file_url"].tolist() == expected["file_url"].tolist()
        assert b["file_url"].tolist() == expected["file_url"].tolist()

    def test_split_returns_tuple(self, patched_parquet):
        out = sm.load_data(split=True)
        assert isinstance(out, tuple) and len(out) == 2
        train, test = out
        sm.assert_no_leakage(train, test)

    def test_local_data_only_loads_parquet_from_cache(self, monkeypatch, tmp_path):
        parquet_path = tmp_path / "data" / "spec_files.parquet"
        parquet_path.parent.mkdir()
        parquet_path.touch()
        read_calls = []

        def fake_read_parquet(path, engine=None):
            read_calls.append((Path(path), engine))
            return pd.DataFrame({"repo_name": ["owner/repo"]})

        monkeypatch.setattr(sm.pd, "read_parquet", fake_read_parquet)

        result = sm.load_data(local_data_only=True, data_dir=tmp_path)

        assert result["repo_name"].tolist() == ["owner/repo"]
        assert read_calls == [(parquet_path, "fastparquet")]

    def test_local_data_only_missing_file_fails_without_remote_fallback(
        self, monkeypatch, tmp_path
    ):
        def unexpected_read(*args, **kwargs):
            raise AssertionError("read_parquet should not be called for a missing cache")

        monkeypatch.setattr(sm.pd, "read_parquet", unexpected_read)

        with pytest.raises(FileNotFoundError, match="data_init.ipynb"):
            sm.load_data(local_data_only=True, data_dir=tmp_path)

    def test_local_data_only_rejects_curated_mode(self, tmp_path):
        with pytest.raises(ValueError, match="curated_only"):
            sm.load_data(
                curated_only=True,
                local_data_only=True,
                data_dir=tmp_path,
            )


def test_download_dataset_parquets_uses_local_huggingface_directory(
    monkeypatch, tmp_path
):
    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        path = tmp_path / kwargs["filename"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        return str(path)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)

    result = sm.download_dataset_parquets(tmp_path)

    assert set(result) == {"spec_files.parquet", "spec_content_features.parquet"}
    assert all(path.is_file() for path in result.values())
    assert [call["filename"] for call in calls] == [
        "data/spec_files.parquet",
        "data/spec_content_features.parquet",
    ]
    assert all(call["repo_id"] == sm.DATASET_REPO_ID for call in calls)
    assert all(call["repo_type"] == "dataset" for call in calls)


def test_download_dataset_parquets_reuses_existing_files(monkeypatch, tmp_path):
    cached = {
        filename: tmp_path / "data" / filename
        for filename in sm.PARQUET_FILENAMES
    }
    for path in cached.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    def unexpected_download(**kwargs):
        raise AssertionError("cached parquet files must not be downloaded")

    monkeypatch.setattr("huggingface_hub.hf_hub_download", unexpected_download)

    assert sm.download_dataset_parquets(tmp_path) == cached


def test_load_data_caches_missing_parquet_and_reuses_it(monkeypatch, tmp_path):
    data = pd.DataFrame({"repo_name": ["owner/repo"]})
    downloads = []
    reads = []

    def fake_download(filename, data_dir):
        downloads.append(filename)
        path = data_dir / "data" / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        return path

    def fake_read(path, engine=None):
        reads.append((Path(path), engine))
        return data.copy()

    monkeypatch.setattr(sm, "_download_dataset_parquet", fake_download)
    monkeypatch.setattr(sm.pd, "read_parquet", fake_read)

    first = sm.load_data(data_dir=tmp_path)
    second = sm.load_data(data_dir=tmp_path)

    expected_path = tmp_path / "data" / "spec_files.parquet"
    assert first.equals(data)
    assert second.equals(data)
    assert downloads == ["spec_files.parquet"]
    assert reads == [
        (expected_path, "fastparquet"),
        (expected_path, "fastparquet"),
    ]


class TestLoadDataCurated:
    def test_returns_same_schema_plus_local_path(
        self, patched_parquet, fake_raw_response, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(sm, "DATA_DIR", tmp_path / "data")
        df = sm.load_data(curated_only=True)
        assert df is not None
        assert len(df) == 4
        for col in ["repo_name", "file_url", "file_path", "file_content_sha"]:
            assert col in df.columns
        assert "local_path" in df.columns
        for p in df["local_path"]:
            assert Path(p).is_file() and Path(p).stat().st_size > 0

    def test_data_dir_argument_controls_curated_download_location(
        self, patched_parquet, fake_raw_response, tmp_path
    ):
        data_dir = tmp_path / "custom-cache"
        df = sm.load_data(curated_only=True, data_dir=data_dir)

        assert df is not None and len(df) == 4
        assert all(Path(path).parent == data_dir for path in df["local_path"])

    def test_filters_to_curated_repos_only(
        self, patched_parquet, fake_raw_response, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(sm, "DATA_DIR", tmp_path / "data")
        df = sm.load_data(curated_only=True)
        assert df is not None and len(df) == 4
        assert set(df["repo_name"]) <= {"alice/repoA", "carol/repoC"}
        assert "bob/repoB" not in set(df["repo_name"])

    def test_downloads_use_raw_url(
        self, patched_parquet, fake_raw_response, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(sm, "DATA_DIR", tmp_path / "data")
        sm.load_data(curated_only=True)
        for url in fake_raw_response:
            if "sample_repos.txt" in url:
                continue
            assert url.startswith("https://raw.githubusercontent.com/")

    def test_second_call_is_idempotent(
        self, patched_parquet, fake_raw_response, tmp_path, monkeypatch
    ):
        data_dir = tmp_path / "data"
        monkeypatch.setattr(sm, "DATA_DIR", data_dir)
        sm.load_data(curated_only=True)
        first_call_count = len(fake_raw_response)
        sm.load_data(curated_only=True)

        assert len(fake_raw_response) == first_call_count
        assert (
            data_dir / "data" / sm.CURATED_METADATA_FILENAME
        ).is_file()

    def test_can_load_curated_metadata_without_downloading_raw_specs(
        self, patched_parquet, fake_raw_response, tmp_path
    ):
        result = sm.load_data(
            curated_only=True,
            data_dir=tmp_path / "cache",
            download_raw_specs=False,
        )

        assert result is not None and len(result) == 4
        assert "local_path" not in result.columns
        assert all("sample_repos.txt" in url for url in fake_raw_response)

    def test_curated_metadata_cache_avoids_rescanning_full_parquet(
        self, patched_parquet, fake_parquet_df, fake_raw_response, monkeypatch, tmp_path
    ):
        curated_df = fake_parquet_df.loc[
            fake_parquet_df["repo_name"].isin({"alice/repoA", "carol/repoC"})
        ].copy()
        read_names = []

        def fake_read_parquet(path, engine=None):
            read_names.append(Path(path).name)
            if Path(path).name == sm.CURATED_METADATA_FILENAME:
                return curated_df.copy()
            return fake_parquet_df.copy()

        monkeypatch.setattr(sm.pd, "read_parquet", fake_read_parquet)

        sm.load_data(
            curated_only=True,
            data_dir=tmp_path,
            download_raw_specs=False,
        )
        sm.load_data(
            curated_only=True,
            data_dir=tmp_path,
            download_raw_specs=False,
        )

        assert read_names == ["spec_files.parquet", sm.CURATED_METADATA_FILENAME]
        assert fake_raw_response == [sm.SAMPLE_REPOS_URL]

    def test_null_path_and_sha_metadata_are_supported(
        self,
        patched_parquet,
        fake_raw_response,
        tmp_path,
        monkeypatch,
        fake_parquet_df,
    ):
        df = fake_parquet_df.iloc[[0]].copy()
        df["file_path"] = pd.NA
        df["file_sha"] = pd.NA
        df["file_last_commit_sha"] = pd.NA
        monkeypatch.setattr(sm.pd, "read_parquet", lambda *a, **k: df.copy())
        monkeypatch.setattr(sm, "DATA_DIR", tmp_path / "data")

        result = sm.load_data(curated_only=True)

        assert result is not None and len(result) == 1
        assert Path(result.iloc[0]["local_path"]).is_file()

    def test_failed_atomic_publish_leaves_no_partial_file(
        self, fake_raw_response, tmp_path, monkeypatch
    ):
        dest = tmp_path / "data" / "spec.md"

        def fail_replace(*args):
            raise OSError("publish failed")

        monkeypatch.setattr(sm.os, "replace", fail_replace)

        with pytest.raises(OSError, match="publish failed"):
            sm._download_file(
                "https://raw.githubusercontent.com/o/r/sha/spec.md", dest
            )

        assert not dest.exists()
        assert list(dest.parent.iterdir()) == []

    def test_partial_temp_write_is_cleaned_up(
        self, fake_raw_response, tmp_path, monkeypatch
    ):
        dest = tmp_path / "data" / "spec.md"
        temp_path = dest.parent / ".partial.tmp"

        class PartialTempFile:
            name = str(temp_path)

            def __enter__(self):
                self.file = temp_path.open("wb")
                return self

            def write(self, content):
                self.file.write(content[:4])
                raise OSError("write failed")

            def __exit__(self, exc_type, exc_value, traceback):
                self.file.close()

        monkeypatch.setattr(
            sm.tempfile,
            "NamedTemporaryFile",
            lambda **kwargs: PartialTempFile(),
        )

        with pytest.raises(OSError, match="write failed"):
            sm._download_file(
                "https://raw.githubusercontent.com/o/r/sha/spec.md", dest
            )

        assert not dest.exists()
        assert list(dest.parent.iterdir()) == []

    def test_download_rejects_redirect_to_loopback(self, tmp_path, monkeypatch):
        calls = []

        class RedirectResponse:
            status_code = 302
            headers = {"Location": "http://127.0.0.1/admin"}

            def raise_for_status(self):
                raise AssertionError("redirect should be rejected before status handling")

        def fake_get(url, timeout=None, allow_redirects=True):
            calls.append((url, allow_redirects))
            return RedirectResponse()

        monkeypatch.setattr(sm.requests, "get", fake_get)

        with pytest.raises(ValueError, match="HTTPS on an approved GitHub host"):
            sm._download_file(
                "https://raw.githubusercontent.com/o/r/sha/spec.md",
                tmp_path / "spec.md",
            )

        assert calls == [
            ("https://raw.githubusercontent.com/o/r/sha/spec.md", False)
        ]

    def test_download_validates_each_github_redirect(self, tmp_path, monkeypatch):
        calls = []

        class Response:
            def __init__(self, status_code, headers=None, content=b"spec"):
                self.status_code = status_code
                self.headers = headers or {"Content-Type": "text/plain"}
                self.content = content

            def raise_for_status(self):
                pass

        responses = [
            Response(
                302,
                {"Location": "/o/r/sha/spec.md", "Content-Type": "text/plain"},
            ),
            Response(200),
        ]

        def fake_get(url, timeout=None, allow_redirects=True):
            calls.append((url, allow_redirects))
            return responses.pop(0)

        monkeypatch.setattr(sm.requests, "get", fake_get)
        dest = tmp_path / "spec.md"

        assert sm._download_file(
            "https://raw.githubusercontent.com/o/r/sha/spec.md", dest
        )
        assert dest.read_bytes() == b"spec"
        assert calls == [
            ("https://raw.githubusercontent.com/o/r/sha/spec.md", False),
            ("https://raw.githubusercontent.com/o/r/sha/spec.md", False),
        ]


@pytest.mark.network
@pytest.mark.skipif(
    os.environ.get("SPECMINE_RUN_NETWORK_TESTS") != "1",
    reason="set SPECMINE_RUN_NETWORK_TESTS=1 to run network smoke tests",
)
class TestNetworkSmoke:
    def test_full_parquet_loads(self):
        df = sm.load_data(sample_size=5, seed=0)
        assert df is not None
        assert len(df) == 5

    def test_curated_one_repo(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sm, "DATA_DIR", tmp_path / "data")
        monkeypatch.setattr(sm, "_fetch_sample_repo_names", lambda: ["PostHog/posthog"])
        df = sm.load_data(curated_only=True, sample_size=3, seed=0)
        assert df is not None and 0 < len(df) <= 3
        for p in df["local_path"]:
            assert Path(p).is_file() and Path(p).stat().st_size > 0
