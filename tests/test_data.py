# type: ignore

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
    for repo, path, url, content_sha in specs:
        rows.append(
            {
                "repo_name": repo,
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

    def fake_get(url, timeout=None):
        calls.append(url)
        if "sample_repos.txt" in url:
            body = "alice/repoA\ncarol/repoC\n"
            return _Resp(body.encode(), "text/plain")
        return _Resp(b"# spec\nhello\n", "text/plain")

    monkeypatch.setattr(sm.requests, "get", fake_get)
    return calls


@pytest.fixture
def patched_parquet(monkeypatch, fake_parquet_df):
    def fake_read_parquet(url, engine=None):
        return fake_parquet_df.copy()

    monkeypatch.setattr(sm.pd, "read_parquet", fake_read_parquet)


class TestBlobToRaw:
    def test_converts_standard_blob_url(self):
        blob = "https://github.com/owner/repo/blob/abc123/path/to/spec.md"
        assert sm._blob_to_raw(blob) == (
            "https://raw.githubusercontent.com/owner/repo/abc123/path/to/spec.md"
        )

    def test_passthrough_when_not_blob(self):
        raw = "https://raw.githubusercontent.com/o/r/abc/spec.md"
        assert sm._blob_to_raw(raw) == raw

    def test_passthrough_for_non_github(self):
        url = "https://example.com/spec.md"
        assert sm._blob_to_raw(url) == url


class TestSafeFilename:
    def test_includes_owner_repo_sha_and_basename(self):
        name = sm._safe_local_filename(
            "owner/repo", "openspec/changes/foo/spec.md", "deadbeef" + "0" * 56
        )
        assert name.startswith("owner__repo__deadbeef__")
        assert name.endswith("spec.md")

    def test_sanitizes_weird_chars(self):
        name = sm._safe_local_filename("o/r", "a b/c:d.md", "abc")
        assert all(c.isalnum() or c in "._-" for c in name)

    def test_no_collision_across_specs_in_same_repo(self):
        a = sm._safe_local_filename("o/r", "a/spec.md", "aaaa1111")
        b = sm._safe_local_filename("o/r", "b/spec.md", "bbbb2222")
        assert a != b


class TestFetchSampleRepoNames:
    def test_parses_lines_and_skips_blanks_and_comments(self, monkeypatch):
        class _Resp:
            text = "# comment\n\nowner1/repo1\nowner2/repo2\n  \n"

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
        full_train, full_test = sm.split_train_test(fake_parquet_df, test_size=0.5)
        test_repos_full = set(full_test["repo_name"])

        subset = fake_parquet_df[
            fake_parquet_df["repo_name"].isin(["alice/repoA", "carol/repoC"])
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
        train = fake_parquet_df.iloc[:3].copy()
        test = fake_parquet_df.iloc[2:4].copy()
        with pytest.raises(AssertionError):
            sm.assert_no_leakage(train, test)


class TestLoadDataFull:
    def test_returns_parquet_df(self, patched_parquet):
        df = sm.load_data()
        assert df is not None and len(df) == 6

    def test_sample_size_limits_rows(self, patched_parquet):
        df = sm.load_data(sample_size=3)
        assert len(df) == 3

    def test_seed_is_reproducible(self, patched_parquet):
        a = sm.load_data(sample_size=3, seed=0)
        b = sm.load_data(sample_size=3, seed=0)
        assert a["repo_name"].tolist() == b["repo_name"].tolist()

    def test_split_returns_tuple(self, patched_parquet):
        out = sm.load_data(split=True)
        assert isinstance(out, tuple) and len(out) == 2
        train, test = out
        sm.assert_no_leakage(train, test)


class TestLoadDataCurated:
    def test_returns_same_schema_plus_local_path(
        self, patched_parquet, fake_raw_response, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(sm, "DATA_DIR", tmp_path / "data")
        df = sm.load_data(curated_only=True)
        assert df is not None
        for col in ["repo_name", "file_url", "file_path", "file_content_sha"]:
            assert col in df.columns
        assert "local_path" in df.columns
        for p in df["local_path"]:
            assert Path(p).exists()

    def test_filters_to_curated_repos_only(
        self, patched_parquet, fake_raw_response, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(sm, "DATA_DIR", tmp_path / "data")
        df = sm.load_data(curated_only=True)
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
        monkeypatch.setattr(sm, "DATA_DIR", tmp_path / "data")
        sm.load_data(curated_only=True)
        n_first = len(fake_raw_response)
        sm.load_data(curated_only=True)
        new_calls = fake_raw_response[n_first:]
        spec_calls = [u for u in new_calls if "sample_repos.txt" not in u]
        assert spec_calls == []


@pytest.mark.network
class TestNetworkSmoke:
    def test_full_parquet_loads(self):
        df = sm.load_data(sample_size=5, seed=0)
        assert df is not None
        assert len(df) == 5

    def test_curated_one_repo(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sm, "DATA_DIR", tmp_path / "data")
        monkeypatch.setattr(sm, "_fetch_sample_repo_names", lambda: ["PostHog/posthog"])
        df = sm.load_data(curated_only=True, sample_size=3, seed=0)
        assert df is not None and len(df) <= 3
        for p in df["local_path"]:
            assert Path(p).exists()
