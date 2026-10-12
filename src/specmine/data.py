import hashlib
import re
import sys
import threading
import time
from pathlib import Path

import pandas as pd
import requests

SAMPLE_REPOS_URL = (
    "https://raw.githubusercontent.com/shyamagarwal13/specmine-official/"
    "main/sample_repos.txt"
)
PARQUET_URL = "hf://datasets/ShyAgarwal/specmine/data/spec_files.parquet"
DATA_DIR = Path("data")
_BLOB_RE = re.compile(
    r"^https?://github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+)/blob/(?P<rest>.+)$"
)


def loading_animation(stop, message="Loading..."):
    spinners = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    i = 0
    while not stop.is_set():
        sys.stdout.write(f"\r{spinners[i]} {message}")
        sys.stdout.flush()
        i = (i + 1) % len(spinners)
        time.sleep(0.1)
    sys.stdout.write("\r" + " " * 140 + "\r")
    sys.stdout.flush()


def _fetch_sample_repo_names() -> list[str]:
    """
    Download the curated list of repo names (owner/repo) from GitHub.
    """
    resp = requests.get(SAMPLE_REPOS_URL, timeout=30)
    resp.raise_for_status()
    names = []
    for line in resp.text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        names.append(line)
    return names


def _blob_to_raw(url: str) -> str:
    """
    Convert a GitHub blob URL to its raw.githubusercontent.com equivalent.
    https://github.com/OWNER/REPO/blob/SHA/path/to/file.md -> https://raw.githubusercontent.com/OWNER/REPO/SHA/path/to/file.md
    """
    m = _BLOB_RE.match(url)
    if not m:
        return url
    return (
        f"https://raw.githubusercontent.com/{m.group('owner')}/"
        f"{m.group('repo')}/{m.group('rest')}"
    )


def _safe_local_filename(repo_name: str, file_path: str, sha: str) -> str:
    """
    Build a filesystem-safe, unique filename.
    Includes a short SHA prefix so multiple spec files per repo don't collide.
    """
    owner, _, repo = repo_name.partition("/")
    short_sha = (sha or "nohash")[:8]
    tail = Path(file_path).name or "spec.md"
    base = f"{owner}__{repo}__{short_sha}__{tail}"
    return re.sub(r"[^A-Za-z0-9._-]+", "_", base)


def _download_file(url: str, dest: Path) -> bool:
    """
    Download a single file, skipping if already present.
    """
    if dest.exists() and dest.stat().st_size > 0:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        ctype = r.headers.get("Content-Type", "")
        if "text/html" in ctype and not url.endswith(".html"):
            raise ValueError(f"got HTML instead of spec file ({ctype})")
        dest.write_bytes(r.content)
        return True
    except Exception as e:
        if dest.exists() and dest.stat().st_size == 0:
            dest.unlink()
        raise e


def _curated_local_dataframe(sample_size=None, seed=None) -> pd.DataFrame:
    """
    Download the curated set of repo spec files to ./data and return the
    same schema as the full parquet, plus a `local_path` column.
    Rows whose file couldn't be downloaded are dropped.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    wanted = set(_fetch_sample_repo_names())
    print(f"Curated repo list: {len(wanted)} unique repos")
    stop = threading.Event()
    t = threading.Thread(
        target=loading_animation,
        args=(stop, "Loading SpecMine parquet for URL lookup..."),
    )
    t.daemon = True
    t.start()
    try:
        full = pd.read_parquet(PARQUET_URL, engine="fastparquet")
    finally:
        stop.set()
        t.join()
    df = full[full["repo_name"].isin(wanted)].copy()
    print(f"Matched {len(df)} spec rows across {df['repo_name'].nunique()} repos")
    if df.empty:
        return df.assign(local_path=pd.Series(dtype="string"))
    if sample_size is not None and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=seed).reset_index(drop=True)
    local_paths = []
    total = len(df)
    failed = 0
    for i, (_, row) in enumerate(df.iterrows(), 1):
        repo = row["repo_name"]
        blob_url = row["file_url"]
        file_path = row.get("file_path", "") or ""
        sha = row.get("file_sha", "") or row.get("file_last_commit_sha", "") or ""
        if not isinstance(blob_url, str) or not blob_url.startswith("http"):
            local_paths.append(None)
            continue
        raw_url = _blob_to_raw(blob_url)
        fname = _safe_local_filename(repo, file_path, sha)
        dest = DATA_DIR / fname
        sys.stdout.write(f"\r  [{i}/{total}] {repo} -> {fname[:70]}")
        sys.stdout.flush()
        try:
            _download_file(raw_url, dest)
            local_paths.append(str(dest))
        except Exception as e:
            local_paths.append(None)
            failed += 1
            if failed <= 5:
                sys.stdout.write("\n")
                print(f"  ! {repo} :: {type(e).__name__}: {e}")
    sys.stdout.write("\r" + " " * 140 + "\r")
    df["local_path"] = local_paths
    before = len(df)
    df = df[df["local_path"].notna()].reset_index(drop=True)
    print(f"Downloaded {len(df)}/{before} files (kept only successful rows)")
    return df


def _load_raw(curated_only=False, sample_size=None, seed=None):
    """Load a single SpecMine DataFrame (no splitting)."""
    if curated_only:
        try:
            return _curated_local_dataframe(sample_size=sample_size, seed=seed)
        except Exception as e:
            print(f"\nError loading curated dataset: {e}")
            print("Check requirements.txt for needed packages")
            return None
    stop = threading.Event()
    spinner_thread = threading.Thread(
        target=loading_animation,
        args=(
            stop,
            "Attempting to connect to Hugging Face and load SpecMine dataset...",
        ),
    )
    spinner_thread.daemon = True
    spinner_thread.start()
    try:
        df = pd.read_parquet(PARQUET_URL, engine="fastparquet")
    except Exception as e:
        print(f"\nError loading dataset: {e}")
        print("Check requirements.txt for needed packages")
        return None
    finally:
        stop.set()
        spinner_thread.join()
    if df is None:
        return None
    if sample_size is not None:
        n = min(sample_size, len(df))
        df = df.sample(n=n, random_state=seed).reset_index(drop=True)
    return df


def _repo_bucket(repo_name: str, salt: str = "specmine-v1") -> float:
    """Deterministic [0,1) bucket for a repo name."""
    h = hashlib.sha256(f"{salt}:{repo_name}".encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def split_train_test(df, test_size=0.2):
    """
    Split a SpecMine DataFrame into (train, test) with no leakage:
    - No repo_name appears in both splits (group split).
    - No file_content_sha appears in both splits (content dedup).
    - Rows with duplicate content are assigned entirely to one side.
    - Repositories connected by duplicate content are kept in one group.
    - Group assignments are deterministic based on the lexically first repo.
    """
    if df is None or df.empty:
        return df, df

    parents = {repo: repo for repo in df["repo_name"].dropna().unique()}

    def find(repo):
        while parents[repo] != repo:
            parents[repo] = parents[parents[repo]]
            repo = parents[repo]
        return repo

    def union(left, right):
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parents[max(left_root, right_root)] = min(left_root, right_root)

    if "file_content_sha" in df.columns:
        repo_by_sha = {}
        for repo, sha in df[["repo_name", "file_content_sha"]].dropna().itertuples(
            index=False, name=None
        ):
            previous_repo = repo_by_sha.setdefault(sha, repo)
            union(repo, previous_repo)

    repos_by_group = {}
    for repo in parents:
        repos_by_group.setdefault(find(repo), []).append(repo)
    repo_is_test = {}
    for repos in repos_by_group.values():
        is_test = _repo_bucket(min(repos)) < test_size
        repo_is_test.update((repo, is_test) for repo in repos)

    df = df.copy()
    df["_is_test"] = df["repo_name"].map(repo_is_test).fillna(False)
    train = df[~df["_is_test"]].drop(columns="_is_test").reset_index(drop=True)
    test = df[df["_is_test"]].drop(columns="_is_test").reset_index(drop=True)
    return train, test


def assert_no_leakage(train, test):
    """Raise if any repo or file content is shared between train and test."""
    shared_repos = set(train["repo_name"]) & set(test["repo_name"])
    assert not shared_repos, f"Repo leakage: {list(shared_repos)[:5]}"
    if "file_content_sha" in train.columns and "file_content_sha" in test.columns:
        shared_content = set(train["file_content_sha"].dropna()) & set(
            test["file_content_sha"].dropna()
        )
        assert not shared_content, f"Content leakage: {len(shared_content)} sha(s)"
    print(
        f"OK — train: {train['repo_name'].nunique()} repos / {len(train)} rows | "
        f"test: {test['repo_name'].nunique()} repos / {len(test)} rows"
    )


def load_data(
    curated_only=False,
    sample_size=None,
    seed=None,
    split=False,
    test_size=0.2,
    return_splits=False,
):
    """
    Load SpecMine data.
    - curated_only=False (default): return the full parquet DataFrame.
    - curated_only=True: download the curated repo spec files to ./data and return the SAME schema as the full parquet, plus a `local_path` column.
    - sample_size=N: random sample of N rows (both modes).
    - seed: random seed for sampling (both modes).
    - split=True (or return_splits=True): return a leak-safe (train, test) tuple instead of one frame. No repo_name and no file_content_sha appears on both sides. `test_size` is the approx. fraction of repos assigned to test.
    """
    df = _load_raw(curated_only=curated_only, sample_size=sample_size, seed=seed)
    if df is None:
        return None
    if split or return_splits:
        train, test = split_train_test(df, test_size=test_size)
        return train, test
    return df
