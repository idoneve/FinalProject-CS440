from __future__ import annotations

import hashlib
import logging
import os
import re
import sys
import tempfile
import threading
import time
from functools import lru_cache
from numbers import Integral
from pathlib import Path
from urllib.parse import SplitResult, urljoin, urlsplit

import pandas as pd
import requests

LOGGER = logging.getLogger(__name__)
ID_COLUMN = "file_url_sha16"
MARKDOWN_COLUMN = "spec_markdown"
SAMPLE_REPOS_URL = (
    "https://raw.githubusercontent.com/shyamagarwal13/specmine-official/"
    "main/sample_repos.txt"
)
PARQUET_URL = "hf://datasets/ShyAgarwal/specmine/data/spec_files.parquet"
FEATURES_PARQUET_URL = (
    "hf://datasets/ShyAgarwal/specmine/data/spec_content_features.parquet"
)
DEFAULT_LOCAL_DATA_DIR = (
    Path(__file__).resolve().parents[2] / "notebooks" / "data"
)
DATA_DIR = DEFAULT_LOCAL_DATA_DIR
DATASET_REPO_ID = "ShyAgarwal/specmine"
PARQUET_FILENAMES = ("spec_files.parquet", "spec_content_features.parquet")
CURATED_METADATA_FILENAME = "curated_spec_files.parquet"
_BLOB_RE = re.compile(
    r"^https://github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+)/blob/(?P<rest>[^?#]+)$",
    re.IGNORECASE,
)
_ALLOWED_GITHUB_HOSTS = {"github.com", "raw.githubusercontent.com"}
_REDIRECT_STATUSES = {301, 302, 303, 307, 308}
_MAX_DOWNLOAD_REDIRECTS = 5


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
    resp = _get_github_response(SAMPLE_REPOS_URL, timeout=30)
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
    parsed = _validate_github_url(url)
    if parsed.hostname == "raw.githubusercontent.com":
        if not parsed.path.strip("/"):
            raise ValueError("GitHub raw URL must include a file path")
        return parsed.geturl()
    m = _BLOB_RE.match(url)
    if not m:
        raise ValueError("Expected a GitHub blob URL or raw.githubusercontent.com URL")
    return (
        f"https://raw.githubusercontent.com/{m.group('owner')}/"
        f"{m.group('repo')}/{m.group('rest')}"
    )


def _validate_github_url(url: str) -> SplitResult:
    """Parse and validate an HTTPS URL whose host is an approved GitHub host."""
    if not isinstance(url, str) or not url or url.strip() != url:
        raise ValueError("Download URL must be a non-empty, whitespace-free string")
    if "\\" in url or any(ord(char) < 32 for char in url):
        raise ValueError("Download URL contains invalid characters")
    try:
        parsed = urlsplit(url)
        hostname = parsed.hostname
        port = parsed.port
    except ValueError as exc:
        raise ValueError("Download URL is malformed") from exc
    if (
        parsed.scheme.lower() != "https"
        or hostname is None
        or hostname.lower() not in _ALLOWED_GITHUB_HOSTS
        or parsed.username is not None
        or parsed.password is not None
        or port is not None
        or parsed.fragment
    ):
        raise ValueError("Download URL must use HTTPS on an approved GitHub host")
    return parsed


def _get_github_response(url: str, *, timeout: int) -> requests.Response:
    """Request a GitHub-hosted URL without following unvalidated redirects."""
    current_url = _validate_github_url(url).geturl()
    for redirect_count in range(_MAX_DOWNLOAD_REDIRECTS + 1):
        response = requests.get(
            current_url,
            timeout=timeout,
            allow_redirects=False,
        )
        if response.status_code not in _REDIRECT_STATUSES:
            return response
        if redirect_count == _MAX_DOWNLOAD_REDIRECTS:
            raise ValueError("Too many redirects while requesting GitHub content")
        location = response.headers.get("Location")
        if not location:
            raise ValueError("Redirect response is missing a Location header")
        current_url = _validate_github_url(urljoin(current_url, location)).geturl()
    raise RuntimeError("GitHub redirect loop ended unexpectedly")


def _safe_local_filename(repo_name: str, file_path: str, sha: str) -> str:
    """
    Build a filesystem-safe, unique filename.
    Includes hashes of the full path and SHA so specs with the same basename
    or abbreviated SHA do not collide.
    """
    owner, _, repo = repo_name.partition("/")
    short_sha = (sha or "nohash")[:8]
    tail = Path(file_path).name or "spec.md"
    path_hash = hashlib.sha256(file_path.encode("utf-8")).hexdigest()
    sha_hash = hashlib.sha256(sha.encode("utf-8")).hexdigest()
    base = f"{owner}__{repo}__{short_sha}__{path_hash}__{sha_hash}__{tail}"
    return re.sub(r"[^A-Za-z0-9._-]+", "_", base)


def _spec_local_path(row: pd.Series, data_dir: Path) -> Path:
    repo = row["repo_name"]
    file_path = row.get("file_path", "")
    file_path = "" if pd.isna(file_path) else str(file_path)
    sha = row.get("file_sha", "")
    sha = "" if pd.isna(sha) else str(sha)
    if not sha:
        sha = row.get("file_last_commit_sha", "")
        sha = "" if pd.isna(sha) else str(sha)
    if not file_path or not sha:
        sha = hashlib.sha256(str(row["file_url"]).encode("utf-8")).hexdigest()
    return data_dir / _safe_local_filename(repo, file_path, sha)


def _download_file(url: str, dest: Path) -> bool:
    """
    Download a single file, skipping if already present.
    """
    if dest.exists() and dest.stat().st_size > 0:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        r = _get_github_response(url, timeout=60)
        r.raise_for_status()
        ctype = r.headers.get("Content-Type", "")
        final_url = getattr(r, "url", url)
        if "text/html" in ctype and not urlsplit(final_url).path.endswith(".html"):
            raise ValueError(f"got HTML instead of spec file ({ctype})")
        with tempfile.NamedTemporaryFile(
            dir=dest.parent,
            prefix=f".{dest.name}.",
            suffix=".tmp",
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)
            temp_file.write(r.content)
        os.replace(temp_path, dest)
        temp_path = None
        return True
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def download_spec_files(df: pd.DataFrame, data_dir=None) -> pd.DataFrame:
    """Download the specs in a SpecMine DataFrame and add their ``local_path``.

    Existing non-empty files are reused. Rows that cannot be downloaded are
    excluded from the result and reported; every row is attempted.
    """
    required_columns = {"repo_name", "file_url"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise KeyError(f"Spec data is missing columns: {sorted(missing_columns)}")

    data_dir = DATA_DIR if data_dir is None else Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    successful_positions = []
    local_paths = []
    failures = []
    total = len(df)
    print(f"Downloading {total} SpecMine files into {data_dir}")

    for position, (_, row) in enumerate(df.iterrows()):
        i = position + 1
        repo = row["repo_name"]
        blob_url = row["file_url"]
        try:
            if not isinstance(repo, str) or not repo:
                raise ValueError("Repository name is missing")
            if not isinstance(blob_url, str):
                raise ValueError("File URL is missing")
            raw_url = _blob_to_raw(blob_url)
            dest = _spec_local_path(row, data_dir)
            _download_file(raw_url, dest)
            successful_positions.append(position)
            local_paths.append(str(dest))
        except Exception as exc:
            failures.append((i, repo, exc))
        if i % 100 == 0 or i == total:
            print(f"Processed {i}/{total} files")

    result = df.iloc[successful_positions].copy().reset_index(drop=True)
    result["local_path"] = local_paths
    print(f"Downloaded or reused {len(result)}/{total} files")
    if failures:
        details = "; ".join(
            f"row {i} ({repo}): {type(exc).__name__}: {exc}"
            for i, repo, exc in failures[:5]
        )
        print(f"Skipped {len(failures)} unavailable files. First failures: {details}")
    return result


def load_spec_markdown(
    metadata: pd.DataFrame, *, local_data_only=False, data_dir=None
) -> pd.DataFrame:
    """Load available spec Markdown by ID, reporting files that are unavailable.

    Cached files are reused. In local-only mode no network request is made;
    otherwise missing files are obtained with :func:`download_spec_files`.
    Returned text is unmodified and only successfully read rows are included.
    """
    if not isinstance(metadata, pd.DataFrame):
        raise TypeError("metadata must be a pandas DataFrame")
    required = {ID_COLUMN, "repo_name", "file_url"}
    missing_columns = required - set(metadata.columns)
    if missing_columns:
        raise KeyError(f"Spec data is missing columns: {sorted(missing_columns)}")

    requested = metadata.copy()
    requested[ID_COLUMN] = requested[ID_COLUMN].astype("string")
    if (
        requested[ID_COLUMN].isna().any()
        or requested[ID_COLUMN].str.strip().eq("").any()
    ):
        raise ValueError("Spec IDs must not be missing or blank")
    if requested[ID_COLUMN].duplicated().any():
        raise ValueError("Spec metadata must contain at most one row per spec ID")
    if requested.empty:
        return pd.DataFrame(columns=[ID_COLUMN, MARKDOWN_COLUMN])

    cache_dir = DATA_DIR if data_dir is None else Path(data_dir)
    if local_data_only:
        cached_rows = []
        for _, row in requested.iterrows():
            repo = row["repo_name"]
            if not isinstance(repo, str) or not repo:
                LOGGER.warning("Missing repository name for spec %s", row[ID_COLUMN])
                continue
            path = _spec_local_path(row, cache_dir)
            if path.is_file() and path.stat().st_size > 0:
                cached_rows.append((row[ID_COLUMN], path))
    else:
        downloaded = download_spec_files(requested, data_dir=cache_dir)
        cached_rows = [
            (row[ID_COLUMN], Path(row["local_path"]))
            for _, row in downloaded.iterrows()
        ]

    records = []
    for spec_id, path in cached_rows:
        try:
            markdown = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            LOGGER.warning("Cannot read Markdown for spec %s: %s", spec_id, exc)
            continue
        if not markdown.strip():
            LOGGER.warning("Markdown for spec %s is empty", spec_id)
            continue
        records.append({ID_COLUMN: spec_id, MARKDOWN_COLUMN: markdown})

    result = pd.DataFrame(records, columns=[ID_COLUMN, MARKDOWN_COLUMN])
    unavailable = requested.loc[
        ~requested[ID_COLUMN].isin(result[ID_COLUMN]), ID_COLUMN
    ].tolist()
    if unavailable:
        LOGGER.warning(
            "Markdown unavailable for %d spec(s): %s",
            len(unavailable),
            ", ".join(unavailable),
        )
    return result


def download_random_spec_sample(
    metadata: pd.DataFrame,
    sample_size: int,
    *,
    seed: int | None = None,
    data_dir=None,
    local_data_only=False,
) -> pd.DataFrame:
    """Return a random sample with at most one spec from each repository.

    Duplicate spec IDs are collapsed before sampling. A random candidate order
    selects one spec per repo ID. In local-only mode, only existing non-empty
    files are used and the result may contain fewer than ``sample_size`` rows.
    Otherwise failed downloads are replaced by later candidates.
    """
    if not isinstance(sample_size, Integral) or isinstance(sample_size, bool):
        raise TypeError("sample_size must be an integer")
    sample_size = int(sample_size)
    if sample_size < 1:
        raise ValueError("sample_size must be a positive integer")
    required = {"repo_id", "repo_name", "file_url_sha16", "file_url"}
    missing = required - set(metadata.columns)
    if missing:
        raise KeyError(f"SpecMine metadata is missing columns: {sorted(missing)}")

    candidates = metadata.dropna(
        subset=["repo_id", "repo_name", "file_url_sha16", "file_url"]
    ).copy()
    candidates["file_url_sha16"] = candidates["file_url_sha16"].astype("string")
    candidates = candidates.loc[candidates["file_url_sha16"].str.strip().ne("")]
    id_repository_pairs = candidates[["file_url_sha16", "repo_id"]].drop_duplicates()
    ambiguous_ids = (
        id_repository_pairs.groupby("file_url_sha16")["repo_id"]
        .nunique()
        .loc[lambda counts: counts > 1]
        .index
    )
    if len(ambiguous_ids):
        raise ValueError(
            "Spec IDs map to multiple repositories: "
            f"{sorted(ambiguous_ids.astype(str))[:10]}"
        )

    candidates = candidates.drop_duplicates(subset=["file_url_sha16"])
    if local_data_only:
        data_dir = DEFAULT_LOCAL_DATA_DIR if data_dir is None else Path(data_dir)
        candidates["local_path"] = candidates.apply(
            lambda row: str(_spec_local_path(row, data_dir)), axis=1
        )
        cached_paths = candidates["local_path"].map(Path)
        candidates = candidates.loc[
            cached_paths.map(lambda path: path.is_file() and path.stat().st_size > 0)
        ].copy()

    if candidates.empty and local_data_only:
        return candidates.assign(local_path=pd.Series(dtype="string"))
    candidates = (
        candidates.groupby("repo_id", group_keys=False, sort=False)
        .sample(n=1, random_state=seed)
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )
    if len(candidates) < sample_size and not local_data_only:
        raise ValueError(
            f"Requested {sample_size} specs from distinct repositories, but only "
            f"{len(candidates)} eligible repositories/specs are available."
        )
    if local_data_only:
        result = candidates.iloc[:sample_size].copy()
        return result.reset_index(drop=True)

    downloaded_batches = []
    cursor = 0
    downloaded_ids = set()
    while len(downloaded_ids) < sample_size and cursor < len(candidates):
        needed = sample_size - len(downloaded_ids)
        batch = candidates.iloc[cursor : cursor + needed]
        cursor += len(batch)
        downloaded = download_spec_files(batch, data_dir=data_dir)
        if not downloaded.empty:
            new_rows = downloaded.loc[
                ~downloaded["file_url_sha16"].astype("string").isin(downloaded_ids)
            ]
            downloaded_batches.append(new_rows)
            downloaded_ids.update(new_rows["file_url_sha16"].astype("string"))

    if len(downloaded_ids) < sample_size:
        raise RuntimeError(
            f"Downloaded {len(downloaded_ids)} of {sample_size} requested specs "
            "from distinct repositories; remaining candidates were unavailable."
        )

    result = pd.concat(downloaded_batches, ignore_index=True)
    result = result.drop_duplicates(subset=["file_url_sha16"]).iloc[:sample_size]
    if result["repo_id"].duplicated().any():
        raise RuntimeError("Random spec sample unexpectedly contains duplicate repositories")
    return result.reset_index(drop=True)


def _curated_metadata_path(data_dir: Path) -> Path:
    """Resolve the persisted curated metadata subset under the parquet cache."""
    return data_dir / "data" / CURATED_METADATA_FILENAME


def _write_parquet_cache(df: pd.DataFrame, path: Path) -> None:
    """Publish a parquet cache file atomically so interrupted writes are ignored."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.stem}.",
            suffix=path.suffix,
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)
        df.to_parquet(temp_path, engine="fastparquet", index=False)
        os.replace(temp_path, path)
        temp_path = None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def _load_curated_metadata(data_dir: Path) -> pd.DataFrame:
    """Load cached curated metadata, creating it from the full parquet once."""
    cached_path = _curated_metadata_path(data_dir)
    if cached_path.is_file():
        return pd.read_parquet(cached_path, engine="fastparquet")

    wanted = set(_fetch_sample_repo_names())
    print(f"Curated repo list: {len(wanted)} unique repos")
    full = _read_dataset_parquet(PARQUET_URL, data_dir=data_dir)
    if "repo_name" not in full:
        raise KeyError("SpecMine metadata is missing the repo_name column")
    curated = full.loc[full["repo_name"].isin(wanted)].copy()
    _write_parquet_cache(curated, cached_path)
    print(
        f"Cached {len(curated)} curated spec rows across "
        f"{curated['repo_name'].nunique()} repos at {cached_path}"
    )
    return curated


def _curated_local_dataframe(
    sample_size=None,
    seed=None,
    data_dir=None,
    download_raw_specs=True,
) -> pd.DataFrame:
    """Load curated metadata and optionally download its raw Markdown files."""
    data_dir = DATA_DIR if data_dir is None else Path(data_dir)
    curated = _load_curated_metadata(data_dir)
    if sample_size is not None and sample_size < len(curated):
        curated = curated.sample(n=sample_size, random_state=seed)
    curated = curated.reset_index(drop=True)
    if not download_raw_specs:
        return curated
    return download_spec_files(curated, data_dir=data_dir)


def _local_parquet_path(url: str, data_dir=None) -> Path:
    """Resolve a cached SpecMine parquet file."""
    data_dir = DEFAULT_LOCAL_DATA_DIR if data_dir is None else Path(data_dir)
    filename = url.rsplit("/", 1)[-1]
    candidates = (data_dir / "data" / filename, data_dir / filename)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    expected = data_dir / "data" / filename
    raise FileNotFoundError(
        f"Local SpecMine parquet not found: {expected}. "
        "Run notebooks/data_init.ipynb or a cache-first loader with network "
        "access to populate the cache before using local_data_only."
    )


def _download_dataset_parquet(filename: str, data_dir: Path) -> Path:
    """Download one missing parquet from the Hugging Face dataset repository."""
    from huggingface_hub import hf_hub_download

    path = Path(
        hf_hub_download(
            repo_id=DATASET_REPO_ID,
            repo_type="dataset",
            filename=f"data/{filename}",
            local_dir=str(data_dir),
        )
    )
    if not path.is_file():
        raise FileNotFoundError(
            f"Hugging Face did not create the expected parquet file: {path}"
        )
    return path


def download_dataset_parquets(data_dir=None) -> dict[str, Path]:
    """Find or download the metadata and feature parquet files in ``data_dir``."""
    data_dir = DEFAULT_LOCAL_DATA_DIR if data_dir is None else Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    resolved = {}
    for filename in PARQUET_FILENAMES:
        url = f"hf://datasets/{DATASET_REPO_ID}/data/{filename}"
        try:
            path = _local_parquet_path(url, data_dir=data_dir)
        except FileNotFoundError:
            path = _download_dataset_parquet(filename, data_dir)
        resolved[filename] = path
    return resolved


def _read_dataset_parquet(url: str, *, local_data_only=False, data_dir=None):
    try:
        source = _local_parquet_path(url, data_dir=data_dir)
    except FileNotFoundError:
        if local_data_only:
            raise
        filename = url.rsplit("/", 1)[-1]
        data_dir = DEFAULT_LOCAL_DATA_DIR if data_dir is None else Path(data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        source = _download_dataset_parquet(filename, data_dir)
    return pd.read_parquet(source, engine="fastparquet")


def _load_raw(
    curated_only=False,
    sample_size=None,
    seed=None,
    data_dir=None,
    local_data_only=False,
    download_raw_specs=True,
):
    """Load a single SpecMine DataFrame (no splitting)."""
    if local_data_only and curated_only:
        raise ValueError(
            "local_data_only cannot be combined with curated_only because "
            "curated mode downloads raw spec files."
        )
    if curated_only:
        try:
            return _curated_local_dataframe(
                sample_size=sample_size,
                seed=seed,
                data_dir=data_dir,
                download_raw_specs=download_raw_specs,
            )
        except Exception as e:
            print(f"\nError loading curated dataset: {e}")
            print("Check requirements.txt for needed packages")
            return None
    if local_data_only:
        df = _read_dataset_parquet(
            PARQUET_URL,
            local_data_only=True,
            data_dir=data_dir,
        )
        if sample_size is not None:
            n = min(sample_size, len(df))
            df = df.sample(n=n, random_state=seed).reset_index(drop=True)
        return df
    stop = threading.Event()
    spinner_thread = threading.Thread(
        target=loading_animation,
        args=(
            stop,
            "Loading SpecMine metadata from the local cache or Hugging Face...",
        ),
    )
    spinner_thread.daemon = True
    spinner_thread.start()
    try:
        df = _read_dataset_parquet(PARQUET_URL, data_dir=data_dir)
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


@lru_cache(maxsize=1)
def _load_feature_table(local_data_only=False, data_dir=None) -> pd.DataFrame:
    """Load and cache the precomputed feature table."""
    return _read_dataset_parquet(
        FEATURES_PARQUET_URL,
        local_data_only=local_data_only,
        data_dir=data_dir,
    )


def load_feature_data(
    file_url_sha16=None, *, local_data_only=False, data_dir=None
) -> pd.DataFrame:
    """Load precomputed ``spec_content_features`` rows from SpecMine.

    Args:
        file_url_sha16: Optional iterable of feature join keys or a DataFrame
            containing ``file_url_sha16``. When supplied, only matching feature
            rows are returned.
        local_data_only: Require a cached parquet without contacting Hugging
            Face. By default, an existing local parquet is reused and a missing
            one is downloaded into the cache.
        data_dir: Cache directory; defaults to ``notebooks/data`` in this repo.
    """
    cache_dir = (
        str(Path(data_dir).resolve()) if data_dir is not None else None
    )
    feature_df = _load_feature_table(local_data_only, cache_dir)
    if file_url_sha16 is None:
        return feature_df.copy()

    if isinstance(file_url_sha16, pd.DataFrame):
        if "file_url_sha16" not in file_url_sha16:
            raise KeyError(
                "DataFrame must contain file_url_sha16 to select feature rows"
            )
        selected_ids = file_url_sha16["file_url_sha16"]
    else:
        selected_ids = file_url_sha16

    requested = pd.Index(selected_ids).dropna().drop_duplicates()
    return feature_df.loc[feature_df["file_url_sha16"].isin(requested)].copy()


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
    - Group assignments are deterministic based on a content SHA when present,
      otherwise on the lexically first repo.
    """
    if not 0 <= test_size <= 1:
        raise ValueError("test_size must be between 0 and 1 inclusive")
    if df is None or df.empty:
        return df, df

    parents = {}

    def find(repo):
        while parents[repo] != repo:
            parents[repo] = parents[parents[repo]]
            repo = parents[repo]
        return repo

    def union(left, right):
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parents[max(left_root, right_root)] = min(left_root, right_root)

    row_nodes = []
    grouping_columns = ["repo_name"]
    if "file_content_sha" in df.columns:
        grouping_columns.append("file_content_sha")
    for values in df[grouping_columns].itertuples(index=False, name=None):
        repo = values[0]
        sha = values[1] if len(values) == 2 else None
        repo_node = ("repo", repo) if pd.notna(repo) else None
        sha_node = (
            ("sha", sha) if "file_content_sha" in df.columns and pd.notna(sha) else None
        )
        if repo_node is not None:
            parents.setdefault(repo_node, repo_node)
        if sha_node is not None:
            parents.setdefault(sha_node, sha_node)
        if repo_node is not None and sha_node is not None:
            union(repo_node, sha_node)
        row_nodes.append(repo_node or sha_node)

    group_repos = {}
    group_shas = {}
    for node in parents:
        root = find(node)
        if node[0] == "sha":
            group_shas.setdefault(root, []).append(node[1])
        else:
            group_repos.setdefault(root, []).append(node[1])
    group_is_test = {}
    for root in group_repos.keys() | group_shas.keys():
        stable_key = (
            f"sha:{min(group_shas[root])}"
            if root in group_shas
            else f"repo:{min(group_repos[root])}"
        )
        group_is_test[root] = _repo_bucket(stable_key) < test_size
    row_is_test = [
        group_is_test[find(node)] if node is not None else False for node in row_nodes
    ]
    df = df.copy()
    df["_is_test"] = row_is_test
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
    data_dir=None,
    local_data_only=False,
    download_raw_specs=True,
):
    """
    Load SpecMine data.
    - curated_only=False (default): return the full parquet DataFrame.
    - curated_only=True: cache the filtered curated metadata and, by default,
      download the selected raw spec files into the cache with a `local_path`
      column. The filtered metadata is reused on later calls.
    - download_raw_specs=False: return curated metadata without downloading
      Markdown files; use this when selecting a smaller sample to download.
    - sample_size=N: random sample of N rows (both modes). The full remote
      metadata parquet is cached before sampling when the cache is missing.
    - seed: random seed for sampling (both modes).
    - data_dir: cache root for parquet and raw spec files; defaults to
      notebooks/data in this repository.
    - local_data_only: require the local metadata parquet and never fall back
      to the network. By default, missing required parquets are downloaded from
      Hugging Face and saved in the cache.
    - split=True (or return_splits=True): return a leak-safe (train, test) tuple instead of one frame. No repo_name and no file_content_sha appears on both sides. `test_size` is the approx. fraction of repos assigned to test.
    """
    df = _load_raw(
        curated_only=curated_only,
        sample_size=sample_size,
        seed=seed,
        data_dir=data_dir,
        local_data_only=local_data_only,
        download_raw_specs=download_raw_specs,
    )
    if df is None:
        return None
    if split or return_splits:
        train, test = split_train_test(df, test_size=test_size)
        return train, test
    return df
