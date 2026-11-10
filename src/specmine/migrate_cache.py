"""Copy an older local cache without publishing partially copied files."""

from __future__ import annotations

import argparse
import filecmp
import os
import shutil
import tempfile
from pathlib import Path
from typing import Sequence


def copy_tree(source: Path, destination: Path) -> int:
    """Copy missing files atomically, rejecting differing existing files."""
    if not source.is_dir():
        raise FileNotFoundError(f"Source directory does not exist: {source}")
    source = source.resolve()
    destination = destination.resolve()
    if destination == source or source in destination.parents:
        raise ValueError("Destination must not be inside the source directory")
    copied = 0
    for path in sorted(source.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Cannot migrate symbolic link: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"Cannot migrate non-regular file: {path}")
        target = destination / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if not target.is_file() or not filecmp.cmp(path, target, shallow=False):
                raise FileExistsError(f"Destination differs from source: {target}")
            continue
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as temp:
                temp_path = Path(temp.name)
                with path.open("rb") as source_file:
                    shutil.copyfileobj(source_file, temp)
            try:
                os.link(temp_path, target)
            except FileExistsError:
                if not target.is_file() or not filecmp.cmp(path, target, shallow=False):
                    raise FileExistsError(f"Destination differs from source: {target}")
            else:
                copied += 1
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
    return copied


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args(argv)
    try:
        copied = copy_tree(args.source, args.destination)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Migration failed: {exc}\n")
    print(f"Copied {copied} file(s); identical existing files were preserved.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
