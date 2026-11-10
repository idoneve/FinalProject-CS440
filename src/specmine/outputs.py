"""Atomic publication of notebook output files."""

from __future__ import annotations

import os
import tempfile
import uuid
from pathlib import Path

import pandas as pd


def write_csv_atomic(frame: pd.DataFrame, destination: Path) -> None:
    """Replace a CSV only after its complete contents have been written."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", dir=destination.parent,
            prefix=f".{destination.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            frame.to_csv(stream, index=False)
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class FigureRun:
    """Stage a complete set of figures and publish it as a unique directory."""

    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.run_id = uuid.uuid4().hex
        self.staging = Path(tempfile.mkdtemp(prefix=".staging-", dir=self.directory))
        self.destination = self.directory / f"run-{self.run_id}"
        self.names: set[str] = set()
        self.failed = False
        self.published = False

    def save(self, figure, name: str) -> None:
        if self.published or self.failed:
            raise RuntimeError("Figure run is already published or failed; start a new run")
        if not name or Path(name).name != name or name in {".", ".."}:
            raise ValueError("Figure name must be a single filename component")
        try:
            for extension in ("png", "pdf"):
                figure.savefig(
                    self.staging / f"{name}.{extension}", dpi=300, bbox_inches="tight"
                )
        except Exception:
            self.failed = True
            raise
        self.names.add(name)

    def publish(self) -> list[Path]:
        if self.failed or self.published or not self.names:
            raise RuntimeError("Cannot publish an empty, failed, or already published figure run")
        for name in self.names:
            for extension in ("png", "pdf"):
                path = self.staging / f"{name}.{extension}"
                if not path.is_file() or path.stat().st_size == 0:
                    raise RuntimeError(f"Staged figure is missing or empty: {path}")
        os.rename(self.staging, self.destination)
        self.published = True
        return [
            self.destination / f"{name}.{extension}"
            for name in sorted(self.names)
            for extension in ("png", "pdf")
        ]
