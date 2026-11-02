"""Exclusive reservation of automatically named output files and directories.

The preferred name is retained when available. Writers own the returned empty
file/directory and may fill it normally; a failed write may leave its new partial
artifact. Explicit user-selected file paths bypass these helpers.
"""

from __future__ import annotations

from pathlib import Path


def reserve_default_file(preferred: str | Path) -> Path:
    """Reserve a default filename without replacing any existing path."""

    original = Path(preferred)
    original.parent.mkdir(parents=True, exist_ok=True)
    candidate = original
    counter = 1
    while True:
        try:
            with candidate.open("xb"):
                pass
        except (FileExistsError, IsADirectoryError, PermissionError) as error:
            # Windows reports EACCES for exclusive creation over a directory.
            # A permission error without that occupied leaf must propagate.
            if not isinstance(error, FileExistsError) and not candidate.is_dir():
                raise
            counter += 1
            candidate = original.with_name(f"{original.stem}_{counter:02d}{original.suffix}")
        else:
            return candidate


def reserve_default_directory(preferred: str | Path) -> Path:
    """Reserve a default batch directory without reusing an existing path."""

    original = Path(preferred)
    original.parent.mkdir(parents=True, exist_ok=True)
    candidate = original
    counter = 1
    while True:
        try:
            candidate.mkdir()
        except FileExistsError:
            counter += 1
            candidate = original.with_name(f"{original.name}_{counter:02d}")
        else:
            return candidate
