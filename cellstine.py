#!/usr/bin/env python3
"""Repository-local launcher for the maintained CELLSTINE CLI."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
root_resolved = ROOT.resolve()
sys.path = [
    entry
    for entry in sys.path
    if Path(entry or ".").resolve() != root_resolved
]
sys.modules.pop("cellstine", None)
sys.path.insert(0, str(SRC))

from cellstine.cli.main import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
