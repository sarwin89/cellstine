#!/usr/bin/env python3
"""Repository-local launcher for the maintained CELLSTINE CLI."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cellstine.cli.main import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
