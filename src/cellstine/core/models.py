"""Shared data models used by the public workflow classes."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict


@dataclass
class CommandResult:
    """Thin user-facing result wrapper shared by the workflow APIs."""

    manifest_path: Path
    run_dir: Path
    artifacts: Dict[str, Any]
    summary: Dict[str, Any]
    payload: Dict[str, Any] = field(default_factory=dict)
