"""Shared data models used by the public workflow classes."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict


@dataclass(frozen=True)
class PrestrainConfig:
    """Pre-commensuration strain definition for a single layer."""

    mode: str = "none"
    magnitude: float = 0.0
    axis: str | None = None


@dataclass
class CommandResult:
    """Thin user-facing result wrapper shared by the workflow APIs."""

    manifest_path: Path
    run_dir: Path
    artifacts: Dict[str, str]
    summary: Dict[str, Any]
    payload: Dict[str, Any] = field(default_factory=dict)
