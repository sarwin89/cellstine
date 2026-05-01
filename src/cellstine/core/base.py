"""Common workflow base class and legacy backend imports."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np

from .dependencies import DependencyManager
from .manifests import RunManifest
from .models import CommandResult


def _slug(value: str, max_length: int = 64) -> str:
    safe = []
    for char in value:
        if char.isalnum() or char in {"-", "_"}:
            safe.append(char)
        else:
            safe.append("_")
    slug = "".join(safe).strip("_") or "run"
    if len(slug) > int(max_length):
        slug = slug[: int(max_length)].rstrip("_-") or "run"
    return slug


def run_output_suffix(run_id: str) -> str:
    """Return the short unique suffix used at the end of generated filenames."""

    parts = str(run_id).split("_")
    if len(parts) >= 2 and parts[-2].isdigit() and len(parts[-2]) == 2:
        return f"{parts[-2]}_{parts[-1]}"
    return parts[-1]


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if is_dataclass(value):
        return {key: _json_ready(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


class Base:
    """Shared workflow base for grouped class-first APIs."""

    workflow_name = "base"

    def __init__(
        self,
        *,
        backend: str = "auto",
        runs_root: str | Path = "runs",
        output_root: str | Path = "output",
        dependency_manager: DependencyManager | None = None,
    ) -> None:
        self.backend = str(backend)
        self.runs_root = Path(runs_root).resolve()
        self.output_root = Path(output_root).resolve()
        self.dependency_manager = dependency_manager or DependencyManager()

    def choose_backend(self, *, feature: str | None = None) -> str:
        return self.dependency_manager.choose_backend(self.backend, feature=feature)

    def create_run_dir(self, stage: str, label: str | None = None) -> tuple[str, Path]:
        timestamp = datetime.now().strftime("%y%m%d-%H%M")
        base_id = _slug(stage)
        if label:
            base_id = f"{base_id}_{_slug(label)}"
        run_id = f"{base_id}_{timestamp}"
        run_dir = self.runs_root / self.workflow_name / run_id
        suffix = 1
        while run_dir.exists():
            suffix += 1
            run_id = f"{base_id}_{suffix:02d}_{timestamp}"
            run_dir = self.runs_root / self.workflow_name / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        return run_id, run_dir

    def write_manifest(
        self,
        *,
        stage: str,
        run_id: str,
        run_dir: Path,
        backend: str,
        inputs: Dict[str, Any] | None = None,
        parameters: Dict[str, Any] | None = None,
        artifacts: Dict[str, Any] | None = None,
        summary: Dict[str, Any] | None = None,
    ) -> Path:
        manifest = RunManifest.create(
            workflow=self.workflow_name,
            stage=stage,
            run_id=run_id,
            backend=backend,
            inputs=_json_ready(inputs or {}),
            parameters=_json_ready(parameters or {}),
            artifacts=_json_ready(artifacts or {}),
            summary=_json_ready(summary or {}),
            dependencies=self.dependency_manager.versions(),
        )
        return manifest.write(run_dir / "manifest.json")

    def result(
        self,
        *,
        manifest_path: Path,
        run_dir: Path,
        artifacts: Dict[str, Any],
        summary: Dict[str, Any],
        payload: Dict[str, Any] | None = None,
    ) -> CommandResult:
        return CommandResult(
            manifest_path=manifest_path.resolve(),
            run_dir=run_dir.resolve(),
            artifacts=_json_ready(artifacts),
            summary=_json_ready(summary),
            payload=_json_ready(payload or {}),
        )

    def resolve_results_file(self, path_or_manifest: str, artifact_keys: tuple[str, ...] | list[str]) -> str:
        candidate = Path(path_or_manifest).resolve()
        if candidate.name == "manifest.json":
            manifest = RunManifest.load(candidate)
            for key in artifact_keys:
                if key in manifest.artifacts:
                    return str(Path(str(manifest.artifacts[key])).resolve())
            raise ValueError(f"{candidate} does not contain any of the requested artifacts: {', '.join(artifact_keys)}")
        return str(candidate)
