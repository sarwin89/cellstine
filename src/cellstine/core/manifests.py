"""Workflow manifest model for stage-to-stage handoff."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


@dataclass
class RunManifest:
    """Serializable run manifest written for each workflow stage."""

    workflow: str
    stage: str
    run_id: str
    backend: str
    created_at: str
    inputs: Dict[str, Any] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, Any] = field(default_factory=dict)
    summary: Dict[str, Any] = field(default_factory=dict)
    dependencies: Dict[str, Any] = field(default_factory=dict)
    path: Path | None = None

    @classmethod
    def create(
        cls,
        *,
        workflow: str,
        stage: str,
        run_id: str,
        backend: str,
        inputs: Dict[str, Any] | None = None,
        parameters: Dict[str, Any] | None = None,
        artifacts: Dict[str, Any] | None = None,
        summary: Dict[str, Any] | None = None,
        dependencies: Dict[str, Any] | None = None,
    ) -> "RunManifest":
        return cls(
            workflow=str(workflow),
            stage=str(stage),
            run_id=str(run_id),
            backend=str(backend),
            created_at=datetime.now(timezone.utc).isoformat(),
            inputs=dict(inputs or {}),
            parameters=dict(parameters or {}),
            artifacts=dict(artifacts or {}),
            summary=dict(summary or {}),
            dependencies=dict(dependencies or {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "workflow": self.workflow,
            "stage": self.stage,
            "run_id": self.run_id,
            "backend": self.backend,
            "created_at": self.created_at,
            "inputs": self.inputs,
            "parameters": self.parameters,
            "artifacts": self.artifacts,
            "summary": self.summary,
            "dependencies": self.dependencies,
        }

    def write(self, path: str | Path) -> Path:
        output_path = Path(path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)
        self.path = output_path
        return output_path

    @classmethod
    def load(cls, path: str | Path) -> "RunManifest":
        source_path = Path(path).resolve()
        with source_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return cls(
            workflow=str(payload["workflow"]),
            stage=str(payload["stage"]),
            run_id=str(payload["run_id"]),
            backend=str(payload["backend"]),
            created_at=str(payload["created_at"]),
            inputs=dict(payload.get("inputs", {})),
            parameters=dict(payload.get("parameters", {})),
            artifacts=dict(payload.get("artifacts", {})),
            summary=dict(payload.get("summary", {})),
            dependencies=dict(payload.get("dependencies", {})),
            path=source_path,
        )
