"""Serializable records for defect analysis workflows."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class DefectSite:
    """One representative defect site in a structure."""

    site_id: str
    species: str | None
    layer_id: int | None
    direct: tuple[float, float, float]
    cartesian: tuple[float, float, float]
    equivalent_indices: list[int] = field(default_factory=list)
    multiplicity: int = 1
    wyckoff: str | None = None
    site_kind: str = "atom"
    backend: str = "native"
    representative_index: int | None = None
    site_family: str | None = None
    pair_indices: list[int] = field(default_factory=list)


@dataclass
class DefectAnalysis:
    """Serializable defect analysis result."""

    structure_path: str
    structure_kind: str
    backend: str
    atom_count: int
    species: list[str]
    counts: list[int]
    layers: list[dict[str, Any]]
    sites: list[DefectSite]
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "cellstine.defect_analysis.v1",
            "structure_path": self.structure_path,
            "structure_kind": self.structure_kind,
            "backend": self.backend,
            "atom_count": self.atom_count,
            "species": list(self.species),
            "counts": [int(value) for value in self.counts],
            "layers": list(self.layers),
            "sites": [asdict(site) for site in self.sites],
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DefectAnalysis":
        return cls(
            structure_path=str(payload["structure_path"]),
            structure_kind=str(payload.get("structure_kind", "auto")),
            backend=str(payload.get("backend", "native")),
            atom_count=int(payload.get("atom_count", 0)),
            species=[str(value) for value in payload.get("species", [])],
            counts=[int(value) for value in payload.get("counts", [])],
            layers=list(payload.get("layers", [])),
            sites=[DefectSite(**dict(site)) for site in payload.get("sites", [])],
            notes=[str(value) for value in payload.get("notes", [])],
        )

