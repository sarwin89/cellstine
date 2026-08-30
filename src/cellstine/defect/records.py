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
    void_radius: float | None = None
    """Radius of the largest empty sphere at an interstitial site, in angstrom."""
    void_kind: str | None = None
    """``"maximum"`` when the empty sphere shrinks along every direction out of
    the site, ``"saddle"`` when it grows along some of them."""
    void_coordination: int | None = None
    """Number of atoms lying on the empty sphere of an interstitial site."""
    members: list[dict[str, Any]] = field(default_factory=list)
    """Every member of the orbit this site represents.

    Each entry carries ``indices`` (the 1-based atoms it removes or replaces,
    empty for an inserted atom), ``direct`` and ``cartesian`` coordinates, and
    ``layer_ids``: the atomic planes it lies in, along the direction the
    analysis was read.  The representative is the first entry.  Splitting an
    orbit into one defect per plane -- see :mod:`cellstine.defect.layers` --
    reads these.
    """


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
    view_direction: dict[str, Any] | None = None
    """The direction of observation the atomic planes were counted along."""
    point_group: str | None = None
    """Point group whose operations were used to identify equivalent sites."""
    operation_count: int = 0
    """Number of space-group operations of the cell used for site equivalence."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "cellstine.defect_analysis.v1",
            "structure_path": self.structure_path,
            "structure_kind": self.structure_kind,
            "backend": self.backend,
            "atom_count": self.atom_count,
            "species": list(self.species),
            "counts": [int(value) for value in self.counts],
            "point_group": self.point_group,
            "operation_count": int(self.operation_count),
            "layers": list(self.layers),
            "sites": [asdict(site) for site in self.sites],
            "notes": list(self.notes),
            "view_direction": None if self.view_direction is None else dict(self.view_direction),
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
            point_group=(
                None if payload.get("point_group") is None else str(payload["point_group"])
            ),
            operation_count=int(payload.get("operation_count", 0)),
            view_direction=(
                None if payload.get("view_direction") is None else dict(payload["view_direction"])
            ),
        )

