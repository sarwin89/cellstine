"""Serializable data models for symmetry workflow results."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class SymmetryOperation:
    """One fractional symmetry operation."""

    rotation: list[list[int]]
    translation: tuple[float, float, float]


@dataclass
class EquivalentAtomGroup:
    """One set of symmetry-equivalent atom indices."""

    group_id: str
    species: str
    representative_index: int
    equivalent_indices: list[int]
    multiplicity: int
    wyckoff: str | None = None


@dataclass
class SymmetryAnalysis:
    """Serializable symmetry analysis result."""

    structure_path: str | None
    backend: str
    atom_count: int
    species: list[str]
    counts: list[int]
    lattice_parameters: dict[str, float]
    space_group_symbol: str | None = None
    space_group_number: int | None = None
    hall_symbol: str | None = None
    point_group: str | None = None
    crystal_system: str | None = None
    lattice_type: str | None = None
    laue: bool | None = None
    operation_count: int = 0
    operations: list[SymmetryOperation] = field(default_factory=list)
    equivalent_groups: list[EquivalentAtomGroup] = field(default_factory=list)
    wyckoffs: list[str] = field(default_factory=list)
    transformation_matrix: list[list[float]] | None = None
    origin_shift: tuple[float, float, float] | None = None
    lattice_point_group: str | None = None
    symmorphic_setting: bool | None = None
    centering_translation_count: int | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "cellstine.symmetry_analysis.v1",
            "structure_path": self.structure_path,
            "backend": self.backend,
            "atom_count": int(self.atom_count),
            "species": list(self.species),
            "counts": [int(value) for value in self.counts],
            "lattice_parameters": dict(self.lattice_parameters),
            "space_group_symbol": self.space_group_symbol,
            "space_group_number": self.space_group_number,
            "hall_symbol": self.hall_symbol,
            "point_group": self.point_group,
            "crystal_system": self.crystal_system,
            "lattice_type": self.lattice_type,
            "lattice_point_group": self.lattice_point_group,
            "symmorphic_setting": self.symmorphic_setting,
            "centering_translation_count": self.centering_translation_count,
            "laue": self.laue,
            "operation_count": int(self.operation_count),
            "operations": [asdict(operation) for operation in self.operations],
            "equivalent_groups": [asdict(group) for group in self.equivalent_groups],
            "wyckoffs": list(self.wyckoffs),
            "transformation_matrix": self.transformation_matrix,
            "origin_shift": None if self.origin_shift is None else [float(value) for value in self.origin_shift],
            "notes": list(self.notes),
        }
