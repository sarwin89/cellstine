"""Canonical internal structure representation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class StructureRecord:
    """Internal structure representation used across workflows."""

    comment: str
    lattice: np.ndarray
    species: List[str]
    counts: List[int]
    positions_direct: np.ndarray
    positions_cartesian: np.ndarray
    coordinate_mode: str = "Direct"
    selective_dynamics: bool = False
    selective_flags: List[Tuple[str, str, str]] | None = None
    source_path: str | None = None
    source_format: str = "vasp"
    metadata: Dict[str, object] = field(default_factory=dict)

    @property
    def natoms(self) -> int:
        return int(sum(int(value) for value in self.counts))

    def copy(self) -> "StructureRecord":
        return StructureRecord(
            comment=str(self.comment),
            lattice=np.array(self.lattice, dtype=float, copy=True),
            species=list(self.species),
            counts=[int(value) for value in self.counts],
            positions_direct=np.array(self.positions_direct, dtype=float, copy=True),
            positions_cartesian=np.array(self.positions_cartesian, dtype=float, copy=True),
            coordinate_mode=str(self.coordinate_mode),
            selective_dynamics=bool(self.selective_dynamics),
            selective_flags=None if self.selective_flags is None else [tuple(flags) for flags in self.selective_flags],
            source_path=None if self.source_path is None else str(Path(self.source_path)),
            source_format=str(self.source_format),
            metadata=dict(self.metadata),
        )
