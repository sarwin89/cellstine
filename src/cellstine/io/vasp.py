"""Native VASP POSCAR/CONTCAR reader and writer."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .models import StructureRecord
from . import native as native_vasp


class VaspIO:
    """Read and write VASP-family structure files."""

    extensions = {".vasp", ".poscar", ".contcar", ""}

    def read(self, path: str) -> StructureRecord:
        data = native_vasp.read_poscar(path)
        return StructureRecord(
            comment=str(data.comment),
            lattice=np.array(data.lattice, dtype=float, copy=True),
            species=list(data.species),
            counts=[int(value) for value in data.counts],
            positions_direct=np.array(data.positions_direct, dtype=float, copy=True),
            positions_cartesian=np.array(data.positions_cartesian, dtype=float, copy=True),
            coordinate_mode=str(data.coordinate_mode),
            selective_dynamics=bool(data.selective_dynamics),
            selective_flags=None if data.selective_flags is None else [tuple(flags) for flags in data.selective_flags],
            source_path=str(Path(path).resolve()),
            source_format="vasp",
            metadata={},
        )

    def write(
        self,
        record: StructureRecord,
        path: str,
        *,
        positions_are_cartesian: bool = False,
        wrap_positions: bool = False,
        comment: str | None = None,
        validate: bool = True,
    ) -> Path:
        """Write ``record`` to ``path`` and return the resolved path.

        The structure is checked first: a degenerate cell, a species list that
        disagrees with the positions, or two atoms on one site are faults that no
        plane-wave code can be given, so they are raised here rather than written
        out and discovered by the calculation.  Pass ``validate=False`` to skip
        the check when a deliberately unusual structure has to be written.
        """

        output_path = Path(path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        positions = record.positions_cartesian if positions_are_cartesian else record.positions_direct
        native_vasp.write_poscar(
            str(output_path),
            np.asarray(record.lattice, dtype=float),
            np.asarray(positions, dtype=float),
            [int(value) for value in record.counts],
            list(record.species),
            comment=comment or record.comment,
            positions_are_cartesian=bool(positions_are_cartesian),
            wrap_positions=bool(wrap_positions),
            selective_flags=record.selective_flags,
            validate=bool(validate),
        )
        return output_path
