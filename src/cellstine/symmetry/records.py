"""Structure-record helper functions for symmetry workflows."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from ..core.lattice import vector_angle_deg
from ..io.models import StructureRecord


def species_type_map(species_by_atom: Sequence[str]) -> tuple[list[int], dict[int, str]]:
    """Return spglib integer species labels and their symbol map."""

    codes: dict[str, int] = {}
    numbers: list[int] = []
    mapping: dict[int, str] = {}
    for value in species_by_atom:
        symbol = str(value)
        if symbol not in codes:
            code = len(codes) + 1
            codes[symbol] = code
            mapping[code] = symbol
        numbers.append(codes[symbol])
    return numbers, mapping


def record_from_spglib_cell(
    source: StructureRecord,
    cell: tuple[Any, Any, Any],
    species_map: dict[int, str],
    *,
    comment: str,
) -> StructureRecord:
    """Return a grouped structure record from a spglib cell tuple."""

    lattice, positions, numbers = cell
    atom_species = [species_map.get(int(number), f"X{int(number)}") for number in list(numbers)]
    return record_from_atoms(source, lattice, positions, atom_species, comment=comment)


def record_from_atoms(
    source: StructureRecord,
    lattice: Any,
    positions: Any,
    atom_species: Sequence[str],
    *,
    comment: str,
) -> StructureRecord:
    """Return a species-grouped record from a per-atom species list."""

    lattice_array = np.asarray(lattice, dtype=float)
    direct = np.mod(np.asarray(positions, dtype=float).reshape(-1, 3), 1.0)
    atom_species = [str(value) for value in atom_species]

    if len(atom_species) != direct.shape[0]:
        raise ValueError("one species label per atom is required")
    grouped: dict[str, list[int]] = {}
    for index, symbol in enumerate(atom_species):
        grouped.setdefault(symbol, []).append(index)
    ordered_species: list[str] = []
    seen: set[str] = set()
    for value in source.species:
        symbol = str(value)
        if symbol in grouped and symbol not in seen:
            ordered_species.append(symbol)
            seen.add(symbol)
    for symbol in grouped:
        if symbol not in seen:
            ordered_species.append(symbol)
            seen.add(symbol)
    counts = [len(grouped[symbol]) for symbol in ordered_species]
    indices = [index for symbol in ordered_species for index in grouped[symbol]]
    output_direct = direct[np.asarray(indices, dtype=np.intp)]
    return StructureRecord(
        comment=comment,
        lattice=lattice_array,
        species=ordered_species,
        counts=counts,
        positions_direct=output_direct,
        positions_cartesian=output_direct @ lattice_array,
        coordinate_mode="Direct",
        selective_dynamics=False,
        selective_flags=None,
        source_path=source.source_path,
        source_format=source.source_format,
        metadata=dict(source.metadata),
    )


def lattice_parameters(lattice: np.ndarray) -> dict[str, float]:
    """Return lengths, angles, and volume for a row-vector lattice."""

    matrix = np.asarray(lattice, dtype=float)
    lengths = [float(np.linalg.norm(matrix[index])) for index in range(3)]
    angles = [
        vector_angle_deg(matrix[first], matrix[second])
        for first, second in ((1, 2), (0, 2), (0, 1))
    ]
    return {
        "a": lengths[0],
        "b": lengths[1],
        "c": lengths[2],
        "alpha": angles[0],
        "beta": angles[1],
        "gamma": angles[2],
        "volume": abs(float(np.linalg.det(matrix))),
    }
