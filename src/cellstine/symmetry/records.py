"""Structure-record helper functions for symmetry workflows."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from ..core.lattice import vector_angle_deg
from ..io.models import StructureRecord


def species_type_map(species_by_atom: Sequence[str]) -> tuple[list[int], dict[int, str]]:
    """Return spglib integer species labels and their symbol map."""

    order: list[str] = []
    numbers: list[int] = []
    mapping: dict[int, str] = {}
    for symbol in species_by_atom:
        if str(symbol) not in order:
            order.append(str(symbol))
            mapping[len(order)] = str(symbol)
        numbers.append(order.index(str(symbol)) + 1)
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

    ordered_species: list[str] = []
    for symbol in source.species:
        if symbol in atom_species and symbol not in ordered_species:
            ordered_species.append(str(symbol))
    for symbol in atom_species:
        if symbol not in ordered_species:
            ordered_species.append(str(symbol))

    grouped_positions: list[np.ndarray] = []
    counts: list[int] = []
    for symbol in ordered_species:
        indices = [index for index, atom_symbol in enumerate(atom_species) if atom_symbol == symbol]
        counts.append(len(indices))
        grouped_positions.extend(np.asarray(direct[index], dtype=float) for index in indices)

    output_direct = np.asarray(grouped_positions, dtype=float) if grouped_positions else np.zeros((0, 3), dtype=float)
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
