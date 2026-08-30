"""In-plane supercells, stacking sequences, and slab assembly."""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import numpy as np

from ...io import native as io_mod
from ...core.lattice import vector_angle_deg
from .surface_cell import (
    _build_native_primitive_surface_cell,
    _primitive_surface_vectors,
    _primitive_translation_lattice,
    _reciprocal_normal,
    _structure_from_transform,
)
from .sequence import shortest_repeating_prefix, stacking_sequence
from .surface_types import PrimitiveSurfaceAnalysis, SurfaceStructureBuild


def repeat_structure_inplane(structure: io_mod.PoscarData, repeat_a: int, repeat_b: int) -> io_mod.PoscarData:
    repeat_a = int(repeat_a)
    repeat_b = int(repeat_b)
    if repeat_a < 1 or repeat_b < 1:
        raise ValueError("in-plane repeats must be at least 1")
    if repeat_a == 1 and repeat_b == 1:
        return structure

    lattice = np.array(structure.lattice, dtype=float, copy=True)
    lattice[0] *= float(repeat_a)
    lattice[1] *= float(repeat_b)

    direct_blocks = []
    flags_out: List[Tuple[str, str, str]] | None = [] if structure.selective_flags is not None else None
    for i_repeat in range(repeat_a):
        for j_repeat in range(repeat_b):
            shifted = np.array(structure.positions_direct, dtype=float, copy=True)
            shifted[:, 0] = (shifted[:, 0] + float(i_repeat)) / float(repeat_a)
            shifted[:, 1] = (shifted[:, 1] + float(j_repeat)) / float(repeat_b)
            direct_blocks.append(shifted)
            if flags_out is not None:
                flags_out.extend(tuple(flags) for flags in structure.selective_flags or [])

    positions_direct = np.vstack(direct_blocks)
    return io_mod.PoscarData(
        comment=f"{structure.comment} | in-plane repeat {repeat_a}x{repeat_b}",
        lattice=lattice,
        species=list(structure.species),
        counts=[int(count) * repeat_a * repeat_b for count in structure.counts],
        positions_direct=positions_direct,
        positions_cartesian=io_mod.direct_to_cartesian(positions_direct, lattice),
        coordinate_mode="Direct",
        selective_dynamics=bool(structure.selective_dynamics),
        selective_flags=flags_out,
    )


def apply_inplane_supercell_matrix(
    structure: io_mod.PoscarData,
    supercell_matrix: Sequence[int] | None,
) -> tuple[io_mod.PoscarData, tuple[int, int, int, int] | None]:
    if supercell_matrix is None:
        return structure, None

    entries = [int(value) for value in supercell_matrix]
    if len(entries) != 4:
        raise ValueError("supercell_matrix must contain exactly four integers")

    matrix = np.array([[entries[0], entries[1], 0], [entries[2], entries[3], 0], [0, 0, 1]], dtype=int)
    determinant = int(round(np.linalg.det(matrix)))
    if determinant == 0:
        raise ValueError("in-plane supercell matrix must have a non-zero determinant")
    if determinant < 0:
        matrix[[0, 1]] = matrix[[1, 0]]

    transformed = _structure_from_transform(structure, matrix)
    applied = (
        int(matrix[0, 0]),
        int(matrix[0, 1]),
        int(matrix[1, 0]),
        int(matrix[1, 1]),
    )
    return transformed, applied


def _resolve_inplane_repeats(
    structure: io_mod.PoscarData,
    repeat_a: int,
    repeat_b: int,
    min_length_a: float | None,
    min_length_b: float | None,
) -> tuple[int, int]:
    resolved_a = max(1, int(repeat_a))
    resolved_b = max(1, int(repeat_b))

    length_a = float(np.linalg.norm(structure.lattice[0]))
    length_b = float(np.linalg.norm(structure.lattice[1]))
    if min_length_a is not None and min_length_a > 0.0 and length_a > 1e-12:
        resolved_a = max(resolved_a, int(math.ceil(float(min_length_a) / length_a)))
    if min_length_b is not None and min_length_b > 0.0 and length_b > 1e-12:
        resolved_b = max(resolved_b, int(math.ceil(float(min_length_b) / length_b)))
    return resolved_a, resolved_b


def analyse_primitive_surface(
    bulk_poscar: str,
    *,
    miller: tuple[int, int, int],
    probe_layers: int = 8,
) -> PrimitiveSurfaceAnalysis:
    structure = io_mod.read_poscar(bulk_poscar)
    primitive_lattice, centering = _primitive_translation_lattice(structure)
    normal = _reciprocal_normal(np.asarray(structure.lattice, dtype=float), miller)
    surface_a, surface_b = _primitive_surface_vectors(
        np.asarray(structure.lattice, dtype=float),
        primitive_lattice,
        normal,
        miller,
    )
    # The probe only reports the stacking sequence, so it asks for a fixed number
    # of layers whatever the stacking period is and does not need the vacuum-free
    # cell to be a whole number of periods.
    probe = _build_native_primitive_surface_cell(
        structure,
        miller,
        layers=max(1, int(probe_layers)),
        vacuum=0.0,
        require_bulk_period=False,
    )
    sequence, atoms_per_layer = stacking_sequence(probe)
    return PrimitiveSurfaceAnalysis(
        miller=(int(miller[0]), int(miller[1]), int(miller[2])),
        centering=centering,
        probe_layers=max(1, int(probe_layers)),
        atoms_per_layer=atoms_per_layer,
        stacking_sequence=sequence,
        stacking_period=shortest_repeating_prefix(sequence),
        inplane_angle_deg=vector_angle_deg(surface_a, surface_b),
        lattice=np.array(probe.lattice, dtype=float, copy=True),
    )


def build_surface_structure(
    bulk_poscar: str,
    *,
    miller: tuple[int, int, int],
    layers: int,
    vacuum: float,
    repeat_a: int = 1,
    repeat_b: int = 1,
    min_length_a: float | None = None,
    min_length_b: float | None = None,
    supercell_matrix: Sequence[int] | None = None,
) -> SurfaceStructureBuild:
    structure = io_mod.read_poscar(bulk_poscar)
    primitive_surface = _build_native_primitive_surface_cell(
        structure,
        miller,
        layers=int(layers),
        vacuum=float(vacuum),
    )
    resolved_repeat_a, resolved_repeat_b = _resolve_inplane_repeats(
        primitive_surface,
        int(repeat_a),
        int(repeat_b),
        min_length_a,
        min_length_b,
    )
    repeated = repeat_structure_inplane(primitive_surface, resolved_repeat_a, resolved_repeat_b)
    surfaced, applied_matrix = apply_inplane_supercell_matrix(repeated, supercell_matrix)
    return SurfaceStructureBuild(
        structure=surfaced,
        repeat_a=int(resolved_repeat_a),
        repeat_b=int(resolved_repeat_b),
        supercell_matrix=applied_matrix,
    )
