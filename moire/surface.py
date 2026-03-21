"""Experimental surface-slab builder for orthogonal bulk POSCAR inputs."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

from . import io as io_mod

DEFAULT_OUTPUT_DIR = Path("output")


@dataclass
class SurfaceRun:
    output_path: Path
    miller: tuple[int, int, int]
    layers: int
    vacuum: float
    total_atoms: int


def _reduce_integer_vector(values: Sequence[int]) -> tuple[int, int, int]:
    entries = [int(value) for value in values]
    divisor = 0
    for entry in entries:
        divisor = math.gcd(divisor, abs(entry))
    divisor = max(divisor, 1)
    return tuple(int(entry // divisor) for entry in entries)


def _is_orthogonal_lattice(lattice: np.ndarray, tolerance: float = 1e-6) -> bool:
    vectors = np.asarray(lattice, dtype=float)
    return (
        abs(float(np.dot(vectors[0], vectors[1]))) <= tolerance
        and abs(float(np.dot(vectors[0], vectors[2]))) <= tolerance
        and abs(float(np.dot(vectors[1], vectors[2]))) <= tolerance
    )


def _choose_in_plane_vectors(miller: tuple[int, int, int]) -> tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]:
    h, k, l = (int(value) for value in miller)
    if h == 0 and k == 0 and l == 0:
        raise ValueError("Miller indices cannot all be zero")

    normal = np.array([h, k, l], dtype=int)
    axis_candidates = [
        np.array([1, 0, 0], dtype=int),
        np.array([0, 1, 0], dtype=int),
        np.array([0, 0, 1], dtype=int),
    ]
    first = None
    for axis in axis_candidates:
        candidate = np.cross(normal, axis)
        if np.any(candidate != 0):
            first = candidate
            break
    if first is None:
        raise ValueError(f"could not build an in-plane basis for Miller indices {miller}")

    second = np.cross(normal, first)
    first_reduced = _reduce_integer_vector(first.tolist())
    second_reduced = _reduce_integer_vector(second.tolist())
    normal_reduced = _reduce_integer_vector(normal.tolist())

    transform = np.array([first_reduced, second_reduced, normal_reduced], dtype=int)
    determinant = int(round(np.linalg.det(transform)))
    if determinant == 0:
        raise ValueError(f"surface transform for Miller indices {miller} is singular")
    if determinant < 0:
        transform[[0, 1]] = transform[[1, 0]]
        first_reduced, second_reduced = second_reduced, first_reduced
    return first_reduced, second_reduced, normal_reduced


def _structure_from_transform(structure: io_mod.PoscarData, transform: np.ndarray, tolerance: float = 1e-8) -> io_mod.PoscarData:
    lattice_old = np.asarray(structure.lattice, dtype=float)
    lattice_new = transform @ lattice_old
    inverse_new = np.linalg.inv(lattice_new)

    search_pad = max(2, int(np.max(np.abs(transform))) + 1)
    collected_positions: List[np.ndarray] = []
    collected_flags: List[Tuple[str, str, str]] | None = [] if structure.selective_flags is not None else None
    expanded_flags = structure.selective_flags or []

    for atom_index, base_direct in enumerate(np.asarray(structure.positions_direct, dtype=float)):
        for shift_a in range(-search_pad, search_pad + 1):
            for shift_b in range(-search_pad, search_pad + 1):
                for shift_c in range(-search_pad, search_pad + 1):
                    image_direct = np.array(
                        [base_direct[0] + shift_a, base_direct[1] + shift_b, base_direct[2] + shift_c],
                        dtype=float,
                    )
                    image_cart = io_mod.direct_to_cartesian(image_direct.reshape(1, 3), lattice_old)[0]
                    new_direct = image_cart @ inverse_new
                    if not np.all((-tolerance <= new_direct) & (new_direct <= 1.0 + tolerance)):
                        continue
                    wrapped = np.mod(new_direct, 1.0)
                    duplicate = False
                    for previous in collected_positions:
                        difference = wrapped - previous
                        if np.all(np.abs(difference - np.round(difference)) <= tolerance):
                            duplicate = True
                            break
                    if duplicate:
                        continue
                    collected_positions.append(wrapped)
                    if collected_flags is not None:
                        collected_flags.append(tuple(expanded_flags[atom_index]))

    if not collected_positions:
        raise ValueError("surface transform did not capture any atoms; try a simpler Miller index")

    positions_direct = np.array(collected_positions, dtype=float)
    positions_cartesian = io_mod.direct_to_cartesian(positions_direct, lattice_new)
    multiplicity = int(round(abs(np.linalg.det(transform))))
    counts = [int(count) * multiplicity for count in structure.counts]
    return io_mod.PoscarData(
        comment=f"{structure.comment} | oriented surface cell",
        lattice=lattice_new,
        species=list(structure.species),
        counts=counts,
        positions_direct=positions_direct,
        positions_cartesian=positions_cartesian,
        coordinate_mode="Direct",
        selective_dynamics=bool(structure.selective_dynamics),
        selective_flags=collected_flags,
    )


def _repeat_structure_inplane(structure: io_mod.PoscarData, repeat_a: int, repeat_b: int) -> io_mod.PoscarData:
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


def _add_vacuum_along_c(structure: io_mod.PoscarData, vacuum: float, padding: float = 0.5) -> io_mod.PoscarData:
    vacuum = float(vacuum)
    if vacuum < 0.0:
        raise ValueError("vacuum must be non-negative")

    lattice = np.array(structure.lattice, dtype=float, copy=True)
    c_vector = lattice[2]
    c_length = float(np.linalg.norm(c_vector))
    if c_length <= 1e-12:
        raise ValueError("surface cell has a zero-length c vector")
    c_unit = c_vector / c_length

    cartesian = np.array(structure.positions_cartesian, dtype=float, copy=True)
    projections = cartesian @ c_unit
    cartesian += (padding - float(projections.min())) * c_unit

    lattice[2] = c_unit * (c_length + vacuum)
    positions_direct = io_mod.cartesian_to_direct(cartesian, lattice)
    return io_mod.PoscarData(
        comment=f"{structure.comment} | vacuum {vacuum:.3f} A",
        lattice=lattice,
        species=list(structure.species),
        counts=[int(count) for count in structure.counts],
        positions_direct=positions_direct,
        positions_cartesian=cartesian,
        coordinate_mode="Direct",
        selective_dynamics=bool(structure.selective_dynamics),
        selective_flags=None if structure.selective_flags is None else [tuple(flags) for flags in structure.selective_flags],
    )


def build_surface(
    bulk_poscar: str,
    *,
    miller: tuple[int, int, int],
    layers: int,
    vacuum: float,
    repeat_a: int = 1,
    repeat_b: int = 1,
    output_path: str | None = None,
) -> SurfaceRun:
    structure = io_mod.read_poscar(bulk_poscar)
    if not _is_orthogonal_lattice(structure.lattice):
        raise ValueError(
            "the current experimental surface builder only supports orthogonal bulk cells "
            "(cubic, tetragonal, or orthorhombic)"
        )

    in_plane_a, in_plane_b, normal = _choose_in_plane_vectors(miller)
    oriented = _structure_from_transform(structure, np.array([in_plane_a, in_plane_b, normal], dtype=int))
    layered = io_mod.repeat_structure_along_c(oriented, int(layers))
    repeated = _repeat_structure_inplane(layered, int(repeat_a), int(repeat_b))
    surfaced = _add_vacuum_along_c(repeated, float(vacuum))

    if output_path is None:
        DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = str(
            DEFAULT_OUTPUT_DIR
            / f"surface_{Path(bulk_poscar).stem}_{int(miller[0])}{int(miller[1])}{int(miller[2])}_layers{int(layers)}.vasp"
        )
    else:
        Path(output_path).resolve().parent.mkdir(parents=True, exist_ok=True)

    io_mod.write_poscar(
        output_path,
        surfaced.lattice,
        surfaced.positions_direct,
        surfaced.counts,
        surfaced.species,
        comment=(
            "Generated by CELLSTINE surface stage | "
            f"Miller ({int(miller[0])} {int(miller[1])} {int(miller[2])}) | Made by Sarwin Chandran"
        ),
        positions_are_cartesian=False,
        wrap_positions=False,
        selective_flags=surfaced.selective_flags,
    )
    return SurfaceRun(
        output_path=Path(output_path).resolve(),
        miller=(int(miller[0]), int(miller[1]), int(miller[2])),
        layers=int(layers),
        vacuum=float(vacuum),
        total_atoms=surfaced.natoms,
    )
