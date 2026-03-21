"""POSCAR helpers shared by the finder and generator flows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np


@dataclass
class PoscarData:
    """Parsed POSCAR/CONTCAR data."""

    comment: str
    lattice: np.ndarray
    species: List[str]
    counts: List[int]
    positions_direct: np.ndarray
    positions_cartesian: np.ndarray
    coordinate_mode: str
    selective_dynamics: bool
    selective_flags: List[Tuple[str, str, str]] | None

    @property
    def natoms(self) -> int:
        return int(sum(self.counts))


def cartesian_to_direct(cartesian: np.ndarray, lattice: np.ndarray) -> np.ndarray:
    """Convert row-vector Cartesian coordinates to direct coordinates."""

    return np.asarray(cartesian, dtype=float) @ np.linalg.inv(np.asarray(lattice, dtype=float))


def direct_to_cartesian(direct: np.ndarray, lattice: np.ndarray) -> np.ndarray:
    """Convert row-vector direct coordinates to Cartesian coordinates."""

    return np.asarray(direct, dtype=float) @ np.asarray(lattice, dtype=float)


def wrap_direct(direct: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """Wrap direct coordinates into the [0, 1) interval."""

    wrapped = np.mod(np.asarray(direct, dtype=float), 1.0)
    wrapped[np.isclose(wrapped, 1.0, atol=tol)] = 0.0
    wrapped[np.isclose(wrapped, 0.0, atol=tol)] = 0.0
    return wrapped


def read_poscar(path: str) -> PoscarData:
    """Read a POSCAR/CONTCAR file."""

    with open(path, "r", encoding="utf-8") as handle:
        lines = handle.read().splitlines()

    if len(lines) < 8:
        raise ValueError(f"{path} does not look like a POSCAR/CONTCAR file")

    comment = lines[0].rstrip("\n")
    scale = float(lines[1].split()[0])
    lattice = np.array([list(map(float, lines[i].split()[:3])) for i in range(2, 5)], dtype=float)
    lattice *= scale

    line_index = 5
    first_tokens = lines[line_index].split()
    species: List[str]
    counts: List[int]
    try:
        counts = [int(token) for token in first_tokens]
        species = []
        line_index += 1
    except ValueError:
        species = first_tokens
        line_index += 1
        counts = [int(token) for token in lines[line_index].split()]
        line_index += 1

    selective_dynamics = False
    if lines[line_index].strip().lower().startswith("s"):
        selective_dynamics = True
        line_index += 1

    coordinate_mode = lines[line_index].strip()
    line_index += 1

    natoms = int(sum(counts))
    raw_positions = []
    selective_flags: List[Tuple[str, str, str]] | None = [] if selective_dynamics else None
    for offset in range(natoms):
        tokens = lines[line_index + offset].split()
        raw_positions.append([float(tokens[0]), float(tokens[1]), float(tokens[2])])
        if selective_dynamics:
            selective_flags.append(tuple(tokens[3:6]))

    raw_array = np.array(raw_positions, dtype=float)
    if coordinate_mode.lower().startswith("d"):
        positions_direct = raw_array
        positions_cartesian = direct_to_cartesian(raw_array, lattice)
    else:
        positions_cartesian = raw_array
        positions_direct = cartesian_to_direct(raw_array, lattice)

    return PoscarData(
        comment=comment,
        lattice=lattice,
        species=species,
        counts=counts,
        positions_direct=positions_direct,
        positions_cartesian=positions_cartesian,
        coordinate_mode=coordinate_mode,
        selective_dynamics=selective_dynamics,
        selective_flags=selective_flags,
    )


def repeat_structure_along_c(structure: PoscarData, repeats: int) -> PoscarData:
    """Return a new structure with the input repeated along the lattice c axis."""

    repeats = int(repeats)
    if repeats < 1:
        raise ValueError("c-axis repeats must be at least 1")
    if repeats == 1:
        return PoscarData(
            comment=str(structure.comment),
            lattice=np.array(structure.lattice, dtype=float, copy=True),
            species=list(structure.species),
            counts=[int(value) for value in structure.counts],
            positions_direct=np.array(structure.positions_direct, dtype=float, copy=True),
            positions_cartesian=np.array(structure.positions_cartesian, dtype=float, copy=True),
            coordinate_mode=str(structure.coordinate_mode),
            selective_dynamics=bool(structure.selective_dynamics),
            selective_flags=None
            if structure.selective_flags is None
            else [tuple(flags) for flags in structure.selective_flags],
        )

    lattice = np.array(structure.lattice, dtype=float, copy=True)
    lattice[2] *= float(repeats)

    direct_blocks = []
    for repeat_index in range(repeats):
        shifted = np.array(structure.positions_direct, dtype=float, copy=True)
        shifted[:, 2] = (shifted[:, 2] + float(repeat_index)) / float(repeats)
        direct_blocks.append(shifted)
    positions_direct = np.vstack(direct_blocks) if direct_blocks else np.zeros((0, 3), dtype=float)
    positions_cartesian = direct_to_cartesian(positions_direct, lattice)

    selective_flags = None
    if structure.selective_flags is not None:
        selective_flags = []
        for _ in range(repeats):
            selective_flags.extend(tuple(flags) for flags in structure.selective_flags)

    return PoscarData(
        comment=f"{structure.comment} | c-repeat x{repeats}",
        lattice=lattice,
        species=list(structure.species),
        counts=[int(count) * repeats for count in structure.counts],
        positions_direct=positions_direct,
        positions_cartesian=positions_cartesian,
        coordinate_mode="Direct",
        selective_dynamics=bool(structure.selective_dynamics),
        selective_flags=selective_flags,
    )


def parse_poscar(path: str) -> Tuple[np.ndarray, np.ndarray, List[int], List[str]]:
    """Backward-compatible POSCAR parser used by older tests and scripts."""

    data = read_poscar(path)
    return data.lattice, data.positions_cartesian, data.counts, data.species


def _normalise_flags(
    selective_flags: Sequence[Sequence[str]] | None,
    natoms: int,
) -> List[Tuple[str, str, str]] | None:
    if selective_flags is None:
        return None
    if len(selective_flags) != natoms:
        raise ValueError("selective_flags length does not match number of atoms")
    return [tuple(str(item) for item in flags[:3]) for flags in selective_flags]


def write_poscar(
    path: str,
    lattice: np.ndarray,
    positions: np.ndarray,
    counts: Sequence[int],
    types: Sequence[str] | None = None,
    comment: str = "Generated by CELLSTINE",
    *,
    positions_are_cartesian: bool = True,
    wrap_positions: bool = True,
    selective_flags: Sequence[Sequence[str]] | None = None,
) -> None:
    """Write a POSCAR using Direct coordinates."""

    lattice = np.asarray(lattice, dtype=float)
    positions_array = np.asarray(positions, dtype=float)
    natoms = int(sum(int(value) for value in counts))
    if positions_array.shape != (natoms, 3):
        raise ValueError("positions shape does not match atom counts")

    if positions_are_cartesian:
        direct = cartesian_to_direct(positions_array, lattice)
    else:
        direct = positions_array
    if wrap_positions:
        direct = wrap_direct(direct)

    normalised_flags = _normalise_flags(selective_flags, natoms)

    with open(path, "w", encoding="utf-8") as handle:
        handle.write(comment.rstrip() + "\n")
        handle.write("1.0\n")
        for vector in lattice:
            handle.write(
                "  {0:21.16f} {1:21.16f} {2:21.16f}\n".format(
                    float(vector[0]),
                    float(vector[1]),
                    float(vector[2]),
                )
            )
        if types:
            handle.write("  " + "  ".join(str(symbol) for symbol in types) + "\n")
        handle.write("  " + "  ".join(str(int(value)) for value in counts) + "\n")
        if normalised_flags is not None:
            handle.write("Selective Dynamics\n")
        handle.write("Direct\n")
        for index, position in enumerate(direct):
            line = "  {0:19.16f} {1:19.16f} {2:19.16f}".format(
                float(position[0]),
                float(position[1]),
                float(position[2]),
            )
            if normalised_flags is not None:
                line += " " + " ".join(normalised_flags[index])
            handle.write(line + "\n")
