"""Native POSCAR helpers shared by the finder and generator flows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
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


def _apply_scaling_factor(lattice: np.ndarray, line: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return the scaled cell, and the scaling to apply to Cartesian positions.

    The second line of a POSCAR is VASP's scaling factor, and it has three legal
    forms:

    * one positive number, which multiplies the cell;
    * one negative number, whose magnitude is the *volume* the cell is to have,
      so the cell is scaled by ``(|s| / V)^(1/3)`` and keeps its shape and its
      handedness -- reading it as a plain multiplier would turn the cell inside
      out and give it the wrong size;
    * three positive numbers, which scale the three Cartesian components.

    In every case the factor is a linear map of space, so it acts on Cartesian
    positions exactly as it acts on the cell and leaves fractional coordinates
    alone.  The second return value is that map, as a per-component factor.

    These claims are proved in ``aristotle-lean-reference/RequestProject/PoscarScaling.lean``:
    ``Cellstine.abs_det_volumeScale_smul`` (the rescaled cell has exactly the
    requested volume), ``Cellstine.det_volumeScale_smul_pos_iff`` together with
    ``Cellstine.det_smul_neg_of_neg`` (the handedness survives, where a plain
    multiplier would flip it), and ``Cellstine.vecMul_scaled_cell`` (fractional
    coordinates are untouched).
    """

    tokens = line.split()
    if not tokens:
        raise ValueError("the POSCAR scaling factor line is empty")
    if len(tokens) not in {1, 3}:
        raise ValueError("a POSCAR needs one or three scaling factors")
    values = [float(token) for token in tokens]
    if not np.all(np.isfinite(values)):
        raise ValueError("POSCAR scaling factors must be finite")
    if len(values) == 3:
        factors = np.array(values, dtype=float)
        if np.any(factors <= 0.0):
            raise ValueError("three POSCAR scaling factors must all be positive")
        return lattice * factors[None, :], factors
    scale = float(values[0])
    if scale == 0.0:
        raise ValueError("a POSCAR scaling factor may not be zero")
    if scale < 0.0:
        volume = abs(float(np.linalg.det(lattice)))
        if volume <= 0.0:
            raise ValueError("a POSCAR that asks for a target volume needs a non-degenerate cell")
        scale = (abs(scale) / volume) ** (1.0 / 3.0)
        if scale == 0.0 or not np.isfinite(scale):
            raise ValueError("the POSCAR target-volume scale exceeds the finite coordinate range")
    factors = np.full(3, scale, dtype=float)
    return lattice * scale, factors


def read_poscar(path: str) -> PoscarData:
    """Read a POSCAR/CONTCAR file."""

    with open(path, "r", encoding="utf-8") as handle:
        def next_line(section: str) -> str:
            line = handle.readline()
            if not line:
                raise ValueError(f"{path}: POSCAR is missing {section}")
            return line.rstrip("\n")

        comment = next_line("the comment")
        scaling = next_line("scaling factors")
        lattice = np.array(
            [list(map(float, next_line("a lattice vector").split()[:3])) for _ in range(3)],
            dtype=float,
        )
        if lattice.shape != (3, 3) or not np.all(np.isfinite(lattice)):
            raise ValueError(f"{path}: POSCAR needs three finite lattice vectors")
        lattice, coordinate_scale = _apply_scaling_factor(lattice, scaling)
        if not np.all(np.isfinite(lattice)):
            raise ValueError(f"{path}: scaled POSCAR lattice exceeds the finite coordinate range")

        first_tokens = next_line("species or atom counts").split()
        species: List[str]
        counts: List[int]
        try:
            counts = [int(token) for token in first_tokens]
            species = []
        except ValueError:
            species = first_tokens
            counts = [int(token) for token in next_line("atom counts").split()]
        if not counts or any(value < 0 for value in counts):
            raise ValueError(f"{path}: POSCAR needs nonnegative atom counts")

        coordinate_mode = next_line("the coordinate mode").strip()
        selective_dynamics = coordinate_mode.lower().startswith("s")
        if selective_dynamics:
            coordinate_mode = next_line("the coordinate mode after Selective Dynamics").strip()
        natoms = int(sum(counts))
        raw_positions = []
        selective_flags: List[Tuple[str, str, str]] | None = [] if selective_dynamics else None
        for offset in range(natoms):
            tokens = next_line(f"atom {offset + 1}").split()
            if len(tokens) < 3:
                raise ValueError(f"{path}: POSCAR atom {offset + 1} is missing coordinates")
            raw_positions.append([float(value) for value in tokens[:3]])
            if selective_dynamics:
                selective_flags.append(tuple(tokens[3:6]))

    raw_array = np.array(raw_positions, dtype=float).reshape(-1, 3)
    if not np.all(np.isfinite(raw_array)):
        raise ValueError(f"{path}: POSCAR positions must be finite")
    # VASP recognizes C/c/K/k as Cartesian; every other initial means Direct.
    if coordinate_mode[:1].lower() not in {"c", "k"}:
        positions_direct = raw_array
        positions_cartesian = direct_to_cartesian(raw_array, lattice)
    else:
        # The scaling factor is a map of space, so it moves Cartesian positions
        # exactly as it moves the cell; fractional positions are unchanged by it.
        positions_cartesian = raw_array * coordinate_scale
        positions_direct = cartesian_to_direct(positions_cartesian, lattice)

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

    counts = [int(value) for value in structure.counts]
    source_positions = np.asarray(structure.positions_direct, dtype=float)
    natoms = sum(counts)
    if any(value < 0 for value in counts) or source_positions.shape != (natoms, 3):
        raise ValueError("positions shape does not match atom counts")
    if structure.selective_flags is not None and len(structure.selective_flags) != natoms:
        raise ValueError("selective_flags length does not match number of atoms")
    positions_direct = np.empty((natoms * repeats, 3), dtype=float)
    selective_flags = None if structure.selective_flags is None else []
    source_offset = output_offset = 0
    for count in counts:
        block = source_positions[source_offset : source_offset + count]
        for repeat_index in range(repeats):
            selected = positions_direct[output_offset : output_offset + count]
            selected[:] = block
            selected[:, 2] = (block[:, 2] + float(repeat_index)) / float(repeats)
            if selective_flags is not None:
                selective_flags.extend(tuple(flags) for flags in
                                       structure.selective_flags[source_offset : source_offset + count])
            output_offset += count
        source_offset += count
    positions_cartesian = direct_to_cartesian(positions_direct, lattice)

    return PoscarData(
        comment=f"{structure.comment} | c-repeat x{repeats}",
        lattice=lattice,
        species=list(structure.species),
        counts=[count * repeats for count in counts],
        positions_direct=positions_direct,
        positions_cartesian=positions_cartesian,
        coordinate_mode="Direct",
        selective_dynamics=bool(structure.selective_dynamics),
        selective_flags=selective_flags,
    )


def _normalise_flags(
    selective_flags: Sequence[Sequence[str]] | None,
    natoms: int,
) -> List[Tuple[str, str, str]] | None:
    if selective_flags is None:
        return None
    if len(selective_flags) != natoms:
        raise ValueError("selective_flags length does not match number of atoms")
    normalised = [tuple(str(item) for item in flags[:3]) for flags in selective_flags]
    if any(len(flags) != 3 for flags in normalised):
        raise ValueError("each atom needs three selective-dynamics flags")
    if any(flag.upper() not in {"T", "F"} for flags in normalised for flag in flags):
        raise ValueError("selective-dynamics flags must be T or F")
    return normalised


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
    validate: bool = True,
) -> None:
    """Write a POSCAR using Direct coordinates.

    Unless ``validate`` is switched off, the structure is checked before the
    file is opened: a degenerate cell, a count list that disagrees with the
    positions, a non-finite coordinate or two atoms on one site are faults that
    no plane-wave code can be given, and it is cheaper to raise them here than
    to find them in a failed calculation.
    """

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

    if validate:
        from ..core.validation import validate_structure

        validate_structure(
            lattice=lattice,
            species=list(types) if types else ["X"] * len(list(counts)),
            counts=[int(value) for value in counts],
            positions_direct=direct,
            context=f"refusing to write {Path(path).name}",
        )

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
