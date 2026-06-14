"""Layer-selection and layer-shift helpers shared across workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from ..io import native as io_mod


@dataclass(frozen=True)
class LayerSelection:
    top_indices: tuple[int, ...]
    bottom_indices: tuple[int, ...]
    z_cutoff: float
    gap_size: float

    @property
    def top_atom_count(self) -> int:
        return len(self.top_indices)

    @property
    def bottom_atom_count(self) -> int:
        return len(self.bottom_indices)


@dataclass(frozen=True)
class LayerShiftRun:
    output_path: Path
    top_atom_count: int
    bottom_atom_count: int
    z_cutoff: float
    gap_size: float
    shift_cartesian: np.ndarray
    shift_direct: np.ndarray


def identify_top_layer(
    structure: io_mod.PoscarData,
    *,
    z_cutoff: float | None = None,
    min_gap: float = 1.0,
) -> LayerSelection:
    positions = np.asarray(structure.positions_cartesian, dtype=float)
    if positions.shape[0] == 0:
        raise ValueError("structure does not contain any atoms")

    z_values = positions[:, 2]
    if z_cutoff is None:
        order = np.argsort(z_values)
        sorted_z = z_values[order]
        if sorted_z.size < 2:
            raise ValueError("at least two atoms are required to isolate an upper layer")
        gaps = np.diff(sorted_z)
        gap_index = int(np.argmax(gaps))
        gap_size = float(gaps[gap_index])
        if gap_size < float(min_gap):
            raise ValueError(
                f"largest internal z gap is only {gap_size:.4f} A; provide --z-cutoff if the upper layer is not cleanly separated"
            )
        z_cutoff = float(0.5 * (sorted_z[gap_index] + sorted_z[gap_index + 1]))
    else:
        gap_size = float("nan")
        z_cutoff = float(z_cutoff)

    top_mask = z_values > float(z_cutoff)
    bottom_mask = ~top_mask
    if not np.any(top_mask):
        raise ValueError(f"no atoms were found above z_cutoff={float(z_cutoff):.6f} A")
    if not np.any(bottom_mask):
        raise ValueError(f"all atoms are above z_cutoff={float(z_cutoff):.6f} A; no bottom layer atoms remain")

    return LayerSelection(
        top_indices=tuple(int(index) for index in np.flatnonzero(top_mask)),
        bottom_indices=tuple(int(index) for index in np.flatnonzero(bottom_mask)),
        z_cutoff=float(z_cutoff),
        gap_size=float(gap_size),
    )


def resolve_shift_vectors(
    lattice: np.ndarray,
    shift_cartesian: Sequence[float] | None,
    shift_direct: Sequence[float] | None,
) -> tuple[np.ndarray, np.ndarray]:
    if shift_cartesian is not None and shift_direct is not None:
        raise ValueError("use either shift_cartesian or shift_direct, not both")

    if shift_direct is not None:
        values = list(shift_direct)
        if len(values) not in {2, 3}:
            raise ValueError("shift_direct must contain either 2 values (du,dv) or 3 values (du,dv,dw)")
        direct_shift = np.zeros(3, dtype=float)
        direct_shift[: len(values)] = np.array(values, dtype=float)
        return io_mod.direct_to_cartesian(direct_shift.reshape(1, 3), lattice)[0], direct_shift

    if shift_cartesian is not None:
        values = list(shift_cartesian)
        if len(values) not in {2, 3}:
            raise ValueError("shift_cartesian must contain either 2 values (dx,dy) or 3 values (dx,dy,dz)")
        cartesian_shift = np.zeros(3, dtype=float)
        cartesian_shift[: len(values)] = np.array(values, dtype=float)
        return cartesian_shift, io_mod.cartesian_to_direct(cartesian_shift.reshape(1, 3), lattice)[0]

    return np.zeros(3, dtype=float), np.zeros(3, dtype=float)


def shift_top_layer(
    poscar_path: str,
    *,
    output_path: str | None = None,
    shift_cartesian: Sequence[float] | None = None,
    shift_direct: Sequence[float] | None = None,
    z_cutoff: float | None = None,
    min_gap: float = 1.0,
) -> LayerShiftRun:
    structure = io_mod.read_poscar(poscar_path)
    selection = identify_top_layer(structure, z_cutoff=z_cutoff, min_gap=min_gap)
    cartesian_shift, direct_shift = resolve_shift_vectors(structure.lattice, shift_cartesian, shift_direct)

    direct_positions = np.array(structure.positions_direct, dtype=float, copy=True)
    top_indices = np.array(selection.top_indices, dtype=int)
    direct_positions[top_indices] += direct_shift

    if output_path is None:
        input_path = Path(poscar_path).resolve()
        output_root = Path("output")
        output_root.mkdir(parents=True, exist_ok=True)
        output_path = str((output_root / f"{input_path.stem}_upper_layer_shifted{input_path.suffix or '.vasp'}").resolve())
    else:
        Path(output_path).resolve().parent.mkdir(parents=True, exist_ok=True)

    io_mod.write_poscar(
        output_path,
        structure.lattice,
        direct_positions,
        structure.counts,
        structure.species,
        comment=f"{structure.comment} | upper layer shifted",
        positions_are_cartesian=False,
        wrap_positions=False,
        selective_flags=structure.selective_flags,
    )

    return LayerShiftRun(
        output_path=Path(output_path).resolve(),
        top_atom_count=selection.top_atom_count,
        bottom_atom_count=selection.bottom_atom_count,
        z_cutoff=selection.z_cutoff,
        gap_size=selection.gap_size,
        shift_cartesian=cartesian_shift,
        shift_direct=direct_shift,
    )
