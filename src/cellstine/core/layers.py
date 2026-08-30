"""Layer-selection and layer-shift helpers shared across workflows.

Which atoms belong to the upper layer is decided by height *along the surface
normal*, not by Cartesian ``z``: the layers of a stack are planes parallel to
``a`` and ``b``, so that is the coordinate that separates them.  For the
convention every stage writes -- ``a`` along ``x``, ``b`` in the ``xy`` plane --
the two agree exactly, and ``z_cutoff`` keeps its usual meaning; for a cell in
any other orientation the normal projection is the one that answers the
question, and it also reads a left-handed cell the right way up.

``layer_partition`` is the *one* rule the whole package uses to decide which
atoms share an atomic plane: sort the heights and cut wherever a consecutive
gap exceeds the tolerance.  That is single linkage, and
``aristotle-lean-reference/RequestProject/LayerPartition.lean`` proves what makes it the right rule --
it is the connected-component partition of "within the tolerance of each
other" (``Cellstine.linked_iff_smallGaps``), the plane numbers grow with height
(``Cellstine.layerIndex_mono``), and moving the origin or reading the structure
from the other end changes no layer, only the numbering, which is exactly
reversed (``Cellstine.linked_add_const``, ``Cellstine.linked_neg``,
``Cellstine.layerIndex_reverse_add``).

That last invariance is the reason the rule is shared.  Comparing each atom
with the *first* member of the growing group, or with its running *mean*, gives
a partition that depends on which end of the slab the sweep starts from: with a
tolerance of ``0.35`` the heights ``0.00, 0.34, 0.50`` are one group read
upwards and two groups read downwards, so the same slab reported a different
layer count, different terminations, and a spurious dipole warning depending on
how it happened to be written.  Cutting at the gaps cannot do that.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from ..io import native as io_mod
from .constants import LAYER_TOLERANCE
from .vacuum import normal_heights

__all__ = [
    "LAYER_TOLERANCE",
    "LayerSelection",
    "LayerShiftRun",
    "identify_top_layer",
    "layer_partition",
    "resolve_shift_vectors",
    "shift_top_layer",
]


def layer_partition(
    heights: Sequence[float] | np.ndarray, tolerance: float
) -> list[tuple[float, list[int]]]:
    """Group atoms into atomic planes by height, bottom plane first.

    Two atoms share a plane when a chain of steps no longer than ``tolerance``
    joins them, which -- the heights being read in order -- is the same as every
    consecutive gap between them being within the tolerance
    (``Cellstine.linked_iff_smallGaps``).  Each plane is returned as its mean
    height together with the indices of its atoms, ordered by height; the
    planes themselves are ordered from the bottom of the structure up
    (``Cellstine.layerIndex_mono``).

    A negative tolerance puts every atom in a plane of its own -- no gap, not
    even a zero one, is within it -- and an empty input has no planes.
    """

    values = np.asarray(heights, dtype=float).reshape(-1)
    if values.size == 0:
        return []
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    cut = np.flatnonzero(np.diff(sorted_values) > float(tolerance)) + 1
    groups = np.split(order, cut)
    return [
        (float(np.mean(values[group])), [int(index) for index in group.tolist()])
        for group in groups
    ]


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

    z_values = normal_heights(structure.lattice, positions)
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
