"""Naming the plane-by-plane stacking sequence of a slab.

A slab is read from the bottom of the cell upwards, one atomic plane at a time,
and every plane is given a letter: planes that sit over one another get the same
letter, planes that do not get a new one, so an fcc (111) slab reads ``ABCABC``
and an hcp (0001) slab reads ``ABAB``.

The letters are decided by *distance*, not by rounding.  Two planes carry the
same letter when their atoms coincide, atom for atom, to within
``POSITION_TOLERANCE`` angstrom, measured along the shortest in-plane periodic
image.  That is the only way to get the answer a person would give: the earlier
implementation of this function snapped fractional coordinates onto a fixed grid
and compared the results exactly, which reported a relaxed ``ABCABC`` slab as
``ABCADC`` because one plane had moved sideways by three thousandths of an
angstrom.

Only positions decide a letter, not species: the sequence describes where the
planes sit, and a rocksalt (111) slab of alternating cation and anion planes
therefore reads by position, as the ``ABC`` convention does.
"""

from __future__ import annotations

import numpy as np

from ...core.geometry import minimum_image_distances
from ...core.vacuum import surface_normal
from .stacking import LAYER_TOLERANCE, POSITION_TOLERANCE, group_layers

__all__ = ["LETTERS", "stacking_sequence", "shortest_repeating_prefix"]

#: Letters handed out to the distinct planes, in order of first appearance.
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _inplane_lattice(lattice: np.ndarray) -> np.ndarray:
    """Return a lattice with the two in-plane vectors and a long normal one.

    Comparing planes is a two-dimensional question, so the third vector is
    replaced by one along the surface normal that is far too long to ever supply
    a shorter image.  The first two rows are untouched, so a fractional
    displacement ``(d1, d2, 0)`` means the same thing as in the original cell.
    """

    array = np.asarray(lattice, dtype=float).reshape(3, 3)
    normal = surface_normal(array)
    height = 4.0 * float(np.max(np.linalg.norm(array[:2], axis=1)))
    return np.array([array[0], array[1], normal * max(height, 1.0)])


def _planes_coincide(
    first: np.ndarray, second: np.ndarray, lattice: np.ndarray, tolerance: float
) -> bool:
    """Return whether two planes hold the same atoms at the same in-plane sites."""

    if len(first) != len(second):
        return False
    remaining = list(range(len(second)))
    for point in first:
        candidates = second[np.asarray(remaining, dtype=int)] - point
        distances = minimum_image_distances(lattice, candidates)
        best = int(np.argmin(distances))
        if float(distances[best]) > float(tolerance):
            return False
        remaining.pop(best)
    return True


def stacking_sequence(
    structure,
    *,
    layer_tolerance: float = LAYER_TOLERANCE,
    tolerance: float = POSITION_TOLERANCE,
) -> tuple[str, tuple[int, ...]]:
    """Return the letter sequence of a slab and the atom count of every plane.

    ``layer_tolerance`` is the half-width in angstrom of an atomic plane along
    the surface normal; ``tolerance`` is how far apart two atoms may be in the
    plane and still count as sitting over one another.
    """

    groups = group_layers(structure, float(layer_tolerance))
    if not groups:
        return "", tuple()
    direct = np.asarray(structure.positions_direct, dtype=float)
    plane_lattice = _inplane_lattice(structure.lattice)

    representatives: list[np.ndarray] = []
    letters: list[str] = []
    for _, indices in groups:
        points = np.mod(direct[np.asarray(indices, dtype=int)], 1.0)
        points[:, 2] = 0.0
        for position, seen in enumerate(representatives):
            if _planes_coincide(seen, points, plane_lattice, tolerance):
                letters.append(LETTERS[position % len(LETTERS)])
                break
        else:
            representatives.append(points)
            letters.append(LETTERS[(len(representatives) - 1) % len(LETTERS)])
    return "".join(letters), tuple(len(indices) for _, indices in groups)


def shortest_repeating_prefix(sequence: str) -> str:
    """Return the shortest block the sequence repeats, e.g. ``ABC`` for ``ABCABCA``."""

    if not sequence:
        return ""
    for size in range(1, len(sequence) + 1):
        prefix = sequence[:size]
        repeats = (prefix * ((len(sequence) // size) + 1))[: len(sequence)]
        if repeats == sequence:
            return prefix
    return sequence
