"""Shared validation helpers.

Besides the two small argument checks the workflows share, this module holds the
sanity check every structure passes on its way to disk.  A generator can be
mathematically correct and still produce a file that no plane-wave code will
accept: a cell whose three vectors are coplanar has no reciprocal lattice, a
species list that disagrees with the number of positions describes a different
structure than the one that was built, and two atoms sitting on the same site --
the classic result of folding a supercell back into the cell it came from -- is
an infinite Coulomb term.  Those are bugs rather than chemistry, so
:func:`validate_structure` raises on them; how *close* two distinct atoms sit is
chemistry, and it is reported by :mod:`cellstine.core.contacts` instead.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from . import geometry

__all__ = [
    "COINCIDENT_TOLERANCE",
    "coincident_site_pairs",
    "ensure_existing_file",
    "ensure_positive",
    "structure_errors",
    "validate_structure",
]

#: Two sites closer than this (in angstrom) are the same site written twice.
COINCIDENT_TOLERANCE = 1e-4


def ensure_existing_file(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(str(resolved))
    return resolved


def ensure_positive(value: float, *, name: str) -> float:
    number = float(value)
    if number <= 0.0:
        raise ValueError(f"{name} must be positive")
    return number


def coincident_site_pairs(
    lattice: np.ndarray,
    positions_direct: np.ndarray,
    *,
    tolerance: float = COINCIDENT_TOLERANCE,
) -> list[tuple[int, int]]:
    """Return the pairs of sites that occupy the same point of the cell.

    The comparison is over the periodic images, so a site at ``x`` and a site at
    ``x + t`` for a lattice translation ``t`` count as the same site: that is
    exactly the duplicate a supercell fold or a wrap at a cell face produces.
    """

    cell = geometry.as_lattice(lattice)
    points = np.asarray(positions_direct, dtype=float).reshape(-1, 3)
    if points.shape[0] < 2:
        return []
    radius = max(float(tolerance), 1e-9)
    first, second = geometry.periodic_neighbour_pairs(cell, points, radius)
    if first.size == 0:
        return []
    distances = geometry.minimum_image_distances(cell, points[first] - points[second])
    keep = distances <= radius
    return [
        (int(a), int(b))
        for a, b in zip(first[keep], second[keep])
    ]


def structure_errors(
    *,
    lattice: Any,
    species: Sequence[str],
    counts: Sequence[int],
    positions_direct: Any,
    tolerance: float = COINCIDENT_TOLERANCE,
) -> list[str]:
    """Return every reason the given structure could not be run as it stands.

    An empty list means the structure is well formed: a non-degenerate cell,
    finite coordinates, one position per atom, and no two atoms on one site.
    """

    problems: list[str] = []

    cell = np.asarray(lattice, dtype=float)
    if cell.shape != (3, 3):
        return [f"the lattice must be a 3x3 matrix of row vectors, got shape {cell.shape}"]
    if not np.all(np.isfinite(cell)):
        return ["the lattice has a non-finite entry"]
    volume = float(np.linalg.det(cell))
    lengths = np.linalg.norm(cell, axis=1)
    scale = float(lengths.prod())
    if scale <= 0.0 or abs(volume) <= 1e-8 * max(scale, 1.0):
        problems.append(
            f"the cell is degenerate: its three vectors span a volume of {volume:.3e} A^3"
        )

    symbols = [str(symbol) for symbol in species]
    numbers = [int(value) for value in counts]
    if len(symbols) != len(numbers):
        problems.append(
            f"the structure lists {len(symbols)} species but {len(numbers)} counts"
        )
    if any(value < 0 for value in numbers):
        problems.append("an atom count is negative")

    points = np.asarray(positions_direct, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        problems.append(
            f"the positions must be an (N, 3) array, got shape {points.shape}"
        )
        return problems
    expected = sum(value for value in numbers if value > 0)
    if points.shape[0] != expected:
        problems.append(
            f"the counts add up to {expected} atoms but {points.shape[0]} positions were given"
        )
    if points.size and not np.all(np.isfinite(points)):
        problems.append("a position has a non-finite coordinate")

    if problems:
        # A duplicate check on a broken cell or a mismatched list would only
        # report the same fault a second time, in less useful words.
        return problems

    duplicates = coincident_site_pairs(cell, points, tolerance=tolerance)
    if duplicates:
        shown = ", ".join(f"{a + 1}/{b + 1}" for a, b in duplicates[:5])
        more = "" if len(duplicates) <= 5 else f" and {len(duplicates) - 5} more"
        problems.append(
            f"{len(duplicates)} pair(s) of atoms sit on the same site (atoms {shown}{more}); "
            f"the structure carries a duplicated atom"
        )
    return problems


def validate_structure(
    *,
    lattice: Any,
    species: Sequence[str],
    counts: Sequence[int],
    positions_direct: Any,
    context: str = "structure",
    tolerance: float = COINCIDENT_TOLERANCE,
) -> None:
    """Raise :class:`ValueError` when the structure could not be run as it stands."""

    problems = structure_errors(
        lattice=lattice,
        species=species,
        counts=counts,
        positions_direct=positions_direct,
        tolerance=tolerance,
    )
    if problems:
        raise ValueError(f"{context}: " + "; ".join(problems))
