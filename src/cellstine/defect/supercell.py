"""Choosing the supercell a point defect should be built in.

A defect calculation models an *isolated* defect, but a periodic cell puts a
copy of the defect at every lattice point.  What decides whether the model is
any good is therefore not the number of atoms but the distance from the defect
to its nearest periodic image, and that distance is the length of a shortest
nonzero translation of the supercell lattice -- see
:mod:`cellstine.defect.dilution`, which measures it for a cell that already
exists.

This module chooses the cell.  For a given number of host cells the supercell
lattice is a sublattice of the host lattice of that index, and every sublattice
of index ``n`` is the row span of exactly one integer matrix in Hermite normal
form: upper triangular, positive diagonal with product ``n``, and every
off-diagonal entry reduced modulo the diagonal entry of its column.  Enumerating
those matrices therefore enumerates the possible supercells once each, with no
sublattice missed and none repeated, and the best of them is the roundest cell
of that size.  The result is usually *not* a diagonal repeat: an fcc primitive
cell repeated ``2x2x2`` separates the images by 2 a, while the eight-cell
sublattice that gives the conventional cubic cell doubled separates them by
``2 a / sqrt 2 * sqrt 2`` -- the search finds whichever is larger without being
told about the crystal system.

Two things keep the answer honest rather than merely arithmetic:

* the shortest translation is computed exactly (Delaunay reduction in three
  dimensions, Lagrange--Gauss in the plane), not estimated from the cell
  lengths, so a long, thin cell is never mistaken for a roomy one;
* Minkowski's convex body theorem bounds what any cell of a given size can
  achieve -- ``(6 n V / pi)^(1/3)`` in three dimensions and
  ``2 sqrt(n A / pi)`` in the plane -- so a request for a separation no cell of
  that size can deliver is refused immediately instead of being searched for,
  and the report can say how close to the best possible the chosen cell is.

The formal statements behind all of this are in
``RequestProject/DefectImageSeparation.lean`` (what the separation is, that it
depends on the lattice rather than on the basis, that a supercell never brings
the images closer, and the two Minkowski bounds) and
``RequestProject/HermiteNormalForm.lean`` (that the Hermite enumeration hits
every sublattice of the given index exactly once).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterator, Sequence

import numpy as np

from ..core.geometry import (
    as_lattice,
    delaunay_reduce,
    plane_reduce,
    shortest_lattice_vector_length,
    shortest_plane_vector_length,
)
from .dilution import SLAB_KINDS

#: Largest number of host cells the search will consider unless asked for more.
from ..core.constants import DEFAULT_CELL_LIMIT  # re-exported for callers

#: How many equally good sublattices are reduced when picking the roundest cell.
_ROUNDNESS_SHORTLIST = 32


def is_slab_kind(structure_kind: str) -> bool:
    """Whether ``structure_kind`` is a cell whose third axis is vacuum."""

    return str(structure_kind).lower() in SLAB_KINDS


# ---------------------------------------------------------------------------
# Hermite normal forms
# ---------------------------------------------------------------------------


def hermite_normal_forms_3d(cells: int) -> Iterator[np.ndarray]:
    """Yield every ``3x3`` Hermite normal form of determinant ``cells``.

    Each matrix is upper triangular with positive diagonal ``a e i`` of product
    ``cells`` and off-diagonal entries reduced modulo the diagonal entry of
    their own column.  Every sublattice of index ``cells`` is the row span of
    exactly one of them, so this enumeration is complete and repeats nothing.
    """

    count = int(cells)
    if count < 1:
        raise ValueError("a supercell must hold at least one host cell")
    for first in _divisors(count):
        rest = count // first
        for second in _divisors(rest):
            third = rest // second
            for upper_ab in range(second):
                for upper_ac in range(third):
                    for upper_bc in range(third):
                        yield np.array(
                            [
                                [first, upper_ab, upper_ac],
                                [0, second, upper_bc],
                                [0, 0, third],
                            ],
                            dtype=np.int64,
                        )


def hermite_normal_forms_2d(cells: int) -> Iterator[np.ndarray]:
    """Yield every ``2x2`` Hermite normal form of determinant ``cells``."""

    count = int(cells)
    if count < 1:
        raise ValueError("a supercell must hold at least one host cell")
    for first in _divisors(count):
        second = count // first
        for upper in range(second):
            yield np.array([[first, upper], [0, second]], dtype=np.int64)


def hermite_normal_form_count(cells: int, *, plane: bool = False) -> int:
    """How many Hermite normal forms -- that is sublattices -- of index ``cells``."""

    count = int(cells)
    if count < 1:
        raise ValueError("a supercell must hold at least one host cell")
    if plane:
        return sum(count // first for first in _divisors(count))
    total = 0
    for first in _divisors(count):
        rest = count // first
        for second in _divisors(rest):
            third = rest // second
            total += second * third * third
    return total


def _divisors(value: int) -> list[int]:
    number = int(value)
    found: list[int] = []
    step = 1
    while step * step <= number:
        if number % step == 0:
            found.append(step)
            if step != number // step:
                found.append(number // step)
        step += 1
    return sorted(found)


def embed_plane_matrix(matrix: Sequence[Sequence[int]]) -> np.ndarray:
    """Return an in-plane ``2x2`` matrix as a ``3x3`` matrix that keeps ``c``."""

    plane = np.asarray(matrix, dtype=np.int64).reshape(2, 2)
    embedded = np.eye(3, dtype=np.int64)
    embedded[:2, :2] = plane
    return embedded


# ---------------------------------------------------------------------------
# what a cell of a given size can possibly achieve
# ---------------------------------------------------------------------------


def cell_volume(lattice: np.ndarray) -> float:
    """The volume of the host cell in cubic angstrom."""

    return abs(float(np.linalg.det(as_lattice(lattice))))


def cell_area(lattice: np.ndarray) -> float:
    """The in-plane area of the host cell in square angstrom."""

    array = as_lattice(lattice)
    return float(np.linalg.norm(np.cross(array[0], array[1])))


def minkowski_bound(measure: float, *, plane: bool) -> float:
    """The largest image separation a lattice of this covolume can have.

    Minkowski's convex body theorem applied to the ball of radius ``r``: a
    symmetric convex body of volume greater than ``2^d`` times the covolume
    contains a nonzero lattice point, so a lattice of covolume ``V`` in three
    dimensions has a nonzero vector shorter than ``r`` whenever
    ``(4 pi / 3) r^3 > 8 V``, that is a shortest vector of length at most
    ``(6 V / pi)^(1/3)``; in the plane the same argument with ``pi r^2 > 4 A``
    gives ``2 sqrt(A / pi)``.
    """

    value = float(measure)
    if value <= 0.0:
        raise ValueError("the covolume of a lattice is positive")
    if plane:
        return 2.0 * math.sqrt(value / math.pi)
    return (6.0 * value / math.pi) ** (1.0 / 3.0)


def cells_needed_lower_bound(
    lattice: np.ndarray, distance: float, *, plane: bool
) -> int:
    """The fewest host cells that could possibly reach ``distance``.

    Inverting :func:`minkowski_bound`: no supercell of fewer cells than this
    can separate the images by ``distance``, whatever its shape, so the search
    starts here instead of at one.
    """

    target = float(distance)
    if target <= 0.0:
        return 1
    if plane:
        measure = cell_area(lattice)
        needed = math.pi * target * target / (4.0 * measure)
    else:
        measure = cell_volume(lattice)
        needed = math.pi * target ** 3 / (6.0 * measure)
    return max(1, int(math.ceil(needed - 1e-9)))


# ---------------------------------------------------------------------------
# the search
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SupercellChoice:
    """A supercell, how big it is, and how well it isolates the defect."""

    matrix: list[list[int]]
    cells: int
    image_distance: float
    periodicity: str
    upper_bound: float
    diagonal_matrix: list[list[int]] | None
    diagonal_distance: float | None

    @property
    def is_diagonal(self) -> bool:
        array = np.asarray(self.matrix, dtype=np.int64)
        return bool(np.array_equal(array, np.diag(np.diag(array))))

    def as_dict(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "matrix": [[int(value) for value in row] for row in self.matrix],
            "cells": int(self.cells),
            "image_distance": float(self.image_distance),
            "image_periodicity": self.periodicity,
            "best_possible_distance": float(self.upper_bound),
            "diagonal": bool(self.is_diagonal),
        }
        if self.diagonal_matrix is not None and self.diagonal_distance is not None:
            record["best_diagonal_matrix"] = [
                [int(value) for value in row] for row in self.diagonal_matrix
            ]
            record["best_diagonal_distance"] = float(self.diagonal_distance)
        return record

    def summary(self) -> str:
        rows = ", ".join(
            "[" + " ".join(f"{int(value):d}" for value in row) + "]" for row in self.matrix
        )
        text = (
            f"{self.cells} host cell(s), supercell matrix {rows}, "
            f"defect-image separation {self.image_distance:.2f} A "
            f"({self.periodicity}; no cell of this size beats "
            f"{self.upper_bound:.2f} A)"
        )
        if (
            self.diagonal_distance is not None
            and self.diagonal_distance + 1e-9 < self.image_distance
        ):
            text += f", against {self.diagonal_distance:.2f} A for the best diagonal repeat"
        return text


def image_distance_of(lattice: np.ndarray, matrix: Sequence[Sequence[int]], *, plane: bool) -> float:
    """The defect-image separation of the supercell ``matrix @ lattice``."""

    array = as_lattice(lattice)
    integers = np.asarray(matrix, dtype=float).reshape(3, 3)
    supercell = integers @ array
    if plane:
        return shortest_plane_vector_length(supercell[:2])
    return shortest_lattice_vector_length(supercell)


def _hermite_parameters_3d(cells: int) -> np.ndarray:
    """The Hermite normal forms of determinant ``cells`` as ``(a, b, c, e, f, i)``."""

    blocks: list[np.ndarray] = []
    count = int(cells)
    for first in _divisors(count):
        rest = count // first
        for second in _divisors(rest):
            third = rest // second
            grid = np.stack(
                np.meshgrid(
                    np.arange(second, dtype=np.int64),
                    np.arange(third, dtype=np.int64),
                    np.arange(third, dtype=np.int64),
                    indexing="ij",
                ),
                axis=-1,
            ).reshape(-1, 3)
            block = np.empty((grid.shape[0], 6), dtype=np.int64)
            block[:, 0] = first
            block[:, 1] = grid[:, 0]
            block[:, 2] = grid[:, 1]
            block[:, 3] = second
            block[:, 4] = grid[:, 2]
            block[:, 5] = third
            blocks.append(block)
    return np.concatenate(blocks, axis=0)


def _hermite_parameters_2d(cells: int) -> np.ndarray:
    """The plane Hermite normal forms of determinant ``cells`` as ``(a, b, d)``."""

    blocks: list[np.ndarray] = []
    count = int(cells)
    for first in _divisors(count):
        second = count // first
        block = np.empty((second, 3), dtype=np.int64)
        block[:, 0] = first
        block[:, 1] = np.arange(second, dtype=np.int64)
        block[:, 2] = second
        blocks.append(block)
    return np.concatenate(blocks, axis=0)


def host_lattice_vectors(
    lattice: np.ndarray, radius: float, *, plane: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Every nonzero host lattice vector no longer than ``radius``, shortest first.

    The integer coefficients of such a vector are bounded by ``radius`` divided
    by the spacing of the corresponding family of lattice planes, so the box
    searched here provably contains them all.
    """

    array = as_lattice(lattice)
    rows = array[:2] if plane else array
    if plane:
        spacings = 1.0 / np.linalg.norm(np.linalg.pinv(rows), axis=0)
    else:
        spacings = 1.0 / np.linalg.norm(np.linalg.inv(rows), axis=0)
    reach = np.floor(float(radius) / spacings + 1e-9).astype(np.int64)
    ranges = [np.arange(-int(value), int(value) + 1, dtype=np.int64) for value in reach]
    grid = np.stack(np.meshgrid(*ranges, indexing="ij"), axis=-1).reshape(-1, len(ranges))
    grid = grid[np.any(grid != 0, axis=1)]
    norms = np.linalg.norm(grid.astype(float) @ rows, axis=1)
    keep = norms <= float(radius) + 1e-9
    grid, norms = grid[keep], norms[keep]
    order = np.argsort(norms)
    return grid[order], norms[order]


def _contains_vector_3d(parameters: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Which Hermite forms have the host vector ``vector`` in their lattice.

    A row ``k`` of integers maps to ``(k1 a, k1 b + k2 e, k1 c + k2 f + k3 i)``,
    so the vector belongs to the sublattice exactly when the three divisions
    below come out exact -- which is an integer test, with no tolerance in it.
    """

    first, upper_ab, upper_ac, second, upper_bc, third = (parameters[:, index] for index in range(6))
    inside = vector[0] % first == 0
    coefficient_one = np.where(inside, vector[0] // first, 0)
    remainder = vector[1] - coefficient_one * upper_ab
    inside &= remainder % second == 0
    coefficient_two = np.where(inside, remainder // second, 0)
    last = vector[2] - coefficient_one * upper_ac - coefficient_two * upper_bc
    inside &= last % third == 0
    return inside


def _contains_vector_2d(parameters: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Which plane Hermite forms have the host vector ``vector`` in their lattice."""

    first, upper, second = (parameters[:, index] for index in range(3))
    inside = vector[0] % first == 0
    coefficient_one = np.where(inside, vector[0] // first, 0)
    remainder = vector[1] - coefficient_one * upper
    inside &= remainder % second == 0
    return inside


def _matrix_from_parameters(parameters: np.ndarray, *, plane: bool) -> np.ndarray:
    if plane:
        first, upper, second = (int(value) for value in parameters)
        return embed_plane_matrix([[first, upper], [0, second]])
    first, upper_ab, upper_ac, second, upper_bc, third = (int(value) for value in parameters)
    return np.array(
        [[first, upper_ab, upper_ac], [0, second, upper_bc], [0, 0, third]], dtype=np.int64
    )


def best_supercell_of_size(
    lattice: np.ndarray, cells: int, *, plane: bool
) -> tuple[np.ndarray, float]:
    """The sublattice of index ``cells`` with the largest image separation.

    Every sublattice of that index is enumerated once, in Hermite normal form.
    Rather than reducing each of them, the host lattice vectors are walked in
    order of increasing length and each candidate is struck out by the first
    one it contains -- which is exactly that candidate's shortest translation.
    The walk is complete because Minkowski's theorem bounds the shortest
    translation of *any* sublattice of this index, so every candidate is struck
    out before the walk ends, and the winner is the last one standing.
    """

    array = as_lattice(lattice)
    count = int(cells)
    if count < 1:
        raise ValueError("a supercell must hold at least one host cell")
    measure = cell_area(array) if plane else cell_volume(array)
    radius = minkowski_bound(measure * count, plane=plane)
    parameters = _hermite_parameters_2d(count) if plane else _hermite_parameters_3d(count)
    vectors, norms = host_lattice_vectors(array, radius, plane=plane)
    contains = _contains_vector_2d if plane else _contains_vector_3d
    alive = parameters
    tied: list[np.ndarray] = [parameters[0]]
    best_distance = 0.0
    for position in range(int(vectors.shape[0])):
        struck = contains(alive, vectors[position])
        if not bool(np.any(struck)):
            continue
        distance = float(norms[position])
        if distance > best_distance + 1e-9:
            tied = []
            best_distance = distance
        tied.extend(alive[struck])
        alive = alive[~struck]
        if int(alive.shape[0]) == 0:
            break
    if int(alive.shape[0]) != 0:  # pragma: no cover - defensive
        raise ArithmeticError(
            "the search radius did not reach every sublattice of this index"
        )
    matrices = [_matrix_from_parameters(row, plane=plane) for row in tied]
    winner = _roundest(array, matrices, plane=plane)
    return reduced_supercell_matrix(array, winner, plane=plane), best_distance


def reduced_supercell_matrix(
    lattice: np.ndarray, matrix: Sequence[Sequence[int]], *, plane: bool
) -> np.ndarray:
    """Rewrite a supercell matrix on a reduced basis of the same supercell.

    The Hermite normal form names a sublattice unambiguously but describes it
    with a badly skewed basis, and a skewed cell is awkward to compute in and
    unreadable on paper.  Reducing the basis -- Delaunay in three dimensions,
    Lagrange--Gauss in the plane -- keeps *exactly* the same lattice, and hence
    the same structure and the same image separation, while giving it the
    shortest, most orthogonal description available.  The rows are ordered so
    that the cell stays right-handed.
    """

    array = as_lattice(lattice)
    supercell = np.asarray(matrix, dtype=float).reshape(3, 3) @ array
    if plane:
        reduced_plane, _ = plane_reduce(supercell[:2])
        rows = np.vstack([reduced_plane, supercell[2]])
    else:
        rows, _ = delaunay_reduce(supercell)
    integers = np.rint(rows @ np.linalg.inv(array)).astype(np.int64)
    if not np.allclose(integers.astype(float) @ array, rows, atol=1e-8):
        raise ArithmeticError(  # pragma: no cover - defensive
            "the reduced basis is not an integer combination of the host lattice"
        )
    if float(np.linalg.det(integers.astype(float))) < 0.0:
        integers[[0, 1]] = integers[[1, 0]]
    return integers


def _roundest(
    lattice: np.ndarray, matrices: Sequence[np.ndarray], *, plane: bool
) -> np.ndarray:
    """Pick the roundest cell among sublattices with the same shortest vector.

    Several sublattices of an index often share the same image separation --
    for face-centred cubic aluminium four primitive cells reach ``a`` in more
    than one way -- and they are not equally good cells to compute in.  The one
    kept is the most orthogonal: since they all enclose the same volume, that is
    the one whose reduced basis vectors have the smallest product, by Hadamard's
    inequality, and then the shortest longest vector.  Ties left after that are
    broken by the entries of the matrix, so the answer never depends on the
    order the candidates were enumerated in.
    """

    array = as_lattice(lattice)
    if len(matrices) > _ROUNDNESS_SHORTLIST:
        rows = np.asarray(matrices, dtype=float) @ array
        if plane:
            rows = rows[:, :2, :]
        cheap = np.prod(np.linalg.norm(rows, axis=2), axis=1)
        order = np.argsort(cheap)[:_ROUNDNESS_SHORTLIST]
        matrices = [matrices[int(index)] for index in order]

    def key(matrix: np.ndarray) -> tuple:
        supercell = np.asarray(matrix, dtype=float) @ array
        if plane:
            reduced, _ = plane_reduce(supercell[:2])
        else:
            reduced, _ = delaunay_reduce(supercell)
        norms = sorted(float(value) for value in np.linalg.norm(reduced, axis=1))
        product = float(np.prod(norms))
        entries = tuple(abs(int(value)) for value in np.asarray(matrix).reshape(-1))
        return (round(product, 9), round(norms[-1], 9), entries)

    return min(matrices, key=key)


def best_diagonal_of_size(
    lattice: np.ndarray, cells: int, *, plane: bool
) -> tuple[np.ndarray | None, float]:
    """The best plain repeat with at most ``cells`` host cells, for comparison."""

    array = as_lattice(lattice)
    best_matrix: np.ndarray | None = None
    best_distance = 0.0
    limit = int(cells)
    for first in range(1, limit + 1):
        for second in range(1, limit // first + 1):
            third_range = [1] if plane else range(1, limit // (first * second) + 1)
            for third in third_range:
                matrix = np.diag([first, second, third]).astype(np.int64)
                distance = image_distance_of(array, matrix, plane=plane)
                if distance > best_distance + 1e-12:
                    best_distance = distance
                    best_matrix = matrix
    return best_matrix, float(best_distance)


def choose_supercell(
    lattice: np.ndarray,
    *,
    structure_kind: str = "bulk",
    min_image_distance: float | None = None,
    max_cells: int | None = None,
    cell_limit: int = DEFAULT_CELL_LIMIT,
    compare_diagonal: bool = True,
) -> SupercellChoice:
    """Choose the supercell a defect should be built in.

    With ``min_image_distance`` the smallest cell that reaches the requested
    separation is returned; with ``max_cells`` the best cell of at most that
    many host cells is returned instead.  Given both, the smallest cell that
    reaches the separation is returned provided it fits inside ``max_cells``.
    """

    array = as_lattice(lattice)
    plane = is_slab_kind(structure_kind)
    periodicity = "in-plane" if plane else "three-dimensional"
    measure = cell_area(array) if plane else cell_volume(array)
    if min_image_distance is None and max_cells is None:
        raise ValueError("give either a minimum image distance or a maximum cell count")
    ceiling = int(max_cells) if max_cells is not None else int(cell_limit)
    if ceiling < 1:
        raise ValueError("a supercell must hold at least one host cell")

    if min_image_distance is not None:
        target = float(min_image_distance)
        start = cells_needed_lower_bound(array, target, plane=plane)
        if start > ceiling:
            raise ValueError(
                f"no supercell of at most {ceiling} host cell(s) can separate the images by "
                f"{target:.2f} A: even the best possible cell of that size reaches only "
                f"{minkowski_bound(measure * ceiling, plane=plane):.2f} A"
            )
        for cells in range(start, ceiling + 1):
            if minkowski_bound(measure * cells, plane=plane) < target:
                continue
            matrix, distance = best_supercell_of_size(array, cells, plane=plane)
            if distance + 1e-9 >= target:
                return _finalise_choice(
                    array,
                    matrix,
                    cells,
                    distance,
                    periodicity,
                    measure,
                    plane=plane,
                    compare_diagonal=compare_diagonal,
                )
        raise ValueError(
            f"no supercell of at most {ceiling} host cell(s) separates the images by "
            f"{target:.2f} A; ask for more cells or a smaller separation"
        )

    best_cells = 1
    best_matrix = np.eye(3, dtype=np.int64)
    best_distance = image_distance_of(array, best_matrix, plane=plane)
    for cells in range(2, ceiling + 1):
        matrix, distance = best_supercell_of_size(array, cells, plane=plane)
        if distance > best_distance + 1e-12:
            best_cells, best_matrix, best_distance = cells, matrix, distance
    return _finalise_choice(
        array,
        best_matrix,
        best_cells,
        best_distance,
        periodicity,
        measure,
        plane=plane,
        compare_diagonal=compare_diagonal,
    )


def _finalise_choice(
    lattice: np.ndarray,
    matrix: np.ndarray,
    cells: int,
    distance: float,
    periodicity: str,
    measure: float,
    *,
    plane: bool,
    compare_diagonal: bool,
) -> SupercellChoice:
    diagonal_matrix: list[list[int]] | None = None
    diagonal_distance: float | None = None
    if compare_diagonal:
        found, value = best_diagonal_of_size(lattice, int(cells), plane=plane)
        if found is not None:
            diagonal_matrix = [[int(entry) for entry in row] for row in found]
            diagonal_distance = float(value)
    return SupercellChoice(
        matrix=[[int(entry) for entry in row] for row in np.asarray(matrix, dtype=np.int64)],
        cells=int(cells),
        image_distance=float(distance),
        periodicity=periodicity,
        upper_bound=float(minkowski_bound(measure * int(cells), plane=plane)),
        diagonal_matrix=diagonal_matrix,
        diagonal_distance=diagonal_distance,
    )


def supercell_table(
    lattice: np.ndarray,
    *,
    structure_kind: str = "bulk",
    max_cells: int = 16,
) -> list[dict[str, Any]]:
    """One row per cell count: the best supercell of that size and its reach.

    Rows that cannot beat a smaller cell are still listed, because a defect
    study often has to pick a size first and then ask what it buys; the
    ``improves`` flag marks the sizes that are actually worth using.
    """

    array = as_lattice(lattice)
    plane = is_slab_kind(structure_kind)
    measure = cell_area(array) if plane else cell_volume(array)
    rows: list[dict[str, Any]] = []
    best_so_far = 0.0
    for cells in range(1, int(max_cells) + 1):
        matrix, distance = best_supercell_of_size(array, cells, plane=plane)
        improves = distance > best_so_far + 1e-9
        best_so_far = max(best_so_far, distance)
        rows.append(
            {
                "cells": int(cells),
                "matrix": [[int(value) for value in row] for row in matrix],
                "image_distance": float(distance),
                "best_possible_distance": float(minkowski_bound(measure * cells, plane=plane)),
                "improves": bool(improves),
            }
        )
    return rows
