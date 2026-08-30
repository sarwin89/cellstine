"""Straight-line migration paths between two structures of one cell.

A transition-state calculation -- a vacancy hop, an adatom diffusing across a
surface, a molecule sliding from one adsorption site to the next -- is set up
from a chain of images that interpolates an initial structure into a final one.
Two things have to be right for the chain to mean anything:

* **which atom becomes which.**  The two POSCARs are two lists of atoms, and
  nothing in the file says that atom 7 of the first is atom 7 of the second.
  Pairing them in file order is only correct when the caller happened to write
  them that way; the honest choice is the pairing that makes the path shortest,
  and that is a *linear assignment problem*, solved here exactly and with a
  certificate (:func:`optimal_assignment`).
* **which image of the cell the atom moves to.**  An atom that leaves through
  one face and returns through the opposite one must be interpolated the short
  way round, so every displacement is taken as the exact minimum image
  (:func:`cellstine.core.geometry.minimum_image_fractional`), never as the
  difference of two wrapped fractional coordinates.

With those two settled the chain itself is a straight line in configuration
space: image ``k`` of ``n`` intermediate images sits at ``x0 + k/(n+1) * d``.
Consecutive images are then exactly ``‖d‖ / (n + 1)`` apart, which is the
even spacing a nudged-elastic-band run expects of its starting chain.

The mathematics is proved in ``RequestProject/MigrationPath.lean``:

* ``Cellstine.assignment_cost_le_of_dual_certificate`` -- the pairing this
  module reports really is a minimum-cost one, because the potentials it
  returns are a feasible dual that is tight on the pairing.
* ``Cellstine.sum_min_image_le`` -- taking the shortest image of every atom
  separately minimises the total path length over all choices of images.
* ``Cellstine.linear_path_spacing`` and ``Cellstine.linear_path_length`` --
  the chain is evenly spaced and its length is the distance between the
  endpoints, so no image is a detour.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

from .geometry import (
    minimum_image_distances,
    minimum_image_fractional,
    shortest_interatomic_distance,
)
from .species import expand_species

__all__ = [
    "Assignment",
    "AtomMatching",
    "MigrationPath",
    "build_migration_path",
    "match_atoms",
    "optimal_assignment",
]


#: Two cells count as the same cell when every entry agrees to this many
#: angstrom.  A nudged-elastic-band run needs one cell for the whole chain, so
#: endpoints that disagree by more than this are refused rather than averaged.
CELL_TOLERANCE = 1e-6

#: Fractional coordinates this close to a cell face are written as ``0``.  An
#: endpoint sits at the far end of a displacement, so rounding can leave it at
#: ``1 - 1e-16``; wrapping that with a plain modulo prints ``1.0``, which is the
#: same site under a different name and makes the last image look unlike the
#: structure it was built from.
FACE_TOLERANCE = 1e-9


def _wrap_to_cell(values: np.ndarray) -> np.ndarray:
    """Return ``values`` in ``[0, 1)``, with the cell faces snapped to zero."""

    wrapped = np.mod(np.asarray(values, dtype=float), 1.0)
    wrapped[wrapped > 1.0 - FACE_TOLERANCE] = 0.0
    wrapped[np.abs(wrapped) < FACE_TOLERANCE] = 0.0
    return wrapped


@dataclass(frozen=True)
class Assignment:
    """A minimum-cost pairing of rows to columns, with its dual certificate.

    ``columns[i]`` is the column paired with row ``i``.  ``row_potentials`` and
    ``column_potentials`` are numbers ``u`` and ``v`` with ``u_i + v_j <= c_ij``
    for every pair and ``u_i + v_{columns[i]} = c_{i,columns[i]}`` on the
    pairing itself.  Any pairing ``t`` then costs at least
    ``sum u + sum v = total_cost``, so no other pairing can be cheaper: the
    certificate turns "the solver said so" into a fact the caller can check,
    and :attr:`certificate_error` is how far from exact the check comes out.
    """

    columns: np.ndarray
    total_cost: float
    row_potentials: np.ndarray
    column_potentials: np.ndarray

    @property
    def size(self) -> int:
        return int(len(self.columns))

    def dual_bound(self) -> float:
        """Return the dual objective, which lower-bounds every pairing."""

        return float(np.sum(self.row_potentials) + np.sum(self.column_potentials))

    def certificate_error(self, cost: np.ndarray) -> float:
        """Return how far the certificate is from feasible and tight, in cost units.

        Zero -- up to floating-point noise -- means the pairing is proved
        optimal for ``cost``.
        """

        matrix = np.asarray(cost, dtype=float)
        slack = matrix - self.row_potentials[:, None] - self.column_potentials[None, :]
        infeasibility = float(max(0.0, -slack.min())) if slack.size else 0.0
        rows = np.arange(self.size)
        tightness = float(np.max(np.abs(slack[rows, self.columns]))) if self.size else 0.0
        return max(infeasibility, tightness)


def optimal_assignment(cost: Sequence[Sequence[float]]) -> Assignment:
    """Return a minimum-cost perfect pairing of the rows and columns of ``cost``.

    This is the Jonker--Volgenant shortest-augmenting-path solver: rows are
    added one at a time and each is joined to the column tree by a shortest
    reduced-cost path, with the potentials shifted so that every reduced cost
    stays non-negative.  It runs in ``O(n^3)`` and, unlike a greedy or
    swap-based heuristic, it returns the *exact* optimum together with the
    potentials that prove it.
    """

    matrix = np.asarray(cost, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("an assignment cost must be a square matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("an assignment cost must be finite")
    size = int(matrix.shape[0])
    if size == 0:
        empty = np.zeros(0, dtype=np.int64)
        return Assignment(empty, 0.0, np.zeros(0), np.zeros(0))

    # One virtual row and column, index 0, carry the augmenting path.
    row_potential = np.zeros(size + 1, dtype=float)
    column_potential = np.zeros(size + 1, dtype=float)
    column_row = np.zeros(size + 1, dtype=np.int64)  # column -> row, 0 when free
    parent = np.zeros(size + 1, dtype=np.int64)

    for row in range(1, size + 1):
        column_row[0] = row
        current = 0
        minimum = np.full(size + 1, np.inf, dtype=float)
        used = np.zeros(size + 1, dtype=bool)
        while True:
            used[current] = True
            source = int(column_row[current])
            free = np.flatnonzero(~used[1:]) + 1
            reduced = matrix[source - 1, free - 1] - row_potential[source] - column_potential[free]
            better = reduced < minimum[free]
            improved = free[better]
            minimum[improved] = reduced[better]
            parent[improved] = current
            best = int(free[int(np.argmin(minimum[free]))])
            delta = float(minimum[best])
            touched = np.flatnonzero(used)
            row_potential[column_row[touched]] += delta
            column_potential[touched] -= delta
            untouched = np.flatnonzero(~used)
            minimum[untouched] -= delta
            current = best
            if column_row[current] == 0:
                break
        while current:
            previous = int(parent[current])
            column_row[current] = column_row[previous]
            current = previous

    columns = np.zeros(size, dtype=np.int64)
    for column in range(1, size + 1):
        columns[int(column_row[column]) - 1] = column - 1
    rows = np.arange(size)
    total = float(np.sum(matrix[rows, columns]))
    # The virtual row and column are bookkeeping only; the potentials of the
    # real rows and columns are the certificate.
    return Assignment(
        columns=columns,
        total_cost=total,
        row_potentials=row_potential[1:].copy(),
        column_potentials=column_potential[1:].copy(),
    )


@dataclass(frozen=True)
class AtomMatching:
    """Which atom of the final structure each atom of the initial one becomes.

    ``partners[i]`` is the index in the final structure paired with atom ``i``
    of the initial one, ``displacements`` the Cartesian minimum-image vectors of
    those pairs, and ``cost`` the sum of their squared lengths, which is the
    square of the straight-line distance the whole chain travels.
    """

    partners: np.ndarray
    displacements: np.ndarray
    cost: float
    certificate_error: float
    identity: bool

    @property
    def distances(self) -> np.ndarray:
        return np.sqrt(np.einsum("ij,ij->i", self.displacements, self.displacements))


def _species_by_atom(species: Sequence[str], counts: Sequence[int]) -> List[str]:
    return [str(symbol) for symbol in expand_species(list(species), list(counts))]


def _pair_cost_matrix(
    lattice: np.ndarray, start_direct: np.ndarray, end_direct: np.ndarray
) -> np.ndarray:
    """Return the squared minimum-image distance of every start/end pair."""

    rows = np.asarray(start_direct, dtype=float).reshape(-1, 3)
    columns = np.asarray(end_direct, dtype=float).reshape(-1, 3)
    deltas = (columns[None, :, :] - rows[:, None, :]).reshape(-1, 3)
    distances = minimum_image_distances(lattice, deltas)
    return (distances * distances).reshape(rows.shape[0], columns.shape[0])


def match_atoms(
    lattice: np.ndarray,
    start_direct: np.ndarray,
    end_direct: np.ndarray,
    species_by_atom: Sequence[str],
    end_species_by_atom: Sequence[str] | None = None,
) -> AtomMatching:
    """Pair the atoms of two structures so that the path between them is shortest.

    Only atoms of the same species may be paired, so the assignment is solved
    once per species; the total is minimal because the species blocks are
    independent.  The cost of a pair is its squared minimum-image distance, and
    the sum of those squares is the squared length of the straight line between
    the two structures in configuration space -- so the pairing returned is the
    one that makes the migration path as short as it can be.
    """

    cell = np.asarray(lattice, dtype=float)
    start = np.asarray(start_direct, dtype=float).reshape(-1, 3)
    end = np.asarray(end_direct, dtype=float).reshape(-1, 3)
    labels = [str(symbol) for symbol in species_by_atom]
    end_labels = labels if end_species_by_atom is None else [str(symbol) for symbol in end_species_by_atom]
    if start.shape[0] != end.shape[0]:
        raise ValueError("the two structures must hold the same number of atoms")
    if len(labels) != start.shape[0] or len(end_labels) != end.shape[0]:
        raise ValueError("one species label is needed per atom")
    if sorted(labels) != sorted(end_labels):
        raise ValueError("the two structures must have the same composition")

    partners = np.zeros(start.shape[0], dtype=np.int64)
    displacements = np.zeros((start.shape[0], 3), dtype=float)
    total = 0.0
    error = 0.0
    for symbol in sorted(set(labels)):
        rows = np.flatnonzero(np.array(labels, dtype=object) == symbol)
        columns = np.flatnonzero(np.array(end_labels, dtype=object) == symbol)
        cost = _pair_cost_matrix(cell, start[rows], end[columns])
        assignment = optimal_assignment(cost)
        error = max(error, assignment.certificate_error(cost))
        total += assignment.total_cost
        partners[rows] = columns[assignment.columns]
    fractional = minimum_image_fractional(cell, end[partners] - start)
    displacements = fractional @ cell
    return AtomMatching(
        partners=partners,
        displacements=displacements,
        cost=float(total),
        certificate_error=float(error),
        identity=bool(np.array_equal(partners, np.arange(start.shape[0]))),
    )


@dataclass(frozen=True)
class MigrationPath:
    """An evenly spaced chain of images from one structure to another.

    ``images`` holds the whole chain, endpoints included, as fractional
    coordinates in the atom order of the initial structure: ``images[0]`` is the
    initial structure and ``images[-1]`` the final one, re-ordered by
    ``matching`` so that the atom in row ``i`` is the same atom throughout.
    """

    lattice: np.ndarray
    species: Tuple[str, ...]
    counts: Tuple[int, ...]
    images: Tuple[np.ndarray, ...]
    matching: AtomMatching
    intermediate_count: int

    @property
    def image_count(self) -> int:
        return len(self.images)

    @property
    def path_length(self) -> float:
        """Return the configuration-space distance between the endpoints, in angstrom."""

        return float(np.sqrt(self.matching.cost))

    @property
    def image_spacing(self) -> float:
        """Return the distance between neighbouring images, in angstrom."""

        return self.path_length / float(self.intermediate_count + 1)

    @property
    def maximum_atom_displacement(self) -> float:
        distances = self.matching.distances
        return float(distances.max()) if distances.size else 0.0

    @property
    def moved_atom_count(self) -> int:
        return int(np.count_nonzero(self.matching.distances > 1e-8))

    def spacings(self) -> np.ndarray:
        """Return the measured distance between each pair of neighbouring images."""

        cell = np.asarray(self.lattice, dtype=float)
        gaps = []
        for first, second in zip(self.images, self.images[1:]):
            steps = minimum_image_fractional(cell, np.asarray(second) - np.asarray(first)) @ cell
            gaps.append(float(np.sqrt(np.sum(steps * steps))))
        return np.asarray(gaps, dtype=float)

    def shortest_contacts(self) -> np.ndarray:
        """Return the shortest interatomic distance inside each image, in angstrom."""

        cell = np.asarray(self.lattice, dtype=float)
        return np.asarray(
            [shortest_interatomic_distance(cell, image) for image in self.images], dtype=float
        )


def _check_same_cell(first: np.ndarray, second: np.ndarray, tolerance: float) -> None:
    difference = float(np.max(np.abs(np.asarray(first, dtype=float) - np.asarray(second, dtype=float))))
    if difference > float(tolerance):
        raise ValueError(
            "the two endpoints must share one cell for a migration path; they differ by "
            f"{difference:.6f} A, so relax them in the same cell or rebuild the final "
            "structure in the cell of the initial one"
        )


def build_migration_path(
    lattice: np.ndarray,
    species: Sequence[str],
    counts: Sequence[int],
    start_direct: np.ndarray,
    end_direct: np.ndarray,
    *,
    end_lattice: np.ndarray | None = None,
    end_species: Sequence[str] | None = None,
    end_counts: Sequence[int] | None = None,
    images: int = 3,
    match: bool = True,
    cell_tolerance: float = CELL_TOLERANCE,
) -> MigrationPath:
    """Return the straight, evenly spaced chain of images between two structures.

    ``images`` counts the intermediate images only, so the returned chain holds
    ``images + 2`` structures.  With ``match`` the atoms are paired by the
    shortest-path assignment; without it they are paired in file order, which is
    what a caller wants when the two files already correspond atom for atom.
    """

    count = int(images)
    if count < 1:
        raise ValueError("a migration path needs at least one intermediate image")
    cell = np.asarray(lattice, dtype=float)
    if end_lattice is not None:
        _check_same_cell(cell, end_lattice, cell_tolerance)
    start = np.asarray(start_direct, dtype=float).reshape(-1, 3)
    end = np.asarray(end_direct, dtype=float).reshape(-1, 3)
    labels = _species_by_atom(species, counts)
    end_labels = labels if end_species is None else _species_by_atom(end_species, end_counts or counts)
    if start.shape[0] != end.shape[0]:
        raise ValueError(
            f"the two endpoints hold {start.shape[0]} and {end.shape[0]} atoms; a migration "
            "path needs the same atoms at both ends"
        )
    if sorted(labels) != sorted(end_labels):
        raise ValueError(
            "the two endpoints have different compositions; a migration path needs the same "
            "atoms at both ends"
        )

    if match:
        matching = match_atoms(cell, start, end, labels, end_labels)
    else:
        if labels != end_labels:
            raise ValueError(
                "without atom matching the two endpoints must list their atoms in the same "
                "species order"
            )
        partners = np.arange(start.shape[0], dtype=np.int64)
        fractional = minimum_image_fractional(cell, end - start)
        displacements = fractional @ cell
        matching = AtomMatching(
            partners=partners,
            displacements=displacements,
            cost=float(np.sum(displacements * displacements)),
            certificate_error=0.0,
            identity=True,
        )

    step = minimum_image_fractional(cell, end[matching.partners] - start)
    chain = []
    for index in range(count + 2):
        fraction = float(index) / float(count + 1)
        chain.append(_wrap_to_cell(start + fraction * step))
    return MigrationPath(
        lattice=cell,
        species=tuple(str(symbol) for symbol in species),
        counts=tuple(int(value) for value in counts),
        images=tuple(chain),
        matching=matching,
        intermediate_count=count,
    )
