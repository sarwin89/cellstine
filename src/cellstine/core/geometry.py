"""Exact periodic geometry: minimum images, neighbour searches and site matching.

Everything here works with *row* lattices -- ``lattice[i]`` is the Cartesian
vector of the ``i``-th basis vector, so a site with fractional coordinates ``x``
(a row) sits at ``x @ lattice`` -- which is the convention of
:class:`cellstine.io.models.StructureRecord` and of the POSCAR format.

Three facts drive the implementations below; all three are proved in
``RequestProject/PeriodicGeometry.lean``.

*Reach bound.*  Write ``b_i`` for the rows of ``inv(lattice).T`` (the reciprocal
basis) and ``d_i = 1 / ‖b_i‖`` for the spacing of the lattice planes normal to
axis ``i``.  A displacement with fractional coordinates ``f`` has
``f_i = b_i · (f @ lattice)``, so Cauchy-Schwarz gives

.. code-block:: text

    |f_i| <= ‖f @ lattice‖ / d_i.

Every enumeration of periodic images in this module takes its range from that
inequality, so it provably misses nothing.

*Minimum image.*  ``f - rint(f)`` -- the textbook "minimum image convention" --
is **not** the shortest image in a skewed cell: in a hexagonal cell it can
overestimate a distance by more than 30 %.  What it does give is an upper bound
``d0`` on the true minimum, and the reach bound then confines the shortest image
to a small box of lattice shifts around it, which this module searches
exhaustively.  The search is done in a Delaunay-reduced basis of the same
lattice, where the box is tiny; the Cartesian answer does not depend on which
basis of the lattice is used.

*Bucket search.*  Points closer than ``r`` differ by at most ``r`` in each
Cartesian coordinate, and by at most ``r / d_i`` in each fractional coordinate,
so a uniform grid of that pitch only ever has to compare neighbouring buckets.
That turns the ``O(n^2)`` scans this library used to run into ``O(n)`` ones.

The reductions the searches rest on -- Niggli, Delaunay, Lagrange--Gauss, and
the integer normal form behind them -- are in :mod:`cellstine.core.reduction`,
and are re-exported from here unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

# Basis validation and reduction live in ``core.reduction``; they are re-exported
# here because every periodic search below starts from a reduced basis, and
# callers have always reached for them through this module.
from .reduction import (
    as_lattice,
    axis_spacings,
    delaunay_reduce,
    integer_lattice_basis,
    niggli_reduce,
    plane_reciprocal_norms,
    plane_reduce,
    rational_lattice_basis,
    reciprocal_norms,
    wrap_fractional,
    wrap_to_cell,
)

__all__ = [
    "as_lattice",
    "wrap_to_cell",
    "wrap_fractional",
    "axis_spacings",
    "reciprocal_norms",
    "niggli_reduce",
    "delaunay_reduce",
    "integer_lattice_basis",
    "rational_lattice_basis",
    "plane_reduce",
    "plane_reciprocal_norms",
    "image_shift_reach",
    "lattice_shifts",
    "plane_shift_reach",
    "plane_shifts",
    "plane_minimum_image",
    "plane_minimum_image_distances",
    "shortest_plane_vector_length",
    "atom_images",
    "minimum_image_displacements",
    "minimum_image_distances",
    "minimum_image_fractional",
    "bounded_minimum_image_squared",
    "periodic_midpoints",
    "pairwise_minimum_image_distances",
    "PeriodicSiteIndex",
    "CartesianGrid",
    "NearestPointQuery",
    "nearest_point_distances",
    "neighbour_images",
    "periodic_neighbour_pairs",
    "shortest_interatomic_distance",
]


# ---------------------------------------------------------------------------
# periodic image enumeration
# ---------------------------------------------------------------------------


def shortest_lattice_vector_length(lattice: np.ndarray) -> float:
    """Return the length of a shortest nonzero translation of ``lattice``.

    This is the distance between a point and its nearest periodic image, so it
    is the separation a defect -- or any other local feature -- has from the
    copies of itself the boundary conditions create.  The search runs over the
    Delaunay-reduced basis: a shortest vector has coefficients in ``{-1, 0, 1}``
    there, and the enumeration below covers ``{-2, ..., 2}``, so the answer is
    exact and not an estimate.
    """

    return _reduced_cell(lattice).shortest


def _shortest_vector_length(reduced: np.ndarray) -> float:
    """Return the shortest nonzero translation of an already reduced basis."""

    span = np.arange(-2, 3, dtype=float)
    grid = np.stack(np.meshgrid(span, span, span, indexing="ij"), axis=-1).reshape(-1, 3)
    nonzero = grid[np.any(grid != 0.0, axis=1)]
    lengths = np.linalg.norm(nonzero @ reduced, axis=1)
    return float(np.min(lengths))


def shortest_plane_vector_length(basis: np.ndarray) -> float:
    """Return the length of a shortest nonzero translation of a plane lattice.

    Lagrange--Gauss reduction puts a shortest vector of the two-dimensional
    lattice first, so no enumeration is needed.
    """

    reduced, _ = plane_reduce(np.asarray(basis, dtype=float).reshape(2, -1))
    return float(np.linalg.norm(reduced[0]))


def image_shift_reach(lattice: np.ndarray, cutoff: float) -> np.ndarray:
    """Return the per-axis range of lattice shifts that can reach ``cutoff``.

    A lattice vector ``n @ lattice`` of length at most ``cutoff`` has
    ``|n_i| <= cutoff / d_i`` with ``d_i`` the spacing of axis ``i``, so shifts
    outside ``[-reach_i, reach_i]`` are provably too long.
    """

    spacings = axis_spacings(lattice)
    reach = np.floor(float(cutoff) / spacings + 1e-12).astype(np.int64)
    return np.maximum(reach, 0)


def lattice_shifts(reach: Sequence[int]) -> np.ndarray:
    """Return every integer shift inside the box ``[-reach_i, reach_i]``."""

    ranges = [np.arange(-int(value), int(value) + 1, dtype=float) for value in reach]
    grid = np.stack(np.meshgrid(*ranges, indexing="ij"), axis=-1)
    return grid.reshape(-1, 3)


def plane_shift_reach(basis: np.ndarray, cutoff: float) -> np.ndarray:
    """Return the per-axis range of in-plane shifts that can reach ``cutoff``.

    The two-dimensional counterpart of :func:`image_shift_reach`: a translation
    ``n @ basis`` no longer than ``cutoff`` has ``|n_i| <= cutoff / d_i`` with
    ``d_i = 1 / |b_i^*|`` the spacing of the rows of lattice points along axis
    ``i``, so shifts outside the box are provably too long
    (``Cellstine.Plane.abs_shift_le_of_euclidNorm_le`` in
    ``RequestProject/PlaneImages.lean``, the pseudo-inverse counterpart of the
    square-basis ``Cellstine.abs_shift_le_of_cartesian_le``).  The rows here
    span a plane in three-dimensional space, so there is no inverse to pair the
    displacement with; ``Cellstine.Plane.exists_rightInverse`` and
    ``Cellstine.Plane.isUnit_gram_of_rightInverse`` show the pseudo-inverse
    ``plane_reciprocal_norms`` uses is exactly the right substitute, available
    whenever the two rows are independent.

    This is what makes a neighbour list on a *given* in-plane cell honest.  The
    fixed ``-1, 0, 1`` box that surface code usually reaches for is complete
    only for a reduced cell; on a sheared one -- a slab handed in as a skew
    supercell of itself, say -- the nearest neighbour of an atom can sit two or
    more cells away, and a fixed box silently reports it as absent.
    """

    spacings = 1.0 / plane_reciprocal_norms(np.asarray(basis, dtype=float).reshape(2, -1))
    reach = np.floor(float(cutoff) / spacings + 1e-12).astype(np.int64)
    return np.maximum(reach, 0)


def plane_shifts(reach: Sequence[int]) -> np.ndarray:
    """Return every integer in-plane shift inside the box ``[-reach_i, reach_i]``."""

    ranges = [np.arange(-int(value), int(value) + 1, dtype=float) for value in reach]
    grid = np.stack(np.meshgrid(*ranges, indexing="ij"), axis=-1)
    return grid.reshape(-1, 2)


def plane_minimum_image(basis: np.ndarray, deltas_uv: np.ndarray) -> np.ndarray:
    """Return the shortest Cartesian representative of each in-plane displacement.

    ``deltas_uv`` holds fractional differences (rows) in the basis ``basis`` of
    two vectors.  Rounding each component -- the usual one-liner -- is *not* the
    shortest image (``Cellstine.rounding_is_not_the_minimum_image``): on a 120
    degree surface cell it can overstate the distance by a tenth of a lattice
    constant, and on a sheared cell by much more.  The search below is exact: it
    rounds in a Lagrange--Gauss reduced basis, which bounds the minimum, and
    then tests the small box of shifts the bound leaves
    (``Cellstine.abs_shift_le_of_le_guess``, in the plane
    ``Cellstine.Plane.abs_shift_le_of_euclidNorm_le``).  The early exit uses
    ``Cellstine.euclidNorm_le_of_two_mul_le_shortest``, which is stated for an
    arbitrary set of lattice vectors and so applies to a plane lattice sitting
    in space unchanged.
    """

    array = np.asarray(basis, dtype=float).reshape(2, -1)
    deltas = np.asarray(deltas_uv, dtype=float).reshape(-1, 2)
    if deltas.shape[0] == 0:
        return np.zeros((0, array.shape[1]), dtype=float)

    reduced, transform = plane_reduce(array)
    fractional = deltas @ np.linalg.inv(transform.astype(float))
    fractional -= np.rint(fractional)
    cartesian = fractional @ reduced
    squared = np.einsum("ij,ij->i", cartesian, cartesian)

    shortest = float(np.linalg.norm(reduced[0]))
    settled = 4.0 * squared <= shortest**2 * (1.0 - 1e-9)
    if bool(settled.all()):
        return cartesian

    bound = float(np.sqrt(squared[~settled]).max())
    reach = np.floor(0.5 + bound * plane_reciprocal_norms(reduced) + 1e-12).astype(np.int64)
    if not np.any(reach):
        return cartesian
    shifts = plane_shifts(reach) @ reduced
    rows = np.flatnonzero(~settled)
    candidates = cartesian[rows][:, None, :] - shifts[None, :, :]
    lengths = np.einsum("ijk,ijk->ij", candidates, candidates)
    best = cartesian.copy()
    best[rows] = candidates[np.arange(rows.shape[0]), np.argmin(lengths, axis=1)]
    return best


def plane_minimum_image_distances(basis: np.ndarray, deltas_uv: np.ndarray) -> np.ndarray:
    """Return the exact minimum-image length of every in-plane displacement."""

    vectors = plane_minimum_image(basis, deltas_uv)
    return np.sqrt(np.einsum("ij,ij->i", vectors, vectors))


def atom_images(lattice: np.ndarray, positions_direct: np.ndarray, cutoff: float) -> np.ndarray:
    """Return the Cartesian coordinates of every image within ``cutoff`` of the cell.

    The shifts run over the box of :func:`image_shift_reach` widened by one, so
    every image of an atom that comes within ``cutoff`` of any point of the base
    cell is included.
    """

    array = as_lattice(lattice)
    reach = image_shift_reach(array, float(cutoff)) + 1
    shifts = lattice_shifts(reach)
    points = np.asarray(positions_direct, dtype=float).reshape(-1, 3)
    return (points[None, :, :] + shifts[:, None, :]).reshape(-1, 3) @ array


# ---------------------------------------------------------------------------
# exact minimum image
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ReducedCell:
    """A Delaunay-reduced basis of a lattice and the data derived from it."""

    lattice: np.ndarray
    to_reduced: np.ndarray
    reciprocal_norms: np.ndarray
    shortest: float


_REDUCED_CACHE: Dict[bytes, _ReducedCell] = {}
_REDUCED_CACHE_LIMIT = 64


def _reduced_cell(lattice: np.ndarray) -> _ReducedCell:
    """Return a cached Delaunay-reduced description of ``lattice``."""

    array = as_lattice(lattice)
    key = array.tobytes()
    cached = _REDUCED_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        reduced, transform = delaunay_reduce(array)
    except (RuntimeError, ValueError):  # pragma: no cover - defensive
        reduced, transform = array, np.eye(3, dtype=np.int64)
    # A row of fractional coordinates f in the input basis is f @ inv(transform)
    # in the reduced basis, because reduced == transform @ lattice.
    entry = _ReducedCell(
        lattice=reduced,
        to_reduced=np.linalg.inv(transform.astype(float)),
        reciprocal_norms=reciprocal_norms(reduced),
        shortest=_shortest_vector_length(reduced),
    )
    if len(_REDUCED_CACHE) >= _REDUCED_CACHE_LIMIT:  # pragma: no cover - cache hygiene
        _REDUCED_CACHE.clear()
    _REDUCED_CACHE[key] = entry
    return entry


def minimum_image_displacements(
    lattice: np.ndarray,
    deltas_direct: np.ndarray,
    *,
    block: int = 65536,
) -> np.ndarray:
    """Return the shortest Cartesian representative of each fractional displacement.

    ``deltas_direct`` holds fractional differences (rows).  For every row the
    returned vector is the shortest ``(delta - n) @ lattice`` over all integer
    shifts ``n``, found exactly: ``rint`` supplies an upper bound on the
    minimum, and the reach bound then leaves only a small box of shifts to test.

    A row whose ``rint`` representative ``c`` is already shorter than half the
    shortest lattice vector ``L`` skips the box search altogether: every other
    image is ``c - s`` with ``s`` a nonzero lattice vector, hence at least
    ``L - |c| > |c|`` long (``Cellstine.euclidNorm_le_of_two_mul_le_shortest`` in
    ``RequestProject/PeriodicGeometry.lean``).  The test is made with a relative
    margin so that the strict inequality survives rounding.
    """

    array = as_lattice(lattice)
    deltas = np.asarray(deltas_direct, dtype=float).reshape(-1, 3)
    if deltas.shape[0] == 0:
        return np.zeros((0, 3), dtype=float)

    cell = _reduced_cell(array)
    fractional = wrap_fractional(deltas @ cell.to_reduced)
    cartesian = fractional @ cell.lattice
    squared = np.einsum("ij,ij->i", cartesian, cartesian)

    # Rows the half-shortest-vector argument settles outright.
    settled = 4.0 * squared <= cell.shortest**2 * (1.0 - 1e-9)
    if settled.all():
        return cartesian
    searched = np.flatnonzero(~settled)

    bound = float(np.sqrt(squared[searched]).max())
    reach = np.floor(0.5 + bound * cell.reciprocal_norms + 1e-12).astype(np.int64)
    if not np.any(reach):
        return cartesian

    shifts = lattice_shifts(reach) @ cell.lattice
    best = cartesian.copy()
    step = max(1, int(block) // max(1, shifts.shape[0]))
    for start in range(0, searched.shape[0], step):
        rows = searched[start : start + step]
        candidates = cartesian[rows][:, None, :] - shifts[None, :, :]
        lengths = np.einsum("ijk,ijk->ij", candidates, candidates)
        best[rows] = candidates[np.arange(rows.shape[0]), np.argmin(lengths, axis=1)]
    return best


def minimum_image_distances(
    lattice: np.ndarray,
    deltas_direct: np.ndarray,
    *,
    block: int = 65536,
) -> np.ndarray:
    """Return the exact minimum-image length of every fractional displacement."""

    vectors = minimum_image_displacements(lattice, deltas_direct, block=block)
    return np.sqrt(np.einsum("ij,ij->i", vectors, vectors))


def minimum_image_fractional(
    lattice: np.ndarray,
    deltas_direct: np.ndarray,
    *,
    block: int = 65536,
) -> np.ndarray:
    """Return the shortest image of each displacement, in fractional coordinates.

    Same result as :func:`minimum_image_displacements` expressed in the basis of
    ``lattice``, which is what a caller needs when it has to go on and build a
    site -- a midpoint, say -- rather than only measure a distance.
    """

    array = as_lattice(lattice)
    vectors = minimum_image_displacements(array, deltas_direct, block=block)
    return vectors @ np.linalg.inv(array)


def bounded_minimum_image_squared(
    lattice: np.ndarray,
    deltas_direct: np.ndarray,
    radius: float,
    *,
    block: int = 65536,
) -> np.ndarray:
    """Return squared minimum-image lengths, exact for everything within ``radius``.

    A site-matching search only ever asks *is this pair closer than the
    tolerance, and if so how close*; it does not care how far apart a pair that
    misses actually is.  That question can be answered without the box search of
    :func:`minimum_image_displacements` for almost every row, as follows.

    Write ``c`` for the ``rint``-wrapped representative of a row in the
    Delaunay-reduced basis and ``L`` for the shortest nonzero lattice vector.
    Any other image is ``c - s`` with ``s`` a nonzero lattice vector, so
    ``|c - s| >= L - |c|``.  Hence when ``2 * radius < L``:

    * ``|c| <= radius``  -- every other image is at least ``L - |c| > radius >=
      |c|`` away, so ``|c|`` *is* the minimum image, exactly;
    * ``radius < |c| < L - radius`` -- every image, ``c`` included, is longer
      than ``radius``, so the row misses whatever ``|c|`` happens to be.

    Both steps are proved in ``RequestProject/PeriodicGeometry.lean``, as
    ``Cellstine.euclidNorm_le_of_le_radius`` and
    ``Cellstine.radius_lt_euclidNorm_sub_of_lt_shortest_sub``.

    Only the rows with ``|c| >= L - radius`` need the exhaustive search, and the
    returned value for them is the true minimum.  Rows that are reported above
    ``radius`` are therefore genuinely above it, and every row reported at or
    below ``radius`` carries its exact squared minimum-image length.  When the
    radius is too large for the argument above, the exact search runs for every
    row.
    """

    array = as_lattice(lattice)
    deltas = np.asarray(deltas_direct, dtype=float).reshape(-1, 3)
    if deltas.shape[0] == 0:
        return np.zeros(0, dtype=float)

    cell = _reduced_cell(array)
    span = float(radius)
    if not 2.0 * span < cell.shortest:
        vectors = minimum_image_displacements(array, deltas, block=block)
        return np.einsum("ij,ij->i", vectors, vectors)

    cartesian = wrap_fractional(deltas @ cell.to_reduced) @ cell.lattice
    squared = np.einsum("ij,ij->i", cartesian, cartesian)
    threshold = (cell.shortest - span) ** 2
    hard = squared >= threshold
    if np.any(hard):
        vectors = minimum_image_displacements(array, deltas[hard], block=block)
        squared[hard] = np.einsum("ij,ij->i", vectors, vectors)
    return squared


def periodic_midpoints(
    lattice: np.ndarray,
    first_direct: np.ndarray,
    second_direct: np.ndarray,
    *,
    block: int = 65536,
) -> np.ndarray:
    """Return the fractional midpoint of each pair, taken along the shortest image.

    The midpoint of two sites is only meaningful once it is said *which* image of
    the second site is meant; the physically relevant one is the closest, and
    that is not in general the one ``rint`` selects.  The result is wrapped into
    ``[0, 1)``.
    """

    array = as_lattice(lattice)
    first = np.asarray(first_direct, dtype=float).reshape(-1, 3)
    second = np.asarray(second_direct, dtype=float).reshape(-1, 3)
    if first.shape != second.shape:
        raise ValueError("midpoints need one second site per first site")
    deltas = minimum_image_fractional(array, second - first, block=block)
    return wrap_to_cell(first + 0.5 * deltas)


def pairwise_minimum_image_distances(
    lattice: np.ndarray,
    positions_direct: np.ndarray,
    *,
    other_direct: np.ndarray | None = None,
    rows: int = 256,
) -> np.ndarray:
    """Return the exact matrix of minimum-image distances between two sets of sites.

    With ``other_direct`` omitted the matrix is the symmetric ``(n, n)`` table of
    the structure's own site separations.
    """

    array = as_lattice(lattice)
    left = np.asarray(positions_direct, dtype=float).reshape(-1, 3)
    right = left if other_direct is None else np.asarray(other_direct, dtype=float).reshape(-1, 3)
    if left.shape[0] == 0 or right.shape[0] == 0:
        return np.zeros((left.shape[0], right.shape[0]), dtype=float)

    result = np.empty((left.shape[0], right.shape[0]), dtype=float)
    step = max(1, int(rows))
    for start in range(0, left.shape[0], step):
        chunk = left[start : start + step]
        deltas = (chunk[:, None, :] - right[None, :, :]).reshape(-1, 3)
        result[start : start + step] = minimum_image_distances(array, deltas).reshape(
            chunk.shape[0], right.shape[0]
        )
    return result


# ---------------------------------------------------------------------------
# bucketed searches
# ---------------------------------------------------------------------------


def _bucket_table(keys: np.ndarray, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(sorted_keys, offsets, ordered_values)`` for a bucket lookup."""

    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    ordered = values[order]
    unique, offsets = np.unique(sorted_keys, return_index=True)
    return unique, np.append(offsets, len(sorted_keys)), ordered


def _bucket_slices(unique: np.ndarray, offsets: np.ndarray, query: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return the start and stop index of the bucket of every query key."""

    if len(unique) == 0:
        zeros = np.zeros(len(query), dtype=np.int64)
        return zeros, zeros
    position = np.clip(np.searchsorted(unique, query), 0, len(unique) - 1)
    hit = unique[position] == query
    start = np.where(hit, offsets[position], 0)
    stop = np.where(hit, offsets[position + 1], 0)
    return start, stop


class PeriodicSiteIndex:
    """Bucketed lookup of fractional sites under periodic boundary conditions.

    ``match`` answers, for a batch of fractional points, which stored site sits
    within ``tolerance`` of it -- the operation a symmetry search performs once
    per candidate operation.  Each site is stored in the bucket it falls in and
    in the neighbouring ones, so one bucket lookup per query point is enough,
    which makes the search ``O(n)`` instead of the ``O(n^2)`` of a full distance
    matrix.  The bucket pitch is at least ``tolerance`` in Cartesian terms, so no
    match can hide in a bucket that is not scanned.
    """

    def __init__(
        self,
        lattice: np.ndarray,
        positions_direct: np.ndarray,
        labels: Sequence[int] | None = None,
        tolerance: float = 1e-5,
    ) -> None:
        self.lattice = as_lattice(lattice)
        self.positions = wrap_to_cell(np.asarray(positions_direct, dtype=float).reshape(-1, 3))
        count = self.positions.shape[0]
        self.labels = (
            np.zeros(count, dtype=np.int64)
            if labels is None
            else np.asarray(labels, dtype=np.int64).reshape(-1)
        )
        if self.labels.shape[0] != count:
            raise ValueError("one label per site is required")
        self.tolerance = float(tolerance)

        spacings = axis_spacings(self.lattice)
        self.bins = np.maximum(np.floor(spacings / max(self.tolerance, 1e-12)).astype(np.int64), 1)
        self.bins = np.minimum(self.bins, 1 << 20)

        if count == 0:
            self._unique = np.zeros(0, dtype=np.int64)
            self._offsets = np.zeros(1, dtype=np.int64)
            self._entries = np.zeros(0, dtype=np.int64)
            return

        base = np.floor(self.positions * self.bins).astype(np.int64) % self.bins
        offsets = np.array(
            [[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)],
            dtype=np.int64,
        )
        spread = (base[:, None, :] + offsets[None, :, :]) % self.bins
        keys = self._encode(spread.reshape(-1, 3))
        entries = np.repeat(np.arange(count, dtype=np.int64), offsets.shape[0])
        # A short axis can wrap several offsets onto the same bucket; keeping one
        # copy of each (site, bucket) pair keeps the buckets tight.
        pairs = np.unique(np.stack([keys, entries], axis=1), axis=0)
        self._unique, self._offsets, self._entries = _bucket_table(pairs[:, 0], pairs[:, 1])

    def _encode(self, cells: np.ndarray) -> np.ndarray:
        strides = np.array([self.bins[1] * self.bins[2], self.bins[2], 1], dtype=np.int64)
        return (np.asarray(cells, dtype=np.int64) * strides).sum(axis=1)

    def match(
        self,
        points_direct: np.ndarray,
        labels: Sequence[int] | None = None,
        *,
        prefer_lowest: bool = False,
    ) -> np.ndarray:
        """Return the index of the site matching each point, or ``-1``.

        A point matches a site when the two are within ``tolerance`` of it in
        Cartesian distance under periodic boundary conditions and, when labels
        are supplied, carries the same label.  The closest such site is
        reported; with ``prefer_lowest`` the lowest-numbered one is, which turns
        the lookup into a canonical representative for collapsing coincident
        sites.
        """

        points = wrap_to_cell(np.asarray(points_direct, dtype=float).reshape(-1, 3))
        queries = points.shape[0]
        wanted = (
            np.zeros(queries, dtype=np.int64)
            if labels is None
            else np.asarray(labels, dtype=np.int64).reshape(-1)
        )
        if wanted.shape[0] != queries:
            raise ValueError("one label per query point is required")
        result = np.full(queries, -1, dtype=np.int64)
        if queries == 0 or self.positions.shape[0] == 0:
            return result

        cells = np.floor(points * self.bins).astype(np.int64) % self.bins
        start, stop = _bucket_slices(self._unique, self._offsets, self._encode(cells))
        width = int((stop - start).max(initial=0))
        if width == 0:
            return result

        columns = np.arange(width, dtype=np.int64)
        indices = start[:, None] + columns[None, :]
        valid = columns[None, :] < (stop - start)[:, None]
        candidates = self._entries[np.where(valid, indices, 0)]

        deltas = (points[:, None, :] - self.positions[candidates]).reshape(-1, 3)
        # Padding entries point at an arbitrary site; zeroing them keeps them out
        # of the exhaustive branch below, and ``valid`` masks them out anyway.
        deltas[~valid.reshape(-1)] = 0.0
        squared = bounded_minimum_image_squared(self.lattice, deltas, self.tolerance).reshape(
            queries, width
        )
        ok = valid & (squared <= self.tolerance**2) & (self.labels[candidates] == wanted[:, None])
        score = np.where(ok, candidates.astype(float), np.inf) if prefer_lowest else np.where(ok, squared, np.inf)
        best = np.argmin(score, axis=1)
        rows = np.arange(queries)
        found = np.isfinite(score[rows, best])
        result[found] = candidates[rows, best][found]
        return result

    def find(self, point: Sequence[float], label: int = 0) -> int:
        """Return the index of the site matching one point, or ``-1``."""

        return int(self.match(np.asarray(point, dtype=float).reshape(1, 3), [int(label)])[0])


class CartesianGrid:
    """Uniform Cartesian bucket grid over a fixed set of points.

    Two points closer than the pitch differ by at most the pitch in every
    Cartesian coordinate, so only the 27 buckets around a query can hold its
    near neighbours -- the standard cell list.  Each stored point is filed in
    those 27 buckets when the grid is built, which leaves one bucket to read per
    query and lets a whole batch of queries be answered by array operations.
    """

    def __init__(self, points: np.ndarray, pitch: float) -> None:
        self.points = np.asarray(points, dtype=float).reshape(-1, 3)
        self.pitch = max(float(pitch), 1e-9)
        if self.points.shape[0] == 0:
            self.origin = np.zeros(3, dtype=float)
            self._unique = np.zeros(0, dtype=np.int64)
            self._offsets = np.zeros(1, dtype=np.int64)
            self._entries = np.zeros(0, dtype=np.int64)
            self._extent = np.ones(3, dtype=np.int64)
            return
        self.origin = self.points.min(axis=0)
        cells = self._cells(self.points)
        self._extent = cells.max(axis=0) + 5
        offsets = np.array(
            [[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)],
            dtype=np.int64,
        )
        spread = (cells[:, None, :] + offsets[None, :, :]).reshape(-1, 3)
        entries = np.repeat(np.arange(self.points.shape[0], dtype=np.int64), offsets.shape[0])
        self._unique, self._offsets, self._entries = _bucket_table(self._encode(spread), entries)

    def _cells(self, points: np.ndarray) -> np.ndarray:
        return np.floor((np.asarray(points, dtype=float) - self.origin) / self.pitch).astype(np.int64)

    def _encode(self, cells: np.ndarray) -> np.ndarray:
        # Cells outside the occupied range are clamped onto the border, which
        # holds no point they could be near, so a distant query reads an empty
        # bucket instead of colliding with an unrelated one.
        clamped = np.clip(np.asarray(cells, dtype=np.int64), -1, self._extent[None, :] - 4) + 1
        strides = np.array([self._extent[1] * self._extent[2], self._extent[2], 1], dtype=np.int64)
        return (clamped * strides).sum(axis=1)

    def candidates(self, queries: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return padded candidate indices and their validity mask.

        Every stored point within ``pitch`` of a query point appears in that
        query's row.
        """

        query_points = np.asarray(queries, dtype=float).reshape(-1, 3)
        count = query_points.shape[0]
        if count == 0 or self.points.shape[0] == 0:
            return np.zeros((count, 0), dtype=np.int64), np.zeros((count, 0), dtype=bool)
        keys = self._encode(self._cells(query_points))
        start, stop = _bucket_slices(self._unique, self._offsets, keys)
        lengths = stop - start
        width = int(lengths.max(initial=0))
        if width == 0:
            return np.zeros((count, 0), dtype=np.int64), np.zeros((count, 0), dtype=bool)
        columns = np.arange(width, dtype=np.int64)[None, :]
        valid = columns < lengths[:, None]
        indices = start[:, None] + np.where(valid, columns, 0)
        return self._entries[indices], valid


def _occupancy_pitch(points: np.ndarray) -> float:
    """Return a bucket pitch that holds about one point per occupied bucket.

    The bounding box of a point set is a poor guide to its density: the periodic
    images of a skewed cell fill a small part of their box, and taking the box
    volume then overestimates the spacing several times over, which leaves the
    occupied buckets crowded and the neighbour scan quadratic in that crowd.
    Starting from the box estimate and dividing by the cube root of the measured
    occupancy fixes that in one or two passes, and costs one sort of the points.
    """

    count = points.shape[0]
    extents = points.max(axis=0) - points.min(axis=0)
    volume = float(np.prod(np.maximum(extents, 1e-9)))
    guess = max((volume / count) ** (1.0 / 3.0), 1e-9)
    origin = points.min(axis=0)
    for _ in range(2):
        cells = np.floor((points - origin) / guess).astype(np.int64)
        extent = cells.max(axis=0) + 1
        strides = np.array([extent[1] * extent[2], extent[2], 1], dtype=np.int64)
        occupied = np.unique((cells * strides).sum(axis=1)).size
        occupancy = count / max(occupied, 1)
        if occupancy <= 1.5:
            break
        guess /= occupancy ** (1.0 / 3.0)
    return guess


class NearestPointQuery:
    """Repeated exact nearest-point queries against one fixed point set.

    A bucket grid of pitch ``p`` finds every stored point within ``p`` of a
    query, so a candidate at distance at most ``p`` is provably the nearest one.
    Queries whose closest candidate is further away -- or which found none at
    all -- are retried on a coarser grid, so the answer is exact while the work
    stays proportional to the number of points rather than to their product.

    The grids are built once and kept.  A branch-and-bound covering-radius sweep
    asks several batches of probes about the *same* atoms, and rebuilding the
    grid for each batch costs more than answering it.

    Each batch is answered in blocks of about ``block`` candidate entries: the
    unblocked candidate table for a hundred thousand probes is hundreds of
    megabytes, while the blocked one stays in cache.
    """

    def __init__(
        self,
        points: np.ndarray,
        *,
        pitch: float | None = None,
        block: int = 1 << 20,
    ) -> None:
        self.points = np.asarray(points, dtype=float).reshape(-1, 3)
        self.block = int(block)
        if pitch is not None:
            self.pitch = max(float(pitch), 1e-6)
        elif self.points.shape[0]:
            self.pitch = max(_occupancy_pitch(self.points), 1e-6)
        else:
            self.pitch = 1.0
        if self.points.shape[0]:
            self.span = float(np.max(self.points.max(axis=0) - self.points.min(axis=0), initial=0.0))
        else:
            self.span = 0.0
        self._grids: List[CartesianGrid] = []

    def _grid(self, level: int) -> CartesianGrid:
        while len(self._grids) <= level:
            self._grids.append(CartesianGrid(self.points, self.pitch * 2.0 ** len(self._grids)))
        return self._grids[level]

    def distances(self, queries: np.ndarray) -> np.ndarray:
        """Return the distance from every query point to the closest stored point."""

        query_points = np.asarray(queries, dtype=float).reshape(-1, 3)
        result = np.full(query_points.shape[0], np.inf, dtype=float)
        if query_points.shape[0] == 0 or self.points.shape[0] == 0:
            return result
        diagonal = self.span + float(
            np.linalg.norm(query_points.max(axis=0) - query_points.min(axis=0))
        )
        pending = np.arange(query_points.shape[0])
        level = 0
        while pending.size:
            grid = self._grid(level)
            current = grid.pitch
            indices, valid = grid.candidates(query_points[pending])
            width = indices.shape[1]
            if width:
                step = max(1, self.block // width)
                for start in range(0, pending.shape[0], step):
                    rows = slice(start, start + step)
                    chunk = indices[rows]
                    deltas = self.points[chunk] - query_points[pending[rows]][:, None, :]
                    squared = np.einsum("ijk,ijk->ij", deltas, deltas)
                    closest = np.min(squared, axis=1, where=valid[rows], initial=np.inf)
                    result[pending[rows]] = np.sqrt(closest)
            pending = pending[result[pending] > current]
            if current > diagonal + 2.0 * self.pitch:  # pragma: no cover - defensive
                break
            level += 1
        if pending.size:  # pragma: no cover - only for degenerate inputs
            deltas = query_points[pending][:, None, :] - self.points[None, :, :]
            result[pending] = np.sqrt(np.einsum("ijk,ijk->ij", deltas, deltas)).min(axis=1)
        return result


def nearest_point_distances(
    queries: np.ndarray,
    points: np.ndarray,
    *,
    pitch: float | None = None,
    block: int = 1 << 20,
) -> np.ndarray:
    """Return the distance from every query point to the closest of ``points``.

    A single-shot :class:`NearestPointQuery`; use the class directly when the
    same point set is asked about more than once.
    """

    return NearestPointQuery(points, pitch=pitch, block=block).distances(queries)


def neighbour_images(
    lattice: np.ndarray,
    positions_direct: np.ndarray,
    cutoff: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return, for every atom of the cell, its neighbouring atom images.

    The result is ``(images, indices, valid)``: ``images`` holds the Cartesian
    coordinates of every periodic image within ``cutoff`` of the cell, and row
    ``i`` of ``indices`` lists -- where ``valid`` is true -- the images that lie
    within ``cutoff`` of atom ``i``, the atom itself included.
    """

    array = as_lattice(lattice)
    points = np.asarray(positions_direct, dtype=float).reshape(-1, 3)
    images = atom_images(array, points, float(cutoff))
    base = points @ array
    if base.shape[0] == 0:
        return images, np.zeros((0, 0), dtype=np.int64), np.zeros((0, 0), dtype=bool)

    grid = CartesianGrid(images, float(cutoff))
    indices, valid = grid.candidates(base)
    if indices.shape[1] == 0:
        return images, indices, valid
    deltas = images[indices] - base[:, None, :]
    squared = np.einsum("ijk,ijk->ij", deltas, deltas)
    valid &= squared <= float(cutoff) ** 2
    return images, indices, valid


def periodic_neighbour_pairs(
    lattice: np.ndarray,
    positions_direct: np.ndarray,
    cutoff: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return every pair of distinct sites whose shortest image is within ``cutoff``.

    The result is a pair of index arrays ``(first, second)`` with
    ``first < second``, each pair listed once.  A cell list over the periodic
    images does the work, so the cost follows the number of pairs that are
    actually close rather than the square of the number of sites; a caller that
    only needs the close pairs -- merging duplicate sites, say -- should use
    this instead of a full distance matrix.
    """

    array = as_lattice(lattice)
    points = np.mod(np.asarray(positions_direct, dtype=float).reshape(-1, 3), 1.0)
    count = points.shape[0]
    empty = (np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64))
    if count < 2 or float(cutoff) <= 0.0:
        return empty

    images, indices, valid = neighbour_images(array, points, float(cutoff))
    if indices.shape[1] == 0:
        return empty
    # ``atom_images`` lays the images out shift by shift, each shift holding the
    # sites in order, so an image belongs to the site at its index modulo the
    # number of sites.
    owners = np.arange(images.shape[0], dtype=np.int64) % count
    partners = owners[indices]
    rows = np.broadcast_to(
        np.arange(count, dtype=np.int64)[:, None], partners.shape
    )
    keep = valid & (partners > rows)
    if not np.any(keep):
        return empty
    first = rows[keep]
    second = partners[keep]
    # The same pair can be reached through more than one image.
    codes = np.unique(first * count + second)
    return codes // count, codes % count


def shortest_interatomic_distance(
    lattice: np.ndarray,
    positions_direct: np.ndarray,
) -> float:
    """Return the shortest distance between two distinct sites of a periodic cell.

    A search radius that finds at least one pair also contains the closest pair,
    so the radius starts at the mean site separation and doubles until a pair
    turns up; the answer is then exact.  Sites that coincide are ignored, which
    keeps a duplicated entry from reporting a zero separation.
    """

    array = as_lattice(lattice)
    points = np.mod(np.asarray(positions_direct, dtype=float).reshape(-1, 3), 1.0)
    count = points.shape[0]
    lattice_shortest = shortest_lattice_vector_length(array)
    if count < 2:
        return float(lattice_shortest)

    volume = abs(float(np.linalg.det(array)))
    radius = max((volume / count) ** (1.0 / 3.0), 1e-6)
    ceiling = float(lattice_shortest)
    while True:
        first, second = periodic_neighbour_pairs(array, points, radius)
        if first.size:
            distances = minimum_image_distances(array, points[first] - points[second])
            positive = distances[distances > 1e-9]
            if positive.size:
                return float(min(positive.min(), lattice_shortest))
        if radius >= ceiling:
            return float(lattice_shortest)
        radius = min(2.0 * radius, ceiling)
