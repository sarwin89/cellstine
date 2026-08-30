"""Wigner-Seitz cells, and with them the first Brillouin zone.

Everything here works with *row* bases, the convention used everywhere else in
CELLSTINE: ``basis[i]`` is the Cartesian vector of the ``i``-th basis vector, so
a point with fractional coordinates ``x`` (a row) sits at ``x @ basis``.

The Wigner-Seitz cell of a lattice is the set of points at least as close to the
origin as to any other lattice point,

.. code-block:: text

    W = { r : |r| <= |r - g| for every lattice vector g }
      = { r : r . g <= |g|^2 / 2 for every lattice vector g },

an intersection of half spaces, hence a convex polytope.  Applied to the
reciprocal lattice it is the first Brillouin zone, and that is what
:func:`brillouin_zone` returns.

Only finitely many of the half spaces matter.  A lattice vector ``g`` bounds the
cell only if its midpoint ``g / 2`` lies in the cell, and in a Delaunay-reduced
basis every such vector has coefficients in ``{-1, 0, 1}``: the Voronoi-relevant
vectors are among the fourteen ``+-a, +-b, +-c, +-(a+b), +-(b+c), +-(a+c),
+-(a+b+c)``.  The search here is run over the wider shell of coefficients in
``{-2, ..., 2}`` and pruned by the midpoint test, so the reduction is used for
speed and never for correctness.

The faces are then read off geometrically: every triple of retained planes is
intersected, the intersections that satisfy all the half spaces are the vertices
of the polytope, and a plane that carries at least three of them is a face.  The
volume computed from that face structure is compared with ``|det(basis)|``,
which the cell must tile, and a mismatch is reported rather than ignored -- it
is the one check that catches a missed face.

The formal statements behind this module are in
``aristotle-lean-reference/RequestProject/BrillouinZone.lean``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .reciprocal import reciprocal_lattice
from .reduction import as_lattice, delaunay_reduce

__all__ = [
    "WignerSeitzCell",
    "wigner_seitz_cell",
    "brillouin_zone",
]


_SHELL = 2


def _candidate_vectors(basis: np.ndarray) -> np.ndarray:
    """Return the nonzero lattice vectors of the search shell, shortest first."""

    reduced, _ = delaunay_reduce(basis)
    span = np.arange(-_SHELL, _SHELL + 1, dtype=np.int64)
    grid = np.stack(np.meshgrid(span, span, span, indexing="ij"), axis=-1).reshape(-1, 3)
    grid = grid[np.any(grid != 0, axis=1)]
    vectors = grid.astype(float) @ reduced
    order = np.argsort(np.einsum("ij,ij->i", vectors, vectors), kind="stable")
    return vectors[order]


def _merge_close_points(points: np.ndarray, tolerance: float) -> np.ndarray:
    """Return one representative per cluster of points within ``tolerance``.

    Every triple of planes through a vertex solves to the same point, so a
    vertex where more than three planes meet -- the four-fold vertices of a
    rhombic dodecahedron, say -- is found several times over, each copy carrying
    its own rounding error.  Sorting and comparing neighbours is *not* enough to
    merge them: the copies differ in the last bits of every coordinate, so the
    sort can interleave them with the other points that share a coordinate.
    The clusters are therefore the connected components of the graph on the
    points that joins two of them when they are within ``tolerance``, and the
    representative of a cluster is its mean.  The result is sorted
    lexicographically, so it does not depend on the order of the input.

    ``aristotle-lean-reference/RequestProject/VertexMerge.lean`` proves that this is exactly right when
    the vertices of the polytope are further apart than the tolerance plus twice
    the rounding error: the components are then precisely the vertices
    (``Cellstine.Merge.linked_iff_true_eq``) and the mean of a component is
    within the rounding error of the vertex it stands for
    (``Cellstine.Merge.dist_centroid_le``).
    """

    count = len(points)
    if count == 0:  # pragma: no cover - defensive
        return points.reshape(0, 3)
    deltas = points[:, None, :] - points[None, :, :]
    close = np.einsum("ijk,ijk->ij", deltas, deltas) <= tolerance * tolerance
    label = np.full(count, -1, dtype=np.int64)
    centres: list[np.ndarray] = []
    for seed in range(count):
        if label[seed] >= 0:
            continue
        component = np.zeros(count, dtype=bool)
        frontier = np.zeros(count, dtype=bool)
        frontier[seed] = True
        while frontier.any():
            component |= frontier
            frontier = np.any(close[frontier], axis=0) & ~component
        label[component] = len(centres)
        centres.append(points[component].mean(axis=0))
    merged = np.asarray(centres, dtype=float)
    order = np.lexsort((merged[:, 2], merged[:, 1], merged[:, 0]))
    return merged[order]


def _polygon_order(points: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """Return the indices that walk ``points`` once around ``normal``."""

    centre = points.mean(axis=0)
    axis = normal / np.linalg.norm(normal)
    offsets = points - centre
    first = offsets[int(np.argmax(np.einsum("ij,ij->i", offsets, offsets)))]
    first = first - axis * float(first @ axis)
    first = first / np.linalg.norm(first)
    second = np.cross(axis, first)
    angles = np.arctan2(offsets @ second, offsets @ first)
    return np.argsort(angles, kind="stable")


@dataclass(frozen=True)
class WignerSeitzCell:
    """The Wigner-Seitz cell of a lattice, as a convex polytope.

    ``face_vectors[i]`` is the lattice vector whose bisector carries face ``i``;
    the face itself lies on ``r . g = |g|^2 / 2``, so ``face_offsets[i]`` is that
    right-hand side and the cell is ``r @ face_vectors.T <= face_offsets``.
    ``face_vertices[i]`` lists the rows of ``vertices`` on face ``i``, ordered
    once around the face by the right-hand rule about the face vector.
    """

    basis: np.ndarray
    face_vectors: np.ndarray
    face_offsets: np.ndarray
    vertices: np.ndarray
    face_vertices: tuple[tuple[int, ...], ...]
    tolerance: float
    volume_error: float = field(default=0.0)

    @property
    def face_count(self) -> int:
        return int(self.face_vectors.shape[0])

    @property
    def vertex_count(self) -> int:
        return int(self.vertices.shape[0])

    @property
    def volume(self) -> float:
        """Return the volume of the cell, which tiles space."""

        return abs(float(np.linalg.det(self.basis)))

    @property
    def inradius(self) -> float:
        """Return the radius of the largest ball centred on the origin inside."""

        return float(np.min(np.linalg.norm(self.face_vectors, axis=1))) / 2.0

    @property
    def circumradius(self) -> float:
        """Return the distance from the origin to the farthest vertex."""

        return float(np.max(np.linalg.norm(self.vertices, axis=1)))

    def edges(self) -> tuple[tuple[int, int], ...]:
        """Return the vertex-index pairs joined by an edge of the polytope."""

        pairs: set[tuple[int, int]] = set()
        for loop in self.face_vertices:
            for position, index in enumerate(loop):
                other = loop[(position + 1) % len(loop)]
                pairs.add((min(index, other), max(index, other)))
        return tuple(sorted(pairs))

    def contains(self, points: Sequence[Sequence[float]], *, tolerance: float | None = None) -> np.ndarray:
        """Return the mask of ``points`` inside the closed cell."""

        array = np.atleast_2d(np.asarray(points, dtype=float))
        if array.shape[-1] != 3:
            raise ValueError("points must be three-dimensional")
        slack = self.tolerance if tolerance is None else float(tolerance)
        return np.all(array @ self.face_vectors.T <= self.face_offsets + slack, axis=1)

    def boundary_scale(self, direction: Sequence[float]) -> float:
        """Return the ``t > 0`` with ``t * direction`` on the boundary.

        The ray from the origin leaves a convex body exactly once, at the
        smallest positive scale allowed by the half spaces it crosses.
        """

        ray = np.asarray(direction, dtype=float).reshape(3)
        if not np.all(np.isfinite(ray)):
            raise ValueError("a direction must be finite")
        projections = self.face_vectors @ ray
        forward = projections > self.tolerance
        if not np.any(forward):
            raise ValueError("a direction must be nonzero")
        return float(np.min(self.face_offsets[forward] / projections[forward]))

    def boundary_point(self, direction: Sequence[float]) -> np.ndarray:
        """Return the point where the ray along ``direction`` meets the boundary."""

        ray = np.asarray(direction, dtype=float).reshape(3)
        return self.boundary_scale(ray) * ray

    def face_centres(self) -> np.ndarray:
        """Return the midpoints ``g / 2`` of the face vectors."""

        return self.face_vectors / 2.0

    def edge_midpoints(self) -> np.ndarray:
        """Return the midpoint of every edge."""

        pairs = self.edges()
        if not pairs:  # pragma: no cover - defensive
            return np.zeros((0, 3))
        first = self.vertices[[pair[0] for pair in pairs]]
        second = self.vertices[[pair[1] for pair in pairs]]
        return (first + second) / 2.0

    def summary(self) -> dict[str, object]:
        """Return a JSON-ready description of the polytope."""

        return {
            "face_count": self.face_count,
            "vertex_count": self.vertex_count,
            "edge_count": len(self.edges()),
            "volume": self.volume,
            "inradius": self.inradius,
            "circumradius": self.circumradius,
            "volume_error": self.volume_error,
        }


def _face_volume(vertices: np.ndarray, vector: np.ndarray) -> float:
    """Return the volume of the pyramid over one face with apex at the origin."""

    apexed = vertices - vertices[0]
    total = np.zeros(3)
    for index in range(1, len(vertices) - 1):
        total = total + np.cross(apexed[index], apexed[index + 1])
    area = float(np.linalg.norm(total)) / 2.0
    height = float(np.linalg.norm(vector)) / 2.0
    return area * height / 3.0


def wigner_seitz_cell(basis: Sequence[Sequence[float]], *, tolerance: float | None = None) -> WignerSeitzCell:
    """Return the Wigner-Seitz cell of the lattice spanned by ``basis``.

    ``tolerance`` is a length in the units of ``basis``; it defaults to a
    relative one taken from the cell volume, which keeps the routine scale free.
    """

    lattice = as_lattice(np.asarray(basis, dtype=float), "basis")
    scale = abs(float(np.linalg.det(lattice))) ** (1.0 / 3.0)
    slack = float(tolerance) if tolerance is not None else 1e-9 * scale
    if slack <= 0.0:
        raise ValueError("the tolerance must be positive")
    # The half-space tests compare a dot product with a squared length, so the
    # length tolerance enters them multiplied by the scale of the lattice.
    area_slack = slack * scale

    vectors = _candidate_vectors(lattice)
    offsets = np.einsum("ij,ij->i", vectors, vectors) / 2.0

    # A bisector carries a face only if its midpoint is strictly inside every
    # other half space -- the classical criterion for a Voronoi-relevant vector.
    # A plane that merely touches the cell (its midpoint sits on another
    # bisector) is dropped here, which is what keeps a prism a prism.
    products = (vectors / 2.0) @ vectors.T
    strict = products < offsets[None, :] - area_slack
    itself = np.eye(len(vectors), dtype=bool)
    keep = np.all(strict | itself, axis=1)
    planes = vectors[keep]
    plane_offsets = offsets[keep]

    count = len(planes)
    if count < 4:  # pragma: no cover - defensive
        raise RuntimeError("the Wigner-Seitz search shell was too small")

    corners: list[np.ndarray] = []
    for first in range(count - 2):
        for second in range(first + 1, count - 1):
            for third in range(second + 1, count):
                matrix = planes[[first, second, third]]
                determinant = float(np.linalg.det(matrix))
                if abs(determinant) <= 1e-10 * scale**3:
                    continue
                point = np.linalg.solve(matrix, plane_offsets[[first, second, third]])
                if np.all(point @ planes.T <= plane_offsets + area_slack):
                    corners.append(point)
    if not corners:  # pragma: no cover - defensive
        raise RuntimeError("the Wigner-Seitz cell has no vertices")

    stacked = np.asarray(corners, dtype=float)
    vertices = _merge_close_points(stacked, 1e-6 * scale)

    face_vectors: list[np.ndarray] = []
    face_offsets: list[float] = []
    face_vertices: list[tuple[int, ...]] = []
    volume = 0.0
    for index in range(count):
        residual = vertices @ planes[index] - plane_offsets[index]
        on_face = np.flatnonzero(np.abs(residual) <= 1e-6 * scale)
        if len(on_face) < 3:
            continue
        loop = on_face[_polygon_order(vertices[on_face], planes[index])]
        face_vectors.append(planes[index])
        face_offsets.append(float(plane_offsets[index]))
        face_vertices.append(tuple(int(item) for item in loop))
        volume += _face_volume(vertices[loop], planes[index])

    expected = abs(float(np.linalg.det(lattice)))
    error = abs(volume - expected) / expected
    if error > 1e-6:  # pragma: no cover - defensive
        raise RuntimeError(
            "the Wigner-Seitz faces do not enclose the cell volume "
            f"(relative error {error:.3e}); the search shell may be too small"
        )

    return WignerSeitzCell(
        basis=lattice,
        face_vectors=np.asarray(face_vectors, dtype=float),
        face_offsets=np.asarray(face_offsets, dtype=float),
        vertices=vertices,
        face_vertices=tuple(face_vertices),
        tolerance=1e-8 * scale,
        volume_error=float(error),
    )


def brillouin_zone(lattice: Sequence[Sequence[float]], *, tolerance: float | None = None) -> WignerSeitzCell:
    """Return the first Brillouin zone of a real-space ``lattice``.

    The zone is the Wigner-Seitz cell of the reciprocal lattice
    ``B = 2 pi inv(A).T``, so its volume is ``(2 pi)^3 / |det A|`` and its faces
    are the Bragg planes of the shortest reciprocal vectors.
    """

    return wigner_seitz_cell(reciprocal_lattice(lattice), tolerance=tolerance)


def zone_boundary_distance(lattice: Sequence[Sequence[float]], direction: Sequence[float]) -> float:
    """Return the distance from Gamma to the zone boundary along ``direction``."""

    zone = brillouin_zone(lattice)
    ray = np.asarray(direction, dtype=float).reshape(3)
    length = float(np.linalg.norm(ray))
    if not math.isfinite(length) or length <= 0.0:
        raise ValueError("a direction must be nonzero and finite")
    return zone.boundary_scale(ray) * length
