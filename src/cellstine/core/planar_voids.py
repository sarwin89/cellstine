"""Hollows of a periodic *planar* point set.

The two-dimensional counterpart of :mod:`cellstine.core.voids`: the adsorption
hollows of a surface layer are the vertices of the Voronoi diagram of the atoms
projected onto the surface plane, that is the circumcentres of Delaunay
triangles whose circle holds no other atom.  They come out of closed-form
geometry rather than a grid search, so a hollow cannot be missed for being
between two grid nodes.

The neighbour cutoff of the enumeration is set by
:func:`cellstine.core.covering.branch_and_bound_maximum`, exactly as in the bulk
search.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

from . import covering, geometry

__all__ = [
    "PlanarVoid",
    "find_planar_voids",
]


@dataclass(frozen=True)
class PlanarVoid:
    """One hollow of a periodic planar point set."""

    uv: Tuple[float, float]
    radius: float
    coordination: int


def _plane_frame(basis: np.ndarray) -> np.ndarray:
    """Return two orthonormal in-plane axes as the rows of a ``(2, 3)`` array."""

    first = np.asarray(basis, dtype=float)[0]
    second = np.asarray(basis, dtype=float)[1]
    normal = np.cross(first, second)
    scale = float(np.linalg.norm(first) * np.linalg.norm(second))
    if float(np.linalg.norm(normal)) <= 1e-12 * max(scale, 1e-12):
        raise ValueError("the two in-plane vectors are parallel")
    axis_u = first / float(np.linalg.norm(first))
    axis_v = np.cross(normal, first)
    axis_v = axis_v / float(np.linalg.norm(axis_v))
    return np.stack([axis_u, axis_v])


def _plane_images(basis: np.ndarray, points_uv: np.ndarray, radius: float) -> np.ndarray:
    """Return the Cartesian images of a planar point set out to ``radius``.

    The range of shifts comes from the reach bound of
    :mod:`cellstine.core.geometry`, so every image within ``radius`` of the cell
    is present.
    """

    reach = np.ceil(
        max(float(radius), 0.0) * geometry.plane_reciprocal_norms(basis)
    ).astype(int) + 1
    shifts = np.array(
        [
            [i, j]
            for i in range(-int(reach[0]), int(reach[0]) + 1)
            for j in range(-int(reach[1]), int(reach[1]) + 1)
        ],
        dtype=float,
    )
    images_uv = (np.asarray(points_uv, dtype=float)[None, :, :] + shifts[:, None, :]).reshape(-1, 2)
    return images_uv @ np.asarray(basis, dtype=float)


def _planar_nearest_neighbour(basis: np.ndarray, points: np.ndarray) -> float:
    """Return the shortest distance between two distinct images of the point set.

    A single point still has neighbours -- its own translates -- so the answer is
    then the shortest lattice vector.
    """

    cartesian = np.asarray(points, dtype=float) @ np.asarray(basis, dtype=float)
    guess = float(np.min(np.linalg.norm(np.asarray(basis, dtype=float), axis=1)))
    for _ in range(12):
        images = _plane_images(basis, points, guess)
        grid = geometry.CartesianGrid(images, guess)
        indices, valid = grid.candidates(cartesian)
        if indices.shape[1]:
            deltas = images[indices] - cartesian[:, None, :]
            distances = np.sqrt(np.einsum("ijk,ijk->ij", deltas, deltas))
            distances[~valid] = np.inf
            distances[distances <= 1e-9] = np.inf
            best = float(distances.min())
            if np.isfinite(best):
                return best
        guess *= 2.0
    raise ArithmeticError("no second image was found within twelve doublings")


def _planar_covering_radius_bound(basis: np.ndarray, points: np.ndarray, spacing: float) -> float:
    """Return an upper bound on the radius of the largest empty circle.

    The distance to the nearest point is 1-Lipschitz, so the same
    branch-and-bound refinement the bulk search uses --
    :func:`covering.branch_and_bound_maximum` -- bounds its maximum over the cell, here
    with two-dimensional boxes.  Every circumcircle of the search is smaller
    than the bound, which is what keeps the enumeration of candidate triangles
    short, so a tight bound is worth having.
    """

    basis = np.asarray(basis, dtype=float)
    lengths = np.linalg.norm(basis, axis=1)
    # Any point of the cell has a point of the set within the cell diameter.
    images = _plane_images(basis, points, float(lengths.sum()))
    query = geometry.NearestPointQuery(images)

    def evaluate(fractional: np.ndarray) -> np.ndarray:
        # The two basis rows are Cartesian, so a planar point is a point of
        # space and the same bucket grid answers for it.
        return query.distances(np.mod(np.asarray(fractional, dtype=float).reshape(-1, 2), 1.0) @ basis)

    counts = np.maximum(np.ceil(lengths / max(2.0 * spacing, 1e-9)).astype(int), 2)
    step = 1.0 / counts.astype(float)
    axes = [(np.arange(int(count), dtype=float) + 0.5) / float(count) for count in counts]
    centres = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 2)
    return covering.branch_and_bound_maximum(
        basis,
        centres,
        step,
        evaluate,
        tolerance=0.02 * max(float(spacing), 1e-9),
        probe_budget=covering.PROBE_BUDGET + covering.PROBE_PER_ATOM * len(np.asarray(points).reshape(-1, 2)),
    )


def _planar_circumcentres(anchor: np.ndarray, first: np.ndarray, second: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return the planar circumcentres of triangles and a mask of the valid ones.

    Rows where the three points are collinear have no circumcentre and are
    reported as invalid.
    """

    ax, ay = anchor[:, 0], anchor[:, 1]
    bx, by = first[:, 0], first[:, 1]
    cx, cy = second[:, 0], second[:, 1]
    twice_area = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    scale = np.maximum.reduce(
        [
            np.abs(bx - ax) + np.abs(by - ay),
            np.abs(cx - ax) + np.abs(cy - ay),
            np.abs(cx - bx) + np.abs(cy - by),
        ]
    )
    good = np.abs(twice_area) > 1e-12 * np.maximum(scale, 1e-12) ** 2
    safe = np.where(good, twice_area, 1.0)
    sq_a = ax * ax + ay * ay
    sq_b = bx * bx + by * by
    sq_c = cx * cx + cy * cy
    centre_x = (sq_a * (by - cy) + sq_b * (cy - ay) + sq_c * (ay - by)) / safe
    centre_y = (sq_a * (cx - bx) + sq_b * (ax - cx) + sq_c * (bx - ax)) / safe
    return np.stack([centre_x, centre_y], axis=1), good


def _is_local_maximum(directions: np.ndarray) -> bool:
    """True when the touching points surround the centre.

    The distance to the nearest point grows in a direction ``d`` exactly when
    ``d`` points away from every touching point, so the centre is a local
    maximum precisely when no half-plane holds all of them -- equivalently when
    the largest angular gap between consecutive touching directions is less than
    a straight angle.  A gap of exactly ``pi``, as at the midpoint of a bond, is
    the saddle case and is rejected.
    """

    if directions.shape[0] < 3:
        return False
    angles = np.sort(np.arctan2(directions[:, 1], directions[:, 0]))
    gaps = np.diff(np.concatenate([angles, angles[:1] + 2.0 * np.pi]))
    return bool(gaps.max() < np.pi - 1e-9)


def _reduced_planar_voids(
    basis: np.ndarray,
    points: np.ndarray,
    merge_distance: float | None,
    coordination_tolerance: float | None,
) -> List[PlanarVoid]:
    """Return the hollows of a point set given in an already reduced plane basis."""

    spacing = _planar_nearest_neighbour(basis, points)
    merge = float(merge_distance) if merge_distance is not None else 1e-4 * spacing
    touch = float(coordination_tolerance) if coordination_tolerance is not None else 0.03 * spacing
    slack = 1e-6 * spacing

    cap = _planar_covering_radius_bound(basis, points, spacing)
    images = _plane_images(basis, points, 2.0 * cap + spacing)
    frame = _plane_frame(basis)
    flat_images = images @ frame.T
    anchors = (np.asarray(points, dtype=float) @ np.asarray(basis, dtype=float)) @ frame.T

    # Every vertex of the Voronoi diagram is the circumcentre of a triangle of
    # points that are mutually within twice the largest empty circle, so the
    # cell list over that cutoff supplies every candidate triangle.
    grid = geometry.CartesianGrid(images, 2.0 * cap + slack)
    indices, valid = grid.candidates(np.asarray(points, dtype=float) @ np.asarray(basis, dtype=float))

    centres: List[np.ndarray] = []
    for anchor_index in range(anchors.shape[0]):
        neighbours = indices[anchor_index][valid[anchor_index]]
        if neighbours.size < 2:
            continue
        offsets = flat_images[neighbours] - anchors[anchor_index][None, :]
        near = np.einsum("ij,ij->i", offsets, offsets) <= (2.0 * cap + slack) ** 2
        neighbours = neighbours[near]
        if neighbours.size < 2:
            continue
        left, right = np.triu_indices(neighbours.size, k=1)
        anchor = np.repeat(anchors[anchor_index][None, :], left.size, axis=0)
        found, good = _planar_circumcentres(anchor, flat_images[neighbours[left]], flat_images[neighbours[right]])
        if not np.any(good):
            continue
        found = found[good]
        radii = np.linalg.norm(found - anchors[anchor_index][None, :], axis=1)
        centres.append(found[radii <= cap + slack])

    if not centres:
        return []
    candidates = np.concatenate(centres, axis=0) @ frame

    # A circumcentre is a Voronoi vertex only when its circle is empty.
    nearest = geometry.nearest_point_distances(candidates, images)
    keep = nearest > slack
    candidates, nearest = candidates[keep], nearest[keep]
    if candidates.shape[0] == 0:
        return []

    fractional = geometry.wrap_to_cell(
        np.column_stack([candidates @ np.linalg.pinv(np.asarray(basis, dtype=float)), np.zeros(candidates.shape[0])])
    )
    normal = np.cross(basis[0], basis[1])
    normal = normal / float(np.linalg.norm(normal))
    box = np.vstack([np.asarray(basis, dtype=float), normal * float(np.linalg.norm(basis, axis=1).max())])
    unique = geometry.PeriodicSiteIndex(box, fractional, tolerance=max(merge, 1e-9))
    representatives = unique.match(fractional, prefer_lowest=True)
    chosen = np.nonzero(representatives == np.arange(fractional.shape[0]))[0]

    contacts = geometry.CartesianGrid(images, cap + touch + slack)
    contact_indices, contact_valid = contacts.candidates(candidates[chosen])

    hollows: List[PlanarVoid] = []
    for row, index in enumerate(chosen):
        radius = float(nearest[index])
        neighbours = contact_indices[row][contact_valid[row]]
        if neighbours.size == 0:
            continue
        offsets = images[neighbours] - candidates[index][None, :]
        distances = np.linalg.norm(offsets, axis=1)
        touching = distances <= radius + slack
        if not _is_local_maximum((offsets[touching] @ frame.T) / distances[touching][:, None]):
            continue
        coordination = int(np.count_nonzero(distances <= radius + touch))
        uv = fractional[index][:2]
        hollows.append(
            PlanarVoid(uv=(float(uv[0]), float(uv[1])), radius=radius, coordination=coordination)
        )
    hollows.sort(key=lambda hollow: (-hollow.radius, hollow.uv))
    return hollows


def find_planar_voids(
    basis: np.ndarray,
    points_uv: np.ndarray,
    *,
    merge_distance: float | None = None,
    coordination_tolerance: float | None = None,
) -> List[PlanarVoid]:
    """Return the hollows of a periodic set of points in a plane.

    A hollow is a local maximum of the distance to the nearest point of the set,
    that is a vertex of its two-dimensional Voronoi diagram.  This is the general
    definition behind the familiar site names: on a triangular lattice the
    maxima are the three-fold hollows, on a square lattice the four-fold hollow,
    and on a honeycomb lattice the centre of the hexagon -- which a search for
    triangles of mutual neighbours misses, because a honeycomb has none.

    ``basis`` holds the two in-plane lattice vectors as rows (Cartesian, three
    components), ``points_uv`` the fractional in-plane coordinates of the atoms.
    Each hollow is reported with the radius of its empty circle and with the
    number of atoms that touch it, which is its coordination.

    The vertices are enumerated in closed form -- as the circumcentres of
    triangles of points, kept when the circle through them is empty and when the
    touching points surround the centre -- so the positions are exact and none
    is lost to a coarse grid.  The search runs in the Lagrange--Gauss reduced
    basis of the same lattice, where the enumeration is smallest.

    ``merge_distance`` sets when two centres count as the same site, and
    ``coordination_tolerance`` how close a point has to be to the empty circle
    to be counted as touching it; both default to fractions of the
    nearest-neighbour spacing of the point set, so the search behaves the same
    way on a metal surface and on a molecular monolayer.
    """

    array = np.asarray(basis, dtype=float).reshape(2, -1)
    points = np.mod(np.asarray(points_uv, dtype=float).reshape(-1, 2), 1.0)
    if points.shape[0] == 0:
        return []

    reduced, transform = geometry.plane_reduce(array)
    inverse = np.linalg.inv(transform.astype(float))
    hollows = _reduced_planar_voids(
        reduced, np.mod(points @ inverse, 1.0), merge_distance, coordination_tolerance
    )
    rebased: List[PlanarVoid] = []
    for hollow in hollows:
        uv = np.mod(np.asarray(hollow.uv, dtype=float) @ transform.astype(float), 1.0)
        rebased.append(
            PlanarVoid(uv=(float(uv[0]), float(uv[1])), radius=hollow.radius, coordination=hollow.coordination)
        )
    return rebased
