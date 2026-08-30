"""Interstitial void search based on the largest-empty-sphere criterion.

An interstitial site of a structure is a point of the cell that is locally as
far as possible from every atom: a local maximum of

.. code-block:: text

    d(x) = min over atoms and periodic images of |x - r_atom|

whose value is large enough to host an atom.  Such a point is equidistant from
at least four atoms and carries an empty sphere through them, so it is a vertex
of the Voronoi diagram of the atoms, equivalently the circumcentre of a
Delaunay tetrahedron.  This module enumerates those circumcentres exactly and
keeps the ones whose sphere is empty, so the sites and their radii come out of
closed-form geometry rather than out of a grid search, and no site is missed
because a grid was too coarse.

For a structure with vacuum -- a slab or a molecule in a box -- the largest
empty spheres sit in the vacuum and are not interstitials at all.  The search
therefore detects vacuum directions and restricts candidates to the region that
actually contains material.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any, List, Sequence, Tuple

import numpy as np

from . import covering, geometry

__all__ = [
    "VoidSite",
    "VoidSearchResult",
    "find_void_sites",
    "material_bounds",
]


@dataclass(frozen=True)
class VoidSite:
    """One interstitial candidate and the empty sphere around it."""

    direct: Tuple[float, float, float]
    cartesian: Tuple[float, float, float]
    radius: float
    kind: str = "maximum"
    """``"maximum"`` for a local maximum of the distance to the nearest atom,
    ``"saddle"`` for a critical point that grows only along some directions.

    Both are sites an interstitial atom can sit at; only a maximum is a hollow
    in every direction.  The octahedral site of a body-centred cubic metal --
    where carbon sits in ferrite -- is a saddle, so a search that keeps only the
    maxima misses it.
    """
    coordination: int = 0
    """Number of atoms lying on the empty sphere."""


@dataclass(frozen=True)
class VoidSearchResult:
    """Interstitial candidates plus the settings that produced them."""

    sites: List[VoidSite]
    minimum_radius: float
    neighbour_cutoff: float
    vacuum_axes: Tuple[int, ...]
    bounds: dict[int, Tuple[float, float]]


_axis_spacings = geometry.axis_spacings


def material_bounds(
    lattice: np.ndarray,
    positions_direct: np.ndarray,
    *,
    vacuum_threshold: float = 5.0,
) -> Tuple[Tuple[int, ...], dict[int, Tuple[float, float]]]:
    """Return the axes that contain vacuum and the occupied fractional range.

    An axis counts as a vacuum direction when the largest gap between
    consecutive atomic projections along it, measured perpendicular to the other
    two axes, exceeds ``vacuum_threshold``.  The reported bounds are the
    fractional interval that holds the material once the cell has been shifted
    so that the gap sits at the cell edge.
    """

    lattice = np.asarray(lattice, dtype=float)
    positions = np.mod(np.asarray(positions_direct, dtype=float).reshape(-1, 3), 1.0)
    spacings = _axis_spacings(lattice)
    vacuum_axes: List[int] = []
    bounds: dict[int, Tuple[float, float]] = {}
    if len(positions) == 0:
        return tuple(), bounds
    for axis in range(3):
        values = np.sort(positions[:, axis])
        gaps = np.diff(np.concatenate([values, values[:1] + 1.0]))
        widest = int(np.argmax(gaps))
        if float(gaps[widest]) * float(spacings[axis]) <= float(vacuum_threshold):
            continue
        start = float(values[(widest + 1) % len(values)])
        if widest + 1 == len(values):
            start = float(values[0])
        lower = start
        upper = start + (1.0 - float(gaps[widest]))
        vacuum_axes.append(axis)
        bounds[axis] = (lower, upper)
    return tuple(vacuum_axes), bounds


def _origin_in_flat_hull(coordinates: np.ndarray, rank: int, tolerance: float) -> bool:
    """Return whether the origin lies in the hull of directions spanning a line or a plane."""

    if rank == 1:
        values = coordinates[:, 0]
        return bool(np.any(values > tolerance) and np.any(values < -tolerance))
    angles = np.sort(np.arctan2(coordinates[:, 1], coordinates[:, 0]))
    gaps = np.diff(np.concatenate([angles, angles[:1] + 2.0 * np.pi]))
    return bool(np.max(gaps, initial=0.0) <= np.pi + tolerance)


def _classify_contact_directions(units: np.ndarray, tolerance: float = 1e-6) -> str | None:
    """Classify a centre from the unit vectors pointing at the atoms it touches.

    The empty-sphere radius is the distance to the nearest atom, so moving along
    ``v`` grows it exactly when ``v`` points away from every touching atom, that
    is when ``v . u < 0`` for each direction ``u`` of the contact set.  Three
    cases follow, and they are decided here exactly rather than by sampling
    directions:

    * some ``v`` moves away from all of them -- the centre is not a critical
      point of the distance function at all, and the sphere grows by sliding.
      This is what happens just under the surface of a slab, where every contact
      lies below;
    * no ``v`` even holds all the distances -- the origin is interior to the
      hull of the contact directions and the centre is a strict local maximum, a
      vertex of the Voronoi diagram;
    * in between, the sphere is stationary along a line or a plane of directions
      and shrinks off it: a saddle.  The octahedral site of a body-centred cubic
      metal is of this kind, its two contacts holding it against motion along
      their axis only.

    The set ``K = {v : v . u <= 0}`` decides all three.  When the contacts span
    space ``K`` is pointed, so it is generated by rays where two of the
    constraints are tight -- the cross products of pairs of contact directions --
    and testing those finitely many rays settles whether ``K`` is trivial, and
    whether it has an interior.  When the contacts span only a line or a plane,
    every direction orthogonal to them holds the distances, so the centre is at
    best a saddle, and the origin lies in their hull exactly when no open
    half-line or half-plane holds them all.
    """

    units = np.asarray(units, dtype=float).reshape(-1, 3)
    if units.shape[0] < 2:
        return None
    _, values, rows = np.linalg.svd(units, full_matrices=False)
    rank = int(np.count_nonzero(values > max(float(values[0]), 1e-12) * 1e-7))
    if rank < 3:
        coordinates = units @ rows[:rank].T
        return "saddle" if _origin_in_flat_hull(coordinates, rank, tolerance) else None

    left, right = np.triu_indices(units.shape[0], k=1)
    rays = np.cross(units[left], units[right])
    lengths = np.linalg.norm(rays, axis=1)
    rays = rays[lengths > 1e-9] / lengths[lengths > 1e-9, None]
    rays = np.concatenate([rays, -rays], axis=0)
    inside = np.all(rays @ units.T <= tolerance, axis=1)
    if not np.any(inside):
        return "maximum"
    # A sum of rays of a pointed cone lies in its relative interior, so it moves
    # away from every contact exactly when the cone is solid -- the case where
    # the sphere really can grow.
    direction = rays[inside].sum(axis=0)
    length = float(np.linalg.norm(direction))
    if length > 1e-9 and np.all((direction / length) @ units.T < -tolerance):
        return None
    return "saddle"


def _critical_point_classes(
    lattice: np.ndarray,
    centres_direct: np.ndarray,
    radii: np.ndarray,
    images: np.ndarray,
    contact_tolerance: float = 1e-3,
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """Return which centres are critical points, of which kind, and their coordination.

    Only atoms on the sphere can hold a centre in place, so a cell list over the
    images supplies the handful of contacts per centre; comparing every centre
    with every image would be the product of two large numbers in a supercell.
    """

    lattice = np.asarray(lattice, dtype=float)
    centres = np.asarray(centres_direct, dtype=float).reshape(-1, 3)
    lengths = np.asarray(radii, dtype=float).reshape(-1)
    verdict = np.zeros(centres.shape[0], dtype=bool)
    kinds: List[str] = [""] * centres.shape[0]
    coordination = np.zeros(centres.shape[0], dtype=int)
    images = np.asarray(images, dtype=float).reshape(-1, 3)
    if centres.shape[0] == 0 or images.shape[0] == 0:
        return verdict, kinds, coordination

    tolerance = float(contact_tolerance)
    limits = lengths * (1.0 + tolerance)
    grid = geometry.CartesianGrid(images, float(np.max(limits, initial=0.0)))
    points = centres @ lattice
    indices, valid = grid.candidates(points)
    if indices.shape[1] == 0:
        return verdict, kinds, coordination
    offsets = images[indices] - points[:, None, :]
    squared = np.einsum("ijk,ijk->ij", offsets, offsets)
    touching = valid & (squared <= (limits ** 2)[:, None])
    for row in np.nonzero(touching.sum(axis=1) >= 2)[0]:
        contacts = offsets[row][touching[row]]
        unit = contacts / np.linalg.norm(contacts, axis=1)[:, None]
        kind = _classify_contact_directions(unit, tolerance)
        if kind is None:
            continue
        verdict[row] = True
        kinds[row] = kind
        coordination[row] = int(contacts.shape[0])
    return verdict, kinds, coordination


def _atom_images(lattice: np.ndarray, atoms_direct: np.ndarray, cutoff: float) -> np.ndarray:
    """Return every periodic image of the atoms within ``cutoff`` of the cell."""

    return geometry.atom_images(lattice, atoms_direct, cutoff)


def _circumcentres(anchor: np.ndarray, triples: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return the circumcentre and circumradius of every tetrahedron formed by
    ``anchor`` and a row of ``triples`` (shape ``(n, 3, 3)``).

    Degenerate tetrahedra -- four coplanar atoms -- get a radius of infinity and
    are dropped by the caller.
    """

    edges = triples - anchor[None, None, :]
    matrix = 2.0 * edges
    rhs = np.einsum("ijk,ijk->ij", edges, edges)
    determinant = np.linalg.det(matrix)
    good = np.abs(determinant) > 1e-9
    centres = np.full((triples.shape[0], 3), np.inf, dtype=float)
    radii = np.full(triples.shape[0], np.inf, dtype=float)
    if np.any(good):
        solved = np.linalg.solve(matrix[good], rhs[good][:, :, None])[:, :, 0]
        centres[good] = anchor[None, :] + solved
        radii[good] = np.linalg.norm(solved, axis=1)
    return centres, radii


#: Guard against a pathological cutoff making the triple enumeration explode.
#: The face test below normally keeps the surviving neighbour count far under
#: this, so the cap is a safety valve rather than part of the mathematics.
_MAX_NEIGHBOURS = 200

_TRIPLE_INDEX_CACHE: dict[int, np.ndarray] = {}


def _triple_indices(count: int) -> np.ndarray:
    """Return every unordered triple of ``range(count)``, cached by size."""

    cached = _TRIPLE_INDEX_CACHE.get(int(count))
    if cached is None:
        cached = np.array(list(itertools.combinations(range(int(count)), 3)), dtype=int).reshape(-1, 3)
        _TRIPLE_INDEX_CACHE[int(count)] = cached
    return cached


def _triangle_circumcentres(anchor: np.ndarray, first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Return the circumcentre of each triangle ``(anchor, anchor + b, anchor + c)``.

    The circumcentre is taken inside the plane of the triangle: it is the point
    of that plane equidistant from the three corners, which is where the
    distance to the nearest atom can be stationary within the plane.
    """

    normal = np.cross(first, second)
    denominator = 2.0 * np.einsum("ij,ij->i", normal, normal)
    lengths_first = np.einsum("ij,ij->i", first, first)
    lengths_second = np.einsum("ij,ij->i", second, second)
    numerator = np.cross(
        lengths_first[:, None] * second - lengths_second[:, None] * first, normal
    )
    return anchor[None, :] + numerator / denominator[:, None]


def _empty_sphere_centres(
    lattice: np.ndarray,
    atoms_direct: np.ndarray,
    cutoff: float,
    radius_cap: float,
    include_saddles: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return the Voronoi vertices of the atoms and the radius of each empty sphere.

    A vertex is the circumcentre of four atoms whose circumsphere contains no
    other atom.  Every such quadruple contains an atom of the base cell, and all
    four atoms lie on a sphere of radius at most ``radius_cap``, so they are
    within ``cutoff = 2 * radius_cap`` of one another: enumerating the mutually
    close triples of neighbours of each base atom finds every vertex, and that
    mutual-distance test is what keeps the enumeration small.

    The same bound makes the emptiness test local.  An atom inside the sphere of
    an accepted vertex is within ``2 * radius`` of the anchor, hence already in
    the anchor's neighbour list, so no global sweep over the cell is needed.
    """

    lattice = np.asarray(lattice, dtype=float)
    atoms = np.asarray(atoms_direct, dtype=float)
    base = atoms @ lattice
    # One cell list over the images answers "which atoms are within the cutoff
    # of this one" for every anchor, instead of scanning all the images once per
    # anchor.
    images, neighbour_index, neighbour_valid = geometry.neighbour_images(lattice, atoms, cutoff)

    centres: List[np.ndarray] = []
    radii: List[np.ndarray] = []
    for anchor_index, anchor in enumerate(base):
        selected = neighbour_index[anchor_index][neighbour_valid[anchor_index]]
        neighbours = images[selected]
        lengths = np.linalg.norm(neighbours - anchor[None, :], axis=1)
        neighbours = neighbours[lengths > 1e-8]
        count = neighbours.shape[0]
        if count > _MAX_NEIGHBOURS:  # pragma: no cover - only for extreme cutoffs
            order = np.argsort(np.linalg.norm(neighbours - anchor[None, :], axis=1))
            neighbours = neighbours[order[:_MAX_NEIGHBOURS]]
            count = neighbours.shape[0]
        if count < (2 if include_saddles else 3):
            continue
        # A face of a tetrahedron is a circle on its circumsphere, so its
        # circumradius never exceeds the circumsphere radius.  Dropping the
        # pairs whose triangle with the anchor already needs a bigger sphere is
        # therefore lossless, and much stronger than only asking the three atoms
        # to be within the cutoff of one another.
        edges = neighbours - anchor[None, :]
        edge_lengths = np.linalg.norm(edges, axis=1)
        cross = np.cross(edges[:, None, :], edges[None, :, :])
        area = np.linalg.norm(cross, axis=2)
        opposite = np.linalg.norm(edges[:, None, :] - edges[None, :, :], axis=2)
        with np.errstate(divide="ignore", invalid="ignore"):
            face_radius = np.where(
                area > 1e-12,
                edge_lengths[:, None] * edge_lengths[None, :] * opposite / (2.0 * area),
                np.inf,
            )
        close = face_radius <= radius_cap
        index = _triple_indices(count)
        index = index[
            close[index[:, 0], index[:, 1]] & close[index[:, 0], index[:, 2]] & close[index[:, 1], index[:, 2]]
        ]
        if index.size:
            centre, radius = _circumcentres(anchor, neighbours[index])
            within = radius <= radius_cap
        elif not include_saddles:
            continue
        else:
            centre = np.empty((0, 3), dtype=float)
            radius = np.empty(0, dtype=float)
            within = np.zeros(0, dtype=bool)
        if include_saddles:
            # A sphere touching only two or three atoms can still be held in
            # place by them, along the axis of a pair or in the plane of a
            # triangle.  Those centres are the midpoints of pairs and the
            # in-plane circumcentres of triangles; the same emptiness test
            # decides which of them carry an empty sphere.
            pairs = edge_lengths <= 2.0 * radius_cap
            if np.any(pairs):
                centre = np.concatenate(
                    [centre, anchor[None, :] + 0.5 * edges[pairs]], axis=0
                )
                radius = np.concatenate([radius, 0.5 * edge_lengths[pairs]])
                within = np.concatenate([within, np.ones(int(np.count_nonzero(pairs)), dtype=bool)])
            left, right = np.triu_indices(count, k=1)
            spanning = close[left, right] & (area[left, right] > 1e-12)
            if np.any(spanning):
                triangles = _triangle_circumcentres(
                    anchor, edges[left[spanning]], edges[right[spanning]]
                )
                centre = np.concatenate([centre, triangles], axis=0)
                triangle_radius = np.linalg.norm(triangles - anchor[None, :], axis=1)
                radius = np.concatenate([radius, triangle_radius])
                within = np.concatenate([within, triangle_radius <= radius_cap])
        if not np.any(within):
            continue
        centre = centre[within]
        radius = radius[within]
        delta = centre[:, None, :] - neighbours[None, :, :]
        nearest = np.sqrt(np.einsum("ijk,ijk->ij", delta, delta)).min(axis=1)
        empty = nearest >= radius - 1e-6
        if not np.any(empty):
            continue
        centres.append(centre[empty])
        radii.append(radius[empty])

    if not centres:
        return np.empty((0, 3)), np.empty(0)
    return np.concatenate(centres, axis=0), np.concatenate(radii, axis=0)




_MINIMUM_IMAGE_SHIFTS = np.array(
    [[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)], dtype=float
)


def _merge_void_centres(
    lattice: np.ndarray,
    candidates: Sequence[Tuple[Any, ...]],
    merge: float,
) -> List[Tuple[Any, ...]]:
    """Collapse candidates that describe the same void, largest sphere first.

    Every vertex is found once per atom that touches it, so the raw list holds
    each site several times over.  Identical copies go first, by rounding, which
    leaves a short list for the distance-based pass.
    """

    if not candidates:
        return []
    lattice = np.asarray(lattice, dtype=float)
    order = sorted(range(len(candidates)), key=lambda index: -candidates[index][1])
    seen: set[tuple[int, int, int]] = set()
    unique: List[int] = []
    for index in order:
        centre = candidates[index][0]
        key = tuple(int(round(float(value) * 1e6)) % 1_000_000 for value in centre)
        if key in seen:
            continue
        seen.add(key)
        unique.append(index)

    # Which candidates are within ``merge`` of which is settled once, by a cell
    # list, instead of by a distance calculation against the growing list of
    # kept centres: the sweep below then only has to mark the neighbours of each
    # centre it keeps, which is linear in the number of close pairs.
    points = np.array([candidates[index][0] for index in unique], dtype=float)
    first, second = geometry.periodic_neighbour_pairs(lattice, points, float(merge))
    close: List[List[int]] = [[] for _ in unique]
    for left, right in zip(first.tolist(), second.tolist()):
        close[left].append(right)
        close[right].append(left)

    suppressed = [False] * len(unique)
    kept: List[Tuple[Any, ...]] = []
    for position, index in enumerate(unique):
        if suppressed[position]:
            continue
        kept.append(candidates[index])
        for neighbour in close[position]:
            suppressed[neighbour] = True
    return kept


def find_void_sites(
    lattice: np.ndarray,
    positions_direct: np.ndarray,
    *,
    minimum_radius: float | None = None,
    vacuum_threshold: float = 5.0,
    merge_distance: float | None = None,
    neighbour_cutoff: float | None = None,
    include_saddles: bool = False,
) -> VoidSearchResult:
    """Return the interstitial voids of a cell, largest empty sphere first.

    ``minimum_radius`` is the smallest empty-sphere radius that still counts as
    an interstitial; it defaults to 40 percent of the shortest interatomic
    distance, which keeps the tetrahedral and octahedral voids of close-packed
    and diamond structures and rejects the shallow dips between neighbours.

    ``neighbour_cutoff`` bounds how far apart the four atoms around a void may
    be.  It defaults to twice a rigorous upper bound on the covering radius of
    the cell, which is exactly what the enumeration needs to be complete: the
    four atoms on an empty sphere of radius ``r`` are at most ``2 r`` apart.
    An open framework therefore costs more than a dense crystal, and nothing
    else changes.

    With ``include_saddles`` the search also returns the centres that are held
    in place by two or three atoms only -- the saddles of the same distance
    function, marked ``kind="saddle"``.  They are stationary points too, and
    they are where an interstitial atom sits whenever the crystal has no local
    maximum nearby: the octahedral site of a body-centred cubic metal, the site
    carbon takes in ferrite, is the midpoint of two second-neighbour atoms and
    is a saddle, not a Voronoi vertex.
    """

    lattice = np.asarray(lattice, dtype=float)
    atoms = np.mod(np.asarray(positions_direct, dtype=float).reshape(-1, 3), 1.0)
    if len(atoms) == 0:
        return VoidSearchResult(sites=[], minimum_radius=0.0, neighbour_cutoff=0.0, vacuum_axes=tuple(), bounds={})

    vacuum_axes, bounds = material_bounds(lattice, atoms, vacuum_threshold=vacuum_threshold)
    shortest = _shortest_interatomic_distance(lattice, atoms)
    if minimum_radius is None:
        minimum_radius = 0.4 * shortest
    minimum_radius = float(minimum_radius)
    merge = float(merge_distance) if merge_distance is not None else max(0.5 * minimum_radius, 0.25)

    if neighbour_cutoff is not None:
        cutoff = float(neighbour_cutoff)
    else:
        # A void vertex touches four atoms that all lie at its empty-sphere
        # radius, so those four sit at most twice that radius apart.  Twice a
        # rigorous bound on the covering radius is therefore a cutoff that
        # cannot hide a void, and no growth loop is needed.
        cutoff = 2.0 * covering.bulk_covering_radius_bound(lattice, atoms, bounds) + 1e-9
    accepted: List[Tuple[np.ndarray, float, str, int]] = []
    inverse = np.linalg.inv(lattice)
    centres, radii = _empty_sphere_centres(
        lattice, atoms, cutoff, 0.5 * cutoff, include_saddles=include_saddles
    )
    # Every vertex turns up once per atom that touches it.  Collapsing the
    # exact repeats here -- the same rounding the merge pass uses, keeping the
    # largest sphere of each -- saves testing the same site four times or more.
    if len(radii):
        keys = np.round(np.mod(centres @ inverse, 1.0) * 1e6).astype(np.int64) % 1_000_000
        order = np.argsort(-radii, kind="stable")
        _, first = np.unique(keys[order], axis=0, return_index=True)
        keep = np.sort(order[first])
        centres, radii = centres[keep], radii[keep]
    if len(radii):
        # The radius and the vacuum bounds are cheap array tests, so they thin
        # the list before the enclosure test runs on what is left.
        fractional = np.mod(centres @ inverse, 1.0)
        # A coordinate a rounding error below the cell edge wraps to 1.0 rather
        # than to 0.0; reporting it as 1.0 would place the site outside the cell.
        fractional[np.abs(fractional - 1.0) < 1e-9] = 0.0
        wanted = radii >= minimum_radius
        for axis, (lower, upper) in (bounds or {}).items():
            wanted &= np.mod(fractional[:, axis] - lower, 1.0) <= (upper - lower) + 1e-9
        chosen = np.nonzero(wanted)[0]
        # The enclosure test only looks at atoms within twice the sphere
        # radius, so one image list built for the largest radius serves every
        # candidate instead of being rebuilt per candidate.
        contact_images = _atom_images(lattice, atoms, 2.0 * float(np.max(radii)))
        critical, kinds, coordination = _critical_point_classes(
            lattice, fractional[chosen], radii[chosen], contact_images
        )
        accepted = [
            (fractional[index], float(radii[index]), kinds[position], int(coordination[position]))
            for position, index in enumerate(chosen)
            if critical[position] and (include_saddles or kinds[position] == "maximum")
        ]

    # Two critical points of different kind are different sites even when they
    # sit close together, so each kind is thinned on its own.
    refined: List[Tuple[np.ndarray, float, str, int]] = []
    for kind in ("maximum", "saddle"):
        refined.extend(
            _merge_void_centres(lattice, [entry for entry in accepted if entry[2] == kind], merge)
        )
    refined.sort(key=lambda entry: -entry[1])

    sites = [
        VoidSite(
            direct=(float(centre[0]), float(centre[1]), float(centre[2])),
            cartesian=tuple(float(value) for value in centre @ lattice),
            radius=float(radius),
            kind=str(kind),
            coordination=int(contacts),
        )
        for centre, radius, kind, contacts in refined
    ]
    return VoidSearchResult(
        sites=sites,
        minimum_radius=minimum_radius,
        neighbour_cutoff=float(cutoff),
        vacuum_axes=vacuum_axes,
        bounds=bounds,
    )


def _shortest_interatomic_distance(lattice: np.ndarray, atoms_direct: np.ndarray) -> float:
    """Return the shortest distance between two atoms, or the shortest lattice vector.

    A cell list answers this in time proportional to the number of atoms, where
    the distance matrix it replaces was quadratic.
    """

    return geometry.shortest_interatomic_distance(lattice, atoms_direct)
