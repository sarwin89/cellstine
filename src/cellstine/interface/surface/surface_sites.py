"""Geometry of adsorption sites on the exposed surface layer."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ...core.geometry import (
    plane_minimum_image_distances,
    plane_shift_reach,
    plane_shifts,
    shortest_plane_vector_length,
)
from ...core.layers import layer_partition
from ...core.planar_voids import PlanarVoid
from ...io import native as io_mod
from .surface_types import AdsorptionSite


def _cluster_projection_levels(values: np.ndarray, tolerance: float) -> list[tuple[float, list[int]]]:
    """Group atoms into atomic planes, bottom first, exactly as the rest of the
    package does (``core.layers.layer_partition``).

    The sites of a surface must not depend on which face is called the top, so
    the layers this reads them off are the flip-invariant single-linkage ones.
    """

    return layer_partition(values, float(tolerance))


def _inplane_cartesian_from_uv(uv: Sequence[float], lattice: np.ndarray) -> np.ndarray:
    basis = np.asarray(lattice, dtype=float)[:2]
    return float(uv[0]) * basis[0] + float(uv[1]) * basis[1]


def _deduplicate_uv_points(points_uv: Sequence[np.ndarray], lattice: np.ndarray, tolerance: float = 1e-4) -> list[np.ndarray]:
    points = [np.mod(np.asarray(point, dtype=float), 1.0) for point in points_uv]
    if not points:
        return []
    basis_2d = np.asarray(lattice, dtype=float)[:2]
    stacked = np.asarray(points, dtype=float)
    # Collapse exact (to ~1e-9) duplicate points first, preserving first
    # occurrence. This is far finer than ``tolerance`` so it cannot merge
    # genuinely distinct sites, but the candidate lists are dominated by exact
    # repeats (the same site found from many anchors), so it removes the bulk of
    # the work before the greedy minimum-image pass.
    if stacked.shape[0] > 1:
        _, first_occurrence = np.unique(np.round(stacked, 9), axis=0, return_index=True)
        order = np.sort(first_occurrence)
        candidates = stacked[order]
    else:
        candidates = stacked
    kept: list[np.ndarray] = []
    kept_array = np.empty((0, 2), dtype=float)
    for point in candidates:
        if kept_array.shape[0]:
            distances = plane_minimum_image_distances(basis_2d, point[None, :] - kept_array)
            if np.any(distances <= tolerance):
                continue
        kept.append(point)
        kept_array = np.vstack((kept_array, point[None, :]))
    return kept


def _expanded_periodic_arrays(
    points_uv: np.ndarray,
    basis_2d: np.ndarray | None = None,
    cutoff: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(points, expanded)`` where ``expanded`` holds every periodic
    image of every point that can come within ``cutoff`` of the cell, in
    base-major / shift order.

    With no basis and cutoff the box is the usual ``-1, 0, 1`` along each axis,
    which is complete only for a reduced in-plane cell.  Given them, the box is
    the one ``core.geometry.plane_shift_reach`` proves complete -- widened by
    one because the two points of a pair are themselves up to a whole cell
    apart -- so a neighbour list on a sheared cell no longer loses the
    neighbours that sit two cells away.
    """

    points = np.asarray(points_uv, dtype=float)
    if basis_2d is None or cutoff is None:
        reach = np.array([1, 1], dtype=np.int64)
    else:
        reach = plane_shift_reach(basis_2d, float(cutoff)) + 1
    shifts = plane_shifts(reach)
    expanded = (points[:, None, :] + shifts[None, :, :]).reshape(-1, 2)
    return points, expanded


def _uv_to_cartesian(uv_array: np.ndarray, basis_2d: np.ndarray) -> np.ndarray:
    """Batched ``uv -> cartesian`` matching ``_inplane_cartesian_from_uv`` exactly
    (``uv[0] * a + uv[1] * b`` with the same elementwise floating-point ops, so
    threshold comparisons are bit-identical to the original scalar code).
    """

    uv_array = np.asarray(uv_array, dtype=float)
    return uv_array[..., 0:1] * basis_2d[0] + uv_array[..., 1:2] * basis_2d[1]


def _anchor_image_distance_matrix(points: np.ndarray, expanded: np.ndarray, lattice: np.ndarray) -> np.ndarray:
    """``(n_points, 9 * n_points)`` Cartesian distances from each point to every
    periodic image, matching ``norm(_inplane_cartesian_from_uv(image - point))``.
    """

    basis_2d = np.asarray(lattice, dtype=float)[:2]
    displacement = expanded[None, :, :] - points[:, None, :]
    cartesian = _uv_to_cartesian(displacement, basis_2d)
    return np.linalg.norm(cartesian, axis=2)


def _nearest_neighbor_distance(points_uv: np.ndarray, lattice: np.ndarray) -> float:
    """Return the shortest in-plane distance between two surface atoms.

    Exactly, and without a search box: a pair of *distinct* atoms is measured
    with ``core.geometry.plane_minimum_image_distances``, and a single atom is
    its own nearest neighbour through the boundary, at the shortest translation
    of the in-plane lattice.  A fixed ``-1, 0, 1`` box misses both on a sheared
    cell, and every site name downstream -- the neighbour cutoff, the bridges,
    the hollow depth tolerance -- is scaled by this number.
    """

    points = np.asarray(points_uv, dtype=float)
    if points.shape[0] == 0:
        raise ValueError("could not determine an in-plane nearest-neighbour distance from the top surface atoms")
    basis_2d = np.asarray(lattice, dtype=float)[:2]
    shortest = shortest_plane_vector_length(basis_2d)
    if points.shape[0] > 1:
        rows, columns = np.triu_indices(points.shape[0], k=1)
        pairs = plane_minimum_image_distances(basis_2d, points[rows] - points[columns])
        distinct = pairs[pairs > 1e-8]
        if distinct.size:
            shortest = min(shortest, float(distinct.min()))
    if not np.isfinite(shortest) or shortest <= 1e-8:
        raise ValueError("could not determine an in-plane nearest-neighbour distance from the top surface atoms")
    return float(shortest)


def _top_layer_coordination_counts(points_uv: np.ndarray, lattice: np.ndarray, neighbour_cutoff: float) -> list[int]:
    points, expanded = _expanded_periodic_arrays(
        points_uv, np.asarray(lattice, dtype=float)[:2], neighbour_cutoff
    )
    if points.shape[0] == 0:
        return []
    distances = _anchor_image_distance_matrix(points, expanded, lattice)
    within = (distances > 1e-8) & (distances <= neighbour_cutoff + 1e-12)
    return [int(value) for value in within.sum(axis=1)]


def _find_bridge_sites(points_uv: np.ndarray, lattice: np.ndarray, neighbour_cutoff: float) -> list[np.ndarray]:
    points, expanded = _expanded_periodic_arrays(
        points_uv, np.asarray(lattice, dtype=float)[:2], neighbour_cutoff
    )
    if points.shape[0] == 0:
        return []
    distances = _anchor_image_distance_matrix(points, expanded, lattice)
    within = (distances > 1e-8) & (distances <= neighbour_cutoff + 1e-12)
    anchor_idx, image_idx = np.nonzero(within)
    if anchor_idx.size == 0:
        return []
    midpoints = np.mod(0.5 * (points[anchor_idx] + expanded[image_idx]), 1.0)
    return _deduplicate_uv_points(list(midpoints), lattice)


def _projection_layer_points(
    positions_direct: np.ndarray,
    projections: np.ndarray,
    side: str,
    tolerance: float,
    max_layers: int | None = None,
) -> list[tuple[float, np.ndarray]]:
    """Return ``(projection, in-plane fractional points)`` for every atomic layer,
    ordered from the exposed surface inwards.  All layers are returned unless
    ``max_layers`` caps the list, so the reported layer count is the real one.
    """

    groups = _cluster_projection_levels(projections, tolerance)
    ordered = sorted(groups, key=lambda item: item[0], reverse=(side == "top"))
    if max_layers is not None:
        ordered = ordered[: int(max_layers)]
    layers: list[tuple[float, np.ndarray]] = []
    for center, indices in ordered:
        layers.append((center, np.mod(np.asarray(positions_direct[indices, :2], dtype=float), 1.0)))
    return layers


def _subsurface_depth_below(
    hollow_uv: np.ndarray,
    lattice: np.ndarray,
    lower_layers: Sequence[tuple[float, np.ndarray]],
    match_tolerance: float,
) -> int | None:
    """Index (1 = first subsurface layer) of the shallowest layer holding an atom
    directly beneath ``hollow_uv``, or ``None`` when the column stays empty.
    """

    basis_2d = np.asarray(lattice, dtype=float)[:2]
    target = np.asarray(hollow_uv, dtype=float)[None, :]
    for depth, (_, layer_points) in enumerate(lower_layers, start=1):
        points = np.asarray(layer_points, dtype=float)
        if points.size == 0:
            continue
        if float(plane_minimum_image_distances(basis_2d, target - points).min()) <= match_tolerance:
            return int(depth)
    return None


def _is_close_packed_stacking(layers: Sequence[tuple[float, np.ndarray]], tolerance: float = 0.25) -> bool:
    """True when the slab stacks like an fcc/hcp close-packed crystal: evenly
    spaced layers each holding the same number of atoms.

    Only then do the names ``fcc_hollow`` and ``hcp_hollow`` mean anything.  A
    buckled bilayer such as Si(111) fails the test, and its hollows are reported
    generically together with the depth of the atom underneath, which is the
    honest description.
    """

    if len(layers) < 3:
        return False
    counts = {int(np.asarray(points).shape[0]) for _, points in layers}
    if len(counts) != 1:
        return False
    spacings = np.abs(np.diff([float(level) for level, _ in layers]))
    if spacings.size < 2:
        return False
    mean = float(np.mean(spacings))
    if mean <= 1e-8:
        return False
    return bool(np.max(np.abs(spacings - mean)) / mean <= float(tolerance))


def _classify_hollow(
    hollow: PlanarVoid,
    depth: int | None,
    close_packed: bool,
) -> str:
    """Name a hollow from its coordination and from what sits below it."""

    if hollow.coordination >= 4:
        return "fourfold_hollow" if hollow.coordination == 4 else "hollow"
    if close_packed and depth == 1:
        return "hcp_hollow"
    if close_packed and depth == 2:
        return "fcc_hollow"
    return "hollow"


def _site_from_uv(
    site_type: str,
    uv: np.ndarray,
    lattice: np.ndarray,
    plane_projection: float,
    normal: np.ndarray,
    *,
    coordination: int | None = None,
    void_radius: float | None = None,
    subsurface_depth: int | None = None,
) -> AdsorptionSite:
    cartesian = _inplane_cartesian_from_uv(uv, lattice) + float(plane_projection) * np.asarray(normal, dtype=float)
    direct = io_mod.wrap_direct(io_mod.cartesian_to_direct(cartesian.reshape(1, 3), lattice))[0]
    return AdsorptionSite(
        site_type=str(site_type),
        direct=(float(direct[0]), float(direct[1]), float(direct[2])),
        cartesian=(float(cartesian[0]), float(cartesian[1]), float(cartesian[2])),
        coordination=None if coordination is None else int(coordination),
        void_radius=None if void_radius is None else float(void_radius),
        subsurface_depth=None if subsurface_depth is None else int(subsurface_depth),
    )
