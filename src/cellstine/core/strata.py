"""Symmetry strata of the Brillouin zone.

A wavevector is carried, as in :mod:`cellstine.core.reciprocal`, as a row ``k``
of fractional reciprocal coordinates, and a crystal operation ``x -> W x + w``
acts on it by the integer matrix ``M = W^-1`` on the right, ``k -> k M``.  Time
reversal contributes ``-M``.  This module is the linear algebra of that action:

* the little co-group ``L(k) = { M : k M = k mod Z^3 }`` of a wavevector,
* its fixed space ``V(k) = { v : v M = v for every M in L(k) }``, whose
  dimension says whether ``k`` is an isolated point, on a symmetry line, in a
  mirror plane, or generic,
* the exhaustive sweep of the grid of denominator :data:`GRID_DENOMINATOR` that
  finds every isolated point,
* which pairs of points are joined by a bare piece of symmetry line, and which
  stratum the interior of a segment lies in.

Nothing here knows about names, Bravais types or paths; that is
:mod:`cellstine.core.kpath`, which is where the prose account of the derivation
(``core/KPATH.md``) and the formal statements (``aristotle-lean-reference/RequestProject/KPath.lean``)
are pointed at from.

Everything is written in bulk: the "which operations fix me" rows are one
``einsum``, distinct rows are found by packing them into bytes, and one fixed
space is computed per *distinct* little co-group rather than per point, which is
what makes the ``48**3`` sweep affordable.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

__all__ = [
    "GRID_DENOMINATOR",
    "kspace_operations",
    "stratum_dimension",
    "grid_strata",
    "symmetry_edges",
    "segment_strata",
]


GRID_DENOMINATOR = 48
"""Denominator of the search grid; see :mod:`cellstine.core.kpath` for why it suffices."""


def _as_row_matrices(operations: Sequence[Sequence[Sequence[int]]]) -> np.ndarray:
    matrices = np.asarray(operations, dtype=np.int64).reshape(-1, 3, 3)
    if matrices.shape[0] == 0:
        raise ValueError("at least one operation is needed")
    determinants = np.rint(np.linalg.det(matrices.astype(float))).astype(np.int64)
    if np.any(np.abs(determinants) != 1):
        raise ValueError("a point-group operation must be unimodular")
    return matrices


def kspace_operations(
    rotations: Sequence[Sequence[Sequence[int]]],
    *,
    time_reversal: bool = True,
) -> np.ndarray:
    """Return the matrices acting on fractional reciprocal coordinates.

    ``rotations`` are the integer matrices ``W`` acting on *column* fractional
    coordinates, as returned by :func:`cellstine.core.symmetry3d.lattice_point_group`
    or by a symmetry analysis of the crystal.  The returned matrices act on
    wavevector rows on the right, ``k -> k M``, with ``M = W^-1``; time reversal
    contributes ``-M`` as well, since bands satisfy ``E(k) = E(-k)``.
    """

    matrices = _as_row_matrices(rotations)
    inverses = np.rint(np.linalg.inv(matrices.astype(float))).astype(np.int64)
    if not np.allclose(np.einsum("kij,kjl->kil", matrices, inverses), np.eye(3), atol=1e-8):
        raise ValueError("the operations are not invertible over the integers")
    if time_reversal:
        inverses = np.concatenate([inverses, -inverses])
    unique = np.unique(inverses.reshape(-1, 9), axis=0)
    return unique.reshape(-1, 3, 3)


def _fixed_space_dimension(matrices: np.ndarray) -> tuple[int, np.ndarray | None]:
    """Return ``dim V`` and, when it is one, a direction spanning ``V``."""

    stacked = np.concatenate([matrix - np.eye(3, dtype=np.int64) for matrix in matrices], axis=1)
    array = stacked.astype(float)
    _, singular, right = np.linalg.svd(array.T)
    largest = float(singular[0]) if singular.size else 0.0
    rank = int(np.sum(singular > 1e-9 * max(largest, 1.0)))
    dimension = 3 - rank
    if dimension != 1:
        return dimension, None
    return dimension, right[-1]


def stratum_dimension(point: Sequence[float], operations: np.ndarray, *, tolerance: float = 1e-8) -> int:
    """Return the dimension of the symmetry stratum through ``point``."""

    wavevector = np.asarray(point, dtype=float).reshape(3)
    images = np.einsum("j,kjl->kl", wavevector, operations.astype(float))
    residual = images - wavevector
    keep = np.all(np.abs(residual - np.rint(residual)) <= tolerance, axis=1)
    dimension, _ = _fixed_space_dimension(operations[keep])
    return dimension


def _primitive_direction(vector: np.ndarray) -> np.ndarray | None:
    """Return the shortest integer vector along ``vector``, if it is rational."""

    values = np.asarray(vector, dtype=float)
    largest = float(np.max(np.abs(values)))
    if largest <= 0.0:  # pragma: no cover - defensive
        return None
    scaled = values / largest
    for denominator in range(1, GRID_DENOMINATOR + 1):
        trial = scaled * denominator
        if np.allclose(trial, np.rint(trial), atol=1e-6):
            integers = np.rint(trial).astype(np.int64)
            divisor = int(np.gcd.reduce(np.abs(integers[integers != 0])))
            integers = integers // divisor
            if integers[int(np.argmax(np.abs(integers)))] < 0:
                integers = -integers
            return integers
    return None  # pragma: no cover - a crystallographic axis is always rational


def _unique_masks(masks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the distinct rows of a boolean matrix, and each row's pattern.

    The rows are packed into bytes before they are sorted.  A little co-group is
    a subset of at most forty-eight operations, so a row fits in six bytes, and
    grouping six-byte keys costs a small fraction of grouping the rows
    themselves -- which matters, because the search grid has ``48**3`` rows.
    """

    if masks.shape[0] == 0:  # pragma: no cover - defensive
        return masks[:0], np.zeros(0, dtype=np.int64)
    packed = np.ascontiguousarray(np.packbits(masks, axis=1))
    view = packed.view([("", packed.dtype)] * packed.shape[1]).ravel()
    _, first, inverse = np.unique(view, return_index=True, return_inverse=True)
    return masks[first], inverse.reshape(-1)


def _fixing_masks(points: np.ndarray, operations: np.ndarray, tolerance: float = 1e-8) -> np.ndarray:
    """Return, for each point, which operations fix it modulo a lattice vector."""

    images = np.einsum("pj,kjl->pkl", np.asarray(points, dtype=float), operations.astype(float))
    residual = images - np.asarray(points, dtype=float)[:, None, :]
    return np.all(np.abs(residual - np.rint(residual)) <= tolerance, axis=2)


def _stratum_dimensions(points: np.ndarray, operations: np.ndarray) -> np.ndarray:
    """Return the stratum dimension of every point of ``points``.

    Points sharing a little co-group share a stratum dimension, so the fixed
    space is computed once per distinct co-group rather than once per point.
    """

    masks = _fixing_masks(points, operations)
    patterns, inverse = _unique_masks(masks)
    dimensions = np.asarray(
        [_fixed_space_dimension(operations[pattern])[0] for pattern in patterns], dtype=np.int64
    )
    return dimensions[inverse]


def symmetry_edges(
    points: np.ndarray,
    cartesian: np.ndarray,
    operations: np.ndarray,
    scale: float,
) -> set[tuple[int, int]]:
    """Return the node pairs joined by a bare piece of symmetry line.

    Two nodes are joined when the midpoint between them lies on a symmetry line
    -- which, the two ends being on it as well, makes the whole segment part of
    the line -- and no third node lies between them on it, so that the segment
    is a piece of the line and not a shortcut across several.
    """

    count = len(points)
    edges: set[tuple[int, int]] = set()
    if count < 2:  # pragma: no cover - defensive
        return edges
    pairs = np.transpose(np.triu_indices(count, k=1))
    block_size = max(1, int(1_000_000 // count))
    for start in range(0, len(pairs), block_size):
        block = pairs[start : start + block_size]
        middles = (points[block[:, 0]] + points[block[:, 1]]) / 2.0
        block = block[_stratum_dimensions(middles, operations) == 1]
        if not len(block):
            continue
        first, second = block[:, 0], block[:, 1]
        direction = points[second] - points[first]
        offsets = points[None, :, :] - points[first][:, None, :]
        area = np.linalg.norm(
            np.cross(
                cartesian[None, :, :] - cartesian[first][:, None, :],
                (cartesian[second] - cartesian[first])[:, None, :],
            ),
            axis=2,
        )
        fraction = np.einsum("mnj,mj->mn", offsets, direction) / np.einsum(
            "mj,mj->m", direction, direction
        )[:, None]
        between = (area <= 1e-6 * scale) & (fraction > 1e-9) & (fraction < 1.0 - 1e-9)
        rows = np.arange(len(block))
        between[rows, first] = False
        between[rows, second] = False
        for pair in block[~np.any(between, axis=1)]:
            edges.add((int(pair[0]), int(pair[1])))
    return edges


_SEGMENT_SAMPLES: tuple[float, ...] = (1.0 / 3.0, 1.0 / 2.0, 2.0 / 3.0)


def segment_strata(
    walk_points: Sequence[np.ndarray],
    operations: np.ndarray,
) -> tuple[int, ...]:
    """Return the stratum dimension of the interior of every segment walked.

    A segment is classified by where its *interior* sits, not by the names of
    its ends: a segment whose interior is fixed by enough operations to leave a
    one-dimensional fixed space runs along a symmetry line, one with a
    two-dimensional fixed space lies in a mirror plane, and one with a
    three-dimensional fixed space is a plain chord of the zone.  The interior is
    sampled at three interior fractions so that an accidental crossing of a
    higher-symmetry point does not decide the segment; the largest dimension
    seen wins, and an isolated fixed point (dimension zero) is reported as a
    line, since a segment cannot consist of one point.
    """

    samples: list[np.ndarray] = []
    for run in walk_points:
        coordinates = np.asarray(run, dtype=float)
        for position in range(len(coordinates) - 1):
            start = coordinates[position]
            end = coordinates[position + 1]
            for fraction in _SEGMENT_SAMPLES:
                samples.append(start + fraction * (end - start))
    if not samples:  # pragma: no cover - defensive
        return ()
    dimensions = _stratum_dimensions(np.asarray(samples, dtype=float), operations)
    grouped = dimensions.reshape(-1, len(_SEGMENT_SAMPLES)).max(axis=1)
    return tuple(int(max(value, 1)) for value in grouped)


_STRATA_CACHE: dict[bytes, tuple[np.ndarray, np.ndarray, list[np.ndarray | None]]] = {}


def grid_strata(operations: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[np.ndarray | None]]:
    """Classify every point of the search grid by its stratum.

    Returns the integer grid (over the denominator :data:`GRID_DENOMINATOR`),
    the dimension of the stratum through each point, and, for the points on a
    symmetry line, the integer direction of that line.

    The classification depends on the point group alone, and the grid is the
    same ``48**3`` points every time, so the answer is kept for reuse: a session
    that asks for the points and then for a path, or for several paths of one
    lattice, sweeps the grid once.
    """

    key = np.ascontiguousarray(operations, dtype=np.int64).tobytes()
    cached = _STRATA_CACHE.get(key)
    if cached is not None:
        return cached

    denominator = GRID_DENOMINATOR
    span = np.arange(denominator, dtype=np.int64)
    grid = np.stack(np.meshgrid(span, span, span, indexing="ij"), axis=-1).reshape(-1, 3)
    masks = np.zeros((len(grid), len(operations)), dtype=bool)
    for index, matrix in enumerate(operations):
        masks[:, index] = np.all((grid @ matrix - grid) % denominator == 0, axis=1)
    patterns, inverse = _unique_masks(masks)
    dimensions = np.zeros(len(patterns), dtype=np.int64)
    directions: list[np.ndarray | None] = [None] * len(patterns)
    for index, pattern in enumerate(patterns):
        dimension, direction = _fixed_space_dimension(operations[pattern])
        dimensions[index] = dimension
        if direction is not None:
            directions[index] = _primitive_direction(direction)
    inverse = inverse.reshape(-1)
    grid.setflags(write=False)
    strata = dimensions[inverse]
    strata.setflags(write=False)
    result = (grid, strata, [directions[position] for position in inverse])
    if len(_STRATA_CACHE) >= 4:  # pragma: no cover - a session uses one or two groups
        _STRATA_CACHE.clear()
    _STRATA_CACHE[key] = result
    return result
