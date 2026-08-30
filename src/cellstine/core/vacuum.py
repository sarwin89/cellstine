"""The vacuum gap of a slab, and how to keep it when atoms are added.

A slab cell is periodic in all three directions, so what separates a slab from
its own image along ``c`` is not the length of ``c`` but the *gap*: the height
of the cell measured along the surface normal, minus the span the atoms occupy
along that normal.

.. code-block:: text

    n    = a x b / ‖a x b‖          surface normal
    h    = |c · n|                  height of the cell along n
    span = max(p · n) - min(p · n)  over the atoms p
    gap  = h - span

That gap is the only quantity a plane-wave calculation feels, and it is what
every CELLSTINE stage reports and preserves.  Adding an adsorbate or an adatom
to a finished slab eats into it, so :func:`fit_cell_to_vacuum` lengthens ``c``
-- keeping its direction, hence the shape of the cell -- until the gap is back
to what it was, and rigidly translates the atoms so that the assembly keeps its
place inside the cell.

``aristotle-lean-reference/RequestProject/VacuumGap.lean`` proves what the two operations here do:
lengthening ``c`` along its own direction leaves the normal and the span alone
and moves the gap by exactly the added height, a rigid translation moves no
atom relative to another, and the fitted cell therefore has the requested gap
(or its original one, when that was already larger).
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "cell_height",
    "fit_cell_to_vacuum",
    "normal_heights",
    "occupied_span",
    "surface_normal",
    "vacuum_gap",
]


def surface_normal(lattice: np.ndarray) -> np.ndarray:
    """Return the unit normal of the ``a``-``b`` plane, oriented along ``+c``."""

    rows = np.asarray(lattice, dtype=float).reshape(3, 3)
    normal = np.cross(rows[0], rows[1])
    length = float(np.linalg.norm(normal))
    if length <= 0.0:
        raise ValueError("the in-plane lattice vectors are parallel; no surface normal exists")
    normal = normal / length
    if float(np.dot(rows[2], normal)) < 0.0:
        normal = -normal
    return normal


def cell_height(lattice: np.ndarray) -> float:
    """Return the height of the cell along its surface normal."""

    rows = np.asarray(lattice, dtype=float).reshape(3, 3)
    return abs(float(np.dot(rows[2], surface_normal(rows))))


def normal_heights(lattice: np.ndarray, positions_cartesian: np.ndarray) -> np.ndarray:
    """Return how high each atom sits above the ``a``-``b`` plane.

    Layers of a slab are planes parallel to ``a`` and ``b``, so the coordinate
    that tells them apart is the projection onto the surface normal, not the
    Cartesian ``z``.  The two agree exactly for the convention every CELLSTINE
    stage writes -- ``a`` along ``x``, ``b`` in the ``xy`` plane -- and differ
    for any other orientation of the same structure.  Because the normal is
    oriented along ``+c``, the height also grows from the bottom of the cell to
    its top whichever way round the basis is, so a left-handed cell (one with a
    negative determinant, which VASP accepts) does not read its layers upside
    down.
    """

    points = np.asarray(positions_cartesian, dtype=float).reshape(-1, 3)
    return points @ surface_normal(lattice)


def occupied_span(lattice: np.ndarray, positions_cartesian: np.ndarray) -> tuple[float, float]:
    """Return ``(lowest, highest)`` atomic projection onto the surface normal."""

    points = np.asarray(positions_cartesian, dtype=float).reshape(-1, 3)
    if points.size == 0:
        return 0.0, 0.0
    projections = normal_heights(lattice, points)
    return float(projections.min()), float(projections.max())


def vacuum_gap(lattice: np.ndarray, positions_cartesian: np.ndarray) -> float:
    """Return the empty height between the slab and its periodic image."""

    lowest, highest = occupied_span(lattice, positions_cartesian)
    return max(cell_height(lattice) - (highest - lowest), 0.0)


def fit_cell_to_vacuum(
    lattice: np.ndarray,
    positions_cartesian: np.ndarray,
    target_gap: float,
    *,
    anchor: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Lengthen ``c`` until the vacuum gap reaches ``target_gap``.

    ``c`` only ever grows, and only along its own direction, so a cell that
    already has enough vacuum is returned untouched.  The atoms are translated
    rigidly along the normal so that the lowest of them sits at ``anchor``
    (by default where the lowest atom already is), which keeps an assembly in
    place when the new atoms hang below the slab.

    Returns the new lattice and the new Cartesian positions.
    """

    rows = np.asarray(lattice, dtype=float).reshape(3, 3).copy()
    points = np.asarray(positions_cartesian, dtype=float).reshape(-1, 3)
    if float(target_gap) < 0.0:
        raise ValueError("target_gap must be non-negative")
    normal = surface_normal(rows)
    height = abs(float(np.dot(rows[2], normal)))
    if points.size == 0:
        return rows, points.copy()

    lowest, highest = occupied_span(rows, points)
    span = highest - lowest
    required = span + float(target_gap)
    if required > height + 1e-12:
        scale = required / height
        rows[2] = rows[2] * scale

    target_low = lowest if anchor is None else float(anchor)
    shifted = points + (target_low - lowest) * normal
    return rows, shifted
