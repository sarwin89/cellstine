"""Provable bounds on the covering radius of a periodic point set.

The covering radius -- the radius of the largest empty sphere -- is what sizes
the neighbour cutoff of the interstitial search in :mod:`cellstine.core.voids`
and of the planar hollow search in :mod:`cellstine.core.planar_voids`.  Both
need a bound that is *certain*, or the search can silently miss a void, and both
pay for the bound being loose, because the cutoff is what their enumerations
cost.

The bound is found by branch and bound.  The distance to the nearest point is
1-Lipschitz, so its value at the centre of a box plus the reach of the box
bounds it over the whole box, while the largest value seen anywhere bounds it
from below.  A box whose own bound does not beat that lower bound cannot hold
the maximum and is dropped unrefined; the rest are cut in half along every axis
and the round repeats.  Stopping -- on the tolerance, or on the probe budget --
only loosens the answer, never invalidates it.

Everything this rests on is proved in ``aristotle-lean-reference/RequestProject/CoveringBound.lean``:
the per-box bound (``Cellstine.le_of_mem_box``,
``Cellstine.infDist_le_of_mem_box``), that the reach of a box is attained at a
corner (``Cellstine.exists_corner_bound``, ``Cellstine.norm_le_corner_sup``),
that the children of a box cover it (``Cellstine.mem_subdivision``), and that
pruning loses no maximum (``Cellstine.branch_and_bound_sound``).
"""

from __future__ import annotations

import itertools
from typing import Tuple

import numpy as np

from . import geometry

__all__ = [
    "grid_box_reach",
    "branch_and_bound_maximum",
    "NearestAtomDistance",
    "bulk_covering_radius_bound",
]


def grid_box_reach(edges: np.ndarray) -> float:
    """Return the largest distance from a point of a grid box to its centre.

    The box is the parallelepiped spanned by ``edges`` (one row per axis), and
    the point of it furthest from the centre is a corner, so the answer is half
    the longest diagonal -- the largest of ``0.5 * ||+-e0 +- e1 ...||`` over the
    sign patterns.  Summing the half-edges instead, as the triangle inequality
    allows, overstates that by up to ``sqrt(3)`` in three dimensions, and the
    overstatement goes straight into the covering-radius bound and from there
    into the neighbour cutoff of the void search.
    """

    rows = np.asarray(edges, dtype=float)
    signs = np.array(list(itertools.product((1.0, -1.0), repeat=rows.shape[0])), dtype=float)
    return 0.5 * float(np.max(np.linalg.norm(signs @ rows, axis=1)))


def branch_and_bound_maximum(
    basis: np.ndarray,
    centres: np.ndarray,
    step: np.ndarray,
    evaluate,
    *,
    tolerance: float,
    probe_budget: int = 1 << 16,
    max_depth: int = 30,
) -> float:
    """Return an upper bound on the maximum of a 1-Lipschitz function on a region.

    The region is given as boxes: ``centres`` holds the fractional centre of
    each, and every box is the image under ``basis`` of a cube of fractional
    edge lengths ``step``.  ``evaluate`` returns the value of the function at a
    batch of fractional points.

    Two facts do the work, and both are proved in
    ``aristotle-lean-reference/RequestProject/CoveringBound.lean``.  A box of centre ``c`` and reach ``r``
    -- the distance from its centre to its furthest corner, ``grid_box_reach``
    -- holds no value above ``f(c) + r``, so ``f(c) + r`` bounds the box
    (``Cellstine.le_of_mem_box``).  And the largest value seen anywhere is a
    lower bound on the maximum, so a box whose own bound does not beat it cannot
    hold the maximum and may be dropped unrefined; the remaining boxes are cut
    into ``2 ** dim`` children, which cover their parent exactly
    (``Cellstine.mem_subdivision`` and ``Cellstine.branch_and_bound_sound``).

    Refining only where the maximum can still be therefore costs a fraction of
    the probes a uniform grid of the same accuracy needs, and returns a much
    tighter bound for the probes it does spend.

    The sweep stops when the two bounds are within ``tolerance`` of each other,
    or when the next round would take the number of evaluated points past
    ``probe_budget``.  Stopping early only makes the answer looser, never wrong:
    whatever round it stops in, what it returns still bounds the whole region.
    """

    basis = np.asarray(basis, dtype=float)
    step = np.asarray(step, dtype=float).reshape(-1)
    dimension = basis.shape[0]
    centres = np.asarray(centres, dtype=float).reshape(-1, dimension)
    offsets = np.array(list(itertools.product((-0.25, 0.25), repeat=dimension)), dtype=float)
    values = np.asarray(evaluate(centres), dtype=float)
    spent = centres.shape[0]
    best = float(values.max(initial=0.0))
    reach = grid_box_reach(basis * step[:, None])
    for _ in range(int(max_depth)):
        # Boxes pruned in an earlier round were bounded by the ``best`` of that
        # round, which only ever grows, so this is a bound over the whole region.
        bound = max(best, float((values + reach).max(initial=best)))
        if bound - best <= tolerance:
            return bound
        live = (values + reach) > best
        if not np.any(live):
            return best
        children = (
            centres[live][:, None, :] + offsets[None, :, :] * step[None, None, :]
        ).reshape(-1, dimension)
        if spent + children.shape[0] > int(probe_budget):
            return bound
        centres = children
        step = step * 0.5
        reach *= 0.5
        values = np.asarray(evaluate(centres), dtype=float)
        spent += centres.shape[0]
        best = max(best, float(values.max(initial=0.0)))
    return max(best, float((values + reach).max(initial=best)))  # pragma: no cover


class NearestAtomDistance:
    """Exact distance from a point of the cell to the nearest atom image.

    Images are collected out to a range that starts at a small multiple of the
    mean atomic separation and doubles until the answer sits strictly inside it,
    at which point every probe has certainly seen its nearest atom.  Taking the
    whole cell diameter straight away would also be exact, but in a large
    supercell it would build hundreds of periodic copies to answer a question
    that is local.

    The bucket grid over the images is built once and reused, so the repeated
    batches of a branch-and-bound sweep cost only the queries.
    """

    def __init__(self, lattice: np.ndarray, atoms_direct: np.ndarray, *, factor: float = 1.5) -> None:
        self._lattice = np.asarray(lattice, dtype=float)
        self._atoms = np.asarray(atoms_direct, dtype=float).reshape(-1, 3)
        self._diameter = float(np.sum(np.linalg.norm(self._lattice, axis=1)))
        volume = abs(float(np.linalg.det(self._lattice)))
        guess = float(factor) * (volume / max(len(self._atoms), 1)) ** (1.0 / 3.0)
        self._reach = min(max(guess, 1e-6), self._diameter)
        self._rebuild()

    def _rebuild(self) -> None:
        self.images = geometry.atom_images(self._lattice, self._atoms, self._reach)
        self._query = geometry.NearestPointQuery(self.images)

    def __call__(self, fractional: np.ndarray) -> np.ndarray:
        # The distance to the nearest atom is lattice periodic, so a probe is
        # answered at its representative in the cell -- which is where the image
        # list is guaranteed complete out to ``self._reach``.
        probes = np.mod(np.asarray(fractional, dtype=float).reshape(-1, 3), 1.0) @ self._lattice
        while True:
            distances = self._query.distances(probes)
            worst = float(distances.max(initial=0.0))
            if worst < self._reach or self._reach >= self._diameter:
                return distances
            self._reach = min(2.0 * self._reach, self._diameter)
            self._rebuild()


#: Fractional edge of the coarse boxes the covering-radius sweep starts from,
#: in Angstrom.  The branch-and-bound refinement decides where to go finer, so
#: this only has to be fine enough to separate the hollows of a cell.
COARSE_BOX = 2.0

#: How many probes the sweep may spend: ``PROBE_BUDGET``, plus
#: ``PROBE_PER_ATOM`` for each atom.  The bound is valid however early the sweep
#: stops, so this only trades the tightness of the cutoff against the cost of
#: finding it; the values were chosen by timing whole searches, where a looser
#: cutoff is paid for several times over in the vertex enumeration.
PROBE_BUDGET = 10_000
PROBE_PER_ATOM = 300


def bulk_covering_radius_bound(
    lattice: np.ndarray,
    atoms_direct: np.ndarray,
    bounds: dict[int, Tuple[float, float]] | None = None,
    tolerance: float = 0.02,
) -> float:
    """Return an upper bound on the radius of the largest empty sphere.

    The distance to the nearest atom is a 1-Lipschitz function of position, so
    its value at the centre of a box plus the reach of that box bounds it over
    the whole box, and the largest value seen bounds it from below.  Starting
    from coarse boxes and refining only those that can still hold the maximum --
    :func:`branch_and_bound_maximum` -- brings the two bounds within
    ``tolerance`` of each other and returns the upper one.

    Sizing the neighbour cutoff of the exact search with a genuine bound is what
    makes that search complete: no heuristic multiple of the packing distance
    can promise the same, because a disordered cell can hold two nearly touching
    atoms next to a wide hollow.  Making the bound tight then matters for speed,
    since the cutoff it sets is what the vertex enumeration pays for.

    The sweep stays inside the material: the biggest sphere of a slab sits in
    its vacuum, and bounding that would set a cutoff far larger than any
    interstitial needs.
    """

    lattice = np.asarray(lattice, dtype=float)
    atoms = np.asarray(atoms_direct, dtype=float).reshape(-1, 3)
    if len(atoms) == 0:
        return 0.0
    lengths = np.linalg.norm(lattice, axis=1)
    lower = np.zeros(3, dtype=float)
    spans = np.ones(3, dtype=float)
    for axis, (low, high) in (bounds or {}).items():
        lower[axis] = float(low)
        spans[axis] = max(float(high) - float(low), 1e-9)
    counts = np.clip(np.ceil(lengths * spans / COARSE_BOX).astype(int), 2, 12)
    step = spans / counts.astype(float)
    axes = [
        lower[axis] + (np.arange(int(counts[axis]), dtype=float) + 0.5) * step[axis]
        for axis in range(3)
    ]
    centres = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    return branch_and_bound_maximum(
        lattice,
        centres,
        step,
        NearestAtomDistance(lattice, atoms),
        tolerance=float(tolerance),
        probe_budget=PROBE_BUDGET + PROBE_PER_ATOM * len(atoms),
    )
