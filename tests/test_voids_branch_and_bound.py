"""The adaptive covering-radius sweep, and the facts it rests on.

``core/covering.py`` bounds the covering radius -- and with it the neighbour cutoff
that makes the interstitial search complete -- by refining boxes only where the
maximum can still be.  The statements the refinement rests on are proved in
``RequestProject/CoveringBound.lean``; this module checks that the
implementation is the algorithm those statements describe:

* ``grid_box_reach`` really is a reach: no point of a box is further from the
  centre (``Cellstine.exists_corner_bound``, ``Cellstine.norm_le_corner_sup``);
* the children of a box cover it (``Cellstine.mem_subdivision``);
* what the sweep returns bounds the function everywhere, at whatever round it is
  stopped (``Cellstine.branch_and_bound_sound``), including when the probe
  budget cuts it short;
* and the reusable nearest-point query answers exactly what the one-shot
  function did.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from cellstine.core import covering, geometry, voids

_BOXES = [
    np.eye(3),
    np.array([[4.1, 0.0, 0.0], [3.3, 2.9, 0.0], [2.7, 1.9, 3.4]]),
    np.array([[2.46, 0.0, 0.0], [-1.23, 2.13, 0.0], [0.0, 0.0, 15.0]]),
]


@pytest.mark.parametrize("edges", _BOXES)
def test_grid_box_reach_bounds_every_point_of_the_box(edges):
    """No point of a box is further from its centre than the reach."""

    reach = covering.grid_box_reach(edges)
    rng = np.random.default_rng(4)
    offsets = rng.uniform(-0.5, 0.5, size=(4000, 3))
    distances = np.linalg.norm(offsets @ edges, axis=1)
    assert distances.max() <= reach + 1e-12
    # And it is attained: the furthest corner is at exactly the reach.
    corners = np.array(list(itertools.product((-0.5, 0.5), repeat=3)), dtype=float)
    assert np.isclose(np.linalg.norm(corners @ edges, axis=1).max(), reach)


@pytest.mark.parametrize("edges", _BOXES)
def test_grid_box_reach_is_tighter_than_the_triangle_inequality(edges):
    """Summing the half-edges is the loose bound the reach replaces."""

    assert covering.grid_box_reach(edges) <= 0.5 * np.linalg.norm(edges, axis=1).sum() + 1e-12


def test_the_children_of_a_box_cover_it():
    """Every point of a box lies in one of the eight half-sized children."""

    rng = np.random.default_rng(11)
    step = np.array([0.3, 0.5, 0.2])
    offsets = np.array(list(itertools.product((-0.25, 0.25), repeat=3)), dtype=float)
    children = offsets * step
    points = rng.uniform(-0.5, 0.5, size=(2000, 3)) * step
    inside = np.abs(points[:, None, :] - children[None, :, :]) <= 0.25 * step + 1e-12
    assert np.all(inside.all(axis=2).any(axis=1))


def _lipschitz_probe(centres_to_peaks):
    """Return a 1-Lipschitz function: the distance to the nearest peak, negated."""

    peaks = np.asarray(centres_to_peaks, dtype=float)

    def evaluate(fractional: np.ndarray) -> np.ndarray:
        points = np.asarray(fractional, dtype=float).reshape(-1, 3)
        deltas = points[:, None, :] - peaks[None, :, :]
        return -np.sqrt(np.einsum("ijk,ijk->ij", deltas, deltas)).min(axis=1)

    return evaluate


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_the_branch_and_bound_result_bounds_the_function_everywhere(seed):
    """Whatever it returns is an upper bound, checked against a dense sample."""

    rng = np.random.default_rng(seed)
    peaks = rng.random((5, 3))
    evaluate = _lipschitz_probe(peaks)
    counts = np.array([4, 4, 4])
    step = 1.0 / counts
    axes = [(np.arange(count) + 0.5) / count for count in counts]
    centres = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    bound = covering.branch_and_bound_maximum(
        np.eye(3), centres, step, evaluate, tolerance=1e-3
    )
    dense = rng.random((200000, 3))
    assert evaluate(dense).max() <= bound + 1e-12
    # The peaks are the maxima of the function, so the bound is nearly attained.
    assert bound <= evaluate(peaks).max() + 1e-2


@pytest.mark.parametrize("budget", [8, 64, 512, 4096])
def test_stopping_early_only_loosens_the_bound(budget):
    """A probe budget that cuts the sweep short still returns a valid bound."""

    rng = np.random.default_rng(17)
    peaks = rng.random((4, 3))
    evaluate = _lipschitz_probe(peaks)
    counts = np.array([3, 3, 3])
    step = 1.0 / counts
    axes = [(np.arange(count) + 0.5) / count for count in counts]
    centres = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    bound = covering.branch_and_bound_maximum(
        np.eye(3), centres, step, evaluate, tolerance=1e-6, probe_budget=budget
    )
    dense = rng.random((100000, 3))
    assert evaluate(dense).max() <= bound + 1e-12


def test_a_larger_budget_never_loosens_the_bound():
    """More probes buy a tighter bound, never a worse one."""

    rng = np.random.default_rng(23)
    peaks = rng.random((6, 3))
    evaluate = _lipschitz_probe(peaks)
    counts = np.array([3, 3, 3])
    step = 1.0 / counts
    axes = [(np.arange(count) + 0.5) / count for count in counts]
    centres = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    bounds = [
        covering.branch_and_bound_maximum(
            np.eye(3), centres, step, evaluate, tolerance=1e-9, probe_budget=budget
        )
        for budget in (32, 256, 2048, 16384)
    ]
    assert all(later <= earlier + 1e-12 for earlier, later in zip(bounds, bounds[1:]))


def test_the_sweep_finds_the_covering_radius_of_a_crystal_closely():
    """The bound has to be close to the radius it bounds, not merely above it.

    ``tests/test_voids_covering_radius.py`` pins the reported radius itself
    against an independent maximisation; what matters here is the gap, because
    the cutoff the bound sets is what the vertex enumeration pays for.  The deep
    hole of a body-centred cubic lattice is the one closed form used, at
    ``a * sqrt(5) / 4``.
    """

    lattice = np.eye(3) * 5.43
    atoms = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.0],
            [0.25, 0.25, 0.25],
            [0.25, 0.75, 0.75],
            [0.75, 0.25, 0.75],
            [0.75, 0.75, 0.25],
        ]
    )
    result = voids.find_void_sites(lattice, atoms)
    exact = max(site.radius for site in result.sites)
    bound = covering.bulk_covering_radius_bound(lattice, atoms)
    assert bound >= exact - 1e-9
    assert bound <= 1.1 * exact

    iron = np.eye(3) * 2.87
    body_centred = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
    # The deep hole of a body-centred cubic lattice has radius a * sqrt(5) / 4.
    assert np.isclose(
        max(site.radius for site in voids.find_void_sites(iron, body_centred).sites),
        2.87 * np.sqrt(5.0) / 4.0,
        atol=1e-6,
    )
    assert covering.bulk_covering_radius_bound(iron, body_centred) >= 2.87 * np.sqrt(5.0) / 4.0


def test_the_periodic_probe_answers_points_outside_the_cell():
    """The distance to the nearest atom is periodic, so a probe wraps."""

    lattice = np.array([[4.1, 0.0, 0.0], [1.3, 3.9, 0.0], [0.7, 0.9, 4.4]])
    atoms = np.random.default_rng(5).random((7, 3))
    probe = covering.NearestAtomDistance(lattice, atoms)
    rng = np.random.default_rng(6)
    fractional = rng.random((64, 3))
    shifted = fractional + rng.integers(-2, 3, size=(64, 3))
    assert np.allclose(probe(fractional), probe(shifted), atol=1e-12)


@pytest.mark.parametrize("seed", [0, 3, 9])
def test_the_reusable_query_matches_brute_force(seed):
    """``NearestPointQuery`` answers exactly, batch after batch."""

    rng = np.random.default_rng(seed)
    points = rng.normal(size=(400, 3)) * 5.0
    query = geometry.NearestPointQuery(points)
    for _ in range(3):
        queries = rng.normal(size=(200, 3)) * 6.0
        deltas = queries[:, None, :] - points[None, :, :]
        reference = np.sqrt(np.einsum("ijk,ijk->ij", deltas, deltas)).min(axis=1)
        assert np.allclose(query.distances(queries), reference, atol=1e-12)
        assert np.allclose(
            geometry.nearest_point_distances(queries, points), reference, atol=1e-12
        )


def test_the_reusable_query_handles_far_away_and_empty_inputs():
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    query = geometry.NearestPointQuery(points)
    assert np.allclose(query.distances(np.array([[500.0, 0.0, 0.0]])), 499.0)
    assert query.distances(np.zeros((0, 3))).shape == (0,)
    empty = geometry.NearestPointQuery(np.zeros((0, 3)))
    assert np.all(np.isinf(empty.distances(np.zeros((3, 3)))))
