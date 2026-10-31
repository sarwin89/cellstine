"""Probe budgets must prevent unneeded child-box allocations."""

import math
import tracemalloc

import numpy as np
import pytest

from cellstine.core.covering import branch_and_bound_maximum
from cellstine.core.planar_voids import _planar_covering_radius_bound, find_planar_voids


def test_probe_budget_is_checked_before_materializing_child_boxes():
    centres = np.zeros((50000, 3))
    evaluated = []

    def constant(points):
        evaluated.append(len(points))
        return np.zeros(len(points))

    tracemalloc.start()
    try:
        bound = branch_and_bound_maximum(
            np.eye(3), centres, np.ones(3), constant,
            tolerance=0.0, probe_budget=len(centres),
        )
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert bound == pytest.approx(math.sqrt(3.0) / 2.0)
    assert evaluated == [len(centres)]
    assert peak < 4 * 1024 * 1024


def test_nearly_touching_planar_points_do_not_expand_the_initial_probe_grid():
    basis = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    spacing = 0.001
    points = np.array([[0.0, 0.0], [spacing, 0.0]])
    tracemalloc.start()
    try:
        bound = _planar_covering_radius_bound(basis, points, spacing)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    # Largest rectangle between the periodically repeated two-point columns.
    expected = 0.5 * math.sqrt((1.0 - spacing)**2 + 1.0)
    assert expected <= bound + 1e-12 <= 1.0
    assert peak < 16 * 1024 * 1024


def test_nearly_touching_planar_points_keep_both_fourfold_hollows():
    basis = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    sites = find_planar_voids(basis, np.array([[0.0, 0.0], [0.001, 0.0]]))
    assert len(sites) == 2
    assert [site.coordination for site in sites] == [4, 4]
    assert np.allclose([site.uv for site in sites], [(0.5005, 0.5), (0.0005, 0.5)], rtol=0.0, atol=1e-10)
    assert [site.radius for site in sites] == pytest.approx([
        0.5 * math.sqrt(0.999**2 + 1.0), 0.5 * math.sqrt(0.001**2 + 1.0),
    ], abs=1e-10)
