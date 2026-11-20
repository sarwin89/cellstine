"""Pair discovery must not allocate a full host-by-host table."""

from __future__ import annotations

import tracemalloc

import numpy as np
import pytest

from cellstine.core import geometry
from cellstine.defect.sites import DefectSiteEnumerationMixin, _divacancy_pairs
from cellstine.io.models import StructureRecord


def _record(direct, lattice=None):
    lattice = np.eye(3) if lattice is None else np.asarray(lattice, dtype=float)
    direct = np.asarray(direct, dtype=float).reshape(-1, 3)
    return StructureRecord("pair resources", lattice, ["C"], [len(direct)], direct, direct @ lattice)


def _sites(record, cutoff):
    return DefectSiteEnumerationMixin()._divacancy_sites(
        record, backend="native", symprec=.01, divacancy_distance=cutoff
    )


def test_sparse_pair_discovery_has_bounded_temporary_storage():
    direct = np.indices((10, 10, 10)).reshape(3, -1).T / 10
    record = _record(direct, 100 * np.eye(3))
    # Warm the shared kernel/cache outside the allocation observation.
    geometry.minimum_image_distances(record.lattice, np.array([[.1, 0, 0]]))
    tracemalloc.start()
    try:
        assert _sites(record, .01) == []
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 16 * 1024**2, f"no-pair discovery allocated {peak} bytes"


def test_pair_discovery_limits_real_kernel_strips(monkeypatch):
    direct = np.indices((9, 9, 9)).reshape(3, -1).T / 9
    record = _record(direct, 100 * np.eye(3))
    original = geometry.pairwise_minimum_image_distances
    shapes = []

    def observe(lattice, left, *, other_direct=None, **kwargs):
        right = left if other_direct is None else other_direct
        shapes.append((len(left), len(right)))
        return original(lattice, left, other_direct=other_direct, **kwargs)

    monkeypatch.setattr(geometry, "pairwise_minimum_image_distances", observe)
    assert _sites(record, .01) == []
    assert len(shapes) > 1
    assert max(a * b for a, b in shapes) <= 65536
    assert sum(a * b for a, b in shapes) < len(direct)**2 * .65


@pytest.mark.parametrize("cutoff", [-1., 0., .25, .5, np.inf, np.nan])
def test_pair_members_keep_upper_triangle_order_and_inclusive_cutoff(cutoff):
    direct = np.array([[0, 0, 0], [0, 0, 0], [.25, 0, 0], [.5, 0, 0], [.75, 0, 0]])
    record = _record(direct)
    distances = geometry.pairwise_minimum_image_distances(record.lattice, direct)
    groups = {}
    for i in range(len(direct)):
        for j in range(i + 1, len(direct)):
            if distances[i, j] <= cutoff:
                groups.setdefault(round(float(distances[i, j]), 2), []).append([i + 1, j + 1])
    actual = _sites(record, cutoff)
    assert [[member["indices"] for member in site.members] for site in actual] == list(groups.values())
    assert [site.pair_indices for site in actual] == [members[0] for members in groups.values()]
    assert [site.multiplicity for site in actual] == [len(members) for members in groups.values()]


@pytest.mark.parametrize("count", [0, 1])
def test_empty_and_single_atom_hosts_have_no_pairs(count):
    assert _sites(_record(np.zeros((count, 3))), np.inf) == []


def test_skew_cell_pairs_use_the_shortest_periodic_image():
    lattice = np.array([[1., 0, 0], [.8, .4, 0], [0, 0, 2.]])
    direct = np.array([[0., 0, 0], [.49, .49, 0]])
    # Rounding fractional deltas gives .903..., while the true image is .228...
    assert np.linalg.norm(direct[1] @ lattice) > .8
    sites = _sites(_record(direct, lattice), .3)
    assert len(sites) == 1
    assert sites[0].pair_indices == [1, 2]
    midpoint = geometry.periodic_midpoints(lattice, direct[0], direct[1])[0]
    np.testing.assert_array_equal(sites[0].direct, midpoint)


@pytest.mark.parametrize("count", [2, 7, 257, 513])
@pytest.mark.parametrize("cutoff", [.0, .15, np.inf])
def test_multiblock_pair_values_are_byte_exact_and_ordered(count, cutoff):
    rng = np.random.default_rng(817)
    direct = rng.random((count, 3))
    direct[-1] = direct[0]
    lattice = np.array([[1., 0, 0], [.7, .6, 0], [.2, .1, 1.]])
    distances = geometry.pairwise_minimum_image_distances(lattice, direct)
    i, j = np.triu_indices(count, k=1)
    keep = distances[i, j] <= cutoff
    expected = [(int(a), int(b), float(d)) for a, b, d in zip(i[keep], j[keep], distances[i[keep], j[keep]])]
    actual = _divacancy_pairs(direct, lattice, cutoff)
    assert [(a, b, d.hex()) for a, b, d in actual] == [(a, b, d.hex()) for a, b, d in expected]


@pytest.mark.parametrize("direction", [-np.inf, np.inf])
def test_pair_cutoff_nextafter_keeps_existing_boundary(direction):
    direct = np.array([[0., 0., 0.], [.25, 0., 0.]])
    cutoff = np.nextafter(.25, direction)
    pairs = _divacancy_pairs(direct, np.eye(3), cutoff)
    assert bool(pairs) is (direction > 0)
