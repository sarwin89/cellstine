"""Dense local clusters must not hide voids supported by farther images."""

import math

import numpy as np
import pytest

from cellstine.core import voids


def test_dense_collinear_cluster_retains_the_known_periodic_deep_hole():
    # Each atom has 201 closer collinear neighbours. A nearest-200 cap drops
    # all transverse periodic images, although they support the largest void.
    lattice = np.eye(3) * 10.0
    atoms = np.full((202, 3), 0.5)
    atoms[:, 0] = np.linspace(0.49, 0.51, len(atoms))
    # The periodic point set separates along x/y/z. Its widest gaps have
    # half-widths 4.9, 5 and 5 Angstrom, giving this exact covering radius.
    expected = math.sqrt(4.9**2 + 5.0**2 + 5.0**2)
    result = voids.find_void_sites(
        lattice, atoms, minimum_radius=1.0, vacuum_threshold=math.inf,
        neighbour_cutoff=18.0,
    )
    assert result.vacuum_axes == ()
    assert result.sites, "dense neighbours must not suppress transverse image support"
    largest = result.sites[0]
    assert largest.radius == pytest.approx(expected, abs=1e-9)
    assert largest.kind == "maximum"
    assert np.allclose(largest.direct, (0.0, 0.0, 0.0), atol=1e-9, rtol=0.0)


def test_triple_indices_do_not_materialize_a_python_tuple_per_combination(monkeypatch):
    import tracemalloc

    monkeypatch.setattr(voids, "_TRIPLE_INDEX_CACHE", {})
    tracemalloc.start()
    try:
        triples = voids._triple_indices(120)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert triples.shape == (math.comb(120, 3), 3)
    assert tuple(triples[0]) == (0, 1, 2)
    assert tuple(triples[-1]) == (117, 118, 119)
    assert peak < 12 * 1024 * 1024


def test_high_coordination_classification_avoids_a_cubic_work_array():
    import tracemalloc

    directions = np.tile(np.concatenate((np.eye(3), -np.eye(3))), (34, 1))
    tracemalloc.start()
    try:
        kind = voids._classify_contact_directions(directions)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert kind == "maximum"
    assert peak < 16 * 1024 * 1024


def _uncapped_anchor_reference(anchor, neighbours, radius_cap, include_saddles):
    import itertools

    edges = neighbours - anchor
    candidates = []
    for indices in itertools.combinations(range(len(neighbours)), 3):
        selected = edges[list(indices)]
        matrix = 2.0 * selected
        if abs(np.linalg.det(matrix)) > 1e-9:
            offset = np.linalg.solve(matrix, np.sum(selected * selected, axis=1))
            candidates.append((anchor + offset, np.linalg.norm(offset)))
    if include_saddles:
        candidates.extend((anchor + edge / 2.0, np.linalg.norm(edge) / 2.0) for edge in edges)
        for first, second in itertools.combinations(edges, 2):
            if np.linalg.norm(np.cross(first, second)) <= 1e-12:
                continue
            basis = np.array([first, second])
            # Solve the rectangular equal-distance equations directly; forming
            # the Gram matrix squares the condition number near duplicate sites.
            offset = np.linalg.lstsq(basis, np.sum(basis * basis, axis=1) / 2.0, rcond=None)[0]
            candidates.append((anchor + offset, np.linalg.norm(offset)))
    accepted = []
    for centre, radius in candidates:
        if radius <= radius_cap and np.min(np.linalg.norm(centre - neighbours, axis=1)) >= radius - 1e-6:
            accepted.append((*centre, radius))
    return np.array(accepted).reshape(-1, 4)


@pytest.mark.parametrize("include_saddles", [False, True])
@pytest.mark.parametrize("seed", [0, 7])
def test_streamed_candidates_match_independent_uncapped_enumeration(seed, include_saddles):
    anchor = np.array([2.0, -1.0, 3.0])
    neighbours = anchor + np.random.default_rng(seed).normal(size=(9, 3))
    expected = _uncapped_anchor_reference(anchor, neighbours, 2.0, include_saddles)
    centres, radii = voids._streamed_void_centres(anchor, neighbours, 2.0, include_saddles)
    actual = np.column_stack((centres, radii))
    assert len(expected)
    assert actual.shape == expected.shape
    assert np.allclose(actual, expected, rtol=0.0, atol=1e-11)


@pytest.mark.parametrize("inside", [0.999e-6, 1.001e-6])
def test_streamed_candidates_keep_the_existing_emptiness_boundary(inside):
    anchor = np.array([1.0, 0.0, 0.0])
    neighbours = np.array([[-1., 0, 0], [0, 1., 0], [0, -1., 0], [0, 0, 1.], [0, 0, -1.], [0, 0, 1. - inside]])
    expected = _uncapped_anchor_reference(anchor, neighbours, 1.1, True)
    centres, radii = voids._streamed_void_centres(anchor, neighbours, 1.1, True)
    actual = np.column_stack((centres, radii))
    assert actual.shape == expected.shape
    assert np.allclose(actual, expected, rtol=0.0, atol=1e-9)
    assert bool(np.any(np.linalg.norm(centres, axis=1) < 1e-12)) == (inside < 1e-6)


def test_large_compatible_triple_stream_is_complete_and_lexicographic():
    edges = np.zeros((203, 3))
    edges[:201, 0] = np.linspace(1.0, 1.2, 201)
    edges[201, 1] = 1.0
    edges[202, 2] = 1.0
    batches = list(voids._void_triple_batches(edges, 2.0, batch=17))
    assert all(len(batch) <= 17 for batch in batches)
    assert np.array_equal(np.concatenate(batches), np.array([[index, 201, 202] for index in range(201)]))


def test_support_bounds_retain_cospherical_and_short_edge_contacts():
    directions = np.concatenate((np.eye(3), -np.eye(3)))
    # Every plane supports the origin for anchor(1,0,0), including co-spherical
    # and near-duplicate contacts. None may be pruned merely by distance/order.
    anchor = np.array([1., 0, 0])
    neighbours = np.concatenate((directions[1:], np.array([[1., 1e-6, 0]])))
    mask, lower, upper = voids._void_support_bounds(neighbours - anchor, 1.1)
    assert mask.all()
    assert np.all(lower <= -anchor)
    assert np.all(upper >= -anchor)


def test_large_coordinate_roundoff_does_not_prune_an_accepted_pair_saddle():
    anchor = np.full(3, 1e10)
    edge = 3 * np.spacing(1e10)
    neighbours = anchor + [[edge, 0.0, 0.0]]
    radius = edge / 2.0
    midpoint = anchor + 0.5 * (neighbours[0] - anchor)
    assert np.linalg.norm(midpoint - neighbours[0]) >= radius - 1e-6
    centres, radii = voids._streamed_void_centres(anchor, neighbours, radius, True)
    assert np.array_equal(centres, midpoint[None, :])
    assert np.array_equal(radii, [radius])


def test_maximum_candidates_already_suppressed_by_merge_are_not_classified(monkeypatch):
    import itertools

    atoms = np.array(list(itertools.product(range(4), repeat=3)), dtype=float) / 4.0
    classified = []
    original = voids._critical_point_classes

    def observed(lattice, centres, radii, images):
        classified.append(len(radii))
        return original(lattice, centres, radii, images)

    monkeypatch.setattr(voids, "_critical_point_classes", observed)
    result = voids.find_void_sites(np.eye(3) * 4.0, atoms, merge_distance=4.0)
    assert len(result.sites) == 1
    assert result.sites[0].direct == (0.875, 0.875, 0.875)
    assert result.sites[0].radius == pytest.approx(math.sqrt(3.0) / 2.0, abs=1e-12)
    assert result.sites[0].coordination == 8
    assert sum(classified) == 1


def test_early_maximum_merge_preserves_equal_radius_order_and_nontransitive_chains(monkeypatch):
    points = np.zeros((4, 3))
    points[:, 0] = [0.0, 0.1, 0.2, 0.7]

    def maxima(lattice, centres, radii, images):
        return np.ones(len(radii), dtype=bool), ["maximum"] * len(radii), np.full(len(radii), 4)

    monkeypatch.setattr(voids, "_critical_point_classes", maxima)
    accepted = voids._maxima_before_merge(np.eye(3), points, np.ones(4), np.empty((0, 3)), 0.11)
    # A suppresses B, but suppressed B must not suppress C. Independent D may
    # classify together with A, yet equal radii still retain A/C/D input order.
    assert np.array_equal(np.array([entry[0] for entry in accepted]), points[[0, 2, 3]])


def test_dense_periodic_line_keeps_all_transverse_maxima():
    lattice = np.eye(3) * 10.0
    atoms = np.full((202, 3), 0.5)
    atoms[:, 0] = np.arange(len(atoms)) / len(atoms)
    result = voids.find_void_sites(
        lattice, atoms, minimum_radius=1.0, vacuum_threshold=math.inf,
        neighbour_cutoff=15.0, merge_distance=0.001,
    )
    expected_radius = math.sqrt(50.0 + (5.0 / 202)**2)
    assert len(result.sites) == 202
    assert [site.radius for site in result.sites] == pytest.approx([expected_radius] * 202, abs=1e-9)
    points = np.array([site.direct for site in result.sites])
    assert np.allclose(points[:, 1:], 0.0, atol=1e-9, rtol=0.0)
    indices = points[:, 0] * 202.0 - 0.5
    assert np.allclose(indices, np.rint(indices), atol=1e-9, rtol=0.0)
    assert np.array_equal(np.sort(np.rint(indices).astype(int) % 202), np.arange(202))
