"""Mathematical checks on the shared periodic-geometry primitives.

The three primitives of :mod:`cellstine.core.geometry` -- the exact minimum
image, the bucketed site index and the Cartesian cell list -- are each compared
against an independent brute-force enumeration on cells that are deliberately
awkward: hexagonal, strongly sheared, and slab-shaped with a long vacuum axis.
The point of the module is that no shortcut is allowed to lose a neighbour, so
the tests assert exact agreement, not agreement to within a tolerance.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from cellstine.core import geometry


HEXAGONAL = np.array(
    [[2.46, 0.0, 0.0], [-1.23, 1.23 * math.sqrt(3.0), 0.0], [0.0, 0.0, 20.0]]
)
SHEARED = np.array([[5.0, 0.0, 0.0], [4.9, 1.0, 0.0], [0.2, 0.3, 4.0]])
CUBIC = 3.9 * np.eye(3)
CELLS = {"cubic": CUBIC, "hexagonal": HEXAGONAL, "sheared": SHEARED}


def brute_force_minimum_image(lattice: np.ndarray, deltas: np.ndarray, reach: int = 5) -> np.ndarray:
    """Return the shortest image length of each row by explicit enumeration."""

    shifts = np.array(list(itertools.product(range(-reach, reach + 1), repeat=3)), dtype=float)
    candidates = (deltas[:, None, :] - shifts[None, :, :]) @ np.asarray(lattice, dtype=float)
    return np.linalg.norm(candidates, axis=2).min(axis=1)


@pytest.mark.parametrize("name", sorted(CELLS))
def test_minimum_image_agrees_with_brute_force(name):
    """The reported shortest image is the shortest image, in every cell."""

    lattice = CELLS[name]
    deltas = np.random.default_rng(20240921).uniform(-2.0, 2.0, size=(400, 3))
    assert np.allclose(
        geometry.minimum_image_distances(lattice, deltas),
        brute_force_minimum_image(lattice, deltas),
        atol=1e-9,
    )


def test_rounding_the_fractional_difference_is_not_the_minimum_image():
    """The textbook ``delta - rint(delta)`` shortcut is wrong in a skewed cell.

    This is why the library computes the shortest image instead of rounding:
    in a hexagonal cell the shortcut can overstate a separation by 30 %.
    """

    delta = np.array([[0.4, 0.6, 0.0]])
    rounded = float(np.linalg.norm((delta - np.rint(delta)) @ HEXAGONAL))
    exact = float(geometry.minimum_image_distances(HEXAGONAL, delta)[0])
    assert exact == pytest.approx(brute_force_minimum_image(HEXAGONAL, delta)[0])
    assert rounded > exact * 1.3


def test_minimum_image_displacement_is_an_image_of_the_input():
    """The returned vector differs from the input by a whole lattice vector."""

    deltas = np.random.default_rng(7).uniform(-3.0, 3.0, size=(50, 3))
    vectors = geometry.minimum_image_displacements(SHEARED, deltas)
    residue = vectors @ np.linalg.inv(SHEARED) - deltas
    assert np.allclose(residue, np.rint(residue), atol=1e-9)


@pytest.mark.parametrize("name", sorted(CELLS))
def test_pairwise_distance_matrix_is_symmetric_and_exact(name):
    lattice = CELLS[name]
    points = np.random.default_rng(3).uniform(0.0, 1.0, size=(40, 3))
    matrix = geometry.pairwise_minimum_image_distances(lattice, points)
    assert np.allclose(matrix, matrix.T, atol=1e-12)
    assert np.allclose(np.diag(matrix), 0.0, atol=1e-12)
    reference = brute_force_minimum_image(
        lattice, (points[:, None, :] - points[None, :, :]).reshape(-1, 3)
    ).reshape(40, 40)
    assert np.allclose(matrix, reference, atol=1e-9)


def test_image_shift_reach_covers_every_short_lattice_vector():
    """No lattice vector shorter than the cutoff lies outside the reported box."""

    cutoff = 7.0
    for lattice in CELLS.values():
        reach = geometry.image_shift_reach(lattice, cutoff)
        wide = np.array(
            list(itertools.product(*(range(-int(value) - 2, int(value) + 3) for value in reach))),
            dtype=float,
        )
        lengths = np.linalg.norm(wide @ lattice, axis=1)
        inside = np.all(np.abs(wide) <= reach[None, :], axis=1)
        assert np.all(lengths[~inside] > cutoff)


@pytest.mark.parametrize("name", sorted(CELLS))
def test_bounded_minimum_image_is_exact_inside_the_radius(name):
    """The screened distance is the true one wherever it claims to be inside.

    ``bounded_minimum_image_squared`` skips the image search for rows the
    shortest-lattice-vector argument already settles.  What it promises is that
    a row it reports at or below the radius carries the exact minimum image, and
    that a row it reports above the radius really is further away than that --
    which is precisely what the site matching relies on.
    """

    lattice = CELLS[name]
    radius = 0.25 * geometry.shortest_lattice_vector_length(lattice)
    rng = np.random.default_rng(4242)
    # A mixture of near-coincident rows, near-lattice-vector rows and generic
    # ones, so that all three branches of the routine are exercised.
    near = rng.normal(scale=1e-3, size=(150, 3))
    images = np.rint(rng.uniform(-2.0, 2.0, size=(150, 3))) + rng.normal(scale=1e-3, size=(150, 3))
    generic = rng.uniform(-1.5, 1.5, size=(200, 3))
    deltas = np.vstack([near, images, generic])

    reported = np.sqrt(geometry.bounded_minimum_image_squared(lattice, deltas, radius))
    exact = brute_force_minimum_image(lattice, deltas)

    inside = reported <= radius
    assert np.array_equal(inside, exact <= radius)
    assert np.allclose(reported[inside], exact[inside], atol=1e-12)
    assert np.all(reported[~inside] > radius)


def test_bounded_minimum_image_falls_back_for_a_large_radius():
    """Too large a radius disables the screen and the exact search runs."""

    lattice = SHEARED
    deltas = np.random.default_rng(7).uniform(-1.0, 1.0, size=(64, 3))
    radius = 10.0 * geometry.shortest_lattice_vector_length(lattice)
    assert np.allclose(
        np.sqrt(geometry.bounded_minimum_image_squared(lattice, deltas, radius)),
        brute_force_minimum_image(lattice, deltas),
        atol=1e-12,
    )


@pytest.mark.parametrize("name", sorted(CELLS))
def test_site_index_finds_every_periodic_match(name):
    """The bucket index answers exactly what a full distance matrix would."""

    lattice = CELLS[name]
    rng = np.random.default_rng(11)
    sites = rng.uniform(0.0, 1.0, size=(120, 3))
    labels = rng.integers(0, 3, size=120)
    index = geometry.PeriodicSiteIndex(lattice, sites, labels, tolerance=1e-4)

    order = rng.permutation(120)
    queries = sites[order] + rng.normal(scale=1e-7, size=(120, 3))
    assert np.array_equal(index.match(queries, labels[order]), order)
    # A label mismatch must not match, and neither must a point far from a site.
    assert np.all(index.match(queries, (labels[order] + 1) % 3) == -1)
    assert index.find(sites[0] + np.array([0.37, 0.21, 0.13]), int(labels[0])) == -1


def test_site_index_reports_the_lowest_index_on_request():
    """``prefer_lowest`` collapses coincident sites onto one representative."""

    sites = np.array([[0.1, 0.2, 0.3], [0.1, 0.2, 0.3], [0.6, 0.6, 0.6]])
    index = geometry.PeriodicSiteIndex(CUBIC, sites, tolerance=1e-5)
    assert list(index.match(sites, prefer_lowest=True)) == [0, 0, 2]


@pytest.mark.parametrize("name", sorted(CELLS))
def test_cell_list_neighbours_match_the_full_image_scan(name):
    """The cell list returns precisely the images inside the cutoff."""

    lattice = CELLS[name]
    points = np.random.default_rng(5).uniform(0.0, 1.0, size=(50, 3))
    cutoff = 4.0
    images, indices, valid = geometry.neighbour_images(lattice, points, cutoff)
    base = points @ lattice
    distances = np.linalg.norm(images[None, :, :] - base[:, None, :], axis=2)
    for atom in range(len(points)):
        expected = set(np.nonzero(distances[atom] <= cutoff)[0].tolist())
        assert set(indices[atom][valid[atom]].tolist()) == expected


@pytest.mark.parametrize("name", sorted(CELLS))
def test_fractional_minimum_image_agrees_with_the_cartesian_one(name):
    """The fractional shortest image is the Cartesian one in the cell's basis."""

    lattice = CELLS[name]
    deltas = np.random.default_rng(3).uniform(-2.0, 2.0, size=(64, 3))
    fractional = geometry.minimum_image_fractional(lattice, deltas)
    assert np.allclose(fractional @ lattice, geometry.minimum_image_displacements(lattice, deltas))
    # A shortest image differs from the input by an integer number of cells.
    assert np.allclose(fractional - deltas, np.rint(fractional - deltas), atol=1e-9)


def test_midpoints_lie_halfway_along_the_shortest_image():
    """A midpoint is equidistant from both sites, at half their true separation."""

    lattice = CELLS["hexagonal"]
    rng = np.random.default_rng(19)
    first = rng.uniform(0.0, 1.0, size=(40, 3))
    second = rng.uniform(0.0, 1.0, size=(40, 3))
    middle = geometry.periodic_midpoints(lattice, first, second)

    separation = geometry.minimum_image_distances(lattice, second - first)
    to_first = geometry.minimum_image_distances(lattice, first - middle)
    to_second = geometry.minimum_image_distances(lattice, second - middle)
    assert np.allclose(to_first, 0.5 * separation, atol=1e-9)
    assert np.allclose(to_second, 0.5 * separation, atol=1e-9)
    assert np.all(middle >= 0.0) and np.all(middle < 1.0)


def test_midpoints_differ_from_the_rounded_shortcut_in_a_skewed_cell():
    """Rounding puts the midpoint of a hexagonal pair in the wrong place."""

    lattice = CELLS["hexagonal"]
    first = np.array([[0.0, 0.0, 0.0]])
    second = np.array([[0.4, 0.6, 0.0]])
    exact = geometry.periodic_midpoints(lattice, first, second)
    delta = second - first
    rounded = np.mod(first + 0.5 * (delta - np.rint(delta)), 1.0)

    separation = float(geometry.minimum_image_distances(lattice, second - first)[0])
    assert float(geometry.minimum_image_distances(lattice, second - exact)[0]) == pytest.approx(
        0.5 * separation, abs=1e-9
    )
    assert float(geometry.minimum_image_distances(lattice, second - rounded)[0]) > 0.5 * separation + 0.1


def _span_contains(basis: np.ndarray, vectors: np.ndarray) -> bool:
    """Return whether every row of ``vectors`` is an integer combination of ``basis``."""

    coefficients = np.asarray(vectors, dtype=float) @ np.linalg.inv(np.asarray(basis, dtype=float))
    return bool(np.allclose(coefficients, np.rint(coefficients), atol=1e-8))


def test_the_integer_basis_spans_exactly_the_generated_lattice():
    generators = np.array([[2, 0, 0], [0, 2, 0], [1, 1, 0], [0, 0, 3], [1, 1, 3]], dtype=np.int64)
    basis = geometry.integer_lattice_basis(generators)
    assert basis.dtype == np.int64
    assert _span_contains(basis, generators), "every generator has to be in the span of the basis"
    assert _span_contains(generators[[0, 3, 2]], basis), "the basis may not enlarge the lattice"
    assert abs(round(float(np.linalg.det(basis.astype(float))))) == 6


def test_no_subset_of_the_generators_need_be_a_basis():
    """The reason a normal form is used rather than a search over triples."""

    generators = np.array([[2, 0, 0], [0, 3, 0], [1, 1, 0], [0, 0, 1]], dtype=np.int64)
    index = abs(round(float(np.linalg.det(geometry.integer_lattice_basis(generators).astype(float)))))
    assert index == 1, "the three generators span the whole of the integer lattice"
    triples = itertools.combinations(range(len(generators)), 3)
    assert all(
        abs(round(float(np.linalg.det(generators[list(triple)].astype(float))))) != index
        for triple in triples
    )


def test_the_integer_basis_does_not_depend_on_the_order_of_the_generators():
    rng = np.random.default_rng(11)
    generators = rng.integers(-4, 5, size=(7, 3))
    while abs(round(float(np.linalg.det(geometry.integer_lattice_basis(generators).astype(float))))) == 0:
        generators = rng.integers(-4, 5, size=(7, 3))
    reference = geometry.integer_lattice_basis(generators)
    for _ in range(20):
        shuffled = generators[rng.permutation(len(generators))]
        assert np.array_equal(geometry.integer_lattice_basis(shuffled), reference)


def test_a_rational_basis_recovers_the_face_centred_lattice():
    centering = np.array([[0.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]])
    generators = np.vstack([np.eye(3), centering])
    basis = geometry.rational_lattice_basis(generators, 4)
    assert abs(abs(float(np.linalg.det(basis))) - 0.25) < 1e-12
    assert _span_contains(basis, generators)


def test_a_rational_basis_rejects_generators_with_the_wrong_denominator():
    with pytest.raises(ValueError):
        geometry.rational_lattice_basis(np.vstack([np.eye(3), [[1.0 / 3.0, 0.0, 0.0]]]), 2)


def test_the_integer_basis_rejects_a_flat_generating_set():
    with pytest.raises(ValueError):
        geometry.integer_lattice_basis(np.array([[1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.int64))


def _niggli_gram(reduced: np.ndarray) -> tuple[float, ...]:
    gram = reduced @ reduced.T
    return (
        float(gram[0, 0]),
        float(gram[1, 1]),
        float(gram[2, 2]),
        2.0 * float(gram[1, 2]),
        2.0 * float(gram[0, 2]),
        2.0 * float(gram[0, 1]),
    )


def test_the_reduced_cell_meets_the_niggli_conditions():
    """The hypotheses of the machine-checked shortest-vector theorem.

    ``RequestProject/NiggliCell.lean`` proves that a cell whose Gram data obeys
    exactly these conditions has no lattice vector shorter than its first basis
    vector.  The conditions are asserted here on the output of the reduction, so
    the proof applies to what the code actually returns.
    """

    generator = np.random.default_rng(2024)
    for _ in range(200):
        lattice = generator.normal(size=(3, 3))
        if abs(float(np.linalg.det(lattice))) < 0.05:
            continue
        reduced, _ = geometry.niggli_reduce(lattice)
        a_value, b_value, c_value, xi, eta, zeta = _niggli_gram(reduced)
        tolerance = 1e-9 * max(a_value, b_value, c_value)
        assert a_value <= b_value + tolerance
        assert b_value <= c_value + tolerance
        assert abs(xi) <= b_value + tolerance
        assert abs(eta) <= a_value + tolerance
        assert abs(zeta) <= a_value + tolerance
        assert a_value + b_value + xi + eta + zeta >= -tolerance
        positive = xi > -tolerance and eta > -tolerance and zeta > -tolerance
        non_positive = xi <= tolerance and eta <= tolerance and zeta <= tolerance
        assert positive or non_positive, "the cross products must share a sign"


def test_the_first_vector_of_the_reduced_cell_is_a_shortest_lattice_vector():
    generator = np.random.default_rng(99)
    shifts = np.array([list(shift) for shift in itertools.product(range(-4, 5), repeat=3)], dtype=float)
    for _ in range(40):
        lattice = generator.normal(size=(3, 3))
        if abs(float(np.linalg.det(lattice))) < 0.05:
            continue
        reduced, _ = geometry.niggli_reduce(lattice)
        lengths = np.linalg.norm(shifts @ reduced, axis=1)
        nonzero = lengths[np.any(shifts != 0.0, axis=1)]
        assert float(np.linalg.norm(reduced[0])) <= float(nonzero.min()) + 1e-9


@pytest.mark.parametrize("name", sorted(CELLS))
def test_periodic_neighbour_pairs_agree_with_the_distance_matrix(name):
    """The cell list finds exactly the pairs a full distance matrix reports."""

    lattice = CELLS[name]
    generator = np.random.default_rng(7)
    points = generator.random((60, 3))
    distances = geometry.pairwise_minimum_image_distances(lattice, points)
    for cutoff in (0.5, 1.5, 3.0, 6.0):
        rows, columns = np.triu_indices(len(points), k=1)
        expected = {
            (int(i), int(j))
            for i, j in zip(rows, columns)
            if distances[i, j] <= cutoff
        }
        first, second = geometry.periodic_neighbour_pairs(lattice, points, cutoff)
        found = {(int(i), int(j)) for i, j in zip(first, second)}
        assert found == expected, f"{name} at cutoff {cutoff}"


@pytest.mark.parametrize("name", sorted(CELLS))
def test_shortest_interatomic_distance_agrees_with_the_distance_matrix(name):
    """The escalating search returns the true minimum separation."""

    lattice = CELLS[name]
    generator = np.random.default_rng(11)
    for count in (2, 5, 40):
        points = generator.random((count, 3))
        distances = geometry.pairwise_minimum_image_distances(lattice, points)
        rows, columns = np.triu_indices(count, k=1)
        expected = min(
            float(distances[rows, columns].min()),
            geometry.shortest_lattice_vector_length(lattice),
        )
        found = geometry.shortest_interatomic_distance(lattice, points)
        assert found == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_shortest_interatomic_distance_of_a_single_site_is_the_lattice_vector():
    """One atom has no partner, so its own nearest image sets the distance."""

    for name, lattice in CELLS.items():
        found = geometry.shortest_interatomic_distance(lattice, np.zeros((1, 3)))
        assert found == pytest.approx(
            geometry.shortest_lattice_vector_length(lattice)
        ), name
