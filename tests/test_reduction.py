"""Mathematical checks on the lattice reductions.

Every reduction in :mod:`cellstine.core.reduction` returns a pair
``(reduced, transform)`` and promises three things: that ``transform`` is an
integer matrix of determinant ``+-1``, so the reduced rows span *exactly* the
same lattice; that ``reduced == transform @ basis``; and that the reduced basis
satisfies the inequalities its reduction is named after.  The searches elsewhere
in CELLSTINE rely on those inequalities to keep an enumeration to a single shell
of neighbours, so they are checked here directly, against brute force.

The formal statements are in ``RequestProject/NiggliCell.lean`` and
``RequestProject/LagrangeGauss.lean``.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from cellstine.core.reduction import (
    as_lattice,
    axis_spacings,
    delaunay_reduce,
    gauss_reduction_multiplier,
    niggli_reduce,
    plane_reciprocal_norms,
    plane_reduce,
    reciprocal_norms,
    wrap_fractional,
    wrap_to_cell,
)

TRIALS = 60


def random_lattices(seed: int, count: int = TRIALS):
    """Yield well-conditioned random row lattices."""

    generator = np.random.default_rng(seed)
    produced = 0
    while produced < count:
        lattice = generator.normal(size=(3, 3)) * generator.uniform(0.5, 5.0)
        if abs(float(np.linalg.det(lattice))) < 1e-2:
            continue
        produced += 1
        yield lattice


def unimodular(generator) -> np.ndarray:
    """Return a random integer basis change of determinant one."""

    matrix = np.eye(3, dtype=np.int64)
    for _ in range(10):
        first, second = generator.choice(3, 2, replace=False)
        matrix[first] += generator.integers(-3, 4) * matrix[second]
    return matrix


def shell(radius: int) -> np.ndarray:
    """Return every nonzero integer triple with entries of size at most ``radius``."""

    span = range(-radius, radius + 1)
    grid = np.array(list(itertools.product(span, span, span)), dtype=float)
    return grid[np.any(grid != 0.0, axis=1)]


SHELL_ONE = shell(1)
SHELL_FOUR = shell(4)


# ---------------------------------------------------------------------------
# validation and wrapping
# ---------------------------------------------------------------------------


def test_a_lattice_must_be_three_independent_finite_rows():
    with pytest.raises(ValueError):
        as_lattice(np.eye(2))
    with pytest.raises(ValueError):
        as_lattice(np.array([[1.0, 0.0, 0.0], [0.0, np.inf, 0.0], [0.0, 0.0, 1.0]]))
    with pytest.raises(ValueError):
        as_lattice(np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 1.0]]))
    assert as_lattice(np.eye(3)).shape == (3, 3)


def test_wrapping_puts_a_coordinate_in_its_window():
    values = np.array([-2.25, -0.5, 0.0, 0.5, 3.75])
    wrapped = wrap_to_cell(values)
    assert np.all((wrapped >= 0.0) & (wrapped < 1.0))
    assert np.allclose(wrapped - values, np.rint(wrapped - values))
    minimum = wrap_fractional(values)
    assert np.all(np.abs(minimum) <= 0.5 + 1e-12)
    assert np.allclose(minimum - values, np.rint(minimum - values))


def test_the_axis_spacing_is_the_height_of_the_cell():
    """The spacing normal to axis ``i`` is the volume over the opposite face area."""

    for lattice in random_lattices(11, 20):
        volume = abs(float(np.linalg.det(lattice)))
        expected = [
            volume / float(np.linalg.norm(np.cross(lattice[(i + 1) % 3], lattice[(i + 2) % 3])))
            for i in range(3)
        ]
        assert axis_spacings(lattice) == pytest.approx(expected, rel=1e-10)
        assert reciprocal_norms(lattice) == pytest.approx(1.0 / np.asarray(expected), rel=1e-10)


# ---------------------------------------------------------------------------
# Niggli
# ---------------------------------------------------------------------------


def test_the_niggli_transform_spans_the_same_lattice():
    for lattice in random_lattices(1):
        reduced, transform = niggli_reduce(lattice)
        assert np.allclose(transform, np.rint(transform), atol=1e-9)
        assert abs(abs(float(np.linalg.det(transform))) - 1.0) < 1e-9
        assert np.allclose(reduced, transform @ lattice, atol=1e-8)


def test_the_niggli_cell_satisfies_the_niggli_inequalities():
    """``a <= b <= c`` and the Buerger conditions on the scalar products."""

    for lattice in random_lattices(2):
        reduced, _ = niggli_reduce(lattice)
        a, b, c = (float(row @ row) for row in reduced)
        xi = 2.0 * float(reduced[1] @ reduced[2])
        eta = 2.0 * float(reduced[0] @ reduced[2])
        zeta = 2.0 * float(reduced[0] @ reduced[1])
        tolerance = 1e-8 * max(a, b, c)
        assert a <= b + tolerance and b <= c + tolerance
        assert abs(xi) <= b + tolerance
        assert abs(eta) <= a + tolerance
        assert abs(zeta) <= a + tolerance


def test_the_niggli_cell_is_canonical():
    """A lattice has one Niggli metric, whatever basis it is handed in."""

    generator = np.random.default_rng(3)
    for lattice in random_lattices(3, 25):
        reference, _ = niggli_reduce(lattice)
        gram = reference @ reference.T
        scale = float(np.max(np.abs(gram)))
        for _ in range(3):
            other, _ = niggli_reduce(unimodular(generator).astype(float) @ lattice)
            assert np.allclose(other @ other.T, gram, atol=1e-7 * scale)


def test_the_niggli_cell_keeps_the_volume():
    for lattice in random_lattices(4, 25):
        reduced, _ = niggli_reduce(lattice)
        assert abs(float(np.linalg.det(reduced))) == pytest.approx(
            abs(float(np.linalg.det(lattice))), rel=1e-10
        )


# ---------------------------------------------------------------------------
# Delaunay
# ---------------------------------------------------------------------------


def test_the_delaunay_transform_spans_the_same_lattice():
    for lattice in random_lattices(5):
        reduced, transform = delaunay_reduce(lattice)
        assert np.allclose(transform, np.rint(transform), atol=1e-9)
        assert abs(abs(float(np.linalg.det(transform))) - 1.0) < 1e-9
        assert np.allclose(reduced, transform @ lattice, atol=1e-8)


def test_a_shortest_vector_is_found_in_the_first_shell():
    """The reason the searches enumerate coefficients in ``{-1, 0, 1}`` only."""

    for lattice in random_lattices(6, 40):
        reduced, _ = delaunay_reduce(lattice)
        far = np.linalg.norm(SHELL_FOUR @ reduced, axis=1)
        near = np.linalg.norm(SHELL_ONE @ reduced, axis=1)
        assert float(np.min(near)) == pytest.approx(float(np.min(far)), rel=1e-12)


def test_the_minimum_image_is_found_in_the_first_shell():
    """A displacement is brought to its true shortest image by one shell."""

    generator = np.random.default_rng(7)
    for lattice in random_lattices(7, 40):
        reduced, _ = delaunay_reduce(lattice)
        offset = generator.uniform(-0.5, 0.5, size=3) @ reduced
        near = min(
            float(np.linalg.norm(offset)),
            float(np.min(np.linalg.norm(offset + SHELL_ONE @ reduced, axis=1))),
        )
        far = min(
            float(np.linalg.norm(offset)),
            float(np.min(np.linalg.norm(offset + SHELL_FOUR @ reduced, axis=1))),
        )
        assert near == pytest.approx(far, rel=1e-12)


# ---------------------------------------------------------------------------
# the plane
# ---------------------------------------------------------------------------


def test_the_plane_reduction_is_lagrange_gauss():
    generator = np.random.default_rng(8)
    for _ in range(TRIALS):
        basis = generator.normal(size=(2, 3)) * generator.uniform(0.5, 5.0)
        if float(np.linalg.norm(np.cross(basis[0], basis[1]))) < 1e-3:
            continue
        reduced, transform = plane_reduce(basis)
        assert np.allclose(transform, np.rint(transform), atol=1e-9)
        assert abs(abs(float(np.linalg.det(transform))) - 1.0) < 1e-9
        assert np.allclose(reduced, transform @ basis, atol=1e-8)
        first = float(reduced[0] @ reduced[0])
        second = float(reduced[1] @ reduced[1])
        assert first <= second + 1e-9 * second
        assert abs(2.0 * float(reduced[0] @ reduced[1])) <= first + 1e-9 * first


def test_the_first_reduced_plane_vector_is_shortest():
    generator = np.random.default_rng(9)
    span = range(-6, 7)
    grid = np.array(list(itertools.product(span, span)), dtype=float)
    grid = grid[np.any(grid != 0.0, axis=1)]
    for _ in range(30):
        basis = generator.normal(size=(2, 3)) * generator.uniform(0.5, 5.0)
        if float(np.linalg.norm(np.cross(basis[0], basis[1]))) < 1e-3:
            continue
        reduced, _ = plane_reduce(basis)
        shortest = float(np.min(np.linalg.norm(grid @ reduced, axis=1)))
        assert float(np.linalg.norm(reduced[0])) == pytest.approx(shortest, rel=1e-12)


def test_the_plane_reduction_terminates_on_a_hexagonal_boundary():
    """A 120 degree cell sits exactly on ``2 |r0 . r1| = |r0|^2``, already reduced.

    The exact rule rounds the ratio ``-0.5`` away from zero once rounding pushes
    it a few ulps past the boundary; the step then only flips the sign of the
    dot product, and the rounds oscillate for ever.  This is the in-plane cell
    of a ``6 x 1`` hexagonal slab supercell, which used to raise.
    """

    side = 2.46
    for repeat in range(1, 9):
        basis = np.array(
            [
                [repeat * side, 0.0, 0.0],
                [-0.5 * side, 0.5 * math.sqrt(3.0) * side, 0.0],
            ]
        )
        reduced, transform = plane_reduce(basis)
        assert abs(abs(float(np.linalg.det(transform))) - 1.0) < 1e-12
        assert np.allclose(reduced, transform @ basis, atol=1e-12)
        first = float(reduced[0] @ reduced[0])
        second = float(reduced[1] @ reduced[1])
        assert first <= second * (1.0 + 1e-12)
        assert 2.0 * abs(float(reduced[0] @ reduced[1])) <= first * (1.0 + 1e-9)
        assert float(np.linalg.norm(reduced[0])) == pytest.approx(side, rel=1e-12)


def test_the_shear_of_a_round_treats_the_boundary_as_reduced():
    """``2 |dot| = norm`` is the reduction condition, and must not take a step."""

    norm = 6.0516
    assert gauss_reduction_multiplier(0.5 * norm, norm) == 0
    assert gauss_reduction_multiplier(-0.5 * norm, norm) == 0
    assert gauss_reduction_multiplier(np.nextafter(0.5 * norm, 1.0), norm) == 0
    assert gauss_reduction_multiplier(-np.nextafter(0.5 * norm, 1.0), norm) == 0
    # Well inside the boundary nothing changes: the nearest integer is returned.
    assert gauss_reduction_multiplier(0.51 * norm, norm) == 1
    assert gauss_reduction_multiplier(-2.4 * norm, norm) == -2
    assert gauss_reduction_multiplier(0.4 * norm, norm) == 0


def test_the_plane_reduction_refuses_a_degenerate_basis():
    with pytest.raises(ValueError):
        plane_reduce(np.zeros((2, 3)))


def test_the_plane_reciprocal_norms_are_the_row_spacings():
    """``1 / d_i``, with ``d_i`` the spacing of the rows of points parallel to ``a_i``."""

    generator = np.random.default_rng(10)
    for _ in range(30):
        basis = generator.normal(size=(2, 3)) * generator.uniform(0.5, 5.0)
        area = float(np.linalg.norm(np.cross(basis[0], basis[1])))
        if area < 1e-3:
            continue
        spacings = 1.0 / plane_reciprocal_norms(basis)
        expected = [area / float(np.linalg.norm(basis[1])), area / float(np.linalg.norm(basis[0]))]
        assert spacings == pytest.approx(expected, rel=1e-10)
