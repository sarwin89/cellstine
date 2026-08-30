"""The lattice points of a Miller plane are built exactly, not searched for.

The surface builder used to find its in-plane vectors by enumerating a box of
integer coefficients, ``|i|, |j|, |k| <= max(5, 2 max|hkl| + 3)``, and keeping the
combinations perpendicular to the plane normal.  That box is a guess: on a skew
cell the short in-plane vectors sit at coefficients far outside it, and the cell
that came back was then a valid but absurd one --- for the lattice below, a
``1.00 x 14.30`` cell at ``1.2`` degrees where the plane lattice really has a
``0.74 x 0.43`` cell at ``103`` degrees.

``cellstine.core.reduction.plane_form_kernel_basis`` replaces the search with one
extended Euclid step, which writes down a basis of ``{m : m . f = 0}`` for the
integer form ``f`` of the plane, and ``RequestProject/PlaneSublattice.lean``
proves that basis spans exactly the lattice points of the plane
(``Cellstine.Plane.exists_kernel_coords``,
``Cellstine.Plane.mem_span_iff_formValue_eq_zero``,
``Cellstine.Plane.kernel_minors``).  The tests here evaluate those statements:
exhaustively for small forms, and on real cells through the surface builder,
where the answer must agree with the old box search everywhere the box was wide
enough and be a reduced cell everywhere else.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from cellstine.core.lattice import vector_angle_deg
from cellstine.core.reduction import plane_form_kernel_basis, plane_reduce
from cellstine.interface.surface.surface_cell import (
    _integer_plane_form,
    _primitive_surface_vectors,
    _primitive_surface_vectors_from_lattice,
    _reciprocal_normal,
    _surface_vector_search_limit,
)


SMALL_FORMS = [
    form
    for form in itertools.product(range(-4, 5), repeat=3)
    if form != (0, 0, 0)
]

CUBIC = np.eye(3) * 3.5
FCC = np.array([[0.0, 1.8, 1.8], [1.8, 0.0, 1.8], [1.8, 1.8, 0.0]])
HEX = np.array(
    [
        [2.46, 0.0, 0.0],
        [-1.23, 2.46 * math.sqrt(3.0) / 2.0, 0.0],
        [0.0, 0.0, 6.7],
    ]
)
TRICLINIC = np.array([[3.1, 0.0, 0.0], [0.7, 3.4, 0.0], [0.4, -0.9, 4.2]])
# A cell whose third vector is nearly parallel to the second: the plane lattice of
# (100) has a short cell that no small box of coefficients contains.
SKEW = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 20.3, 0.31]])

LATTICES = {"cubic": CUBIC, "fcc": FCC, "hexagonal": HEX, "triclinic": TRICLINIC}
MILLERS = [
    (1, 0, 0),
    (0, 0, 1),
    (1, 1, 0),
    (1, 1, 1),
    (2, 1, 0),
    (1, -1, 1),
    (2, 1, 1),
    (3, 2, 1),
]
CASES = [(name, miller) for name in LATTICES for miller in MILLERS]


def _plane_area(lattice: np.ndarray, miller: tuple[int, int, int]) -> float:
    """The area of a primitive cell of the ``(hkl)`` plane lattice: ``V / d``."""

    volume = abs(float(np.linalg.det(lattice)))
    reciprocal = np.linalg.inv(np.asarray(lattice, dtype=float)).T
    normal = sum(float(index) * row for index, row in zip(miller, reciprocal))
    divisor = math.gcd(math.gcd(abs(miller[0]), abs(miller[1])), abs(miller[2]))
    return volume * float(np.linalg.norm(normal)) / float(divisor)


# ---------------------------------------------------------------------------
# the kernel basis itself
# ---------------------------------------------------------------------------


def test_both_basis_vectors_lie_in_the_plane():
    for form in SMALL_FORMS:
        values = np.array(form, dtype=np.int64)
        basis = plane_form_kernel_basis(values)
        assert basis.dtype == np.int64
        assert basis.shape == (2, 3)
        assert (basis @ values == 0).all()


def test_the_two_vectors_are_independent():
    for form in SMALL_FORMS:
        basis = plane_form_kernel_basis(np.array(form, dtype=np.int64))
        assert np.any(np.cross(basis[0], basis[1]) != 0)


def test_the_minors_are_the_reduced_form():
    """``Cellstine.Plane.kernel_minors``: the pair has index one in the plane."""

    for form in SMALL_FORMS:
        values = np.array(form, dtype=np.int64)
        divisor = math.gcd(math.gcd(abs(int(values[0])), abs(int(values[1]))), abs(int(values[2])))
        basis = plane_form_kernel_basis(values)
        cross = np.cross(basis[0], basis[1])
        # The cross product of a basis of the plane lattice is the primitive form
        # of the plane, up to sign: an index-``n`` sublattice gives ``n`` times it.
        assert np.array_equal(np.abs(cross) * divisor, np.abs(values))


def test_every_lattice_point_of_the_plane_is_an_integer_combination():
    """``Cellstine.Plane.exists_kernel_coords``, checked by exhaustion.

    The coefficients are recovered with exact integer arithmetic --- a `2 x 2`
    minor of the basis and Cramer's rule --- so the check itself introduces no
    floating point.
    """

    points = np.array(list(itertools.product(range(-5, 6), repeat=3)), dtype=np.int64)
    for form in SMALL_FORMS[::3]:
        values = np.array(form, dtype=np.int64)
        basis = plane_form_kernel_basis(values)
        first, second = basis[0], basis[1]
        minors = np.cross(first, second)
        axis = int(np.argmax(np.abs(minors)))
        rows = [index for index in range(3) if index != axis]
        determinant = int(
            first[rows[0]] * second[rows[1]] - first[rows[1]] * second[rows[0]]
        )
        assert determinant != 0
        for point in points[points @ values == 0]:
            numerator_x = int(point[rows[0]] * second[rows[1]] - point[rows[1]] * second[rows[0]])
            numerator_y = int(first[rows[0]] * point[rows[1]] - first[rows[1]] * point[rows[0]])
            assert numerator_x % determinant == 0
            assert numerator_y % determinant == 0
            x = numerator_x // determinant
            y = numerator_y // determinant
            assert np.array_equal(x * first + y * second, point)


def test_a_zero_form_is_refused():
    with pytest.raises(ValueError):
        plane_form_kernel_basis(np.zeros(3, dtype=np.int64))


def test_a_form_along_the_third_axis_gives_the_first_two_axes():
    basis = plane_form_kernel_basis(np.array([0, 0, 5], dtype=np.int64))
    assert np.array_equal(basis, np.array([[1, 0, 0], [0, 1, 0]], dtype=np.int64))


# ---------------------------------------------------------------------------
# the plane as seen from the primitive basis
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("miller", MILLERS)
def test_the_form_of_a_primitive_cell_is_the_miller_index_itself(miller):
    assert _integer_plane_form(CUBIC, CUBIC, miller) == tuple(miller)


@pytest.mark.parametrize("miller", MILLERS)
def test_the_form_transforms_with_the_centring_matrix(miller):
    """An ``F``-centred cubic cell, read on its own primitive rhombohedron."""

    conventional = np.eye(3) * 3.6
    primitive = np.array([[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]]) @ conventional
    form = _integer_plane_form(conventional, primitive, miller)
    assert form is not None
    # A primitive-basis vector is in the plane exactly when the form annihilates it.
    normal = _reciprocal_normal(conventional, miller)
    for coefficients in itertools.product(range(-3, 4), repeat=3):
        vector = np.asarray(coefficients, dtype=float) @ primitive
        in_plane = abs(float(vector @ normal)) <= 1e-9
        assert in_plane == (int(np.dot(coefficients, form)) == 0)


# ---------------------------------------------------------------------------
# the surface cell the builder returns
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name,miller", CASES)
def test_the_exact_cell_agrees_with_the_box_search_where_the_box_was_wide_enough(name, miller):
    lattice = LATTICES[name]
    normal = _reciprocal_normal(lattice, miller)
    box = _primitive_surface_vectors_from_lattice(
        lattice, normal, search=_surface_vector_search_limit(miller)
    )
    exact = _primitive_surface_vectors(lattice, lattice, normal, miller)
    assert np.allclose(box[0], exact[0], atol=1e-12)
    assert np.allclose(box[1], exact[1], atol=1e-12)


@pytest.mark.parametrize("name,miller", CASES)
def test_the_exact_cell_is_primitive_and_reduced(name, miller):
    lattice = LATTICES[name]
    normal = _reciprocal_normal(lattice, miller)
    first, second = _primitive_surface_vectors(lattice, lattice, normal, miller)
    area = float(np.linalg.norm(np.cross(first, second)))
    assert math.isclose(area, _plane_area(lattice, miller), rel_tol=1e-9)
    assert 60.0 + 1e-8 < vector_angle_deg(first, second) <= 120.0 + 1e-8
    assert float(np.cross(first, second) @ normal) > 0.0


def test_a_skew_cell_no_longer_returns_a_sliver():
    """The motivating case: the box search misses the short in-plane vectors."""

    miller = (1, 0, 0)
    normal = _reciprocal_normal(SKEW, miller)
    box = _primitive_surface_vectors_from_lattice(
        SKEW, normal, search=_surface_vector_search_limit(miller)
    )
    exact = _primitive_surface_vectors(SKEW, SKEW, normal, miller)
    # Both span the plane lattice --- the areas agree with V / d ...
    expected_area = _plane_area(SKEW, miller)
    for pair in (box, exact):
        assert math.isclose(
            float(np.linalg.norm(np.cross(*pair))), expected_area, rel_tol=1e-9
        )
    # ... but only the exact one is a reduced cell.
    assert vector_angle_deg(*box) < 2.0
    assert max(float(np.linalg.norm(vector)) for vector in box) > 14.0
    assert 60.0 < vector_angle_deg(*exact) <= 120.0
    assert max(float(np.linalg.norm(vector)) for vector in exact) < 1.0


def test_the_skew_cell_is_the_lagrange_gauss_reduction_of_the_plane_lattice():
    form = _integer_plane_form(SKEW, SKEW, (1, 0, 0))
    assert form == (1, 0, 0)
    coefficients = plane_form_kernel_basis(np.array(form, dtype=np.int64))
    reduced, transform = plane_reduce(coefficients.astype(float) @ SKEW)
    first, second = reduced
    # Lagrange--Gauss: the first row is a shortest vector, and |2 a.b| <= |a|^2.
    assert float(first @ first) <= float(second @ second)
    assert 2.0 * abs(float(first @ second)) <= float(first @ first) + 1e-12
    assert abs(int(round(float(np.linalg.det(transform))))) == 1
    exact = _primitive_surface_vectors(SKEW, SKEW, _reciprocal_normal(SKEW, (1, 0, 0)), (1, 0, 0))
    got = sorted(float(np.linalg.norm(vector)) for vector in exact)
    want = sorted(float(np.linalg.norm(vector)) for vector in reduced)
    assert np.allclose(got, want, atol=1e-12)


@pytest.mark.parametrize("miller", [(3, 2, 1), (5, 3, 1), (7, 5, 3), (9, 7, 5)])
def test_a_high_index_plane_needs_no_box_at_all(miller):
    """The exact path costs one Euclid step whatever the Miller indices are."""

    normal = _reciprocal_normal(CUBIC, miller)
    exact = _primitive_surface_vectors(CUBIC, CUBIC, normal, miller)
    box = _primitive_surface_vectors_from_lattice(
        CUBIC, normal, search=_surface_vector_search_limit(miller)
    )
    assert np.allclose(exact[0], box[0], atol=1e-12)
    assert np.allclose(exact[1], box[1], atol=1e-12)
    assert math.isclose(
        float(np.linalg.norm(np.cross(*exact))), _plane_area(CUBIC, miller), rel_tol=1e-9
    )
