"""The in-plane cell of a Miller surface is a primitive cell of the plane lattice.

`interface/surface/surface_cell.py` chooses the two surface vectors by minimising
the longer of their two lengths, then their total length, then the area.  Nothing
in that rule mentions the *index* of the pair, so the question the tests below
answer is whether the cell it returns spans the whole plane lattice or only a
sublattice of it -- a sublattice would multiply the atom count of every slab.

``RequestProject/SurfacePlaneBasis.lean`` proves it is exact:
``Cellstine.abs_pairDet_eq_one_of_minimal`` (a pair minimal for that rule is a
basis, because a reduced cell has area at least ``(sqrt 3 / 2) * l1 * l2`` while
any pair has area at most ``l1 * l2``, so its index is below ``2 / sqrt 3``), and
``Cellstine.hexagonal_of_cos_ge_half`` with
``Cellstine.shear_gram_of_hexagonal`` (only a hexagonal cell can come out at 60
degrees or sharper, and the ``b -> b - a`` step then turns it into the
conventional 120-degree cell without changing either length).

The checks here are the same statements, evaluated on real lattices: the two
returned vectors have integer coordinates in the crystal lattice, every in-plane
lattice vector is an integer combination of them, the pair realises the two
successive minima of the plane lattice, and the cell angle always lands in
(60, 120] degrees.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from cellstine.core.lattice import vector_angle_deg
from cellstine.interface.surface.surface_cell import (
    _primitive_surface_vectors_from_lattice,
    _reciprocal_normal,
    _surface_vector_search_limit,
)


CUBIC = np.eye(3) * 3.5
FCC = np.array([[0.0, 1.8, 1.8], [1.8, 0.0, 1.8], [1.8, 1.8, 0.0]])
BCC = np.array([[-1.6, 1.6, 1.6], [1.6, -1.6, 1.6], [1.6, 1.6, -1.6]])
HEX = np.array(
    [
        [2.46, 0.0, 0.0],
        [-1.23, 2.46 * math.sqrt(3.0) / 2.0, 0.0],
        [0.0, 0.0, 6.7],
    ]
)
TRICLINIC = np.array([[3.1, 0.0, 0.0], [0.7, 3.4, 0.0], [0.4, -0.9, 4.2]])

LATTICES = {"cubic": CUBIC, "fcc": FCC, "bcc": BCC, "hexagonal": HEX, "triclinic": TRICLINIC}
MILLERS = [
    (1, 0, 0),
    (0, 1, 0),
    (0, 0, 1),
    (1, 1, 0),
    (1, 1, 1),
    (2, 1, 0),
    (1, -1, 1),
    (2, 1, 1),
]
CASES = [(name, miller) for name in LATTICES for miller in MILLERS]


def _surface_pair(lattice: np.ndarray, miller: tuple[int, int, int]):
    normal = _reciprocal_normal(lattice, miller)
    limit = _surface_vector_search_limit(miller)
    return normal, _primitive_surface_vectors_from_lattice(lattice, normal, search=limit)


def _integer_coordinates(lattice: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """The coordinates of ``vector`` in the lattice, which must be integral."""

    coefficients = np.linalg.solve(np.asarray(lattice, dtype=float).T, vector)
    rounded = np.round(coefficients)
    assert np.allclose(coefficients, rounded, atol=1e-8)
    return rounded.astype(int)


def _in_plane_lattice_vectors(lattice: np.ndarray, normal: np.ndarray, limit: int):
    """Every lattice vector inside the search box that lies in the plane."""

    span = range(-limit, limit + 1)
    for i, j, k in itertools.product(span, span, span):
        if i == j == k == 0:
            continue
        vector = i * lattice[0] + j * lattice[1] + k * lattice[2]
        if abs(float(vector @ normal)) <= 1e-7:
            yield np.array([i, j, k]), vector


@pytest.mark.parametrize("name,miller", CASES)
def test_the_surface_vectors_are_lattice_vectors_in_the_plane(name, miller):
    lattice = LATTICES[name]
    normal, (surface_a, surface_b) = _surface_pair(lattice, miller)
    for vector in (surface_a, surface_b):
        _integer_coordinates(lattice, vector)
        assert abs(float(vector @ normal)) <= 1e-7
    # Right handed about the outward normal.
    assert float(np.cross(surface_a, surface_b) @ normal) > 0.0


@pytest.mark.parametrize("name,miller", CASES)
def test_the_surface_cell_is_primitive(name, miller):
    lattice = LATTICES[name]
    normal, (surface_a, surface_b) = _surface_pair(lattice, miller)
    limit = _surface_vector_search_limit(miller)
    # Every in-plane lattice vector is an integer combination of the two.
    basis = np.column_stack([surface_a, surface_b, normal])
    for _, vector in _in_plane_lattice_vectors(lattice, normal, limit):
        coefficients = np.linalg.solve(basis, vector)
        assert abs(coefficients[2]) <= 1e-8
        assert np.allclose(coefficients[:2], np.round(coefficients[:2]), atol=1e-8)


@pytest.mark.parametrize("name,miller", CASES)
def test_the_surface_cell_has_the_least_possible_area(name, miller):
    lattice = LATTICES[name]
    normal, (surface_a, surface_b) = _surface_pair(lattice, miller)
    limit = _surface_vector_search_limit(miller)
    area = float(np.linalg.norm(np.cross(surface_a, surface_b)))
    vectors = [vector for _, vector in _in_plane_lattice_vectors(lattice, normal, limit)]
    best = min(
        (
            value
            for value in (
                float(np.linalg.norm(np.cross(u, v))) for u, v in itertools.combinations(vectors, 2)
            )
            if value > 1e-7
        ),
        default=None,
    )
    assert best is not None
    assert math.isclose(area, best, rel_tol=1e-9)


@pytest.mark.parametrize("name,miller", CASES)
def test_the_two_lengths_are_the_successive_minima(name, miller):
    lattice = LATTICES[name]
    normal, (surface_a, surface_b) = _surface_pair(lattice, miller)
    limit = _surface_vector_search_limit(miller)
    vectors = [vector for _, vector in _in_plane_lattice_vectors(lattice, normal, limit)]
    lengths = sorted(float(np.linalg.norm(vector)) for vector in vectors)
    shortest = lengths[0]
    # The second minimum: the shortest vector independent of the shortest one.
    reference = min(vectors, key=lambda v: float(np.linalg.norm(v)))
    second = min(
        float(np.linalg.norm(v))
        for v in vectors
        if float(np.linalg.norm(np.cross(reference, v))) > 1e-7
    )
    got = sorted((float(np.linalg.norm(surface_a)), float(np.linalg.norm(surface_b))))
    # ``b -> b - a`` on a hexagonal cell keeps both lengths, so this holds there too.
    assert math.isclose(got[0], shortest, rel_tol=1e-9)
    assert math.isclose(got[1], second, rel_tol=1e-9)


@pytest.mark.parametrize("name,miller", CASES)
def test_the_cell_angle_is_between_sixty_and_a_hundred_and_twenty_degrees(name, miller):
    lattice = LATTICES[name]
    _, (surface_a, surface_b) = _surface_pair(lattice, miller)
    angle = vector_angle_deg(surface_a, surface_b)
    # A reduced cell has |cos gamma| <= 1/2, and the shear removes the one sharp
    # case gamma = 60, so the reported angle is always in (60, 120].
    assert 60.0 + 1e-8 < angle <= 120.0 + 1e-8


def test_a_hexagonal_plane_comes_out_at_a_hundred_and_twenty_degrees():
    _, (surface_a, surface_b) = _surface_pair(HEX, (0, 0, 1))
    assert math.isclose(
        float(np.linalg.norm(surface_a)), float(np.linalg.norm(surface_b)), rel_tol=1e-12
    )
    assert math.isclose(vector_angle_deg(surface_a, surface_b), 120.0, abs_tol=1e-9)
    # The 60-degree cell it starts from is the same lattice, one shear away.
    assert math.isclose(
        float(np.linalg.norm(surface_b + surface_a)),
        float(np.linalg.norm(surface_a)),
        rel_tol=1e-12,
    )


def test_the_cell_of_a_cubic_hundred_face_is_the_square_face():
    _, (surface_a, surface_b) = _surface_pair(CUBIC, (0, 0, 1))
    assert math.isclose(float(np.linalg.norm(surface_a)), 3.5, rel_tol=1e-12)
    assert math.isclose(float(np.linalg.norm(surface_b)), 3.5, rel_tol=1e-12)
    assert math.isclose(vector_angle_deg(surface_a, surface_b), 90.0, abs_tol=1e-9)


def test_the_fcc_hundred_and_eleven_face_is_the_hexagonal_close_packed_mesh():
    _, (surface_a, surface_b) = _surface_pair(FCC, (1, 1, 1))
    nearest = 1.8 * math.sqrt(2.0)
    assert math.isclose(float(np.linalg.norm(surface_a)), nearest, rel_tol=1e-12)
    assert math.isclose(float(np.linalg.norm(surface_b)), nearest, rel_tol=1e-12)
    assert math.isclose(vector_angle_deg(surface_a, surface_b), 120.0, abs_tol=1e-9)
