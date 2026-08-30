"""The hexagonal boundary of the plane reduction has one canonical answer.

A hexagonal lattice sits exactly on the boundary ``2 |a . b| = |a|^2`` of the
Lagrange--Gauss reduction condition, so the sixty and the hundred-and-twenty
degree descriptions of one and the same lattice are *both* reduced.  Left
unresolved, which of the two comes out is decided by the last bit of a dot
product, and the reported cell angle of a hexagonal surface, moire or interface
cell flips between 60 and 120 degrees when an upstream sum is reassociated.

``core.reduction.plane_reduce`` and ``moire.search.gram_report._reduce_common_basis``
both resolve it the same way -- the obtuse choice, matching the ``(60, 120]``
range CELLSTINE reports elsewhere -- by the shear ``b -> b - a`` that
``Cellstine.gaussStep_boundary_involutive`` shows cycles on the boundary, which
unlike negating ``b`` leaves the handedness of the pair alone.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from cellstine.core.lattice import vector_angle_deg
from cellstine.core.reduction import plane_reduce
from cellstine.moire.search.gram_report import _reduce_common_basis


def _hexagonal_pair(constant: float, *, acute: bool) -> np.ndarray:
    """Return the two vectors of a hexagonal cell, at 60 or at 120 degrees."""

    sign = 1.0 if acute else -1.0
    return np.array(
        [
            [constant, 0.0],
            [sign * constant / 2.0, constant * math.sqrt(3.0) / 2.0],
        ],
        dtype=float,
    )


@pytest.mark.parametrize("acute", [True, False])
@pytest.mark.parametrize("constant", [1.0, 2.467, 12.5])
def test_plane_reduce_reports_the_obtuse_hexagonal_cell(constant: float, acute: bool) -> None:
    reduced, transform = plane_reduce(_hexagonal_pair(constant, acute=acute))
    angle = vector_angle_deg(reduced[0], reduced[1])
    assert angle == pytest.approx(120.0, abs=1e-9)
    assert float(np.linalg.norm(reduced[0])) == pytest.approx(constant, rel=1e-12)
    assert float(np.linalg.norm(reduced[1])) == pytest.approx(constant, rel=1e-12)
    assert abs(int(round(np.linalg.det(transform.astype(float))))) == 1


def test_every_rewriting_of_one_hexagonal_lattice_reduces_the_same_way() -> None:
    """The answer is a property of the lattice, not of how it was handed over."""

    generator = np.random.default_rng(20260829)
    base = _hexagonal_pair(2.467, acute=True)
    for _ in range(64):
        unimodular = np.eye(2, dtype=np.int64)
        for _ in range(6):
            row, column = int(generator.integers(2)), int(generator.integers(2))
            if row == column:
                continue
            unimodular[row] += int(generator.integers(-2, 3)) * unimodular[column]
        reduced, _ = plane_reduce(unimodular.astype(float) @ base)
        assert vector_angle_deg(reduced[0], reduced[1]) == pytest.approx(120.0, abs=1e-9)
        assert float(np.linalg.norm(reduced[0])) == pytest.approx(2.467, rel=1e-12)


def test_plane_reduce_leaves_a_genuinely_acute_cell_alone() -> None:
    """Only the boundary is canonicalised; 70 degrees is reduced as it stands."""

    angle = math.radians(70.0)
    basis = np.array([[1.0, 0.0], [math.cos(angle), math.sin(angle)]], dtype=float)
    reduced, _ = plane_reduce(basis)
    assert vector_angle_deg(reduced[0], reduced[1]) == pytest.approx(70.0, abs=1e-9)


def test_plane_reduce_leaves_a_square_cell_alone() -> None:
    reduced, _ = plane_reduce(np.eye(2))
    assert vector_angle_deg(reduced[0], reduced[1]) == pytest.approx(90.0, abs=1e-9)


@pytest.mark.parametrize("acute", [True, False])
def test_plane_reduce_spans_the_same_lattice(acute: bool) -> None:
    basis = _hexagonal_pair(2.467, acute=acute)
    reduced, transform = plane_reduce(basis)
    assert np.allclose(reduced, transform.astype(float) @ basis, atol=1e-12)


def test_common_basis_reduction_agrees_with_plane_reduce_on_the_boundary() -> None:
    """The vectorised moire reduction makes the same choice, on both signs."""

    constant = 2.467
    g11 = np.array([constant**2, constant**2], dtype=float)
    g22 = np.array([constant**2, constant**2], dtype=float)
    g12 = np.array([constant**2 / 2.0, -(constant**2) / 2.0], dtype=float)
    signs = np.array([1, 1], dtype=np.int64)
    transform = _reduce_common_basis(g11, g12, g22, signs)

    for index, acute in enumerate((True, False)):
        basis = _hexagonal_pair(constant, acute=acute)
        # ``_reduce_common_basis`` acts on the right of a column basis.
        columns = basis.T @ transform[index].astype(float)
        assert vector_angle_deg(columns[:, 0], columns[:, 1]) == pytest.approx(120.0, abs=1e-9)
        assert int(round(float(np.linalg.det(transform[index].astype(float))))) == 1


def test_common_basis_reduction_leaves_a_square_cell_alone() -> None:
    g11 = np.array([4.0], dtype=float)
    g22 = np.array([4.0], dtype=float)
    g12 = np.array([0.0], dtype=float)
    transform = _reduce_common_basis(g11, g12, g22, np.array([1], dtype=np.int64))
    assert np.array_equal(transform[0], np.eye(2, dtype=np.int64))
