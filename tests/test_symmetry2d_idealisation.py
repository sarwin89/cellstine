"""The planar point-group search and the cell idealisation it feeds.

Every claim checked here has a machine-checked counterpart in
``RequestProject/PlanarPointGroup.lean``; the names are quoted in the individual
docstrings.  The last section records the one claim that turned out to be
*false* --- that the group average is the Frobenius-nearest invariant metric ---
so it is not quietly reintroduced.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from cellstine.core.symmetry2d import (
    close_group,
    group_has_mirror,
    lattice_point_group,
    proper_subgroup,
    rotation_order,
    symmetrised_basis,
)


def _basis(a: float, b: float, gamma_deg: float) -> np.ndarray:
    """Return the 2x2 *column* basis of a planar cell."""

    gamma = math.radians(gamma_deg)
    return np.array([[a, b * math.cos(gamma)], [0.0, b * math.sin(gamma)]], dtype=float)


HEXAGONAL = _basis(1.0, 1.0, 120.0)
SQUARE = _basis(1.0, 1.0, 90.0)
RECTANGULAR = _basis(1.0, 1.7, 90.0)
CENTRED = _basis(1.0, 1.0, 70.0)
OBLIQUE = _basis(1.0, 1.31, 71.0)


# --------------------------------------------------------------------------
# The search: what it tests, and that it misses nothing
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "basis,order",
    [(HEXAGONAL, 12), (SQUARE, 8), (RECTANGULAR, 4), (CENTRED, 4), (OBLIQUE, 2)],
)
def test_the_search_finds_exactly_the_metric_preserving_matrices(basis, order):
    """``Cellstine.gram_preserving_iff_columns``, checked against a brute-force sweep."""

    metric = basis.T @ basis
    found = {tuple(np.asarray(g).ravel().tolist()) for g in lattice_point_group(basis)}

    reference = set()
    for entries in itertools.product(range(-3, 4), repeat=4):
        candidate = np.array(entries, dtype=float).reshape(2, 2)
        if np.allclose(candidate.T @ metric @ candidate, metric, atol=1e-9):
            reference.add(tuple(int(value) for value in entries))

    assert found == reference
    assert len(found) == order


@pytest.mark.parametrize("basis", [HEXAGONAL, SQUARE, RECTANGULAR, CENTRED, OBLIQUE])
def test_every_reported_operation_is_unimodular(basis):
    """``Cellstine.det_sq_eq_one_of_gram_preserving``."""

    for element in lattice_point_group(basis):
        determinant = int(round(float(np.linalg.det(np.asarray(element, dtype=float)))))
        assert determinant in (1, -1)


@pytest.mark.parametrize("basis", [HEXAGONAL, SQUARE, RECTANGULAR, CENTRED, OBLIQUE])
def test_the_reported_group_really_is_a_group(basis):
    group = lattice_point_group(basis)
    keys = {tuple(np.asarray(g).ravel().tolist()) for g in group}
    assert tuple(np.eye(2, dtype=int).ravel().tolist()) in keys
    for left in group:
        for right in group:
            assert tuple((left @ right).ravel().tolist()) in keys
    assert len(close_group(group)) == len(group)


# --------------------------------------------------------------------------
# The crystallographic restriction
# --------------------------------------------------------------------------


@pytest.mark.parametrize("basis", [HEXAGONAL, SQUARE, RECTANGULAR, CENTRED, OBLIQUE])
def test_no_rotation_has_a_trace_outside_the_five_allowed_values(basis):
    """``Cellstine.planar_trace_sq_le_four`` and ``Cellstine.int_trace_mem_of_sq_le_four``."""

    for element in proper_subgroup(lattice_point_group(basis)):
        trace = int(element[0, 0] + element[1, 1])
        assert trace in (-2, -1, 0, 1, 2)


@pytest.mark.parametrize(
    "basis,order", [(HEXAGONAL, 6), (SQUARE, 4), (RECTANGULAR, 2), (CENTRED, 2), (OBLIQUE, 2)]
)
def test_the_rotation_order_is_one_of_the_crystallographic_ones(basis, order):
    assert rotation_order(lattice_point_group(basis)) == order
    assert rotation_order(lattice_point_group(basis)) in (1, 2, 3, 4, 6)


def test_a_five_fold_metric_is_not_a_lattice_metric():
    """A regular pentagon's metric admits no order-five integer rotation."""

    for entries in itertools.product(range(-4, 5), repeat=4):
        candidate = np.array(entries, dtype=np.int64).reshape(2, 2)
        if abs(int(round(float(np.linalg.det(candidate.astype(float)))))) != 1:
            continue
        power = np.eye(2, dtype=np.int64)
        order = 0
        for step in range(1, 8):
            power = power @ candidate
            if np.array_equal(power, np.eye(2, dtype=np.int64)):
                order = step
                break
        assert order != 5 and order != 7


@pytest.mark.parametrize("basis", [HEXAGONAL, SQUARE, RECTANGULAR, CENTRED, OBLIQUE])
def test_no_planar_group_exceeds_twelve_elements(basis):
    assert len(lattice_point_group(basis)) <= 12


# --------------------------------------------------------------------------
# The idealisation: what the group average does guarantee
# --------------------------------------------------------------------------


def _average(basis: np.ndarray, group: np.ndarray) -> np.ndarray:
    metric = basis.T @ basis
    closure = close_group(group).astype(float)
    return sum(element.T @ metric @ element for element in closure) / len(closure)


@pytest.mark.parametrize("basis", [HEXAGONAL, SQUARE, RECTANGULAR, CENTRED, OBLIQUE])
def test_the_idealised_metric_is_exactly_invariant(basis):
    """``Cellstine.averageMetric_invariant``."""

    group = lattice_point_group(basis)
    idealised, _ = symmetrised_basis(basis, group)
    metric = idealised.T @ idealised
    for element in close_group(group).astype(float):
        assert np.allclose(element.T @ metric @ element, metric, atol=1e-12, rtol=0.0)


@pytest.mark.parametrize("basis", [HEXAGONAL, SQUARE, RECTANGULAR, CENTRED, OBLIQUE])
def test_an_already_ideal_cell_is_returned_unchanged(basis):
    """``Cellstine.averageMetric_eq_self``."""

    group = lattice_point_group(basis)
    idealised, deviation = symmetrised_basis(basis, group)
    assert deviation == pytest.approx(0.0, abs=1e-12)
    assert np.allclose(idealised, basis, atol=1e-12)


def test_the_idealised_metric_is_positive_definite_and_keeps_the_orientation():
    """``Cellstine.averageMetric_pos``, plus the Procrustes step."""

    rng = np.random.default_rng(2024)
    for _ in range(200):
        perturbed = HEXAGONAL + 0.01 * rng.normal(size=(2, 2))
        group = lattice_point_group(perturbed, tolerance=5e-2)
        idealised, deviation = symmetrised_basis(perturbed, group)
        metric = idealised.T @ idealised
        assert metric[0, 0] > 0.0
        assert np.linalg.det(metric) > 0.0
        # Handedness is preserved, so reported twist angles are not flipped.
        assert np.sign(np.linalg.det(idealised)) == np.sign(np.linalg.det(perturbed))
        assert deviation >= 0.0


def test_the_idealisation_never_moves_an_entry_further_than_the_group_does():
    """``Cellstine.averageMetric_sub_apply_le``."""

    rng = np.random.default_rng(7)
    for _ in range(200):
        perturbed = HEXAGONAL + 0.01 * rng.normal(size=(2, 2))
        group = close_group(lattice_point_group(perturbed, tolerance=5e-2)).astype(float)
        metric = perturbed.T @ perturbed
        worst = max(
            float(np.max(np.abs(element.T @ metric @ element - metric))) for element in group
        )
        averaged = _average(perturbed, group.astype(np.int64))
        assert float(np.max(np.abs(averaged - metric))) <= worst + 1e-12


# --------------------------------------------------------------------------
# What the group average is NOT
# --------------------------------------------------------------------------


def _invariant_metric_projection(metric: np.ndarray, group: np.ndarray) -> np.ndarray:
    """Return the Frobenius-orthogonal projection of ``metric`` onto the invariants."""

    def vec(m):
        return np.array([m[0, 0], m[0, 1], m[1, 1]], dtype=float)

    def mat(v):
        return np.array([[v[0], v[1]], [v[1], v[2]]], dtype=float)

    rows = []
    for element in np.asarray(group, dtype=float):
        block = np.zeros((3, 3))
        for index, unit in enumerate(np.eye(3)):
            block[:, index] = vec(element.T @ mat(unit) @ element)
        rows.append(block - np.eye(3))
    _, singular, right = np.linalg.svd(np.vstack(rows))
    span = right[int(np.sum(singular > 1e-9)) :].T
    # The Frobenius inner product on symmetric 2x2 matrices weights the
    # off-diagonal entry twice.
    weight = np.diag([1.0, 2.0, 1.0])
    coefficients = np.linalg.solve(span.T @ weight @ span, span.T @ weight @ vec(metric))
    return mat(span @ coefficients)


def test_the_group_average_is_not_the_frobenius_nearest_invariant_metric():
    """The docstring of ``symmetrised_basis`` used to claim that it is; it is not.

    Averaging is the orthogonal projection onto the invariant metrics only when
    the group is closed under transposition.  A hexagonal integer point group in
    its natural (non-orthogonal) basis is not, and the average then sits strictly
    further from the input than the true minimiser.  The average is kept anyway
    because it is canonical, exact on ideal input and always positive definite.
    """

    group = close_group(lattice_point_group(HEXAGONAL))
    metric = np.array([[1.03, -0.48], [-0.48, 0.99]], dtype=float)
    averaged = sum(
        element.T @ metric @ element for element in group.astype(float)
    ) / len(group)
    projected = _invariant_metric_projection(metric, group)

    # Both are invariant.
    for element in group.astype(float):
        assert np.allclose(element.T @ averaged @ element, averaged, atol=1e-12)
        assert np.allclose(element.T @ projected @ element, projected, atol=1e-12)

    distance_average = float(np.linalg.norm(averaged - metric, "fro"))
    distance_projection = float(np.linalg.norm(projected - metric, "fro"))
    assert distance_projection < distance_average - 1e-9

    # ... and it is not an isolated case.
    rng = np.random.default_rng(3)
    strictly_worse = 0
    trials = 0
    for _ in range(200):
        candidate = np.array(
            [[1.0, -0.5], [-0.5, 1.0]], dtype=float
        ) + 0.05 * np.array([[1.0, 0.0], [0.0, 1.0]]) * rng.normal(size=(2, 2))
        candidate = 0.5 * (candidate + candidate.T)
        if np.linalg.det(candidate) <= 0.0 or candidate[0, 0] <= 0.0:
            continue
        trials += 1
        averaged = sum(
            element.T @ candidate @ element for element in group.astype(float)
        ) / len(group)
        projected = _invariant_metric_projection(candidate, group)
        if np.linalg.norm(averaged - candidate, "fro") > np.linalg.norm(
            projected - candidate, "fro"
        ) + 1e-12:
            strictly_worse += 1
    assert trials > 100
    assert strictly_worse == trials
