"""Independent mirror certificates and stacking-consumer compatibility."""

import itertools

import numpy as np
import pytest

from cellstine.interface.surface import stacking
from cellstine.io.models import StructureRecord

# Literal sets obtained with exact integer Gram/ellipsoid certificates, not
# with the implementation or its private bound/search helper.
CASES = [('square',
  [[1, 0], [0, 1]],
  [[[-1, 0], [0, 1]], [[0, -1], [-1, 0]], [[0, 1], [1, 0]], [[1, 0], [0, -1]]]),
 ('rectangle', [[2, 0], [0, 1]], [[[-1, 0], [0, 1]], [[1, 0], [0, -1]]]),
 ('oblique', [[2, 1], [0, 2]], [[[-1, 1], [0, 1]], [[1, -1], [0, -1]]]),
 ('sheared_square',
  [[1, 5], [0, 1]],
  [[[-5, 24], [-1, 5]], [[-1, 10], [0, 1]], [[1, -10], [0, -1]], [[5, -24], [1, -5]]]),
 ('sheared_rectangle', [[2, 10], [0, 1]], [[[-1, 20], [0, 1]], [[1, -20], [0, -1]]]),
 ('reverse_sheared_square',
  [[1, 0], [5, 1]],
  [[[-5, 1], [-24, 5]], [[-1, 0], [-10, 1]], [[1, 0], [10, -1]], [[5, -1], [24, -5]]]),
 ('left_handed_square',
  [[-1, -5], [0, 1]],
  [[[-5, -24], [1, 5]], [[-1, -10], [0, 1]], [[1, 10], [0, -1]], [[5, 24], [-1, -5]]])]


def _keys(matrices):
    return [tuple(int(value) for value in matrix.reshape(-1)) for matrix in matrices]


@pytest.mark.parametrize("name,basis,expected", CASES, ids=[row[0] for row in CASES])
def test_complete_integer_mirrors_match_independent_certificates(name, basis, expected):
    lattice = np.array([[*basis[0], 0], [*basis[1], 0], [0, 0, 12]], dtype=float)
    found = stacking.inplane_mirror_matrices(lattice)
    expected_keys = [tuple(sum(matrix, [])) for matrix in expected]
    assert _keys(found) == expected_keys
    gram = lattice[:2] @ lattice[:2].T
    for matrix in found:
        assert matrix.dtype.kind in "iu" and matrix.flags.writeable
        assert int(matrix[0, 0]) * int(matrix[1, 1]) - int(matrix[0, 1]) * int(matrix[1, 0]) == -1
        assert np.array_equal(matrix @ gram @ matrix.T, gram)


def test_default_sheared_compound_mirror_is_isometry_and_preserves_records():
    lattice = np.array([[1., 5., 0.], [0., 1., 0.], [0., 0., 12.]])
    direct = np.array([[.1, .2, .2], [.4, .3, .4], [.2, .7, .6], [.8, .9, .8]])
    flags = [("F", "T", "F"), ("T", "F", "T"), ("F", "F", "T"), ("T", "T", "F")]
    source = StructureRecord(comment="compound mirror", lattice=lattice, species=["Na", "Cl"],
        counts=[2, 2], positions_direct=direct, positions_cartesian=direct @ lattice,
        selective_dynamics=True, selective_flags=flags, metadata={"nested": ["retained"]})
    result = stacking.mirror_structure(source)
    assert result is not source and result.species == ["Na", "Cl"] and result.counts == [2, 2]
    assert result.selective_flags == flags and result.metadata == source.metadata
    assert result.metadata is not source.metadata and result.metadata["nested"] is source.metadata["nested"]
    before = np.linalg.norm(source.positions_cartesian[:, None] - source.positions_cartesian[None, :], axis=2)
    after = np.linalg.norm(result.positions_cartesian[:, None] - result.positions_cartesian[None, :], axis=2)
    assert np.allclose(after, before, rtol=0., atol=1e-12)
    twice = stacking.mirror_structure(result)
    assert np.allclose(twice.positions_direct, direct, rtol=0., atol=1e-12)
    assert np.array_equal(source.positions_direct, direct) and source.selective_flags == flags


def test_mirror_arrays_are_owned_across_rows_calls_and_input():
    lattice = np.eye(3); original = lattice.copy()
    first = stacking.inplane_mirror_matrices(lattice)
    expected = _keys(first)
    first[0][:] = 99
    assert _keys(stacking.inplane_mirror_matrices(lattice)) == expected
    assert _keys(first[1:]) == expected[1:] and np.array_equal(lattice, original)


@pytest.mark.parametrize("shear", [100, 300])
def test_integer_annulus_remains_complete_for_larger_shears(shear):
    lattice = np.array([[1., float(shear), 0.], [0., 1., 0.], [0., 0., 12.]])
    # Conjugate the four square reflections by [[1,shear],[0,1]].
    expected = sorted([(-1, 2 * shear, 0, 1), (1, -2 * shear, 0, -1),
        (shear, 1 - shear * shear, 1, -shear),
        (-shear, shear * shear - 1, -1, shear)])
    assert _keys(stacking.inplane_mirror_matrices(lattice)) == expected


@pytest.mark.parametrize("lattice,expected", [
    (np.diag([1., 1e12, 12.]), [(-1, 0, 0, 1), (1, 0, 0, -1)]),
    (np.array([[1., 0., 0.], [1., 1e-6, 0.], [0., 0., 12.]]),
     [(-1, 0, -2, 1), (1, 0, 2, -1)]),
])
def test_extreme_metric_keeps_scan_bounded_and_valid_reflections(lattice, expected, monkeypatch):
    # Catch an infeasible search immediately, while still running real bounds,
    # final predicates and returned matrices. Never launch the trillion loop.
    builtin_range = range
    def bounded_range(*args):
        result = builtin_range(*args)
        assert len(result) < 10000, "mirror enumeration started an unbounded coordinate scan"
        return result
    monkeypatch.setattr(stacking, "range", bounded_range, raising=False)
    assert _keys(stacking.inplane_mirror_matrices(lattice, tolerance=0.)) == expected


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_metrics_keep_empty_legacy_result(value):
    lattice = np.eye(3); lattice[0, 0] = value
    with np.errstate(invalid="ignore"):
        assert stacking.inplane_mirror_matrices(lattice) == []


@pytest.mark.parametrize("tolerance", [-1., float("nan")])
def test_empty_legacy_result_for_invalid_tolerance(tolerance):
    assert stacking.inplane_mirror_matrices(np.eye(3), tolerance) == []


@pytest.mark.parametrize("lattice,tolerance", [
    (np.zeros((3, 3)), 1e-6), (np.eye(3), float("inf")),
    (np.eye(3) * 1e-12, 1e-6),
])
def test_unresolved_or_broad_metric_keeps_legacy_domain(lattice, tolerance):
    found = stacking.inplane_mirror_matrices(lattice, tolerance)
    assert len(found) == 116 and _keys(found) == sorted(_keys(found))


@pytest.mark.parametrize("delta", [-2e-6, 0., 2e-6])
@pytest.mark.parametrize("tolerance", [0., 1e-6, 1e-5])
def test_near_hexagonal_tolerance_preserves_original_float_acceptance(delta, tolerance):
    lattice = np.array([[2.5, 0., 0.], [-1.25 + delta, 2.5 * np.sqrt(3.) / 2., 0.], [0., 0., 12.]])
    gram = lattice[:2] @ lattice[:2].T
    threshold = tolerance * max(float(np.max(np.abs(gram))), 1.)
    expected = []
    for values in itertools.product(range(-3, 4), repeat=4):
        matrix = np.array(values, dtype=int).reshape(2, 2)
        if int(round(np.linalg.det(matrix))) == -1 and float(np.max(np.abs(matrix @ gram @ matrix.T - gram))) <= threshold:
            expected.append(values)
    assert _keys(stacking.inplane_mirror_matrices(lattice, tolerance)) == expected


def test_ordinary_search_filters_rows_before_matrix_determinants(monkeypatch):
    original = np.linalg.det
    calls = 0
    def counted(matrix):
        nonlocal calls
        calls += 1
        return original(matrix)
    monkeypatch.setattr(np.linalg, "det", counted)
    assert len(stacking.inplane_mirror_matrices(np.eye(3))) == 4
    assert calls < 100
