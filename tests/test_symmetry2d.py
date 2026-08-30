"""Planar point-group helpers used to fold moire searches."""

from __future__ import annotations

import math

import numpy as np
import pytest

from cellstine.core.symmetry2d import (
    DEFAULT_SYMMETRY_TOLERANCE,
    cartesian_mirror_angles,
    close_group,
    cartesian_rotation_angles,
    column_basis_from_lattice,
    equivalence_period_radians,
    group_has_mirror,
    lattice_point_group,
    layer_point_group,
    proper_subgroup,
    rotation_order,
    symmetrised_basis,
)
from cellstine.core.species import expand_species
from cellstine.io import native as io

from conftest import hexagonal_basis


def _group(path) -> np.ndarray:
    record = io.read_poscar(str(path))
    return layer_point_group(
        record.lattice,
        record.positions_direct,
        expand_species(record.species, record.counts),
    )


def test_lattice_point_groups_have_the_expected_orders():
    assert len(lattice_point_group(hexagonal_basis(2.46))) == 12
    assert len(lattice_point_group(np.diag([3.9, 3.9]))) == 8
    assert len(lattice_point_group(np.diag([3.9, 5.1]))) == 4
    oblique = np.array([[3.9, 1.1], [0.0, 5.1]])
    assert len(lattice_point_group(oblique)) == 2


def test_group_elements_preserve_the_metric():
    basis = hexagonal_basis(2.46)
    metric = basis.T @ basis
    for element in lattice_point_group(basis):
        transformed = element.T.astype(float) @ metric @ element.astype(float)
        assert np.allclose(transformed, metric, atol=1e-10)


def test_graphene_layer_is_six_fold_and_hbn_is_three_fold(graphene_poscar, hbn_poscar):
    graphene = _group(graphene_poscar)
    hbn = _group(hbn_poscar)
    assert rotation_order(graphene) == 6
    assert rotation_order(hbn) == 3
    assert group_has_mirror(graphene) and group_has_mirror(hbn)
    assert len(graphene) == 12
    assert len(hbn) == 6


def test_mos2_layer_is_three_fold(mos2_poscar):
    """The two sulphur heights break the six-fold rotation of the bare lattice."""

    group = _group(mos2_poscar)
    assert rotation_order(group) == 3
    assert len(group) == 6


def test_layer_group_is_a_subgroup_of_the_lattice_group(hbn_poscar):
    record = io.read_poscar(str(hbn_poscar))
    basis = column_basis_from_lattice(record.lattice)
    lattice = {tuple(element.ravel().tolist()) for element in lattice_point_group(basis)}
    layer = {tuple(element.ravel().tolist()) for element in _group(hbn_poscar)}
    assert layer <= lattice


def test_twist_period_follows_the_rotation_orders():
    assert equivalence_period_radians(6, 6) == pytest.approx(math.pi / 3.0)
    assert equivalence_period_radians(3, 3) == pytest.approx(2.0 * math.pi / 3.0)
    assert equivalence_period_radians(6, 3) == pytest.approx(math.pi / 3.0)
    assert equivalence_period_radians(4, 4) == pytest.approx(math.pi / 2.0)


def test_cartesian_angles_match_the_integer_action():
    basis = hexagonal_basis(2.46)
    group = lattice_point_group(basis)
    proper = proper_subgroup(group)
    angles = cartesian_rotation_angles(basis, proper)
    assert len(angles) == 6
    multiples = np.sort(np.mod(angles, 2.0 * math.pi) / (math.pi / 3.0))
    assert np.allclose(multiples, np.arange(6), atol=1e-9)
    improper = np.array(
        [
            element
            for element in group
            if element[0, 0] * element[1, 1] - element[0, 1] * element[1, 0] < 0
        ]
    )
    doubled = cartesian_mirror_angles(basis, improper)
    for element, angle in zip(improper, doubled):
        reflection = basis @ element.astype(float) @ np.linalg.inv(basis)
        axis = np.array([math.cos(0.5 * angle), math.sin(0.5 * angle)])
        assert np.allclose(reflection @ axis, axis, atol=1e-9)


def test_column_basis_rejects_non_planar_cells():
    lattice = np.array([[2.46, 0.0, 0.3], [-1.23, 2.13, 0.0], [0.0, 0.0, 20.0]])
    with pytest.raises(ValueError):
        column_basis_from_lattice(lattice)


def _six_decimal_hexagonal(a: float) -> np.ndarray:
    """Return a hexagonal column basis rounded the way a POSCAR prints it."""

    height = round(a * math.sqrt(3) / 2.0, 6)
    return np.array([[round(a, 6), -round(a / 2.0, 6)], [0.0, height]])


def test_default_tolerance_survives_poscar_rounding():
    """A cell printed with six decimals must keep its hexagonal point group.

    At machine-epsilon tolerance the rounding of a printed POSCAR already breaks
    the six-fold symmetry, which silently costs the moire search its angle
    folding, so the default must be a physical tolerance instead.
    """

    basis = _six_decimal_hexagonal(2.468)
    assert len(lattice_point_group(basis, tolerance=1e-9)) == 4
    assert len(lattice_point_group(basis)) == 12
    assert DEFAULT_SYMMETRY_TOLERANCE >= 1e-6


def test_close_group_completes_a_partial_generating_set():
    sixfold = np.array([[1, -1], [1, 0]], dtype=np.int64)
    closure = close_group(sixfold)
    assert len(closure) == 6
    assert any(np.array_equal(element, np.eye(2, dtype=np.int64)) for element in closure)
    products = {
        tuple((left @ right).ravel().tolist()) for left in closure for right in closure
    }
    assert products == {tuple(element.ravel().tolist()) for element in closure}


def test_symmetrised_basis_is_exactly_invariant_and_barely_moves():
    basis = _six_decimal_hexagonal(2.468)
    group = lattice_point_group(basis)
    idealised, deviation = symmetrised_basis(basis, group)
    metric = idealised.T @ idealised
    scale = float(np.max(np.abs(metric)))
    for element in close_group(group).astype(float):
        assert np.allclose(element.T @ metric @ element, metric, atol=1e-12 * scale)
    assert deviation < 1e-6
    assert np.allclose(idealised, basis, atol=1e-5)
    lengths = np.linalg.norm(idealised, axis=0)
    assert lengths[0] == pytest.approx(lengths[1], rel=1e-14)
    cosine = metric[0, 1] / math.sqrt(metric[0, 0] * metric[1, 1])
    assert math.degrees(math.acos(cosine)) == pytest.approx(120.0, abs=1e-12)


def test_symmetrised_basis_preserves_orientation_and_handedness():
    basis = _six_decimal_hexagonal(2.468)
    group = lattice_point_group(basis)
    idealised, _ = symmetrised_basis(basis, group)
    assert np.sign(np.linalg.det(idealised)) == np.sign(np.linalg.det(basis))
    first = idealised[:, 0]
    assert abs(math.degrees(math.atan2(first[1], first[0]))) < 1e-4


def test_symmetrised_basis_is_a_fixed_point_for_an_ideal_cell():
    basis = hexagonal_basis(2.46)
    group = lattice_point_group(basis)
    idealised, deviation = symmetrised_basis(basis, group)
    assert deviation == pytest.approx(0.0, abs=1e-14)
    assert np.allclose(idealised, basis, atol=1e-12)
