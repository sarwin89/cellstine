"""Checks on the native three-dimensional symmetry engine.

Every assertion here is a statement that can be verified independently of the
implementation: the point group of a known crystal, the invariance of the
Niggli cell under a change of basis, the group axioms for the returned
operations, and the volume of a primitive cell.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from cellstine.core import symmetry3d as sym


def _fcc(constant: float) -> np.ndarray:
    return constant * np.array([[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]])


def _hexagonal(constant: float, height: float) -> np.ndarray:
    return np.array(
        [
            [constant, 0.0, 0.0],
            [-0.5 * constant, 0.5 * math.sqrt(3.0) * constant, 0.0],
            [0.0, 0.0, height],
        ]
    )


def _silicon() -> tuple[np.ndarray, np.ndarray, list[str]]:
    return _fcc(5.43), np.array([[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]]), ["Si", "Si"]


def _silicon_conventional() -> tuple[np.ndarray, np.ndarray, list[str]]:
    lattice = 5.43 * np.eye(3)
    base = np.array([[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]])
    centering = np.array([[0.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]])
    positions = np.mod((base[:, None, :] + centering[None, :, :]).reshape(-1, 3), 1.0)
    return lattice, positions, ["Si"] * len(positions)


def _rock_salt(constant: float) -> tuple[np.ndarray, np.ndarray, list[str]]:
    lattice = _fcc(constant)
    return lattice, np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]), ["Na", "Cl"]


def _body_centred_tungsten(constant: float) -> tuple[np.ndarray, np.ndarray, list[str]]:
    lattice = constant * np.eye(3)
    return lattice, np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]), ["W", "W"]


def _hcp_magnesium() -> tuple[np.ndarray, np.ndarray, list[str]]:
    lattice = _hexagonal(3.209, 5.211)
    positions = np.array([[1.0 / 3.0, 2.0 / 3.0, 0.25], [2.0 / 3.0, 1.0 / 3.0, 0.75]])
    return lattice, positions, ["Mg", "Mg"]


def _graphene() -> tuple[np.ndarray, np.ndarray, list[str]]:
    lattice = _hexagonal(2.46, 20.0)
    positions = np.array([[1.0 / 3.0, 2.0 / 3.0, 0.5], [2.0 / 3.0, 1.0 / 3.0, 0.5]])
    return lattice, positions, ["C", "C"]


def _hexagonal_boron_nitride() -> tuple[np.ndarray, np.ndarray, list[str]]:
    lattice = _hexagonal(2.504, 20.0)
    positions = np.array([[1.0 / 3.0, 2.0 / 3.0, 0.5], [2.0 / 3.0, 1.0 / 3.0, 0.5]])
    return lattice, positions, ["B", "N"]


def _molybdenum_disulfide() -> tuple[np.ndarray, np.ndarray, list[str]]:
    lattice = _hexagonal(3.16, 20.0)
    height = 1.56 / 20.0
    positions = np.array(
        [
            [1.0 / 3.0, 2.0 / 3.0, 0.5],
            [2.0 / 3.0, 1.0 / 3.0, 0.5 + height],
            [2.0 / 3.0, 1.0 / 3.0, 0.5 - height],
        ]
    )
    return lattice, positions, ["Mo", "S", "S"]


def _rutile() -> tuple[np.ndarray, np.ndarray, list[str]]:
    lattice = np.diag([4.594, 4.594, 2.959])
    offset = 0.3053
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [offset, offset, 0.0],
            [1.0 - offset, 1.0 - offset, 0.0],
            [0.5 + offset, 0.5 - offset, 0.5],
            [0.5 - offset, 0.5 + offset, 0.5],
        ]
    )
    return lattice, positions, ["Ti", "Ti", "O", "O", "O", "O"]


POINT_GROUP_CASES = [
    ("silicon primitive", _silicon(), "m-3m", 48, "cubic"),
    ("silicon conventional", _silicon_conventional(), "m-3m", 192, "cubic"),
    ("rock salt", _rock_salt(5.64), "m-3m", 48, "cubic"),
    ("tungsten", _body_centred_tungsten(3.165), "m-3m", 96, "cubic"),
    ("magnesium", _hcp_magnesium(), "6/mmm", 24, "hexagonal"),
    ("graphene", _graphene(), "6/mmm", 24, "hexagonal"),
    ("boron nitride", _hexagonal_boron_nitride(), "-6m2", 12, "hexagonal"),
    ("molybdenum disulfide", _molybdenum_disulfide(), "-6m2", 12, "hexagonal"),
    ("rutile", _rutile(), "4/mmm", 16, "tetragonal"),
]


@pytest.mark.parametrize("name,case,symbol,order,system", POINT_GROUP_CASES)
def test_point_group_and_operation_count(name, case, symbol, order, system):
    lattice, positions, species = case
    dataset = sym.analyse_symmetry(lattice, positions, species)
    assert dataset.point_group == symbol, name
    assert dataset.operation_count == order, name
    assert dataset.crystal_system == system, name


@pytest.mark.parametrize("name,case,symbol,order,system", POINT_GROUP_CASES)
def test_operations_form_a_group(name, case, symbol, order, system):
    """The returned operations must be closed under composition modulo a lattice
    translation, and every one of them must map the structure onto itself."""

    lattice, positions, species = case
    dataset = sym.analyse_symmetry(lattice, positions, species)
    rotations = dataset.rotations
    translations = dataset.translations

    keys = {
        (tuple(rotation.reshape(-1).tolist()), tuple(np.round(np.mod(shift, 1.0), 5).tolist()))
        for rotation, shift in zip(rotations, translations)
    }
    assert len(keys) == len(rotations), name

    for index_a in range(len(rotations)):
        for index_b in range(len(rotations)):
            product = rotations[index_a] @ rotations[index_b]
            shift = rotations[index_a] @ translations[index_b] + translations[index_a]
            key = (tuple(product.reshape(-1).tolist()), tuple(np.round(np.mod(shift, 1.0), 5).tolist()))
            assert key in keys, f"{name}: composition left the group"


@pytest.mark.parametrize("name,case,symbol,order,system", POINT_GROUP_CASES)
def test_operations_map_the_structure_onto_itself(name, case, symbol, order, system):
    lattice, positions, species = case
    wrapped = np.mod(np.asarray(positions, dtype=float), 1.0)
    dataset = sym.analyse_symmetry(lattice, positions, species)
    labels = np.asarray(species)
    for rotation, shift in zip(dataset.rotations, dataset.translations):
        image = np.mod(wrapped @ np.asarray(rotation, dtype=float).T + shift, 1.0)
        for point, label in zip(image, labels):
            delta = wrapped - point
            delta -= np.round(delta)
            distance = np.linalg.norm(delta @ np.asarray(lattice, dtype=float), axis=1)
            nearest = int(np.argmin(distance))
            assert distance[nearest] < 1e-4, name
            assert labels[nearest] == label, name


def test_niggli_cell_is_independent_of_the_input_basis():
    """The Niggli cell is a function of the lattice, not of the basis chosen for
    it, so a unimodular change of basis must leave it unchanged."""

    generator = np.random.default_rng(20240822)
    lattices = [
        _fcc(5.43),
        _hexagonal(3.209, 5.211),
        np.diag([4.594, 4.594, 2.959]),
        np.array([[3.0, 0.0, 0.0], [0.4, 4.0, 0.0], [0.7, 0.9, 5.0]]),
    ]
    for lattice in lattices:
        reference, _ = sym.niggli_reduce(lattice)
        reference_metric = reference @ reference.T
        for _ in range(20):
            transform = np.eye(3, dtype=int)
            for _ in range(6):
                row, column = generator.choice(3, size=2, replace=False)
                transform[row] += int(generator.integers(-2, 3)) * transform[column]
            assert abs(round(float(np.linalg.det(transform)))) == 1
            reduced, integer_transform = sym.niggli_reduce(transform @ lattice)
            assert np.allclose(reduced, integer_transform @ (transform @ lattice), atol=1e-8)
            assert np.allclose(reduced @ reduced.T, reference_metric, atol=1e-6)


def test_delaunay_cell_keeps_the_lattice():
    lattice = np.array([[3.0, 0.0, 0.0], [0.4, 4.0, 0.0], [0.7, 0.9, 5.0]])
    reduced, transform = sym.delaunay_reduce(lattice)
    assert np.allclose(reduced, transform @ lattice, atol=1e-9)
    assert abs(abs(float(np.linalg.det(transform))) - 1.0) < 1e-9
    assert np.linalg.norm(reduced, axis=1).sum() <= np.linalg.norm(lattice, axis=1).sum() + 1e-9


def test_primitive_cell_of_a_conventional_silicon_cell():
    lattice, positions, species = _silicon_conventional()
    primitive_lattice, primitive_positions, primitive_species = sym.primitive_cell(lattice, positions, species)
    volume = abs(float(np.linalg.det(primitive_lattice)))
    assert abs(volume - abs(float(np.linalg.det(lattice))) / 4.0) < 1e-6
    assert len(primitive_positions) == 2
    assert set(primitive_species) == {"Si"}
    dataset = sym.analyse_symmetry(primitive_lattice, primitive_positions, primitive_species)
    assert dataset.point_group == "m-3m"
    assert dataset.operation_count == 48


def test_centering_translations_of_a_face_centred_cell():
    lattice, positions, species = _silicon_conventional()
    dataset = sym.analyse_symmetry(lattice, positions, species)
    assert len(dataset.primitive_translations) == 4
    expected = {(0.0, 0.0, 0.0), (0.0, 0.5, 0.5), (0.5, 0.0, 0.5), (0.5, 0.5, 0.0)}
    found = {tuple(np.round(np.mod(shift, 1.0), 6).tolist()) for shift in dataset.primitive_translations}
    assert found == expected


def test_equivalent_atoms_of_molybdenum_disulfide():
    lattice, positions, species = _molybdenum_disulfide()
    dataset = sym.analyse_symmetry(lattice, positions, species)
    orbits = dataset.equivalent_atoms
    assert orbits[1] == orbits[2], "the two sulfur atoms are related by the mirror plane"
    assert orbits[0] != orbits[1]
    assert len(set(orbits.tolist())) == 2


def test_diamond_silicon_is_not_symmorphic_in_the_standard_setting():
    lattice, positions, species = _silicon()
    dataset = sym.analyse_symmetry(lattice, positions, species)
    assert dataset.has_inversion
    assert not dataset.symmorphic_setting


def _supercell(
    lattice: np.ndarray,
    positions: np.ndarray,
    species: list[str],
    transform: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return the supercell ``transform @ lattice`` with every image of every atom."""

    multiple = int(round(abs(float(np.linalg.det(transform.astype(float))))))
    inverse = np.linalg.inv(transform.astype(float))
    reach = int(np.ceil(np.abs(transform).sum())) + 1
    shifts = np.array(
        [[x, y, z] for x in range(-reach, reach + 1) for y in range(-reach, reach + 1) for z in range(-reach, reach + 1)],
        dtype=float,
    )
    new_positions: list[np.ndarray] = []
    new_species: list[str] = []
    for point, symbol in zip(positions, species):
        images = np.mod((point + shifts) @ inverse, 1.0)
        kept: list[np.ndarray] = []
        for image in images:
            if not any(np.allclose(np.mod(image - other + 0.5, 1.0) - 0.5, 0.0, atol=1e-8) for other in kept):
                kept.append(image)
        assert len(kept) == multiple
        new_positions.extend(kept)
        new_species.extend([symbol] * multiple)
    return transform.astype(float) @ lattice, np.asarray(new_positions), new_species


@pytest.mark.parametrize(
    "transform",
    [
        np.array([[1, 0, 0], [0, 1, 0], [0, 0, 2]]),
        np.array([[1, 2, 0], [0, 1, 3], [2, 0, 5]]),
        np.array([[3, 1, 1], [1, 3, 1], [-2, 1, 4]]),
    ],
)
def test_the_primitive_cell_of_a_skewed_supercell_is_the_reduced_cell(transform):
    """A primitive cell is only useful if it is also a *short* cell.

    Any basis of the translation lattice has the right volume, but a sheared one
    wastes plane waves and k-points.  Whatever supercell is handed in, the
    answer has to be the Niggli cell of the primitive lattice, so its metric is
    the metric of the ideal face-centred cubic primitive cell.
    """

    lattice, positions, species = _silicon()
    reference, _ = sym.niggli_reduce(lattice)
    big_lattice, big_positions, big_species = _supercell(lattice, positions, species, transform)
    primitive_lattice, primitive_positions, primitive_species = sym.primitive_cell(
        big_lattice, big_positions, big_species
    )
    assert len(primitive_positions) == 2
    assert primitive_species == ["Si", "Si"]
    assert abs(abs(float(np.linalg.det(primitive_lattice))) - abs(float(np.linalg.det(lattice)))) < 1e-6
    assert float(np.linalg.det(primitive_lattice)) > 0.0, "the reported cell must stay right-handed"
    assert np.allclose(
        primitive_lattice @ primitive_lattice.T, reference @ reference.T, atol=1e-6
    ), "the primitive cell must be the reduced one, not an arbitrary basis of the same lattice"
    dataset = sym.analyse_symmetry(primitive_lattice, primitive_positions, primitive_species)
    assert dataset.operation_count == 48


def _direct_symmorphic_scan(translations: np.ndarray, centering: np.ndarray) -> bool:
    """The straight ``|operations| x |centering|`` comparison, for reference."""

    residues = np.asarray(translations, dtype=float)[:, None, :] - np.asarray(
        centering, dtype=float
    )[None, :, :]
    residues -= np.rint(residues)
    return bool(np.all(np.any(np.all(np.abs(residues) <= 1e-6, axis=2), axis=1)))


@pytest.mark.parametrize(
    "builder",
    [
        _silicon,
        _molybdenum_disulfide,
        lambda: (_fcc(4.05), np.zeros((1, 3)), ["Al"]),
        lambda: (np.eye(3) * 3.0, np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]), ["W", "W"]),
    ],
)
def test_the_symmorphic_test_agrees_with_the_direct_scan(builder):
    """The bucketed lookup answers exactly what comparing everything answers."""

    lattice, positions, species = builder()
    for transform in (np.eye(3, dtype=int), np.diag([2, 1, 1]), np.diag([2, 2, 1])):
        cell, sites, kinds = _supercell(lattice, positions, species, transform)
        dataset = sym.analyse_symmetry(cell, sites, kinds)
        centering = sym.pure_translations(dataset.rotations, dataset.translations)
        assert dataset.symmorphic_setting == _direct_symmorphic_scan(
            dataset.translations, centering
        )


def test_the_symmorphic_test_rejects_a_translation_that_is_not_a_centering():
    centering = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.0]])
    assert sym._translations_are_centering(np.array([[0.5, 0.5, 0.0]]), centering)
    assert sym._translations_are_centering(np.array([[1.5, -0.5, 2.0]]), centering)
    assert not sym._translations_are_centering(np.array([[0.25, 0.5, 0.0]]), centering)
    assert not sym._translations_are_centering(np.array([[0.5, 0.5, 0.0]]), np.zeros((0, 3)))
