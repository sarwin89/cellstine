"""Mathematical checks on the symmetry workflow stages.

The engine itself is checked in ``test_symmetry3d``; what is checked here is the
workflow layer that the CLI drives: that the reported operations really map the
written structure onto itself, that the orbits of equivalent atoms partition the
cell, that ``symmetry reduce`` returns a genuine primitive cell (right volume,
right density, sublattice relation, unchanged distance spectrum), and that
``symmetry lattice-reduce`` returns the same lattice in a reduced basis.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from cellstine.io import native as io_mod
from cellstine.symmetry.symmetry import Symmetry

from conftest import write_poscar


def _diamond_conventional(path, constant: float = 5.43):
    lattice = constant * np.eye(3)
    corners = np.array([[0.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]])
    positions = np.vstack([corners, corners + 0.25])
    return str(write_poscar(path, lattice, ["Si"], [8], positions, comment="diamond silicon"))


def _minimum_image_distances(lattice: np.ndarray, direct: np.ndarray) -> np.ndarray:
    """Return the sorted spectrum of periodic interatomic distances."""

    lattice = np.asarray(lattice, dtype=float)
    direct = np.mod(np.asarray(direct, dtype=float), 1.0)
    shifts = np.array(list(itertools.product((-1, 0, 1), repeat=3)), dtype=float) @ lattice
    distances = []
    count = len(direct)
    cartesian = direct @ lattice
    for first in range(count):
        for second in range(count):
            offsets = cartesian[second] - cartesian[first] + shifts
            norms = np.linalg.norm(offsets, axis=1)
            norms = norms[norms > 1e-8]
            distances.append(float(norms.min()))
    return np.sort(np.asarray(distances))


def _successive_minima(lattice: np.ndarray, reach: int = 3) -> np.ndarray:
    """Return the three successive minima of a lattice by direct enumeration."""

    lattice = np.asarray(lattice, dtype=float)
    coefficients = np.array(list(itertools.product(range(-reach, reach + 1), repeat=3)), dtype=float)
    vectors = coefficients @ lattice
    lengths = np.linalg.norm(vectors, axis=1)
    order = np.argsort(lengths)
    chosen: list[np.ndarray] = []
    minima: list[float] = []
    for index in order:
        if lengths[index] <= 1e-12:
            continue
        candidate = np.vstack(chosen + [coefficients[index]])
        if np.linalg.matrix_rank(candidate, tol=1e-9) != len(candidate):
            continue
        chosen.append(coefficients[index])
        minima.append(float(lengths[index]))
        if len(minima) == 3:
            break
    return np.asarray(minima)


@pytest.fixture(scope="module")
def workflow(tmp_path_factory) -> Symmetry:
    workspace = tmp_path_factory.mktemp("symmetry-workflow")
    return Symmetry(runs_root=str(workspace / "runs"), output_root=str(workspace / "output"))


@pytest.fixture(scope="module")
def diamond_path(tmp_path_factory) -> str:
    return _diamond_conventional(tmp_path_factory.mktemp("symmetry-structures") / "si8.vasp")


def test_conventional_diamond_carries_the_full_space_group(workflow, diamond_path):
    """Fd-3m in its conventional setting has 48 x 4 = 192 operations."""

    analysis = workflow.analyse(diamond_path).payload["analysis"]
    assert analysis["operation_count"] == 192
    assert analysis["point_group"] == "m-3m"
    assert analysis["crystal_system"] == "cubic"
    assert analysis["centering_translation_count"] == 4
    assert analysis["laue"] is True
    assert analysis["symmorphic_setting"] is False


def test_the_reported_operations_map_the_structure_onto_itself(workflow, diamond_path):
    record = io_mod.read_poscar(diamond_path)
    direct = np.mod(np.asarray(record.positions_direct, dtype=float), 1.0)
    analysis = workflow.analyse(diamond_path).payload["analysis"]
    for operation in analysis["operations"]:
        rotation = np.asarray(operation["rotation"], dtype=float)
        translation = np.asarray(operation["translation"], dtype=float)
        image = np.mod(direct @ rotation.T + translation, 1.0)
        difference = image[:, None, :] - direct[None, :, :]
        difference -= np.round(difference)
        matched = np.linalg.norm(difference, axis=2) < 1e-6
        assert matched.sum(axis=1).min() == 1, "an operation moved an atom off the structure"
        assert sorted(int(row.argmax()) for row in matched) == list(range(len(direct)))


def test_the_operations_close_under_composition(workflow, diamond_path):
    """The reported set is a group: closed modulo lattice translations."""

    analysis = workflow.analyse(diamond_path).payload["analysis"]
    operations = [
        (np.asarray(item["rotation"], dtype=int), np.asarray(item["translation"], dtype=float))
        for item in analysis["operations"]
    ]
    known = {
        (rotation.tobytes(), tuple(np.round(np.mod(translation, 1.0), 5)))
        for rotation, translation in operations
    }
    generator = operations[: 8]
    for first_rotation, first_translation in generator:
        for second_rotation, second_translation in operations:
            rotation = first_rotation @ second_rotation
            translation = np.mod(first_translation @ second_rotation + second_translation, 1.0)
            key = (rotation.tobytes(), tuple(np.round(translation, 5)))
            assert key in known


def test_equivalent_groups_partition_the_atoms(workflow, mos2_poscar):
    analysis = workflow.analyse(str(mos2_poscar)).payload["analysis"]
    groups = analysis["equivalent_groups"]
    collected = sorted(index for group in groups for index in group["equivalent_indices"])
    assert collected == list(range(1, analysis["atom_count"] + 1))
    assert sum(group["multiplicity"] for group in groups) == analysis["atom_count"]
    multiplicities = sorted(group["multiplicity"] for group in groups)
    assert multiplicities == [1, 2], "the two sulphur atoms of 1H-MoS2 are mirror images"
    for group in groups:
        assert group["representative_index"] in group["equivalent_indices"]


def test_primitive_reduction_of_the_conventional_diamond_cell(workflow, diamond_path, tmp_path):
    source = io_mod.read_poscar(diamond_path)
    destination = tmp_path / "si_primitive.vasp"
    result = workflow.reduce(diamond_path, cell="primitive", output_path=str(destination))
    assert result.summary["atom_count"] == 2

    reduced = io_mod.read_poscar(str(destination))
    source_volume = abs(np.linalg.det(np.asarray(source.lattice, dtype=float)))
    reduced_volume = abs(np.linalg.det(np.asarray(reduced.lattice, dtype=float)))
    assert reduced_volume == pytest.approx(source_volume / 4.0, rel=1e-12)
    assert reduced.natoms / reduced_volume == pytest.approx(source.natoms / source_volume, rel=1e-12)

    # The primitive basis must generate the conventional one.
    transform = np.asarray(source.lattice, dtype=float) @ np.linalg.inv(np.asarray(reduced.lattice, dtype=float))
    assert np.allclose(transform, np.round(transform), atol=1e-9)
    assert round(abs(np.linalg.det(transform))) == 4

    nearest = _minimum_image_distances(reduced.lattice, reduced.positions_direct).min()
    assert nearest == pytest.approx(5.43 * math.sqrt(3.0) / 4.0, rel=1e-9)


def test_primitive_reduction_leaves_a_primitive_cell_alone(workflow, silicon_poscar, tmp_path):
    source = io_mod.read_poscar(str(silicon_poscar))
    destination = tmp_path / "si_primitive_again.vasp"
    workflow.reduce(str(silicon_poscar), cell="primitive", output_path=str(destination))
    reduced = io_mod.read_poscar(str(destination))
    assert reduced.natoms == source.natoms
    assert abs(np.linalg.det(np.asarray(reduced.lattice, dtype=float))) == pytest.approx(
        abs(np.linalg.det(np.asarray(source.lattice, dtype=float))), rel=1e-12
    )
    assert np.allclose(
        _minimum_image_distances(reduced.lattice, reduced.positions_direct),
        _minimum_image_distances(source.lattice, source.positions_direct),
        atol=1e-9,
    )


def test_primitive_reduction_undoes_a_supercell(workflow, silicon_poscar, tmp_path):
    source = io_mod.read_poscar(str(silicon_poscar))
    lattice = np.asarray(source.lattice, dtype=float).copy()
    lattice[0] *= 2.0
    direct = np.asarray(source.positions_direct, dtype=float)
    positions = np.vstack([direct * [0.5, 1.0, 1.0], direct * [0.5, 1.0, 1.0] + [0.5, 0.0, 0.0]])
    supercell = tmp_path / "si_super.vasp"
    write_poscar(supercell, lattice, ["Si"], [4], positions)

    destination = tmp_path / "si_from_super.vasp"
    result = workflow.reduce(str(supercell), cell="primitive", output_path=str(destination))
    assert result.summary["atom_count"] == 2
    reduced = io_mod.read_poscar(str(destination))
    assert abs(np.linalg.det(np.asarray(reduced.lattice, dtype=float))) == pytest.approx(
        abs(np.linalg.det(lattice)) / 2.0, rel=1e-12
    )
    assert np.allclose(
        _minimum_image_distances(reduced.lattice, reduced.positions_direct),
        _minimum_image_distances(source.lattice, source.positions_direct),
        atol=1e-9,
    )


@pytest.mark.parametrize("reduction", ["niggli", "delaunay"])
def test_lattice_reduction_keeps_the_lattice_and_the_structure(workflow, silicon_poscar, tmp_path, reduction):
    source = io_mod.read_poscar(str(silicon_poscar))
    skewed = np.asarray(source.lattice, dtype=float).copy()
    skew = np.array([[1, 0, 0], [3, 1, 0], [-2, 5, 1]], dtype=float)
    skewed = skew @ skewed
    skewed_path = tmp_path / f"si_skew_{reduction}.vasp"
    direct = np.mod(np.asarray(source.positions_cartesian, dtype=float) @ np.linalg.inv(skewed), 1.0)
    write_poscar(skewed_path, skewed, ["Si"], [2], direct)

    destination = tmp_path / f"si_{reduction}.vasp"
    workflow.lattice_reduce(str(skewed_path), reduction=reduction, output_path=str(destination))
    reduced = io_mod.read_poscar(str(destination))
    reduced_lattice = np.asarray(reduced.lattice, dtype=float)

    assert abs(np.linalg.det(reduced_lattice)) == pytest.approx(abs(np.linalg.det(skewed)), rel=1e-10)
    transform = reduced_lattice @ np.linalg.inv(skewed)
    assert np.allclose(transform, np.round(transform), atol=1e-9)
    assert round(abs(np.linalg.det(transform))) == 1

    lengths = np.linalg.norm(reduced_lattice, axis=1)
    assert lengths.max() < np.linalg.norm(skewed, axis=1).max()
    assert np.allclose(
        _minimum_image_distances(reduced_lattice, reduced.positions_direct),
        _minimum_image_distances(source.lattice, source.positions_direct),
        atol=1e-9,
    )


def test_niggli_reduction_satisfies_the_reduction_conditions(workflow, silicon_poscar, tmp_path):
    source = io_mod.read_poscar(str(silicon_poscar))
    skewed = np.array([[1, 0, 0], [4, 1, 0], [7, -3, 1]], dtype=float) @ np.asarray(source.lattice, dtype=float)
    skewed_path = tmp_path / "si_skew_conditions.vasp"
    direct = np.mod(np.asarray(source.positions_cartesian, dtype=float) @ np.linalg.inv(skewed), 1.0)
    write_poscar(skewed_path, skewed, ["Si"], [2], direct)

    destination = tmp_path / "si_niggli_conditions.vasp"
    workflow.lattice_reduce(str(skewed_path), reduction="niggli", output_path=str(destination))
    lattice = np.asarray(io_mod.read_poscar(str(destination)).lattice, dtype=float)

    gram = lattice @ lattice.T
    first, second, third = gram[0, 0], gram[1, 1], gram[2, 2]
    tolerance = 1e-8 * max(first, second, third)
    assert first <= second + tolerance <= third + 2.0 * tolerance
    assert 2.0 * abs(gram[1, 2]) <= second + tolerance
    assert 2.0 * abs(gram[0, 2]) <= first + tolerance
    assert 2.0 * abs(gram[0, 1]) <= first + tolerance
    assert np.allclose(np.sqrt([first, second, third]), _successive_minima(lattice), atol=1e-9)


def test_delaunay_reduction_reaches_the_successive_minima(workflow, silicon_poscar, tmp_path):
    source = io_mod.read_poscar(str(silicon_poscar))
    skewed = np.array([[1, 0, 0], [2, 1, 0], [-5, 3, 1]], dtype=float) @ np.asarray(source.lattice, dtype=float)
    skewed_path = tmp_path / "si_skew_delaunay.vasp"
    direct = np.mod(np.asarray(source.positions_cartesian, dtype=float) @ np.linalg.inv(skewed), 1.0)
    write_poscar(skewed_path, skewed, ["Si"], [2], direct)

    destination = tmp_path / "si_delaunay.vasp"
    workflow.lattice_reduce(str(skewed_path), reduction="delaunay", output_path=str(destination))
    lattice = np.asarray(io_mod.read_poscar(str(destination)).lattice, dtype=float)

    lengths = np.sort(np.linalg.norm(lattice, axis=1))
    assert np.allclose(lengths, _successive_minima(lattice), atol=1e-9)
    assert lengths[0] == pytest.approx(5.43 / math.sqrt(2.0), rel=1e-12)
