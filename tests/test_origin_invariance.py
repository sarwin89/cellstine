"""Where the origin of the cell is put is not physics, and must change nothing.

Every quantity below is a property of the crystal, not of the coordinates it
arrives in: a rigid shift of all the atoms -- by half a cell, by a third, by a
generic amount, or by a hair *below* zero, where wrapping has to choose between
``0`` and ``1`` and a boundary atom can be counted twice or lost -- leaves the
space-group order, the Bravais symbol, the slab a Miller cut produces and the
adsorption sites on it exactly as they were.

The one quantity that is allowed to move is the covering-radius *bound*: it is
the output of a branch and bound that stops within a tolerance of the true
maximum, and where the boxes fall relative to the atoms depends on the origin.
It is checked to agree to within that tolerance, which is the promise the
routine makes.

The same argument applies one level up, to the basis itself: a lattice handed
over in a different (unimodular) basis, or rigidly rotated into a different
frame, is the same lattice, and has to be classified the same way.
"""

from __future__ import annotations

import numpy as np
import pytest

from cellstine.core.bravais import bravais_symbol
from cellstine.core.covering import bulk_covering_radius_bound
from cellstine.core.symmetry3d import analyse_symmetry
from cellstine.interface.surface.backend import build_surface_structure, find_adsorption_sites

from conftest import write_poscar

SQRT3 = np.sqrt(3.0)

CRYSTALS = {
    "fcc_copper": (
        3.6 * np.array([[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]]),
        ["Cu"],
        [1],
        np.array([[0.0, 0.0, 0.0]]),
    ),
    "hcp_magnesium": (
        np.array([[3.21, 0.0, 0.0], [-1.605, 3.21 * SQRT3 / 2.0, 0.0], [0.0, 0.0, 5.21]]),
        ["Mg"],
        [2],
        np.array([[1 / 3, 2 / 3, 0.25], [2 / 3, 1 / 3, 0.75]]),
    ),
    "rocksalt_sodium_chloride": (
        5.64 * np.array([[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]]),
        ["Na", "Cl"],
        [1, 1],
        np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]),
    ),
    "triclinic_carbon": (
        np.array([[4.0, 0.0, 0.0], [1.1, 3.7, 0.0], [0.6, 0.9, 4.3]]),
        ["C"],
        [1],
        np.array([[0.0, 0.0, 0.0]]),
    ),
}

SHIFTS = {
    "half": np.full(3, 0.5),
    "just_below_zero": np.full(3, -1e-13),
    "third": np.full(3, 1 / 3),
    "generic": np.array([0.137, 0.911, 0.4242]),
}

MILLERS = [(1, 0, 0), (1, 1, 0), (1, 1, 1), (2, 1, 0)]

COVERING_TOLERANCE = 0.02


def _labels(species, counts):
    return [symbol for symbol, count in zip(species, counts) for _ in range(int(count))]


def _slab_measurements(path, miller):
    build = build_surface_structure(
        str(path), miller=miller, layers=5, vacuum=12.0, repeat_a=1, repeat_b=1
    )
    slab = build.structure
    lattice = np.asarray(slab.lattice, dtype=float)
    cartesian = np.asarray(slab.positions_direct, dtype=float) @ lattice
    sites = find_adsorption_sites(slab, surface_side="top")
    return {
        "atoms": int(sum(slab.counts)),
        "area": float(np.linalg.norm(np.cross(lattice[0], lattice[1]))),
        "thickness": float(cartesian[:, 2].max() - cartesian[:, 2].min()),
        "sites": tuple(sorted(sites.site_counts.items())),
    }


def _written(directory, name, crystal, shift):
    lattice, species, counts, positions = crystal
    shifted = np.mod(np.asarray(positions, dtype=float) + shift, 1.0)
    path = write_poscar(directory / f"{name}.vasp", lattice, species, counts, shifted)
    return path, shifted


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    return tmp_path_factory.mktemp("origin-invariance")


@pytest.mark.parametrize("name", sorted(CRYSTALS))
@pytest.mark.parametrize("shift", sorted(SHIFTS))
def test_the_space_group_and_bravais_type_do_not_move_with_the_origin(workspace, name, shift):
    lattice, species, counts, positions = CRYSTALS[name]
    labels = _labels(species, counts)
    reference = analyse_symmetry(lattice, positions, labels)
    _path, shifted = _written(workspace, f"{name}_{shift}", CRYSTALS[name], SHIFTS[shift])
    moved = analyse_symmetry(lattice, shifted, labels)
    assert len(moved.rotations) == len(reference.rotations)
    assert moved.point_group == reference.point_group
    assert moved.crystal_system == reference.crystal_system
    assert moved.has_inversion == reference.has_inversion


@pytest.mark.parametrize("name", sorted(CRYSTALS))
@pytest.mark.parametrize("shift", sorted(SHIFTS))
@pytest.mark.parametrize("miller", MILLERS)
def test_a_slab_and_its_sites_do_not_move_with_the_origin(workspace, name, shift, miller):
    base, _ = _written(workspace, f"{name}_origin", CRYSTALS[name], np.zeros(3))
    moved, _ = _written(workspace, f"{name}_{shift}", CRYSTALS[name], SHIFTS[shift])
    reference = _slab_measurements(base, miller)
    measured = _slab_measurements(moved, miller)
    assert measured["atoms"] == reference["atoms"]
    assert measured["area"] == pytest.approx(reference["area"], rel=1e-12)
    assert measured["thickness"] == pytest.approx(reference["thickness"], abs=1e-9)
    assert measured["sites"] == reference["sites"]


@pytest.mark.parametrize("name", sorted(CRYSTALS))
@pytest.mark.parametrize("shift", sorted(SHIFTS))
def test_the_covering_bound_agrees_to_its_own_tolerance(workspace, name, shift):
    lattice, _species, _counts, positions = CRYSTALS[name]
    array = np.asarray(lattice, dtype=float)
    reference = bulk_covering_radius_bound(array, np.asarray(positions, dtype=float))
    shifted = np.mod(np.asarray(positions, dtype=float) + SHIFTS[shift], 1.0)
    measured = bulk_covering_radius_bound(array, shifted)
    assert abs(measured - reference) <= COVERING_TOLERANCE


UNIMODULAR = {
    "identity": np.eye(3, dtype=int),
    "shear": np.array([[1, 0, 0], [1, 1, 0], [0, 0, 1]]),
    "swap": np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]]),
    "tangle": np.array([[1, 2, 0], [0, 1, 0], [3, -1, 1]]),
}


def _rotation(axis, angle):
    unit = np.asarray(axis, dtype=float)
    unit = unit / np.linalg.norm(unit)
    cross = np.array(
        [[0.0, -unit[2], unit[1]], [unit[2], 0.0, -unit[0]], [-unit[1], unit[0], 0.0]]
    )
    return (
        np.cos(angle) * np.eye(3)
        + np.sin(angle) * cross
        + (1.0 - np.cos(angle)) * np.outer(unit, unit)
    )


ROTATIONS = {
    "none": np.eye(3),
    "z_thirty": _rotation((0.0, 0.0, 1.0), np.pi / 6.0),
    "diagonal": _rotation((1.0, 1.0, 1.0), 0.7),
    "generic": _rotation((0.3, -0.7, 0.2), 1.9),
}


@pytest.mark.parametrize("name", sorted(CRYSTALS))
@pytest.mark.parametrize("basis", sorted(UNIMODULAR))
@pytest.mark.parametrize("turn", sorted(ROTATIONS))
def test_the_bravais_symbol_is_a_property_of_the_lattice_not_of_its_basis(name, basis, turn):
    """A lattice does not know which basis or which frame it was handed over in.

    Left-multiplying by a unimodular integer matrix picks a different basis of
    the very same lattice, and right-multiplying by a rotation carries the
    whole lattice rigidly into another orientation.  Neither is physics, so the
    Bravais classification has to come out the same every time.
    """

    lattice = np.asarray(CRYSTALS[name][0], dtype=float)
    reference = bravais_symbol(lattice)
    changed = UNIMODULAR[basis] @ lattice @ ROTATIONS[turn].T
    assert abs(abs(np.linalg.det(changed)) - abs(np.linalg.det(lattice))) < 1e-9
    assert bravais_symbol(changed) == reference
