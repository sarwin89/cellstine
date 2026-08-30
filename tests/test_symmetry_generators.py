"""The reduced generating set must generate exactly the same space group.

Grouping sites -- or pairs of sites -- into symmetry orbits is a
connected-component problem, and a connected component is decided by any
generating set of the acting group.  A supercell carries one operation per
(rotation, lattice translation) pair, so an ``n x n x n`` supercell of a cubic
crystal has ``48 n^3`` operations, of which all but a few dozen are redundant;
:func:`cellstine.core.symmetry3d.generating_operations` returns the few.

The reduction is only allowed if it is exact, so these tests close the returned
set under composition and demand set equality with the full operation list, on
crystals that exercise centring (fcc, body-centred), a non-symmorphic group
(diamond, hcp) and a two-species basis (rocksalt).  A second test checks the
consequence that actually matters: the orbits of the atoms are unchanged.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from cellstine.core import geometry, symmetry3d


SIMPLE_CUBIC = (3.0 * np.eye(3), [[0.0, 0.0, 0.0]], ["Po"])
FCC = (
    np.array([[0.0, 2.0, 2.0], [2.0, 0.0, 2.0], [2.0, 2.0, 0.0]]),
    [[0.0, 0.0, 0.0]],
    ["Al"],
)
BODY_CENTRED = (3.0 * np.eye(3), [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]], ["Fe", "Fe"])
DIAMOND = (
    np.array([[0.0, 2.7, 2.7], [2.7, 0.0, 2.7], [2.7, 2.7, 0.0]]),
    [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
    ["Si", "Si"],
)
ROCKSALT = (
    np.array([[0.0, 2.1, 2.1], [2.1, 0.0, 2.1], [2.1, 2.1, 0.0]]),
    [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    ["Na", "Cl"],
)
HCP = (
    np.array([[3.2, 0.0, 0.0], [-1.6, 1.6 * math.sqrt(3.0), 0.0], [0.0, 0.0, 5.2]]),
    [[1.0 / 3.0, 2.0 / 3.0, 0.25], [2.0 / 3.0, 1.0 / 3.0, 0.75]],
    ["Mg", "Mg"],
)
CRYSTALS = {
    "simple_cubic": SIMPLE_CUBIC,
    "fcc": FCC,
    "body_centred": BODY_CENTRED,
    "diamond": DIAMOND,
    "rocksalt": ROCKSALT,
    "hcp": HCP,
}
SUPERCELLS = [(1, 1, 1), (2, 1, 1), (2, 2, 2)]
# Closing a group by hand is quadratic in its order, so the element-for-element
# comparison stops at the cells whose groups are small; the orbit comparison,
# which is the property the callers rely on, runs on all of them.
CLOSURE_SUPERCELLS = [(1, 1, 1), (2, 1, 1)]


def build_supercell(crystal, repeats):
    """Return the lattice, fractional positions and species of a supercell."""

    lattice, basis, species = crystal
    counts = np.asarray(repeats, dtype=float)
    positions = []
    labels: list[str] = []
    for first in range(repeats[0]):
        for second in range(repeats[1]):
            for third in range(repeats[2]):
                shift = np.array([first, second, third], dtype=float)
                positions.append((np.asarray(basis, dtype=float) + shift) / counts)
                labels.extend(species)
    return np.asarray(lattice, dtype=float) * counts[:, None], np.vstack(positions), labels


def operation_key(rotation, translation):
    """A hashable exact-enough identity for one space-group operation."""

    matrix = tuple(int(round(value)) for value in np.asarray(rotation).ravel())
    shift = tuple(
        int(round(value)) % 1000000
        for value in np.round(np.mod(np.asarray(translation, dtype=float), 1.0), 6) * 1000000
    )
    return matrix, shift


def group_closure(rotations, translations, limit=4096):
    """Close a set of operations under composition and return the keys."""

    found = {}
    frontier = []
    for rotation, translation in zip(rotations, translations):
        matrix = np.asarray(rotation, dtype=np.int64)
        shift = np.mod(np.asarray(translation, dtype=float), 1.0)
        key = operation_key(matrix, shift)
        if key not in found:
            found[key] = (matrix, shift)
            frontier.append((matrix, shift))
    while frontier:
        grown = []
        for left_matrix, left_shift in frontier:
            for right_matrix, right_shift in list(found.values()):
                matrix = left_matrix @ right_matrix
                shift = np.mod(left_matrix @ right_shift + left_shift, 1.0)
                key = operation_key(matrix, shift)
                if key not in found:
                    assert len(found) < limit, "closure did not terminate"
                    found[key] = (matrix, shift)
                    grown.append((matrix, shift))
        frontier = grown
    return set(found)


def orbits_from_operations(lattice, positions, rotations, translations):
    """Connected components of the point set under the given operations."""

    count = len(positions)
    parent = list(range(count))

    def find(value):
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    finder = geometry.PeriodicSiteIndex(lattice, positions, tolerance=1e-3)
    for rotation, translation in zip(rotations, translations):
        images = positions @ np.asarray(rotation, dtype=float).T + np.asarray(
            translation, dtype=float
        )
        targets = finder.match(images)
        for source in range(count):
            target = int(targets[source])
            if target < 0:
                continue
            root_a, root_b = find(source), find(target)
            if root_a != root_b:
                parent[max(root_a, root_b)] = min(root_a, root_b)
    grouped: dict[int, list[int]] = {}
    for index in range(count):
        grouped.setdefault(find(index), []).append(index)
    return sorted(tuple(sorted(members)) for members in grouped.values())


@pytest.mark.parametrize("name", sorted(CRYSTALS))
@pytest.mark.parametrize("repeats", CLOSURE_SUPERCELLS)
def test_generators_close_to_the_full_space_group(name, repeats):
    """Composing the kept operations reproduces the group, element for element."""

    lattice, positions, species = build_supercell(CRYSTALS[name], repeats)
    dataset = symmetry3d.analyse_symmetry(lattice, positions, species)
    rotations, translations = symmetry3d.generating_operations(
        dataset.rotations, dataset.translations
    )
    assert len(rotations) <= len(dataset.rotations)
    full = group_closure(dataset.rotations, dataset.translations)
    assert len(full) == len(dataset.rotations)
    assert group_closure(rotations, translations) == full


@pytest.mark.parametrize("name", sorted(CRYSTALS))
@pytest.mark.parametrize("repeats", SUPERCELLS)
def test_generators_give_the_same_site_orbits(name, repeats):
    """The orbits used by the defect analysis are unchanged by the reduction."""

    lattice, positions, species = build_supercell(CRYSTALS[name], repeats)
    dataset = symmetry3d.analyse_symmetry(lattice, positions, species)
    rotations, translations = symmetry3d.generating_operations(
        dataset.rotations, dataset.translations
    )
    assert orbits_from_operations(
        lattice, positions, rotations, translations
    ) == orbits_from_operations(lattice, positions, dataset.rotations, dataset.translations)


def test_generators_of_a_cell_without_symmetry_are_trivial():
    """A cell whose only operation is the identity needs no generator at all."""

    lattice = np.array([[3.0, 0.0, 0.0], [0.1, 3.1, 0.0], [0.2, 0.3, 3.2]])
    positions = np.array([[0.0, 0.0, 0.0], [0.31, 0.17, 0.53]])
    dataset = symmetry3d.analyse_symmetry(lattice, positions, ["Na", "Cl"])
    assert len(dataset.rotations) == 1
    rotations, translations = symmetry3d.generating_operations(
        dataset.rotations, dataset.translations
    )
    assert rotations.shape == (0, 3, 3)
    assert translations.shape == (0, 3)
