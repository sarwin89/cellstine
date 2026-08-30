"""The divacancy classes are the true orbits of the pairs.

``defect/sites.py`` never walks the space group over the candidate pairs.  It
keeps a generating set, applies each generator once to every pair inside the
cutoff, and reports the connected components of the resulting graph.  Three
things have to be true for that to be the right answer, and
``RequestProject/PairOrbits.lean`` proves all three:

* the induced action on unordered pairs is a group homomorphism, so a chain of
  generator steps reaches exactly the orbit of the generated group
  (``Cellstine.pairLinked_iff_exists_symmetry``),
* a chain never needs to leave the set of pairs inside the cutoff, because a
  symmetry preserves distances (``Cellstine.siteLinkedOn_iff_siteLinked``),
* the integer address ``min * natoms + max`` identifies a pair
  (``Cellstine.pairCode_injOn``).

The tests below check the same statements numerically on real cells: the
partition the workflow reports is compared, class for class, against a
brute-force sweep that applies *every* symmetry operation to *every* candidate
pair and closes the relation by hand.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from cellstine.core import symmetry3d
from cellstine.defect.workflow import Defect

from conftest import write_poscar


def _candidate_pairs(
    lattice: np.ndarray, positions: np.ndarray, cutoff: float
) -> list[tuple[int, int]]:
    """Every unordered pair of atoms within ``cutoff`` of one another."""

    shifts = np.array(list(itertools.product((-2, -1, 0, 1, 2), repeat=3)), dtype=float)
    pairs = []
    for first in range(len(positions)):
        for second in range(first + 1, len(positions)):
            delta = positions[second] - positions[first]
            images = (delta + shifts) @ lattice
            if float(np.min(np.linalg.norm(images, axis=1))) <= cutoff:
                pairs.append((first, second))
    return pairs


def _brute_force_pair_classes(
    permutations: np.ndarray, pairs: list[tuple[int, int]]
) -> set[frozenset[tuple[int, int]]]:
    """Close the candidate pairs under the *whole* operation list."""

    members = set(pairs)
    remaining = set(pairs)
    classes: set[frozenset[tuple[int, int]]] = set()
    while remaining:
        seed = min(remaining)
        component = {seed}
        frontier = [seed]
        while frontier:
            first, second = frontier.pop()
            for permutation in permutations:
                image_first = int(permutation[first])
                image_second = int(permutation[second])
                image = (
                    min(image_first, image_second),
                    max(image_first, image_second),
                )
                if image in members and image not in component:
                    component.add(image)
                    frontier.append(image)
        classes.add(frozenset(component))
        remaining -= component
    return classes


def _reported_pair_classes(analysis) -> set[frozenset[tuple[int, int]]]:
    """The partition the divacancy sites of an analysis describe."""

    classes = set()
    for site in analysis.sites:
        if site.site_kind != "divacancy":
            continue
        members = {
            (min(entry["indices"]) - 1, max(entry["indices"]) - 1)
            for entry in site.members
        }
        assert len(members) == site.multiplicity
        classes.add(frozenset(members))
    return classes


def _run(tmp_path, name, lattice, species, counts, positions, cutoff):
    path = write_poscar(
        tmp_path / f"{name}.vasp", lattice, species, counts, positions, comment=name
    )
    tool = Defect(
        runs_root=str(tmp_path / "runs"), output_root=str(tmp_path / "output")
    )
    analysis = tool._analyse_record(
        str(path),
        structure_kind="bulk",
        backend="native",
        surface_side="top",
        layer_tolerance=0.35,
        symprec=1e-4,
        divacancy_distance=cutoff,
    )
    dataset = symmetry3d.analyse_symmetry(
        lattice,
        positions,
        [name for name, count in zip(species, counts) for _ in range(count)],
        symprec=1e-4,
    )
    permutations = symmetry3d.site_permutations(
        lattice,
        positions,
        [name for name, count in zip(species, counts) for _ in range(count)],
        dataset.rotations,
        dataset.translations,
        symprec=1e-4,
    )
    return analysis, permutations


def _simple_cubic_supercell(constant: float, repeats: int):
    lattice = np.eye(3) * constant * repeats
    grid = np.arange(repeats) / repeats
    positions = np.array(list(itertools.product(grid, grid, grid)), dtype=float)
    return lattice, positions


CASES = [
    # Face-centred cubic aluminium, conventional cell: every nearest-neighbour
    # pair is equivalent, so the six candidates form a single class.
    (
        "fcc",
        np.eye(3) * 4.05,
        ["Al"],
        [4],
        np.array([[0.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]]),
        3.0,
        [6],
    ),
    # Rocksalt: two species, so like and unlike pairs cannot mix.
    (
        "rocksalt",
        np.eye(3) * 4.2,
        ["Na", "Cl"],
        [4, 4],
        np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.5, 0.5],
                [0.5, 0.0, 0.5],
                [0.5, 0.5, 0.0],
                [0.5, 0.5, 0.5],
                [0.5, 0.0, 0.0],
                [0.0, 0.5, 0.0],
                [0.0, 0.0, 0.5],
            ]
        ),
        3.2,
        [6, 6, 12],
    ),
    # A 2x2x1 tetragonal supercell: the edge pairs and the face-diagonal pairs
    # are inequivalent, so this separates a real grouping from "one class".
    (
        "tetragonal",
        np.diag([6.0, 6.0, 4.4]),
        ["Cu"],
        [4],
        np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.5, 0.5, 0.0]]),
        4.5,
        [2, 4],
    ),
]


@pytest.mark.parametrize("name,lattice,species,counts,positions,cutoff,sizes", CASES)
def test_divacancy_classes_match_the_whole_group(
    tmp_path, name, lattice, species, counts, positions, cutoff, sizes
):
    analysis, permutations = _run(
        tmp_path, name, lattice, species, counts, positions, cutoff
    )
    pairs = _candidate_pairs(lattice, positions, cutoff)
    expected = _brute_force_pair_classes(permutations, pairs)
    reported = _reported_pair_classes(analysis)
    assert reported, "the cutoff was chosen so that there are neighbour pairs"
    assert reported == expected
    # Pinned so that the comparison above cannot pass by both sides collapsing.
    assert sorted(len(members) for members in reported) == sizes


def test_generators_reach_the_same_classes_as_the_full_operation_list(tmp_path):
    """A generating set draws a graph with the same components as the group.

    This is the shortcut the workflow takes, isolated: the connected components
    of the graph the *generators* draw on the pairs are those of the graph the
    whole group draws.
    """

    lattice, positions = _simple_cubic_supercell(2.8, 2)
    labels = ["Po"] * len(positions)
    dataset = symmetry3d.analyse_symmetry(lattice, positions, labels, symprec=1e-4)
    generator_rotations, generator_translations = symmetry3d.generating_operations(
        dataset.rotations, dataset.translations
    )
    assert len(generator_rotations) < len(dataset.rotations), (
        "this cell must genuinely have more operations than generators"
    )
    full = symmetry3d.site_permutations(
        lattice, positions, labels, dataset.rotations, dataset.translations, symprec=1e-4
    )
    generated = symmetry3d.site_permutations(
        lattice,
        positions,
        labels,
        generator_rotations,
        generator_translations,
        symprec=1e-4,
    )
    pairs = _candidate_pairs(lattice, positions, 3.0)
    assert pairs
    assert _brute_force_pair_classes(generated, pairs) == _brute_force_pair_classes(
        full, pairs
    )


def test_the_pair_address_is_injective_on_the_upper_triangle():
    """``min * natoms + max`` separates the pairs the enumeration produces.

    The workflow binary searches this single integer instead of a pair of them;
    ``Cellstine.pairCode_injOn`` is the same statement.
    """

    for natoms in range(2, 12):
        codes = [
            first * natoms + second
            for first in range(natoms)
            for second in range(first + 1, natoms)
        ]
        assert len(set(codes)) == len(codes)
        assert codes == sorted(codes)
