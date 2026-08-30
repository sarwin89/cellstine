"""The folded in-plane cell must depend on the layer, not on how it was written.

Lagrange--Gauss reduction of a plane basis leaves four right-handed choices --
``(u, v)``, ``(-u, -v)``, ``(v, -u)`` and ``(-v, u)`` -- and they all describe
the same lattice, so any of them is a correct answer.  Which one comes out of
the reduction, though, depends on the cell the layer was handed in on, and the
matrices a moire search reports are written in that cell: two files holding the
same hBN layer, one primitive and one a ``2 x 1`` supercell, folded onto a 120
and a 60 degree cell respectively, and then reported two candidate lists that
could not be compared row by row.

These tests pin the canonical choice: the orbit is the same set whichever member
the reduction returns, so selecting from the orbit by a rule that never looks at
the input makes the folded cell a function of the lattice alone.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from cellstine.core.species import group_species
from cellstine.core.symmetry3d import planar_primitive_layer, planar_translation_basis
from cellstine.io import native as io_mod
from cellstine.moire.search.find import run_find

from conftest import hexagonal_basis


def _layer(rows: np.ndarray, basis_positions: list[list[float]], species: list[str], vacuum: float = 20.0):
    lattice = np.zeros((3, 3))
    lattice[:2, :2] = np.asarray(rows, dtype=float)
    lattice[2, 2] = vacuum
    positions = np.array([[point[0], point[1], 0.5] for point in basis_positions], dtype=float)
    return lattice, positions, list(species)


LAYERS = {
    "hexagonal": _layer(
        hexagonal_basis(2.504).T, [[1 / 3, 2 / 3], [2 / 3, 1 / 3]], ["B", "N"]
    ),
    "square": _layer(np.array([[3.1, 0.0], [0.0, 3.1]]), [[0.0, 0.0]], ["Cu"]),
    "rectangular": _layer(np.array([[3.1, 0.0], [0.0, 4.7]]), [[0.0, 0.0]], ["Cu"]),
    "oblique": _layer(
        np.array([[3.1, 0.0], [1.1, 4.7]]), [[0.0, 0.0], [0.5, 0.25]], ["Cu", "Au"]
    ),
}

# Unimodular rewritings of a cell: the same lattice, a different pair of rows.
GAUGES = [
    np.array([[1, 0], [0, 1]]),
    np.array([[0, 1], [-1, 0]]),
    np.array([[-1, 0], [0, -1]]),
    np.array([[1, 1], [0, 1]]),
    np.array([[2, 1], [1, 1]]),
    np.array([[1, 0], [-3, 1]]),
]


def _rewrite(lattice, positions, gauge):
    """Return the layer on the unimodular rewriting ``gauge`` of its cell."""

    transform = np.eye(3)
    transform[:2, :2] = np.asarray(gauge, dtype=float)
    new_lattice = transform @ np.asarray(lattice, dtype=float)
    # x = f A = f' A' with A' = T A, so f' = f T^{-1}.
    inverse = np.linalg.inv(transform)
    new_positions = np.asarray(positions, dtype=float) @ inverse
    return new_lattice, np.mod(new_positions, 1.0)


def _repeat(lattice, positions, species, na: int, nb: int):
    grown = np.diag([float(na), float(nb), 1.0]) @ np.asarray(lattice, dtype=float)
    points = []
    labels = []
    for ia in range(na):
        for ib in range(nb):
            for point, label in zip(np.asarray(positions, dtype=float), species):
                points.append([(point[0] + ia) / na, (point[1] + ib) / nb, point[2]])
                labels.append(label)
    return grown, np.array(points, dtype=float), labels


@pytest.mark.parametrize("name", sorted(LAYERS))
@pytest.mark.parametrize("na, nb", [(2, 1), (1, 2), (2, 2), (3, 1), (2, 3)])
@pytest.mark.parametrize("gauge_index", range(len(GAUGES)))
def test_the_folded_cell_does_not_depend_on_the_cell_the_layer_was_written_in(
    name, na, nb, gauge_index
):
    """Fold ``na x nb`` repeats, written on six different cells, onto one answer."""

    lattice, positions, species = LAYERS[name]
    reference, _, _, _ = planar_primitive_layer(
        *_repeat(lattice, positions, species, 2, 2), symprec=1e-6
    )
    grown, points, labels = _repeat(lattice, positions, species, na, nb)
    rewritten, rewritten_points = _rewrite(grown, points, GAUGES[gauge_index])
    folded, _, _, index = planar_primitive_layer(
        rewritten, rewritten_points, labels, symprec=1e-6
    )
    assert index == na * nb
    assert np.allclose(folded, reference, atol=1e-8)


@pytest.mark.parametrize("name", sorted(LAYERS))
def test_the_canonical_cell_is_reduced_right_handed_and_obtuse(name):
    """The chosen member of the orbit obeys the convention it is chosen by."""

    lattice, positions, species = LAYERS[name]
    grown, points, labels = _repeat(lattice, positions, species, 2, 3)
    basis, index = planar_translation_basis(grown, points, labels, symprec=1e-6)
    assert index == 6
    rows = (basis[:2] @ np.asarray(grown, dtype=float))[:, :2]
    first, second = rows
    lengths = (float(np.linalg.norm(first)), float(np.linalg.norm(second)))

    assert float(np.linalg.det(rows)) > 0.0, "right handed"
    assert lengths[0] <= lengths[1] + 1e-9, "the shorter row comes first"
    assert 2.0 * abs(float(first @ second)) <= lengths[0] ** 2 + 1e-9, "Lagrange-Gauss reduced"
    if abs(lengths[0] - lengths[1]) <= 1e-6 * lengths[1]:
        assert float(first @ second) <= 1e-9, "the obtuse choice is taken where there is one"
    assert (first[0], first[1]) >= (-1e-9, 0.0), "the first row points along +x"


def _write(path, lattice, positions, species) -> str:
    ordered, counts, order = group_species(list(species))
    io_mod.write_poscar(
        str(path),
        np.asarray(lattice, dtype=float),
        np.asarray(positions, dtype=float)[order],
        [int(value) for value in counts],
        list(ordered),
        "layer",
        positions_are_cartesian=False,
    )
    return str(path)


def _candidate_digest(candidates) -> list[tuple]:
    return [
        (
            tuple(tuple(int(value) for value in row) for row in candidate["top_matrix"]),
            tuple(tuple(int(value) for value in row) for row in candidate["bottom_matrix"]),
            int(candidate["atom_count"]),
            round(float(candidate["angle_deg"]), 9),
            (round(float(candidate["moire_a"]), 9), round(float(candidate["moire_b"]), 9)),
            round(float(candidate["moire_gamma_deg"]), 9),
        )
        for candidate in candidates
    ]


def test_a_bilayer_search_reports_the_same_rows_for_a_supercell_input(tmp_path):
    """The end the user sees: the candidate list is a property of the lattices."""

    graphene = _layer(hexagonal_basis(2.46).T, [[1 / 3, 2 / 3], [2 / 3, 1 / 3]], ["C", "C"])
    hbn = LAYERS["hexagonal"]
    files = {
        "graphene": _write(tmp_path / "graphene.vasp", *graphene),
        "hbn": _write(tmp_path / "hbn.vasp", *hbn),
        "hbn21": _write(tmp_path / "hbn_2x1.vasp", *_repeat(*hbn, 2, 1)),
        "hbn32": _write(tmp_path / "hbn_3x2.vasp", *_repeat(*hbn, 3, 2)),
    }
    reference = run_find(
        top_poscar=files["hbn"],
        bottom_poscar=files["graphene"],
        max_length=14.0,
        top_strain=0.03,
        bottom_strain=0.0,
        max_atoms=300,
        output_root=str(tmp_path / "reference"),
    )
    for name in ("hbn21", "hbn32"):
        run = run_find(
            top_poscar=files[name],
            bottom_poscar=files["graphene"],
            max_length=14.0,
            top_strain=0.03,
            bottom_strain=0.0,
            max_atoms=300,
            output_root=str(tmp_path / name),
        )
        assert _candidate_digest(run.candidates) == _candidate_digest(reference.candidates)
        assert run.candidates, "the search must still find something"


def test_a_hexagonal_layer_folds_onto_the_120_degree_cell():
    """The case that motivated the rule: never report the 60 degree twin."""

    lattice, positions, species = LAYERS["hexagonal"]
    for na, nb in itertools.product((1, 2, 3), repeat=2):
        if na * nb == 1:
            continue
        folded, _, _, _ = planar_primitive_layer(
            *_repeat(lattice, positions, species, na, nb), symprec=1e-6
        )
        rows = np.asarray(folded, dtype=float)[:2, :2]
        angle = math.degrees(
            math.acos(
                float(rows[0] @ rows[1])
                / float(np.linalg.norm(rows[0]) * np.linalg.norm(rows[1]))
            )
        )
        assert angle == pytest.approx(120.0, abs=1e-6)
        assert np.allclose(rows, np.asarray(lattice, dtype=float)[:2, :2], atol=1e-9)
