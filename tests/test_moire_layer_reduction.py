"""A layer handed in as a supercell of itself must not coarsen the moire search.

The commensurate cells of a bilayer are a property of the two *lattices*, not of
whichever cell of them happens to be in the file.  Before the fold below was put
in, a ``2 x 2`` graphene cell given as a layer made every reported cell four
times too large --- and, worse, deleted the small ones altogether, because no
supercell of a ``2 x 2`` cell realises a 4-atom stack.  These tests pin both the
in-plane reduction itself and the end-to-end promise that the search of a
supercell input reproduces the primitive one exactly.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from cellstine.core.species import expand_species, group_species
from cellstine.core.symmetry3d import planar_primitive_layer, planar_translation_basis
from cellstine.io import native as io
from cellstine.moire.builder.make import generate_from_results
from cellstine.moire.search.find import run_find

from conftest import hexagonal_basis


def _graphene(a: float = 2.46, vacuum: float = 20.0):
    lattice = np.zeros((3, 3))
    lattice[:2, :2] = hexagonal_basis(a).T
    lattice[2, 2] = vacuum
    positions = np.array([[0.0, 0.0, 0.5], [1.0 / 3.0, 2.0 / 3.0, 0.5]])
    return lattice, positions, ["C", "C"]


def _repeat(lattice, positions, species, na: int, nb: int):
    """Return the ``na x nb`` in-plane supercell of a layer."""

    grown = np.diag([float(na), float(nb), 1.0]) @ lattice
    points = []
    labels = []
    for ia in range(na):
        for ib in range(nb):
            for point, label in zip(positions, species):
                points.append([(point[0] + ia) / na, (point[1] + ib) / nb, point[2]])
                labels.append(label)
    return grown, np.array(points, dtype=float), labels


def _write(path, lattice, positions, species, comment="layer"):
    ordered, counts, order = group_species(list(species))
    io.write_poscar(
        str(path),
        np.asarray(lattice, dtype=float),
        np.asarray(positions, dtype=float)[order],
        [int(value) for value in counts],
        list(ordered),
        comment,
        positions_are_cartesian=False,
    )
    return path


def _cell_area(lattice) -> float:
    return float(abs(np.linalg.det(np.asarray(lattice, dtype=float)[:2, :2])))


@pytest.mark.parametrize("na, nb", [(1, 1), (2, 1), (1, 3), (2, 2), (2, 3), (3, 3)])
def test_a_graphene_supercell_folds_back_onto_the_two_atom_cell(na, nb):
    """The index is the number of repeats, and the folded cell is the primitive one."""

    lattice, positions, species = _graphene()
    grown, points, labels = _repeat(lattice, positions, species, na, nb)

    basis, index = planar_translation_basis(grown, points, labels, symprec=1e-4)
    assert index == na * nb
    # The basis is a basis of a lattice containing the input one with that index,
    # so its determinant is the reciprocal.
    assert float(np.linalg.det(basis)) == pytest.approx(1.0 / (na * nb), rel=1e-9)
    # The layer normal is untouched: the third row is c itself.
    assert np.allclose(basis[2], (0.0, 0.0, 1.0))
    assert np.allclose(basis[:2, 2], 0.0)

    folded, folded_points, folded_species, folded_index = planar_primitive_layer(
        grown, points, labels, symprec=1e-4
    )
    assert folded_index == na * nb
    assert len(folded_species) == 2
    assert sorted(folded_species) == ["C", "C"]
    # Same atomic density, same c axis, same in-plane area as the 1x1 cell.
    assert np.allclose(folded[2], grown[2])
    assert _cell_area(folded) == pytest.approx(_cell_area(lattice), rel=1e-9)
    # And the reduction is short: both in-plane vectors have the graphene length.
    assert np.allclose(np.linalg.norm(folded[:2, :2], axis=1), 2.46, rtol=1e-9)
    # No atom is lost or invented: the folded cell holds exactly one site per
    # ``index`` sites of the input, at the same heights.
    assert len(folded_points) * (na * nb) == len(points)
    assert np.allclose(np.asarray(folded_points)[:, 2], 0.5)


def test_a_layer_that_repeats_along_the_normal_is_left_alone():
    """An AA bilayer given as one 'layer' must not be thinned into a monolayer."""

    lattice, positions, species = _graphene()
    stacked_positions = np.vstack(
        [
            np.column_stack((positions[:, 0], positions[:, 1], np.full(2, 0.4))),
            np.column_stack((positions[:, 0], positions[:, 1], np.full(2, 0.4 + 0.15))),
        ]
    )
    stacked_species = ["C"] * 4

    _, index = planar_translation_basis(lattice, stacked_positions, stacked_species, symprec=1e-4)
    assert index == 1
    folded, points, symbols, folded_index = planar_primitive_layer(
        lattice, stacked_positions, stacked_species, symprec=1e-4
    )
    assert folded_index == 1
    assert len(symbols) == 4
    assert np.allclose(folded, lattice)


def test_a_layer_with_a_broken_sublattice_is_not_folded():
    """Two inequivalent species on the two halves defeat the translation."""

    lattice, positions, species = _graphene()
    grown, points, labels = _repeat(lattice, positions, species, 2, 1)
    labels = list(labels)
    labels[0] = "N"  # break the 2 -> 1 translation by decorating one site

    _, index = planar_translation_basis(grown, points, labels, symprec=1e-4)
    assert index == 1


def test_a_supercell_input_reproduces_the_primitive_search(tmp_path):
    """``run_find`` on a 2x2 layer must give exactly the 1x1 candidate list."""

    lattice, positions, species = _graphene()
    primitive = _write(tmp_path / "graphene.vasp", lattice, positions, species)
    grown, points, labels = _repeat(lattice, positions, species, 2, 2)
    supercell = _write(tmp_path / "graphene_2x2.vasp", grown, points, labels)

    common = dict(max_length=14.0, top_strain=0.0, bottom_strain=0.0, max_atoms=200)
    base = run_find(
        top_poscar=str(primitive),
        bottom_poscar=str(primitive),
        output_root=str(tmp_path / "runs_primitive"),
        **common,
    )
    folded = run_find(
        top_poscar=str(supercell),
        bottom_poscar=str(supercell),
        output_root=str(tmp_path / "runs_supercell"),
        **common,
    )

    assert base.layer_index == {"top": 1, "bottom": 1}
    assert folded.layer_index == {"top": 4, "bottom": 4}
    assert len(folded.candidates) == len(base.candidates)
    assert len(base.candidates) > 0

    def fingerprint(candidates):
        return sorted(
            (
                round(float(row["angle_deg"]), 6),
                int(row["atom_count"]),
                round(float(row["moire_a"]), 6),
                round(float(row["moire_b"]), 6),
            )
            for row in candidates
        )

    assert fingerprint(folded.candidates) == fingerprint(base.candidates)
    # The smallest cell of graphene on graphene is the 4-atom untwisted stack.
    assert min(int(row["atom_count"]) for row in folded.candidates) == 4


def test_the_reduced_layer_is_written_and_recorded(tmp_path):
    """Every matrix a run reports must refer to a POSCAR that exists on disk."""

    lattice, positions, species = _graphene()
    grown, points, labels = _repeat(lattice, positions, species, 2, 2)
    supercell = _write(tmp_path / "graphene_2x2.vasp", grown, points, labels)

    run = run_find(
        top_poscar=str(supercell),
        bottom_poscar=str(supercell),
        max_length=12.0,
        top_strain=0.0,
        bottom_strain=0.0,
        max_atoms=100,
        output_root=str(tmp_path / "runs"),
    )
    document = json.loads(run.result_path.read_text())
    metadata = document["metadata"]
    assert metadata["top_layer_index"] == 4
    assert metadata["bottom_layer_index"] == 4
    assert metadata["top_poscar_source"] == str(supercell.resolve())
    assert metadata["bottom_poscar_source"] == str(supercell.resolve())

    for key, name in (("top_poscar", "top"), ("bottom_poscar", "bottom")):
        recorded = run.result_path.parent / f"primitive_{name}.vasp"
        assert recorded.exists()
        assert document["search"][key] == str(recorded)
        used = io.read_poscar(document["search"][key])
        assert used.natoms == 2
        assert _cell_area(used.lattice) == pytest.approx(_cell_area(lattice), rel=1e-6)
        # The reduced layer carries the same species as the input.
        assert set(expand_species(used.species, used.counts)) == {"C"}


def test_keeping_the_layer_cells_reproduces_the_old_coarse_search(tmp_path):
    """``reduce_layers=False`` still searches exactly the cells that were given."""

    lattice, positions, species = _graphene()
    grown, points, labels = _repeat(lattice, positions, species, 2, 2)
    supercell = _write(tmp_path / "graphene_2x2.vasp", grown, points, labels)

    run = run_find(
        top_poscar=str(supercell),
        bottom_poscar=str(supercell),
        max_length=14.0,
        top_strain=0.0,
        bottom_strain=0.0,
        max_atoms=200,
        reduce_layers=False,
        output_root=str(tmp_path / "runs"),
    )
    assert run.layer_index == {"top": 1, "bottom": 1}
    # Every cell is a supercell of the 8-atom input layer pair, so nothing
    # smaller than 16 atoms can appear.
    assert min(int(row["atom_count"]) for row in run.candidates) >= 16
    document = json.loads(run.result_path.read_text())
    assert document["metadata"]["top_poscar_source"] is None
    assert document["search"]["top_poscar"] == str(supercell.resolve())


def test_the_builder_can_still_make_a_bilayer_from_a_reduced_run(tmp_path):
    """The whole find -> make pipeline must work off the folded layer POSCARs."""

    lattice, positions, species = _graphene()
    grown, points, labels = _repeat(lattice, positions, species, 2, 2)
    supercell = _write(tmp_path / "graphene_2x2.vasp", grown, points, labels)

    run = run_find(
        top_poscar=str(supercell),
        bottom_poscar=str(supercell),
        max_length=12.0,
        top_strain=0.0,
        bottom_strain=0.0,
        max_atoms=100,
        output_root=str(tmp_path / "runs"),
    )
    smallest = min(run.candidates, key=lambda row: int(row["atom_count"]))
    built = generate_from_results(
        str(run.result_path),
        index=int(smallest["index"]),
        interlayer_distance=3.35,
        output_dir=str(tmp_path / "built"),
    )
    made = io.read_poscar(str(built.output_path))
    assert made.natoms == int(smallest["atom_count"])
    assert set(expand_species(made.species, made.counts)) == {"C"}
    # A commensurate graphene bilayer keeps the two-atom-per-cell density of each
    # layer, so the in-plane area per atom is half the monolayer cell area.
    assert _cell_area(made.lattice) / made.natoms == pytest.approx(
        _cell_area(lattice) / 4.0, rel=1e-6
    )
