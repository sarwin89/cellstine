"""A supercell input must not coarsen the *multi-layer* moire search either.

``moire search`` folds each layer onto its own primitive in-plane cell before it
searches, because the commensurate cells of a stack are a property of the
lattices and not of whichever cell happens to be in the file.  ``moire stack-search``
inherited the bilayer engine but not that fold, and the mismatch was worse than
a missed option: the per-pair matrices came back in the *folded* base cell while
the shared lattice, the cell lengths and the atom counts were still computed in
the *given* one, so a ``2 x 2`` graphene base reported a 4.92 A cell holding an
8-atom base layer and a 2-atom layer above it -- three numbers that cannot all
describe one structure -- and every twisted candidate was filtered out because
its length was measured twice too long.

These tests pin the fold for ``stack-search``: a layer given as a supercell of itself
reproduces the primitive search exactly, candidate for candidate; the folded
cells are written next to the results and are the files the reported matrices
refer to; the builder rebuilds the same structure from either document; and
``reduce_layers=False`` still searches the cells exactly as given.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cellstine.core.species import group_species
from cellstine.io import native as io_mod
from cellstine.moire.builder.nlayer import generate_from_results
from cellstine.moire.search.nlayer import read_nlayer_results, run_findn

from conftest import hexagonal_basis


def _honeycomb(constant: float, species: list[str], vacuum: float = 20.0):
    lattice = np.zeros((3, 3))
    lattice[:2, :2] = hexagonal_basis(constant).T
    lattice[2, 2] = vacuum
    positions = np.array([[1.0 / 3.0, 2.0 / 3.0, 0.5], [2.0 / 3.0, 1.0 / 3.0, 0.5]])
    return lattice, positions, list(species)


def _repeat(lattice, positions, species, na: int, nb: int):
    """Return the ``na x nb`` in-plane supercell of a layer."""

    grown = np.diag([float(na), float(nb), 1.0]) @ np.asarray(lattice, dtype=float)
    points = []
    labels = []
    for ia in range(na):
        for ib in range(nb):
            for point, label in zip(np.asarray(positions, dtype=float), species):
                points.append([(point[0] + ia) / na, (point[1] + ib) / nb, point[2]])
                labels.append(label)
    return grown, np.array(points, dtype=float), labels


def _write(path: Path, lattice, positions, species) -> str:
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


def _digest(document: dict) -> list[tuple]:
    """Return the part of a results document that describes the structures."""

    digest = []
    for candidate in document["candidates"]:
        digest.append(
            (
                tuple(tuple(int(value) for value in row) for row in candidate["base_matrix"]),
                int(candidate["base_atom_count"]),
                int(candidate["total_atoms"]),
                int(candidate["coincidence_index"]),
                tuple(round(float(value), 9) for value in candidate["cell_lengths"]),
                round(float(candidate["cell_angle_deg"]), 9),
                tuple(
                    (
                        tuple(tuple(int(value) for value in row) for row in layer["matrix"]),
                        int(layer["atom_count"]),
                        round(float(layer["angle_deg"]), 9),
                    )
                    for layer in candidate["layers"]
                ),
            )
        )
    return digest


@pytest.fixture(scope="module")
def layers(tmp_path_factory):
    """A primitive and a 2x2 graphene, plus a primitive and a 2x1 hBN."""

    workspace = tmp_path_factory.mktemp("nlayer-reduction-inputs")
    graphene = _honeycomb(2.46, ["C", "C"])
    hbn = _honeycomb(2.504, ["B", "N"])
    return {
        "graphene": _write(workspace / "graphene.vasp", *graphene),
        "graphene22": _write(workspace / "graphene_2x2.vasp", *_repeat(*graphene, 2, 2)),
        "hbn": _write(workspace / "hbn.vasp", *hbn),
        "hbn21": _write(workspace / "hbn_2x1.vasp", *_repeat(*hbn, 2, 1)),
    }


def _run(tmp_path: Path, base: str, uppers: list[str], **kwargs):
    run = run_findn(
        base_poscar=base,
        upper_poscars=uppers,
        max_length=12.0,
        layer_strains=0.02,
        max_atoms=400,
        per_layer_limit=8,
        max_candidates=8,
        output_root=str(tmp_path),
        **kwargs,
    )
    return read_nlayer_results(run.result_path)


@pytest.mark.parametrize(
    "base_key, upper_keys",
    [
        ("graphene22", ["graphene"]),
        ("graphene", ["graphene22"]),
        ("graphene22", ["hbn21"]),
        ("graphene22", ["hbn", "graphene22"]),
    ],
)
def test_a_supercell_input_reproduces_the_primitive_search(tmp_path, layers, base_key, upper_keys):
    """Folding makes the answer a property of the lattices, not of the file."""

    primitive = {"graphene22": "graphene", "hbn21": "hbn"}
    reference = _run(
        tmp_path / "reference",
        layers[primitive.get(base_key, base_key)],
        [layers[primitive.get(key, key)] for key in upper_keys],
    )
    coarse_input = _run(
        tmp_path / "folded", layers[base_key], [layers[key] for key in upper_keys]
    )
    assert _digest(coarse_input) == _digest(reference)
    assert coarse_input["candidates"], "the search must still find something"


def test_the_folded_cells_are_written_and_recorded(tmp_path, layers):
    """Every matrix refers to a file on disk, and the user's file is kept beside it."""

    document = _run(tmp_path, layers["graphene22"], [layers["hbn21"], layers["graphene"]])
    search = document["search"]

    assert search["reduce_layers"] is True
    assert search["base_cell_multiplicity"] == 4
    assert search["base_poscar_source"] == layers["graphene22"]
    assert Path(search["base_poscar"]).is_file()
    assert io_mod.read_poscar(search["base_poscar"]).natoms == 2

    # The recorded upper-layer paths are the folded ones; the sources are the input.
    assert search["upper_poscar_sources"] == [layers["hbn21"], layers["graphene"]]
    assert io_mod.read_poscar(search["upper_poscars"][0]).natoms == 2
    # A layer that is already primitive is used exactly as handed in.
    assert search["upper_poscars"][1] == layers["graphene"]

    candidate = document["candidates"][0]
    assert [layer["cell_multiplicity"] for layer in candidate["layers"]] == [2, 1]
    assert candidate["layers"][0]["poscar_source"] == layers["hbn21"]
    assert candidate["layers"][1]["poscar_source"] is None
    for layer in candidate["layers"]:
        assert Path(layer["poscar"]).is_file()


def test_the_builder_rebuilds_the_same_stack_from_either_document(tmp_path, layers):
    """The structure a supercell input builds is the one the primitive input builds."""

    reference = _run(tmp_path / "reference", layers["graphene"], [layers["hbn"]])
    folded = _run(tmp_path / "folded", layers["graphene22"], [layers["hbn21"]])

    index = min(len(reference["candidates"]), len(folded["candidates"]))
    assert index >= 1
    built = []
    for name, document in (("reference", reference), ("folded", folded)):
        run = generate_from_results(
            str(Path(document["search"]["pair_results"][0]).parents[1] / "results_nlayer.json"),
            index=1,
            interlayers=[3.35],
            output_dir=str(tmp_path / f"build_{name}"),
        )
        built.append(io_mod.read_poscar(str(run.output_path)))

    first, second = built
    assert first.natoms == second.natoms
    assert np.allclose(first.lattice, second.lattice, atol=1e-9)
    assert np.allclose(
        np.sort(first.positions_direct, axis=0),
        np.sort(second.positions_direct, axis=0),
        atol=1e-9,
    )


def test_keeping_the_layer_cells_searches_the_given_cells(tmp_path, layers):
    """``--keep-layer-cells`` is still available, and is a different search."""

    document = _run(
        tmp_path, layers["graphene22"], [layers["graphene"]], reduce_layers=False
    )
    search = document["search"]
    assert search["reduce_layers"] is False
    assert search["base_poscar"] == layers["graphene22"]
    assert search["base_poscar_source"] is None
    assert search["base_cell_multiplicity"] == 1
    assert search["base_atom_count"] == 8
    # Every cell of the unfolded search is a repeat of the 2x2 base cell, so the
    # smallest one it can report is that cell itself.
    for candidate in document["candidates"]:
        assert min(candidate["cell_lengths"]) >= 4.92 - 1e-6
