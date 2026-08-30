"""The one-call functional API of each subpackage, driven end to end.

Besides the command line and the workflow classes, every stage is also exposed
as a plain function -- ``cellstine.interface.workflow.build.build``,
``cellstine.adsorbate.placement.place.place``, and so on.  Those are the entry
points the package map advertises, so they are run here on real inputs and their
output structures are checked, rather than only their existence.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from cellstine.adsorbate.assemble import assemble
from cellstine.adsorbate.placement.place import place
from cellstine.adsorbate.transform.move import move
from cellstine.interface.surface.sites import sites
from cellstine.interface.surface.surface import Surface
from cellstine.interface.workflow.build import build
from cellstine.interface.workflow.match import match
from cellstine.io import native as io_mod
from cellstine.moire.transform.translate import translate
from cellstine.moire.transform.translaten import translaten

from conftest import write_poscar


@pytest.fixture()
def workspace(tmp_path, monkeypatch) -> Path:
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture()
def aluminium(workspace) -> str:
    lattice = 4.05 * np.eye(3)
    positions = np.array([[0.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]])
    return str(write_poscar(workspace / "al.vasp", lattice, ["Al"], [4], positions))


@pytest.fixture()
def silicon(workspace) -> str:
    lattice = 5.431 * np.eye(3)
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.0],
            [0.25, 0.25, 0.25],
            [0.25, 0.75, 0.75],
            [0.75, 0.25, 0.75],
            [0.75, 0.75, 0.25],
        ]
    )
    return str(write_poscar(workspace / "si.vasp", lattice, ["Si"], [8], positions))


@pytest.fixture()
def carbon_monoxide(workspace) -> str:
    positions = np.array([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5 + 1.128 / 12.0]])
    return str(
        write_poscar(workspace / "co.vasp", np.diag([12.0, 12.0, 12.0]), ["C", "O"], [1, 1], positions)
    )


@pytest.fixture()
def aluminium_slab(aluminium) -> str:
    result = Surface().surface(bulk_poscar=aluminium, miller="1,1,1", layers=4, vacuum=15.0)
    return str(result.artifacts["slab_poscar"])


def _labelled_cartesian(structure):
    labels = [
        symbol for symbol, count in zip(structure.species, structure.counts) for _ in range(count)
    ]
    return labels, np.asarray(structure.positions_cartesian, dtype=float)


def test_sites_reports_the_four_sites_of_a_close_packed_face(aluminium_slab):
    """``interface.surface.sites.sites`` on Al(111)."""

    result = sites(slab_poscar=aluminium_slab)
    report = json.loads(Path(result.artifacts["sites_json"]).read_text())
    assert report["site_counts"] == {"top": 1, "bridge": 3, "hcp_hollow": 1, "fcc_hollow": 1}


def test_match_then_build_makes_the_matched_cell(aluminium, silicon):
    """``interface.workflow.match.match`` feeds ``...build.build``."""

    matched = match(
        bottom_bulk=aluminium,
        top_bulk=silicon,
        bottom_millers=["1,1,1"],
        top_millers=["1,1,1"],
        max_length=13.0,
        preview_limit=0,
    )
    matches_path = Path(matched.artifacts["matches_json"])
    best = json.loads(matches_path.read_text())["matches"][0]

    built = build(
        bottom_input=aluminium,
        top_input=silicon,
        bottom_kind="bulk",
        top_kind="bulk",
        match_json=str(matches_path),
        match_index=1,
        gap=2.5,
    )
    structure = io_mod.read_poscar(str(built.artifacts["interface_poscar"]))
    assert sum(structure.counts) == int(best["total_atoms"])
    lattice = np.asarray(structure.lattice, dtype=float)
    assert float(np.linalg.norm(lattice[0])) == pytest.approx(float(best["cell_a"]), rel=1e-6)
    assert float(np.linalg.norm(lattice[1])) == pytest.approx(float(best["cell_b"]), rel=1e-6)


def test_place_then_move_keeps_the_molecule_rigid(aluminium_slab, carbon_monoxide):
    """``adsorbate.placement.place.place`` then ``adsorbate.transform.move.move``."""

    placed = place(
        substrate_poscar=aluminium_slab,
        molecule_poscar=carbon_monoxide,
        substrate_kind="slab",
        site_type="fcc_hollow",
        height=2.0,
        substrate_repeat_a=2,
        substrate_repeat_b=2,
    )
    structure = io_mod.read_poscar(str(placed.artifacts["output_poscar"]))
    labels, cartesian = _labelled_cartesian(structure)
    metal = cartesian[[i for i, label in enumerate(labels) if label == "Al"]]
    molecule = cartesian[[i for i, label in enumerate(labels) if label in {"C", "O"}]]
    assert float(molecule[:, 2].min() - metal[:, 2].max()) == pytest.approx(2.0, abs=1e-6)
    assert float(np.linalg.norm(molecule[0] - molecule[1])) == pytest.approx(1.128, abs=1e-6)

    moved = move(
        poscar_path=str(placed.artifacts["output_poscar"]),
        target_cartesian=[1.0, 1.0],
        rotation_deg=30.0,
    )
    after = io_mod.read_poscar(str(moved.artifacts["output_poscar"]))
    labels, cartesian = _labelled_cartesian(after)
    metal_after = cartesian[[i for i, label in enumerate(labels) if label == "Al"]]
    molecule_after = cartesian[[i for i, label in enumerate(labels) if label in {"C", "O"}]]
    # The substrate does not move, and the molecule stays rigid.
    assert np.allclose(np.sort(metal_after[:, 2]), np.sort(metal[:, 2]), atol=1e-9)
    assert float(np.linalg.norm(molecule_after[0] - molecule_after[1])) == pytest.approx(
        1.128, abs=1e-6
    )


def _bilayer(path: Path) -> str:
    """Two square monolayers 3 Angstrom apart in a 20 Angstrom cell."""

    lattice = np.array([[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 20.0]])
    positions = np.array([[0.0, 0.0, 0.25], [0.0, 0.0, 0.4]])
    return str(write_poscar(path, lattice, ["C"], [2], positions))


def test_translate_moves_only_the_upper_layer(workspace):
    """``moire.transform.translate.translate`` shifts what is above the gap."""

    structure_path = _bilayer(workspace / "bilayer.vasp")
    before = io_mod.read_poscar(structure_path)
    result = translate(poscar_path=structure_path, shift_cartesian=[0.5, 0.25], z_cutoff=6.0)
    after = io_mod.read_poscar(str(result.artifacts["output_poscar"]))

    start = np.asarray(before.positions_cartesian, dtype=float)
    end = np.asarray(after.positions_cartesian, dtype=float)
    lower, upper = np.argsort(start[:, 2])
    assert np.allclose(end[lower], start[lower], atol=1e-9)
    assert np.allclose(end[upper][:2] - start[upper][:2], [0.5, 0.25], atol=1e-9)
    assert end[upper][2] == pytest.approx(start[upper][2], abs=1e-9)


def test_translaten_finds_the_cutoff_by_itself(workspace):
    """``moire.transform.translaten.translaten`` without a cutoff.

    The widest gap of the structure is the interlayer gap, so the upper layer is
    the part that moves, and the answer is the same as asking for that cutoff.
    """

    structure_path = _bilayer(workspace / "stack.vasp")
    before = io_mod.read_poscar(structure_path)
    result = translaten(poscar_path=structure_path, shift_direct=[0.25, 0.0])
    after = io_mod.read_poscar(str(result.artifacts["output_poscar"]))

    start = np.asarray(before.positions_cartesian, dtype=float)
    end = np.asarray(after.positions_cartesian, dtype=float)
    lower, upper = np.argsort(start[:, 2])
    assert np.allclose(end[lower], start[lower], atol=1e-9)
    assert np.allclose(end[upper][:2] - start[upper][:2], [0.25 * 3.0, 0.0], atol=1e-9)


def test_assemble_reports_substrate_supercells_that_fit_the_target(aluminium_slab):
    """``adsorbate.assemble.assemble`` searches substrate supercells."""

    result = assemble(
        substrate_poscar=aluminium_slab,
        a_length=10.0,
        max_length=20.0,
        top_strain=0.05,
        bottom_strain=0.05,
        preview_limit=3,
    )
    document = json.loads(Path(result.artifacts["results_json"]).read_text())
    assert document["candidates"], "no substrate supercell was offered"

    # The synthetic target the search ran against is written out alongside the
    # results, so the reported top cells can be checked against it directly.
    target = io_mod.read_poscar(str(result.artifacts["target_poscar"]))
    target_basis = np.asarray(target.lattice, dtype=float)[:2, :2]
    assert np.linalg.norm(target_basis[0]) == pytest.approx(10.0, rel=1e-9)
    assert np.linalg.norm(target_basis[1]) == pytest.approx(10.0, rel=1e-9)
    target_gram = target_basis @ target_basis.T

    for candidate in document["candidates"]:
        # Every reported top cell is an integer supercell of that target, and
        # its Gram matrix is exactly the one the integers predict.
        matrix = np.asarray(candidate["top_matrix"], dtype=float)
        assert np.allclose(matrix, np.rint(matrix), atol=0.0)
        expected = matrix @ target_gram @ matrix.T
        reported = candidate["top_gram"]
        assert float(reported[0]) == pytest.approx(expected[0, 0], rel=1e-9)
        assert float(reported[1]) == pytest.approx(expected[0, 1], abs=1e-9, rel=1e-9)
        assert float(reported[2]) == pytest.approx(expected[1, 1], rel=1e-9)
        # Every reported cell honours the strain budget it was searched under,
        # and fits inside the length limit.
        assert max(abs(value) for value in candidate["top_layer_strain"]) <= 0.05 + 1e-9
        assert max(abs(value) for value in candidate["bottom_layer_strain"]) <= 0.05 + 1e-9
        assert float(candidate["moire_a"]) <= 20.0 + 1e-6
        assert float(candidate["moire_b"]) <= 20.0 + 1e-6
