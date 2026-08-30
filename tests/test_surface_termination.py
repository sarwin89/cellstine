"""A slab of a compound has to say what it exposes.

Two properties decide whether a cut of a compound is a usable surface model, and
neither shows up in the atom count: whether the cell still holds a whole number
of formula units, and whether its two faces are the same termination.  The
rocksalt (1 1 1) cut fails the second (Tasker's type III polar surface) and its
odd-layer cuts fail the first; the perovskite (0 0 1) cut fails the first while
keeping both faces alike.  An elemental crystal passes both, always.
"""

from __future__ import annotations

import numpy as np
import pytest

from cellstine.interface.surface import backend as surface
from cellstine.interface.surface.termination import (
    formula_unit,
    layer_species,
    termination_report,
)

from conftest import write_poscar


def _report(path, species, counts, miller, layers, *, vacuum: float = 15.0):
    build = surface.build_surface_structure(
        str(path), miller=miller, layers=layers, vacuum=vacuum
    )
    structure = build.structure
    return termination_report(
        bulk_species=species,
        bulk_counts=counts,
        slab_lattice=structure.lattice,
        slab_positions_cartesian=structure.positions_cartesian,
        slab_species=structure.species,
        slab_counts=structure.counts,
    )


@pytest.fixture(scope="module")
def rocksalt(tmp_path_factory) -> str:
    """Conventional NaCl: two interpenetrating fcc lattices."""

    workspace = tmp_path_factory.mktemp("termination")
    positions = np.array(
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
    )
    return str(
        write_poscar(workspace / "nacl.vasp", 5.64 * np.eye(3), ["Na", "Cl"], [4, 4], positions)
    )


@pytest.fixture(scope="module")
def perovskite(tmp_path_factory) -> str:
    """Cubic SrTiO3, Sr at the corner and Ti at the body centre."""

    workspace = tmp_path_factory.mktemp("termination-perovskite")
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5],
        ]
    )
    return str(
        write_poscar(
            workspace / "srtio3.vasp", 3.905 * np.eye(3), ["Sr", "Ti", "O"], [1, 1, 3], positions
        )
    )


@pytest.fixture(scope="module")
def aluminium(tmp_path_factory) -> str:
    workspace = tmp_path_factory.mktemp("termination-metal")
    positions = np.array([[0.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]])
    return str(write_poscar(workspace / "al.vasp", 4.05 * np.eye(3), ["Al"], [4], positions))


def test_the_formula_unit_is_the_reduced_ratio():
    assert formula_unit({"Na": 108, "Cl": 108}) == {"Na": 1, "Cl": 1}
    assert formula_unit({"Sr": 4, "Ti": 4, "O": 12}) == {"Sr": 1, "Ti": 1, "O": 3}
    assert formula_unit({"Fe": 4, "O": 6}) == {"Fe": 2, "O": 3}
    assert formula_unit({}) == {}


def test_layers_are_grouped_by_height_along_the_normal():
    lattice = np.array([[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 20.0]])
    positions = np.array([[0.0, 0.0, 0.0], [1.5, 1.5, 0.1], [0.0, 0.0, 2.5]])
    layers = layer_species(lattice, positions, ["Na", "Cl", "Na"])
    assert [counts for _, counts in layers] == [{"Cl": 1, "Na": 1}, {"Na": 1}]
    assert layers[0][0] < layers[1][0], "layers are reported from the bottom of the cell upwards"


def test_a_metal_slab_is_stoichiometric_and_symmetric(aluminium):
    report = _report(aluminium, ["Al"], [4], (1, 1, 1), 4)
    assert report.stoichiometric
    assert report.symmetric_terminations
    assert report.excess == {}
    assert report.notes == []
    assert report.formula_units == pytest.approx(4.0)


def test_the_nonpolar_rocksalt_face_is_clean(rocksalt):
    for layers in (4, 5):
        report = _report(rocksalt, ["Na", "Cl"], [4, 4], (1, 0, 0), layers)
        assert report.slab_counts == {"Na": layers, "Cl": layers}
        assert report.stoichiometric
        assert report.symmetric_terminations
        assert report.notes == []


def test_the_polar_rocksalt_face_is_reported_as_two_different_terminations(rocksalt):
    report = _report(rocksalt, ["Na", "Cl"], [4, 4], (1, 1, 1), 4)
    assert report.stoichiometric, "an even cut holds whole formula units"
    assert not report.symmetric_terminations
    assert {report.bottom_termination == {"Na": 1}, report.top_termination == {"Cl": 1}} == {True}
    assert any("dipole" in note for note in report.notes)


def test_an_odd_polar_cut_is_also_short_of_a_formula_unit(rocksalt):
    report = _report(rocksalt, ["Na", "Cl"], [4, 4], (1, 1, 1), 3)
    assert not report.stoichiometric
    assert report.excess == {"Na": 1}
    assert report.formula_units == pytest.approx(1.0)
    assert any("formula unit" in note for note in report.notes)


def test_the_perovskite_cut_keeps_its_faces_but_loses_its_stoichiometry(perovskite):
    report = _report(perovskite, ["Sr", "Ti", "O"], [1, 1, 3], (0, 0, 1), 5)
    assert report.bulk_formula == {"Sr": 1, "Ti": 1, "O": 3}
    assert report.symmetric_terminations, "both faces of an odd cut are the same SrO plane"
    assert not report.stoichiometric
    assert report.excess == {"O": 1, "Sr": 1}
    assert report.notes and all("dipole" not in note for note in report.notes)


def test_the_surface_stage_reports_the_termination(rocksalt, tmp_path):
    from cellstine.interface.surface.surface import Surface

    workflow = Surface(output_root=str(tmp_path / "output"), runs_root=str(tmp_path / "runs"))
    result = workflow.surface(
        bulk_poscar=rocksalt, miller="111", layers=4, vacuum=15.0
    )
    assert result.summary["stoichiometric"] is True
    assert result.summary["symmetric_terminations"] is False
    assert any("dipole" in note for note in result.summary["warnings"])
    termination = result.payload["termination"]
    assert termination["bulk_formula"] == {"Na": 1, "Cl": 1}
    assert termination["layer_count"] == 4
