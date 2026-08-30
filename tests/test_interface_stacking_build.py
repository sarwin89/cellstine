"""End-to-end checks of stacking reversal and registry selection in interfaces.

Two identical ``ABC``-stacked slabs offer five distinct interfaces out of the
twelve labelled combinations, and every removed combination has to reproduce
the structure it was declared congruent to.  Both statements are checked here
against the structures the builder actually writes, using a fingerprint that no
isometry can change: for each atom the sorted list of its minimum-image
distances to every atom, with the rows themselves sorted.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pytest

from cellstine.interface.surface import registry, stacking
from cellstine.interface.workflow.interface import Interface
from cellstine.io.converters import StructureConverter

from conftest import write_poscar


@pytest.fixture(scope="module")
def aluminium_bulk(tmp_path_factory) -> str:
    path = tmp_path_factory.mktemp("stacking-build") / "al.vasp"
    lattice = 4.05 * np.eye(3)
    positions = np.array([[0.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]])
    return str(write_poscar(path, lattice, ["Al"], [4], positions))


@pytest.fixture(scope="module")
def slab_111(tmp_path_factory, aluminium_bulk) -> str:
    """An Al(111) slab with six close-packed layers."""

    workspace = tmp_path_factory.mktemp("stacking-slab")
    from cellstine.interface.surface.surface import Surface

    tool = Surface(runs_root=str(workspace / "runs"), output_root=str(workspace / "output"))
    result = tool.surface(bulk_poscar=aluminium_bulk, miller="111", layers=6, vacuum=12.0)
    return str(result.artifacts["slab_poscar"])


@pytest.fixture()
def workflow(tmp_path) -> Interface:
    return Interface(runs_root=str(tmp_path / "runs"), output_root=str(tmp_path / "output"))


def _fingerprint(path: str) -> np.ndarray:
    """A distance fingerprint that no isometry can change.

    Each atom contributes the sorted list of its minimum-image distances to
    every atom, and the rows are then sorted, so the result is invariant under
    translation, rotation, reflection and atom relabelling.  It separates the
    fcc and hcp contacts, which the plain multiset of all distances does not:
    an offset ``h`` and its negative ``-h`` give the same distances between two
    layers, so only per-atom environments see the difference.
    """

    record = StructureConverter().read(str(path))
    lattice = np.asarray(record.lattice, dtype=float)
    direct = np.asarray(record.positions_direct, dtype=float)
    shifts = np.array(
        [[i, j, 0] for i in (-1, 0, 1) for j in (-1, 0, 1)], dtype=float
    ) @ lattice
    cartesian = direct @ lattice
    count = cartesian.shape[0]
    rows = np.zeros((count, count), dtype=float)
    for first in range(count):
        for second in range(count):
            difference = cartesian[second] - cartesian[first] + shifts
            rows[first, second] = float(np.min(np.linalg.norm(difference, axis=1)))
        rows[first] = np.sort(rows[first])
    rows = np.round(rows, 4)
    order = np.lexsort(rows.T[::-1])
    return rows[order]


def test_the_generated_slab_is_close_packed(slab_111):
    analysis = stacking.analyse_stacking(StructureConverter().read(slab_111))
    assert analysis.close_packed
    assert analysis.sequence == "ABCABC"
    assert analysis.sense == 1


def test_registries_report_lists_five_distinct_options(workflow, slab_111):
    result = workflow.registries(bottom_input=slab_111, top_input=slab_111, bottom_kind="slab", top_kind="slab")
    assert result.summary["slabs_interchangeable"] is True
    assert result.summary["distinct_options"] == 5
    assert result.summary["listed_options"] == 5
    assert result.summary["labelled_combinations"] == 12
    assert result.summary["bottom_sequence"] == "ABCABC"
    assert Path(result.artifacts["registries_json"]).is_file()
    assert "contact" in result.payload["registry_table"]


def test_registries_report_can_show_the_removed_duplicates(workflow, slab_111):
    result = workflow.registries(
        bottom_input=slab_111,
        top_input=slab_111,
        bottom_kind="slab",
        top_kind="slab",
        include_equivalent=True,
    )
    assert result.summary["distinct_options"] == 5
    assert result.summary["labelled_combinations"] == 12
    assert result.summary["listed_options"] == 12
    table = result.payload["registry_table"]
    assert table.count("mirror image of") == 5
    assert table.count("same interface turned over of") == 2


def _build(workflow, slab, index, tmp_path, *, include_equivalent=False):
    output = Path(tmp_path) / f"interface_{index}.vasp"
    workflow.build(
        bottom_input=slab,
        top_input=slab,
        bottom_kind="slab",
        top_kind="slab",
        gap=2.34,
        vacuum=12.0,
        registry=index,
        include_equivalent=include_equivalent,
        output_path=str(output),
    )
    return output


def test_the_five_options_are_five_different_structures(workflow, slab_111, tmp_path):
    fingerprints = [_fingerprint(_build(workflow, slab_111, index, tmp_path)) for index in range(1, 6)]
    for first, second in itertools.combinations(range(5), 2):
        assert not np.allclose(fingerprints[first], fingerprints[second]), (
            f"options {first + 1} and {second + 1} produced the same structure"
        )


def test_the_removed_options_rebuild_the_options_they_duplicate(workflow, slab_111, tmp_path):
    analysis = stacking.analyse_stacking(StructureConverter().read(slab_111))
    options = registry.enumerate_registry_options(
        analysis, analysis, include_equivalent=True, slabs_interchangeable=True
    )
    assert len(options) == 12
    fingerprints = {
        option.index: _fingerprint(
            _build(workflow, slab_111, option.index, tmp_path, include_equivalent=True)
        )
        for option in options
    }
    pairs = [(option.index, option.equivalent_to) for option in options if option.equivalent_to is not None]
    assert len(pairs) == 7
    for index, duplicated in pairs:
        assert np.allclose(fingerprints[index], fingerprints[duplicated]), (
            f"option {index} should be congruent to option {duplicated}"
        )


def test_the_continuing_registry_rebuilds_the_bulk_crystal(workflow, slab_111, tmp_path):
    """ABC on ABC at the fcc hollow is a single fcc crystal across the contact."""

    output = Path(tmp_path) / "continued.vasp"
    result = workflow.build(
        bottom_input=slab_111,
        top_input=slab_111,
        bottom_kind="slab",
        top_kind="slab",
        gap=4.05 / np.sqrt(3.0),
        vacuum=12.0,
        top_stacking="abc",
        registry="fcc",
        output_path=str(output),
    )
    assert result.summary["stacking"]["kind"] == "fcc_hollow"
    assert result.summary["stacking"]["top_sense"] == "ABC"
    analysis = stacking.analyse_stacking(StructureConverter().read(str(output)))
    assert analysis.sequence == "ABCABCABCABC"
    assert analysis.sense == 1
    heights = np.array([layer.height for layer in analysis.layers])
    spacings = np.diff(heights)
    assert np.allclose(spacings, spacings[0], atol=1e-6)


def test_a_reversed_top_slab_builds_a_twin(workflow, slab_111, tmp_path):
    output = Path(tmp_path) / "twin.vasp"
    result = workflow.build(
        bottom_input=slab_111,
        top_input=slab_111,
        bottom_kind="slab",
        top_kind="slab",
        gap=4.05 / np.sqrt(3.0),
        vacuum=12.0,
        top_stacking="cba",
        registry="hcp",
        output_path=str(output),
    )
    assert result.summary["stacking"]["top_sense"] == "CBA"
    assert result.summary["stacking"]["top_mirrored"] is True
    analysis = stacking.analyse_stacking(StructureConverter().read(str(output)))
    assert analysis.sequence == "ABCABCBACBAC"


def test_eclipsed_registry_puts_the_two_faces_on_top_of_each_other(workflow, slab_111, tmp_path):
    output = Path(tmp_path) / "eclipsed.vasp"
    result = workflow.build(
        bottom_input=slab_111,
        top_input=slab_111,
        bottom_kind="slab",
        top_kind="slab",
        gap=2.5,
        vacuum=12.0,
        registry="eclipsed",
        output_path=str(output),
    )
    assert result.summary["stacking"]["delta"] == 0
    assert result.summary["stacking"]["contact"] in {"A-A", "B-B", "C-C"}
    analysis = stacking.analyse_stacking(StructureConverter().read(str(output)))
    assert analysis.close_packed
    assert analysis.increments[5] == 0


def test_the_contact_letters_are_only_meaningful_as_a_difference(workflow, slab_111, tmp_path):
    """A-A, B-B and C-C name one and the same option."""

    analysis = stacking.analyse_stacking(StructureConverter().read(slab_111))
    options = registry.enumerate_registry_options(analysis, analysis)
    contacts = [option.contact for option in options]
    assert len(set(contacts)) == 3
    for contact in contacts:
        first, second = contact.split("-")
        assert registry.select_registry_option(options, contact).delta == (
            "ABC".index(second) - "ABC".index(first)
        ) % 3


def test_a_registry_is_refused_for_a_matched_supercell(workflow, slab_111, tmp_path):
    with pytest.raises(ValueError, match="same in-plane cell"):
        workflow.build(match_json="unused.json", registry="fcc")


def test_stacking_choices_are_recorded_in_the_manifest(workflow, slab_111, tmp_path):
    import json

    result = workflow.build(
        bottom_input=slab_111,
        top_input=slab_111,
        bottom_kind="slab",
        top_kind="slab",
        gap=2.4,
        vacuum=12.0,
        top_stacking="cba",
        registry="fcc",
        output_path=str(Path(tmp_path) / "recorded.vasp"),
    )
    manifest = json.loads(Path(result.manifest_path).read_text())
    assert manifest["parameters"]["top_stacking"] == "cba"
    assert manifest["parameters"]["registry"] == "fcc"
    assert manifest["summary"]["stacking"]["distinct_options"] == 5


def test_a_bulk_cell_given_to_build_is_cut_into_a_slab(workflow, aluminium_bulk, slab_111):
    """A cell with no vacuum is bulk, whatever the caller forgot to say.

    Reading the conventional fcc cell as a finished slab would stack two bulk
    cells face to face: the interface would have no surfaces, no vacuum, and
    the requested Miller indices and layer count would be silently discarded.
    The detected build must instead agree with the one from the explicit slab.
    """

    detected = workflow.build(
        bottom_input=aluminium_bulk,
        top_input=aluminium_bulk,
        bottom_miller="111",
        top_miller="111",
        bottom_layers=6,
        top_layers=6,
        bottom_vacuum=12.0,
        top_vacuum=12.0,
        gap=2.34,
    )
    assert detected.summary["bottom_kind"] == "bulk (detected)"
    assert detected.summary["top_kind"] == "bulk (detected)"
    assert int(detected.summary["total_atoms"]) == 12

    from_slabs = workflow.build(
        bottom_input=slab_111,
        top_input=slab_111,
        bottom_kind="slab",
        top_kind="slab",
        gap=2.34,
    )
    assert np.allclose(
        _fingerprint(detected.artifacts["interface_poscar"]),
        _fingerprint(from_slabs.artifacts["interface_poscar"]),
    )


def test_a_slab_given_to_build_is_left_alone(workflow, slab_111):
    """Detection must not re-cut a structure that already has vacuum."""

    detected = workflow.build(bottom_input=slab_111, top_input=slab_111, gap=2.34)
    assert detected.summary["bottom_kind"] == "surface (detected)"
    declared = workflow.build(
        bottom_input=slab_111,
        top_input=slab_111,
        bottom_kind="slab",
        top_kind="slab",
        gap=2.34,
    )
    assert np.allclose(
        _fingerprint(detected.artifacts["interface_poscar"]),
        _fingerprint(declared.artifacts["interface_poscar"]),
    )
