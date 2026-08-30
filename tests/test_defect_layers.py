"""Defects plane by plane, and the direction the planes are counted along.

The assertions are crystallographic.  For the conventional face-centred cubic
cell of aluminium the planes perpendicular to ``(h k l)`` are
``a / sqrt(h^2 + k^2 + l^2)`` apart, and the atoms of the cell fall on the
sub-planes that the face-centring translations add; a symmetric five-layer
Al(100) slab has five atomic planes, of which symmetry leaves only three
distinct, so a vacancy in *every* plane must be five structures and a vacancy
per inequivalent site must be three; and looking the other way along the same
direction must group the same atoms together while numbering the planes from
the other end.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from cellstine.core.directions import resolve_direction
from cellstine.defect.layers import parse_layer_selection, resolve_layer_ids
from cellstine.defect.workflow import Defect
from cellstine.io import native as io_mod

from conftest import write_poscar

ALUMINIUM_CONSTANT = 4.05


@pytest.fixture(scope="module")
def aluminium_bulk_path(tmp_path_factory) -> str:
    path = tmp_path_factory.mktemp("layer-bulk") / "al4.vasp"
    lattice = ALUMINIUM_CONSTANT * np.eye(3)
    positions = np.array([[0.0, 0.0, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]])
    return str(write_poscar(path, lattice, ["Al"], [4], positions, comment="fcc aluminium"))


@pytest.fixture(scope="module")
def aluminium_slab_path(tmp_path_factory) -> str:
    """A five-layer Al(100) slab: an odd number of planes, mirror-symmetric."""

    path = tmp_path_factory.mktemp("layer-slabs") / "al_slab5.vasp"
    spacing = 0.5 * ALUMINIUM_CONSTANT
    height = 4.0 * spacing + 15.0
    lattice = np.diag([ALUMINIUM_CONSTANT, ALUMINIUM_CONSTANT, height])
    cartesian = []
    for layer in range(5):
        z = 7.5 + layer * spacing
        if layer % 2 == 0:
            cartesian += [[0.0, 0.0, z], [spacing, spacing, z]]
        else:
            cartesian += [[spacing, 0.0, z], [0.0, spacing, z]]
    direct = np.asarray(cartesian, dtype=float) @ np.linalg.inv(lattice)
    return str(write_poscar(path, lattice, ["Al"], [10], direct, comment="Al(100) five-layer slab"))


@pytest.fixture()
def workflow(tmp_path) -> Defect:
    return Defect(runs_root=str(tmp_path / "runs"), output_root=str(tmp_path / "output"))


def _analysis(workflow: Defect, path: str, **kwargs):
    return workflow._analyse_record(
        path,
        structure_kind="auto",
        backend="native",
        surface_side="top",
        layer_tolerance=0.35,
        symprec=0.01,
        **kwargs,
    )


# --- the direction of observation -------------------------------------------


@pytest.mark.parametrize(
    "spec, indices",
    [("001", (0, 0, 1)), ("110", (1, 1, 0)), ("111", (1, 1, 1)), ("(1,1,2)", (1, 1, 2))],
)
def test_plane_spacing_is_the_textbook_cubic_value(spec, indices):
    lattice = ALUMINIUM_CONSTANT * np.eye(3)
    direction = resolve_direction(lattice, spec)
    expected = ALUMINIUM_CONSTANT / math.sqrt(sum(value * value for value in indices))
    assert direction.miller == indices
    assert direction.spacing == pytest.approx(expected, rel=1e-12)


def test_a_negated_direction_is_the_opposite_unit_vector():
    lattice = ALUMINIUM_CONSTANT * np.eye(3)
    forward = resolve_direction(lattice, "111")
    backward = resolve_direction(lattice, "-111")
    assert np.allclose(backward.unit, -forward.unit)
    assert backward.spacing == pytest.approx(forward.spacing)


def test_a_direction_along_no_lattice_plane_is_reported_as_such():
    lattice = ALUMINIUM_CONSTANT * np.eye(3)
    direction = resolve_direction(lattice, "cart:1,0.31,0.107")
    assert direction.miller is None
    assert direction.spacing is None
    assert direction.notes


def test_the_lattice_projections_are_multiples_of_the_reported_spacing():
    """Every lattice translation must land on one of the reported planes."""

    lattice = np.array([[3.1, 0.0, 0.0], [-1.55, 2.6846787, 0.0], [0.0, 0.0, 12.3]])
    for spec in ("001", "100", "110", "111", "a", "c"):
        direction = resolve_direction(lattice, spec)
        assert direction.spacing is not None
        for shift in ((1, 0, 0), (0, 1, 0), (0, 0, 1), (2, -3, 1)):
            projection = float(np.asarray(shift, dtype=float) @ lattice @ direction.unit)
            multiple = projection / direction.spacing
            assert multiple == pytest.approx(round(multiple), abs=1e-6)


# --- the plane census --------------------------------------------------------


def test_a_five_layer_slab_has_five_planes_one_interlayer_spacing_apart(workflow, aluminium_slab_path):
    analysis = _analysis(workflow, aluminium_slab_path)
    heights = [float(layer["projection"]) for layer in analysis.layers]
    assert len(heights) == 5
    gaps = np.diff(heights)
    assert np.allclose(gaps, 0.5 * ALUMINIUM_CONSTANT)
    assert all(int(layer["atom_count"]) == 2 for layer in analysis.layers)


def test_the_bulk_cell_read_along_110_shows_the_face_centred_sub_planes(workflow, aluminium_bulk_path):
    analysis = _analysis(workflow, aluminium_bulk_path, view_direction="110")
    heights = [float(layer["projection"]) for layer in analysis.layers]
    step = ALUMINIUM_CONSTANT / (2.0 * math.sqrt(2.0))
    assert len(heights) == 3
    assert np.allclose(np.diff(heights), step)
    assert [int(layer["atom_count"]) for layer in analysis.layers] == [1, 2, 1]


def test_looking_the_other_way_keeps_the_planes_and_reverses_their_numbering(
    workflow, aluminium_slab_path
):
    forward = _analysis(workflow, aluminium_slab_path, view_direction="c*")
    backward = _analysis(workflow, aluminium_slab_path, view_direction="-c*")
    assert len(forward.layers) == len(backward.layers)
    count = len(forward.layers)
    for index, layer in enumerate(forward.layers):
        mirrored = backward.layers[count - 1 - index]
        assert sorted(layer["atom_indices"]) == sorted(mirrored["atom_indices"])
        assert float(layer["projection"]) == pytest.approx(-float(mirrored["projection"]))


def test_the_planes_do_not_move_when_the_structure_is_translated(workflow, tmp_path, aluminium_slab_path):
    original = io_mod.read_poscar(aluminium_slab_path)
    shifted = np.asarray(original.positions_direct, dtype=float) + np.array([0.13, -0.27, 0.05])
    path = write_poscar(
        tmp_path / "shifted.vasp",
        np.asarray(original.lattice, dtype=float),
        list(original.species),
        [int(value) for value in original.counts],
        shifted,
        comment="translated slab",
    )
    moved = _analysis(workflow, str(path))
    fixed = _analysis(workflow, aluminium_slab_path)
    assert len(moved.layers) == len(fixed.layers)
    for left, right in zip(moved.layers, fixed.layers):
        assert sorted(left["atom_indices"]) == sorted(right["atom_indices"])


# --- choosing the planes -----------------------------------------------------


def test_plane_selections_are_read_as_written():
    layers = [{"layer_id": index} for index in range(1, 6)]
    assert resolve_layer_ids(layers, None) is None
    assert resolve_layer_ids(layers, "all") == (1, 2, 3, 4, 5)
    assert resolve_layer_ids(layers, "top") == (5,)
    assert resolve_layer_ids(layers, "bottom") == (1,)
    assert resolve_layer_ids(layers, "surface") == (1, 5)
    assert resolve_layer_ids(layers, "interior") == (2, 3, 4)
    assert resolve_layer_ids(layers, "middle") == (3,)
    assert resolve_layer_ids(layers, "1,3") == (1, 3)
    assert resolve_layer_ids(layers, "2-4") == (2, 3, 4)
    assert resolve_layer_ids(layers, "-1") == (5,)
    assert resolve_layer_ids(layers, [2, 2, 1]) == (1, 2)
    assert parse_layer_selection("2..4") == ("explicit", (2, 3, 4))


def test_a_plane_that_does_not_exist_is_refused():
    layers = [{"layer_id": index} for index in range(1, 4)]
    with pytest.raises(ValueError):
        resolve_layer_ids(layers, "9")
    with pytest.raises(ValueError):
        resolve_layer_ids(layers, "0")
    with pytest.raises(ValueError):
        resolve_layer_ids([{"layer_id": 1}], "interior")


# --- one defect per plane ----------------------------------------------------


def _generated_layers(result) -> list[int | None]:
    return [record["layer_id"] for record in result.payload["generated"]]


def test_every_plane_of_a_slab_gets_its_own_vacancy(workflow, aluminium_slab_path):
    without = workflow.generate(aluminium_slab_path, "vacancy")
    with_layers = workflow.generate(aluminium_slab_path, "vacancy", layers="all")
    # Symmetry ties plane 1 to plane 5 and plane 2 to plane 4, so the orbits
    # are three and the planes are five.
    assert int(without.summary["generated"]) == 3
    assert int(with_layers.summary["generated"]) == 5
    assert sorted(value for value in _generated_layers(with_layers)) == [1, 2, 3, 4, 5]


def test_a_vacancy_removes_exactly_the_atom_of_the_plane_it_names(workflow, aluminium_slab_path):
    host = io_mod.read_poscar(aluminium_slab_path)
    heights = np.asarray(host.positions_cartesian, dtype=float)[:, 2]
    analysis = _analysis(workflow, aluminium_slab_path)
    plane_height = {
        int(layer["layer_id"]): float(layer["projection"]) for layer in analysis.layers
    }
    result = workflow.generate(aluminium_slab_path, "vacancy", layers="all")
    for record in result.payload["generated"]:
        written = io_mod.read_poscar(record["output_path"])
        assert written.natoms == host.natoms - 1
        remaining = np.asarray(written.positions_cartesian, dtype=float)[:, 2]
        expected = plane_height[int(record["layer_id"])]
        # Exactly one atom of that plane is gone, and no other plane changed.
        before = int(np.sum(np.isclose(heights, expected, atol=1e-6)))
        after = int(np.sum(np.isclose(remaining, expected, atol=1e-6)))
        assert after == before - 1
        assert written.natoms == host.natoms - 1


def test_the_outermost_planes_are_the_surfaces(workflow, aluminium_slab_path):
    host = io_mod.read_poscar(aluminium_slab_path)
    heights = np.asarray(host.positions_cartesian, dtype=float)[:, 2]
    top = workflow.generate(aluminium_slab_path, "vacancy", layers="top")
    bottom = workflow.generate(aluminium_slab_path, "vacancy", layers="-1")
    assert _generated_layers(top) == [5]
    # -1 counts from the top, so it is the same plane as "top".
    assert _generated_layers(bottom) == [5]
    written = io_mod.read_poscar(top.payload["generated"][0]["output_path"])
    remaining = np.asarray(written.positions_cartesian, dtype=float)[:, 2]
    assert int(np.sum(np.isclose(remaining, heights.max(), atol=1e-6))) == 1
    lowest = workflow.generate(aluminium_slab_path, "vacancy", layers="bottom")
    assert _generated_layers(lowest) == [1]


def test_the_plane_numbering_follows_the_direction_of_observation(workflow, aluminium_slab_path):
    """Plane 1 seen one way is plane N seen the other way, and is the same atom."""

    forward = workflow.generate(aluminium_slab_path, "vacancy", layers="1", view_direction="c*")
    backward = workflow.generate(aluminium_slab_path, "vacancy", layers="1", view_direction="-c*")
    first = io_mod.read_poscar(forward.payload["generated"][0]["output_path"])
    second = io_mod.read_poscar(backward.payload["generated"][0]["output_path"])
    lowest = np.asarray(first.positions_cartesian, dtype=float)[:, 2]
    highest = np.asarray(second.positions_cartesian, dtype=float)[:, 2]
    host = io_mod.read_poscar(aluminium_slab_path)
    host_heights = np.asarray(host.positions_cartesian, dtype=float)[:, 2]
    # Reading upwards, plane 1 is the bottom; reading downwards it is the top.
    assert int(np.sum(np.isclose(lowest, host_heights.min(), atol=1e-6))) == 1
    assert int(np.sum(np.isclose(highest, host_heights.max(), atol=1e-6))) == 1


def test_substitution_can_be_placed_in_a_chosen_plane(workflow, aluminium_slab_path):
    result = workflow.generate(
        aluminium_slab_path,
        "substitution",
        substitution_species="Mg",
        layers="middle",
    )
    assert _generated_layers(result) == [3]
    written = io_mod.read_poscar(result.payload["generated"][0]["output_path"])
    assert "Mg" in list(written.species)
    magnesium = [
        position
        for symbol, position in zip(
            [
                species
                for species, count in zip(written.species, written.counts)
                for _ in range(int(count))
            ],
            np.asarray(written.positions_cartesian, dtype=float),
        )
        if symbol == "Mg"
    ]
    assert len(magnesium) == 1
    analysis = _analysis(workflow, aluminium_slab_path)
    middle = float(analysis.layers[2]["projection"])
    assert float(magnesium[0][2]) == pytest.approx(middle, abs=1e-6)


def test_the_written_analysis_remembers_the_direction_it_was_read_along(workflow, aluminium_bulk_path):
    result = workflow.analyse(aluminium_bulk_path, view_direction="111")
    payload = result.payload["analysis"]
    assert payload["view_direction"]["miller"] == [1, 1, 1]
    assert payload["view_direction"]["spacing"] == pytest.approx(
        ALUMINIUM_CONSTANT / math.sqrt(3.0)
    )
    stored = Path(result.artifacts["analysis_json"])
    assert stored.is_file()


def test_a_saved_analysis_is_reread_when_another_direction_is_asked_for(workflow, aluminium_bulk_path):
    saved = workflow.analyse(aluminium_bulk_path, view_direction="001")
    analysis_json = str(saved.artifacts["analysis_json"])
    again = workflow.preview(analysis_json, view_direction="111")
    assert again.payload["analysis"]["view_direction"]["miller"] == [1, 1, 1]
    same = workflow.preview(analysis_json, view_direction="001")
    assert same.payload["analysis"]["view_direction"]["miller"] == [0, 0, 1]


# --- one defect per equivalent atom ------------------------------------------


def test_generating_all_sites_writes_one_structure_per_atom(workflow, aluminium_slab_path):
    """``generate='all'`` covers every atom, not one representative per orbit."""

    host = io_mod.read_poscar(aluminium_slab_path)
    result = workflow.generate(aluminium_slab_path, "vacancy", generate="all")
    assert int(result.summary["generated"]) == host.natoms
    removed = []
    for record in result.payload["generated"]:
        written = io_mod.read_poscar(record["output_path"])
        assert written.natoms == host.natoms - 1
        removed.append(tuple(np.round(record["defect_position_direct"], 8)))
    # Each structure is missing a different atom.
    assert len(set(removed)) == host.natoms


def test_generating_all_sites_respects_the_plane_selection(workflow, aluminium_slab_path):
    """Restricted to one plane, only the atoms of that plane are used."""

    result = workflow.generate(aluminium_slab_path, "vacancy", generate="all", layers="top")
    assert int(result.summary["generated"]) == 2
    assert sorted(_generated_layers(result)) == [5, 5]


def test_an_unknown_generation_mode_is_refused(workflow, aluminium_slab_path):
    with pytest.raises(ValueError, match="generate must be"):
        workflow.generate(aluminium_slab_path, "vacancy", generate="everything")
