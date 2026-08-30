"""End-to-end runs of the Brillouin-zone sampling stage.

The command line is driven exactly as a user drives it, and the KPOINTS file it
writes is read back and checked against the cell it belongs to: the weights must
add up to the unreduced mesh, the divisions must meet the requested spacing, and
a supercell of the same crystal must be sampled proportionally more coarsely so
that the two runs describe the same physical sampling density.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from cellstine.cli.main import execute_namespace
from cellstine.cli.parsers import build_parser
from cellstine.core import reciprocal as rc
from cellstine.io import kpoints as kpoints_io
from cellstine.io import native as io_mod

from conftest import write_poscar


def run_cli(*argv: str):
    return execute_namespace(build_parser().parse_args(list(argv)))


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
def graphene(workspace) -> str:
    constant = 2.46
    lattice = np.array(
        [
            [constant, 0.0, 0.0],
            [-0.5 * constant, 0.5 * math.sqrt(3.0) * constant, 0.0],
            [0.0, 0.0, 20.0],
        ]
    )
    positions = np.array([[0.0, 0.0, 0.5], [1.0 / 3.0, 2.0 / 3.0, 0.5]])
    return str(write_poscar(workspace / "graphene.vasp", lattice, ["C"], [2], positions))


def test_the_stage_writes_a_mesh_that_meets_the_requested_spacing(aluminium):
    result = run_cli("symmetry", "kpoints", aluminium, "--spacing", "0.25")
    path = Path(result.artifacts["kpoints"])
    assert path.is_file()
    lattice = np.asarray(io_mod.read_poscar(aluminium).lattice, dtype=float)
    divisions = tuple(result.summary["divisions"])
    assert divisions == rc.mesh_divisions_for_spacing(lattice, 0.25)
    assert max(rc.mesh_spacings(lattice, divisions)) <= 0.25 + 1e-12
    parsed = kpoints_io.read_kpoints(path)
    assert parsed.mode == "explicit"
    assert int(parsed.weights.sum()) == int(result.summary["full_point_count"])
    assert parsed.point_count == int(result.summary["irreducible_point_count"])


def test_the_manifest_records_the_mesh_it_wrote(aluminium):
    result = run_cli("symmetry", "kpoints", aluminium, "--divisions", "6,6,6")
    manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
    assert manifest["parameters"]["divisions"] == [6, 6, 6]
    assert manifest["summary"]["full_point_count"] == 216
    assert manifest["summary"]["weight_total"] == 216
    assert manifest["stage"] == "kpoints"


def test_a_supercell_is_sampled_proportionally_more_coarsely(aluminium, workspace):
    """The same spacing on a 2x2x2 supercell gives half the divisions."""

    primitive = run_cli("symmetry", "kpoints", aluminium, "--spacing", "0.2")
    record = io_mod.read_poscar(aluminium)
    lattice = np.asarray(record.lattice, dtype=float)
    repeats = np.diag([2, 2, 2])
    positions = []
    for i in range(2):
        for j in range(2):
            for k in range(2):
                positions.append(
                    (np.asarray(record.positions_direct, dtype=float) + [i, j, k]) / 2.0
                )
    stacked = np.concatenate(positions, axis=0)
    supercell_path = write_poscar(
        workspace / "al222.vasp", repeats @ lattice, ["Al"], [len(stacked)], stacked
    )
    supercell = run_cli("symmetry", "kpoints", str(supercell_path), "--spacing", "0.2")
    assert tuple(supercell.summary["divisions"]) == rc.supercell_divisions(
        tuple(primitive.summary["divisions"]), repeats
    )
    assert supercell.summary["points_per_zone_volume"] == pytest.approx(
        primitive.summary["points_per_zone_volume"], rel=1e-9
    )


def test_the_surface_flag_samples_the_normal_with_one_point(graphene):
    result = run_cli("symmetry", "kpoints", graphene, "--spacing", "0.2", "--surface")
    assert result.summary["divisions"][2] == 1
    plain = run_cli("symmetry", "kpoints", graphene, "--spacing", "0.2")
    assert plain.summary["divisions"][2] > 1, (
        "20 angstrom of vacuum still leaves a zone longer than the requested spacing, "
        "which is exactly what --surface is for"
    )
    tighter = run_cli("symmetry", "kpoints", graphene, "--spacing", "0.05", "--surface")
    assert tighter.summary["divisions"][2] == 1
    assert tighter.summary["divisions"][0] > 1


def test_the_monkhorst_mesh_is_offset_and_still_adds_up(aluminium):
    result = run_cli(
        "symmetry", "kpoints", aluminium, "--divisions", "4,4,4", "--mesh", "monkhorst"
    )
    assert result.summary["shift"] == [0.5, 0.5, 0.5]
    assert result.summary["weight_total"] == result.summary["full_point_count"] == 64
    points = np.asarray(result.payload["points"], dtype=float)
    assert not np.any(np.all(np.isclose(points, 0.0), axis=1))


def test_switching_off_the_symmetry_leaves_only_time_reversal(aluminium):
    reduced = run_cli("symmetry", "kpoints", aluminium, "--divisions", "4,4,4")
    plain = run_cli("symmetry", "kpoints", aluminium, "--divisions", "4,4,4", "--no-symmetry")
    bare = run_cli(
        "symmetry",
        "kpoints",
        aluminium,
        "--divisions",
        "4,4,4",
        "--no-symmetry",
        "--no-time-reversal",
    )
    assert bare.summary["irreducible_point_count"] == 64
    assert reduced.summary["irreducible_point_count"] < plain.summary["irreducible_point_count"]
    assert plain.summary["irreducible_point_count"] < 64


def test_the_layout_can_be_forced_either_way(aluminium):
    listed = run_cli("symmetry", "kpoints", aluminium, "--divisions", "4,4,4", "--list-points")
    automatic = run_cli("symmetry", "kpoints", aluminium, "--divisions", "4,4,4", "--automatic")
    assert kpoints_io.read_kpoints(Path(listed.artifacts["kpoints"])).mode == "explicit"
    parsed = kpoints_io.read_kpoints(Path(automatic.artifacts["kpoints"]))
    assert parsed.mode == "gamma"
    assert parsed.divisions == (4, 4, 4)


def test_an_output_path_is_honoured(aluminium, workspace):
    destination = workspace / "KPOINTS"
    result = run_cli(
        "symmetry", "kpoints", aluminium, "--divisions", "3,3,3", "--output", str(destination)
    )
    assert Path(result.artifacts["kpoints"]) == destination.resolve()
    assert destination.is_file()


def test_a_mesh_needs_a_spacing_or_divisions(aluminium):
    with pytest.raises(ValueError):
        run_cli("symmetry", "kpoints", aluminium)


def test_the_interactive_flow_offers_the_sampling_stage(monkeypatch):
    from cellstine.cli.interactive import build_defect

    monkeypatch.setattr(build_defect, "_prompt_path", lambda *a, **k: "structure.vasp")
    # The two menus are answered in order: how to size the mesh, then which mesh.
    menu_answers = iter(["spacing", "gamma"])
    monkeypatch.setattr(build_defect, "_choice", lambda *a, **k: next(menu_answers))
    monkeypatch.setattr(build_defect, "_prompt_float", lambda *a, **k: 0.3)
    monkeypatch.setattr(build_defect, "_prompt_yes_no", lambda *a, **k: False)
    monkeypatch.setattr(build_defect, "_print_title", lambda *a, **k: None)
    argv = build_defect._build_symmetry_kpoints()
    assert argv[:3] == ["symmetry", "kpoints", "structure.vasp"]
    assert "--spacing" in argv and "0.3" in argv
    parsed = build_parser().parse_args(argv)
    assert parsed.group == "symmetry" and parsed.stage == "kpoints"
    assert parsed.spacing == pytest.approx(0.3)


@pytest.fixture()
def primitive_aluminium(workspace) -> str:
    """The one-atom fcc cell of the same crystal as the ``aluminium`` fixture."""

    lattice = 0.5 * 4.05 * np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    positions = np.zeros((1, 3))
    return str(write_poscar(workspace / "al_primitive.vasp", lattice, ["Al"], [1], positions))


def test_a_non_primitive_cell_is_reported_as_folded(aluminium, primitive_aluminium):
    """A mesh over a centred cell samples a zone that is smaller by the index.

    The conventional cell of aluminium holds four primitive cells, so its zone
    has a quarter of the volume and each of its wavevectors carries four of the
    primitive zone.  The stage says so; the primitive cell of the same crystal
    draws no such note.
    """

    conventional = run_cli("symmetry", "kpoints", aluminium, "--divisions", "6,6,6")
    note = conventional.summary.get("note", "")
    assert "4-fold non-primitive" in note

    primitive = run_cli("symmetry", "kpoints", primitive_aluminium, "--divisions", "6,6,6")
    assert "non-primitive" not in primitive.summary.get("note", "")

    conventional_lattice = np.asarray(io_mod.read_poscar(aluminium).lattice, dtype=float)
    primitive_lattice = np.asarray(io_mod.read_poscar(primitive_aluminium).lattice, dtype=float)
    index = abs(np.linalg.det(conventional_lattice) / np.linalg.det(primitive_lattice))
    assert index == pytest.approx(4.0)


def test_the_band_path_of_a_non_primitive_cell_is_reported_as_folded(
    aluminium, primitive_aluminium
):
    """The band path of a centred cell carries the folding warning.

    The conventional cell of aluminium is simple cubic as a lattice, so its walk
    is the cubic one and every band of the crystal appears on it four times over;
    the primitive cell gives the face-centred walk and no warning.
    """

    conventional = run_cli("symmetry", "kpath", aluminium, "--divisions", "10")
    assert conventional.summary["bravais_symbol"] == "cP"
    assert any(
        "4-fold non-primitive" in warning for warning in conventional.summary["warnings"]
    )

    primitive = run_cli("symmetry", "kpath", primitive_aluminium, "--divisions", "10")
    assert primitive.summary["bravais_symbol"] == "cF"
    assert not any(
        "non-primitive" in warning for warning in primitive.summary.get("warnings", [])
    )
