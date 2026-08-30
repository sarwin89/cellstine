"""Pictures taken along a chosen direction of observation.

The visualiser may be told which way to look at a structure.  Turning the
structure that way must be a rigid motion -- no distance and no angle in the
crystal may change -- and the plan view must then really be the projection an
observer standing on that axis would see.  These tests check the frame that
does the turning, the files the two backends write, and the command lines that
reach them.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from cellstine.cli.main import execute_namespace
from cellstine.cli.parsers import build_parser
from cellstine.core.directions import orthonormal_frame, resolve_direction
from cellstine.visualize.visualize import Visualize

from conftest import write_poscar


def _slab(path: Path, *, constant: float = 4.05, layers: int = 5, spacing: float = 2.0) -> Path:
    """A simple stack of single atoms, one per plane, in a tall cell."""

    height = spacing * layers + 12.0
    lattice = np.diag([constant, constant, height])
    positions = np.array([[0.0, 0.0, (index * spacing + 4.0) / height] for index in range(layers)])
    return write_poscar(path, lattice, ["Al"], [layers], positions, comment="stack")


@pytest.fixture()
def slab(tmp_path) -> Path:
    return _slab(tmp_path / "POSCAR")


def _run_cli(*argv: str):
    return execute_namespace(build_parser().parse_args(list(argv)))


CUBIC = np.diag([4.05, 4.05, 4.05])


@pytest.mark.parametrize("spec", ["a", "c", "111", "(110)", "[112]", "cart:1,2,3", "x"])
def test_the_view_frame_is_a_rotation(spec):
    """The frame is orthonormal and right-handed, so it moves nothing."""

    direction = resolve_direction(CUBIC, spec)
    frame = direction.frame(CUBIC)
    assert np.allclose(frame @ frame.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(frame) == pytest.approx(1.0, abs=1e-12)
    assert np.allclose(frame[2], direction.unit, atol=1e-12)


def test_the_view_frame_preserves_every_distance():
    """Rotating a structure into the frame leaves all its distances alone."""

    rng = np.random.default_rng(20240517)
    points = rng.normal(size=(12, 3)) * 3.0
    frame = resolve_direction(CUBIC, "112").frame(CUBIC)
    turned = points @ frame.T
    before = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    after = np.linalg.norm(turned[:, None, :] - turned[None, :, :], axis=-1)
    assert np.allclose(before, after, atol=1e-12)


def test_the_height_axis_of_the_frame_is_the_projection():
    """The third coordinate in the frame is the depth along the direction."""

    direction = resolve_direction(CUBIC, "110")
    frame = direction.frame(CUBIC)
    points = np.array([[1.0, 0.0, 0.0], [0.0, 2.5, -1.0], [3.0, 3.0, 3.0]])
    assert np.allclose((points @ frame.T)[:, 2], direction.project(points), atol=1e-12)


def test_the_in_plane_axis_follows_the_cell():
    """``u`` is the part of the ``a`` vector that survives the projection."""

    lattice = np.array([[3.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 5.0]])
    direction = resolve_direction(lattice, "001")
    frame = direction.frame(lattice)
    expected = lattice[0] - float(np.dot(lattice[0], direction.unit)) * direction.unit
    assert np.allclose(frame[0], expected / np.linalg.norm(expected), atol=1e-12)


def test_the_frame_may_be_built_without_a_cell():
    """With no cell the frame falls back to the Cartesian axes."""

    frame = orthonormal_frame(np.array([0.0, 0.0, 2.0]))
    assert np.allclose(frame @ frame.T, np.eye(3), atol=1e-12)
    assert np.allclose(frame[2], [0.0, 0.0, 1.0], atol=1e-12)
    assert np.allclose(frame[0], [1.0, 0.0, 0.0], atol=1e-12)


def test_a_zero_direction_is_refused():
    with pytest.raises(ValueError):
        orthonormal_frame(np.zeros(3))


def test_a_direction_along_a_gives_a_perpendicular_first_axis():
    """When ``a`` is the view direction the frame falls back to another axis."""

    direction = resolve_direction(CUBIC, "a")
    frame = direction.frame(CUBIC)
    assert abs(float(np.dot(frame[0], direction.unit))) < 1e-12
    assert np.allclose(np.cross(frame[0], frame[1]), frame[2], atol=1e-12)


def test_reversing_the_direction_keeps_a_right_handed_frame():
    forward = resolve_direction(CUBIC, "111").frame(CUBIC)
    backward = resolve_direction(CUBIC, "-111").frame(CUBIC)
    for frame in (forward, backward):
        assert np.linalg.det(frame) == pytest.approx(1.0, abs=1e-12)
    assert np.allclose(forward[2], -backward[2], atol=1e-12)


def test_the_static_picture_records_the_direction_it_used(slab, tmp_path):
    """The multiview PNG names the direction and the spacing of its planes."""

    pytest.importorskip("matplotlib")

    tool = Visualize(runs_root=str(tmp_path / "runs"), output_root=str(tmp_path / "out"))
    result = tool.structure(structure_path=str(slab), view_direction="110")
    png = Path(str(result.artifacts["png"]))
    assert png.exists() and png.stat().st_size > 0
    assert result.summary["view_direction"] == "(1 1 0) plane normal"
    assert result.summary["plane_spacing"] == pytest.approx(4.05 / math.sqrt(2.0), abs=1e-3)


def test_without_a_direction_the_picture_is_the_file_as_written(slab, tmp_path):
    pytest.importorskip("matplotlib")

    tool = Visualize(runs_root=str(tmp_path / "runs"), output_root=str(tmp_path / "out"))
    result = tool.structure(structure_path=str(slab))
    assert "view_direction" not in result.summary
    assert Path(str(result.artifacts["png"])).exists()


def test_the_html_viewer_opens_looking_along_the_direction(slab, tmp_path):
    """The camera sits on the axis, opposite the way the observer looks."""

    tool = Visualize(runs_root=str(tmp_path / "runs"), output_root=str(tmp_path / "out"))
    result = tool.structure(structure_path=str(slab), plotly=True, view_direction="[001]")
    html = Path(str(result.artifacts["html"])).read_text()
    assert '"eye"' in html
    assert '"z": -2.0' in html
    assert "observed along the" in html


def test_the_html_viewer_omits_the_camera_when_no_direction_is_asked_for(slab, tmp_path):
    tool = Visualize(runs_root=str(tmp_path / "runs"), output_root=str(tmp_path / "out"))
    result = tool.structure(structure_path=str(slab), plotly=True)
    html = Path(str(result.artifacts["html"])).read_text()
    assert '"camera": null' in html


def test_the_root_view_command_can_draw_a_structure_along_a_direction(slab, tmp_path, monkeypatch):
    """Structure visualization is routed through the root view command."""

    pytest.importorskip("matplotlib")

    monkeypatch.chdir(tmp_path)
    output = tmp_path / "structure.png"
    result = _run_cli("view", str(slab), "--output", str(output), "--view-direction", "111")
    assert output.exists() and output.stat().st_size > 0
    assert result.summary["view_direction"] == "(1 1 1) plane normal"


def test_the_manifest_keeps_the_direction_that_was_asked_for(slab, tmp_path):
    import json

    pytest.importorskip("matplotlib")

    tool = Visualize(runs_root=str(tmp_path / "runs"), output_root=str(tmp_path / "out"))
    result = tool.structure(structure_path=str(slab), view_direction="1x1x0")
    manifest = json.loads(Path(result.manifest_path).read_text())
    assert manifest["parameters"]["view_direction"] == "1x1x0"
    assert manifest["summary"]["view_direction"] == "(-1 -1 0) plane normal"


TRICLINIC = np.array([[4.0, 0.0, 0.0], [1.1, 3.7, 0.0], [0.6, 0.9, 4.3]])


@pytest.mark.parametrize(
    "lattice,spec",
    [
        (CUBIC, "auto"),
        (CUBIC, "a"),
        (CUBIC, "111"),
        (CUBIC, "110"),
        (CUBIC, "-112"),
        (TRICLINIC, "auto"),
        (TRICLINIC, "111"),
        (TRICLINIC, "c*"),
        (TRICLINIC, "2,1,3"),
    ],
)
def test_a_plane_normal_stacks_the_whole_lattice_at_its_own_spacing(lattice, spec):
    """What makes a direction a plane normal is that the crystal repeats along it.

    Every lattice vector then projects onto a whole number of interplanar
    spacings, so the lattice really is a stack of ``(h k l)`` planes ``d``
    apart, and ``d`` is ``1 / |G|`` for the reciprocal vector of the family.
    """

    direction = resolve_direction(lattice, spec)
    assert direction.is_lattice_plane_normal
    assert direction.miller is not None
    assert direction.spacing is not None and direction.spacing > 0.0
    heights = np.asarray(lattice, dtype=float) @ direction.unit
    steps = heights / direction.spacing
    assert steps == pytest.approx(np.round(steps), abs=1e-9)
    assert np.any(np.abs(np.round(steps)) == 1)
    reciprocal = np.linalg.inv(np.asarray(lattice, dtype=float)).T
    normal = np.asarray(direction.miller, dtype=float) @ reciprocal
    assert direction.spacing == pytest.approx(1.0 / np.linalg.norm(normal), rel=1e-9)


@pytest.mark.parametrize(
    "spec", ["x", "y", "a", "[111]", "cart:0.123456,0.98765,0.3"]
)
def test_a_direction_the_triclinic_crystal_does_not_stack_along_says_so(spec):
    """A direction with no small-integer plane family has no spacing to report.

    Reporting one anyway would invent a periodicity the crystal does not have,
    and the layers seen along it would be an artefact of the cell supplied.
    """

    direction = resolve_direction(TRICLINIC, spec)
    assert not direction.is_lattice_plane_normal
    assert direction.miller is None
    assert direction.spacing is None
    assert "not" in direction.describe().lower() or "no " in direction.describe().lower()
