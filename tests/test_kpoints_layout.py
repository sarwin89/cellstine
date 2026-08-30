"""An automatic KPOINTS file must stand for the mesh it was written from.

``Monkhorst`` is not a label: it already offsets the grid by half a step along
every axis with an even division, so writing the same half step on the shift
line as well moves the mesh by a whole step and quietly hands back the
Gamma-centred grid.  These tests compare the points the file describes against
the points of the mesh object, so a spelling that means something else fails.
"""

from __future__ import annotations

import numpy as np
import pytest

from cellstine.core import reciprocal as rc
from cellstine.io import kpoints as kpoints_io


def _cubic(constant: float = 4.0) -> np.ndarray:
    return constant * np.eye(3)


def _sorted(points: np.ndarray) -> np.ndarray:
    rounded = np.round(np.asarray(points, dtype=float), 9) + 0.0
    order = np.lexsort((rounded[:, 2], rounded[:, 1], rounded[:, 0]))
    return rounded[order]


@pytest.mark.parametrize(
    "divisions, mode",
    [
        ((4, 4, 4), "gamma"),
        ((4, 4, 4), "monkhorst"),
        ((5, 5, 5), "monkhorst"),
        ((4, 4, 3), "monkhorst"),
        ((6, 2, 1), "monkhorst"),
        ((1, 1, 1), "monkhorst"),
    ],
)
def test_the_automatic_layout_describes_the_same_grid(tmp_path, divisions, mode):
    mesh = rc.build_mesh(_cubic(), divisions=divisions, mode=mode, time_reversal=False)
    path = kpoints_io.write_mesh(tmp_path / "KPOINTS", mesh, explicit=False)
    parsed = kpoints_io.read_kpoints(path)
    assert parsed.divisions == tuple(divisions)
    assert np.allclose(parsed.total_shift, np.asarray(mesh.shift) % 1.0)
    assert np.allclose(_sorted(parsed.mesh_points()), _sorted(mesh.points), atol=1e-12)


def test_a_half_shifted_mesh_is_never_written_as_monkhorst_plus_a_half_step(tmp_path):
    mesh = rc.build_mesh(_cubic(), divisions=(4, 4, 4), mode="monkhorst", time_reversal=False)
    text = kpoints_io.write_mesh(tmp_path / "KPOINTS", mesh, explicit=False).read_text(
        encoding="utf-8"
    )
    lines = text.splitlines()
    assert lines[2].lower().startswith("m")
    assert np.allclose([float(item) for item in lines[4].split()], 0.0), (
        "the mode word already carries the half step"
    )


def test_the_gamma_spelling_of_the_same_mesh_agrees_point_for_point(tmp_path):
    path = kpoints_io.write_automatic_kpoints(
        tmp_path / "KPOINTS", (4, 4, 4), shift=(0.5, 0.5, 0.5), gamma_centred=True
    )
    gamma = kpoints_io.read_kpoints(path)
    other = tmp_path / "KPOINTS-mp"
    other.write_text("mp\n0\nMonkhorst-Pack\n4 4 4\n0 0 0\n", encoding="utf-8")
    monkhorst = kpoints_io.read_kpoints(other)
    assert np.allclose(gamma.total_shift, monkhorst.total_shift)
    assert np.allclose(_sorted(gamma.mesh_points()), _sorted(monkhorst.mesh_points()))


def test_monkhorst_with_a_half_step_is_read_back_as_the_gamma_grid(tmp_path):
    """The wrong spelling is not an error, but it does mean the unshifted mesh."""

    path = tmp_path / "KPOINTS"
    path.write_text("double shifted\n0\nMonkhorst\n4 4 4\n0.5 0.5 0.5\n", encoding="utf-8")
    parsed = kpoints_io.read_kpoints(path)
    assert np.allclose(parsed.total_shift, 0.0)
    assert np.allclose(
        _sorted(parsed.mesh_points()), _sorted(rc.mesh_points((4, 4, 4), (0.0, 0.0, 0.0)))
    )


def test_an_odd_mesh_is_written_gamma_centred_whatever_the_mode_word(tmp_path):
    mesh = rc.build_mesh(_cubic(), divisions=(3, 3, 3), mode="monkhorst", time_reversal=False)
    assert mesh.shift == (0.0, 0.0, 0.0)
    parsed = kpoints_io.read_kpoints(
        kpoints_io.write_mesh(tmp_path / "KPOINTS", mesh, explicit=False)
    )
    assert parsed.mode == "gamma"
    assert np.allclose(parsed.total_shift, 0.0)


def test_an_explicit_file_reports_its_own_points(tmp_path):
    mesh = rc.build_mesh(_cubic(), divisions=(4, 4, 4), time_reversal=True)
    parsed = kpoints_io.read_kpoints(
        kpoints_io.write_mesh(tmp_path / "KPOINTS", mesh, explicit=True)
    )
    assert np.allclose(parsed.mesh_points(), mesh.points, atol=1e-10)
    with pytest.raises(ValueError, match="automatic"):
        _ = parsed.total_shift
