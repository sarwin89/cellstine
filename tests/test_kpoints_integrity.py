"""KPOINTS input integrity and bounded line-path assembly."""

import tracemalloc

import numpy as np
import pytest

from cellstine.io import kpoints as k


@pytest.mark.parametrize("text", [
    "mesh\n0\nGamma\n0 2 2\n",
    "mesh\n0\nGamma\n2.5 2 2\n",
    "mesh\n0\nGamma\n2 2 2\n.5 .5\n",
    "list\n1.5\nReciprocal\n0 0 0 1\n",
    "list\n1\nReciprocal\n0 0\n0 0 0 1\n",
    "list\n1\nReciprocal\nnan 0 0 1\n",
    "line\n-2\nLine-mode\nReciprocal\n0 0 0\n.5 0 0\n",
    "line\n2\nLine-mode\nReciprocal\n0 0\n0 0 0\n.5 0 0\n",
])
def test_malformed_kpoints_cannot_silently_change_the_requested_sampling(tmp_path, text):
    path = tmp_path / "KPOINTS"
    path.write_text(text)
    with pytest.raises(ValueError):
        k.read_kpoints(path)


@pytest.mark.parametrize("kind", ["automatic", "explicit", "line"])
def test_kpoints_writers_reject_nonfinite_data_before_replacing_existing_file(tmp_path, kind):
    path = tmp_path / "KPOINTS"
    path.write_text("preserve")
    with pytest.raises(ValueError):
        if kind == "automatic":
            k.write_automatic_kpoints(path, (2, 2, 2), shift=(float("nan"), 0., 0.))
        elif kind == "explicit":
            k.write_explicit_kpoints(path, [[float("nan"), 0., 0.]], [1.])
        else:
            k.write_line_mode_kpoints(path, [(("G", [0., 0., 0.]), ("X", [float("nan"), 0., 0.]))], 4)
    assert path.read_text() == "preserve"


def test_line_points_avoids_retaining_all_segment_arrays_before_concatenation():
    ends = np.tile(np.array([[0., 0., 0.], [.5, .25, 0.]]), (64, 1))
    parsed = k.KpointsFile("path", "line", line_divisions=2048, endpoints=ends)
    tracemalloc.start()
    try:
        points = parsed.line_points()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert points.shape == (64 * 2048, 3)
    assert np.array_equal(points[0], ends[0]) and np.array_equal(points[-1], ends[-1])
    assert peak < 4 * 1024 * 1024
