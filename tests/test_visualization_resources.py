"""HTML data stays data; selected views and failed exports keep bounded work."""

from __future__ import annotations

from collections import Counter
from html.parser import HTMLParser
import json
import re
import tracemalloc

import numpy as np
import pytest

from cellstine.io.models import StructureRecord
from cellstine.visualize.backends import matplotlib as static
from cellstine.visualize.backends import plotly as structure_html
from cellstine.visualize.results import plotly as results_html


class _Document(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tags = []

    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, attrs))


def _constant(text, name):
    found = re.search(r"const " + name + r" = (.*?);\n", text, re.S)
    assert found, name
    assert "<" not in found.group(1), "inline JSON must escape HTML raw-text delimiters"
    return json.loads(found.group(1))


def _record(title="ordinary"):
    direct = np.arange(30, dtype=float).reshape(10, 3) / 30
    lattice = np.array([[3., 0., 0.], [.2, 4., 0.], [.3, .4, 5.]])
    return StructureRecord(title, lattice, ["Fe_pv", "C", "Fe_pv", "Ghost"], [3, 5, 2, 0], direct, direct @ lattice)


ATTACKS = ["</script><img src=x onerror=alert(1)>",
           "</ScRiPt ><svg onload=alert(1)>",
           "<!-- <script> & \"quotes\" ' \\ \u2028\u2029 \u03b8 \U0001f600"]


@pytest.mark.parametrize("attack", ATTACKS)
@pytest.mark.parametrize("field", ["comment", "title"])
def test_structure_html_keeps_titles_inside_the_script_value(tmp_path, attack, field):
    record = _record(attack if field == "comment" else "ordinary")
    before = record.positions_cartesian.tobytes()
    output = structure_html.write_structure_html(record, output_path=tmp_path / "viewer.html",
                                                 title=attack if field == "title" else None)
    text = output.read_text(encoding="utf-8")
    document = _Document()
    document.feed(text)
    assert [tag for tag, _ in document.tags].count("script") == 2
    assert not any(tag in {"img", "svg"} for tag, _ in document.tags)
    payload = _constant(text, "payload")
    assert payload["title"] == attack
    np.testing.assert_array_equal(np.column_stack([payload["atoms"][a] for a in "xyz"]), record.positions_cartesian)
    assert record.positions_cartesian.tobytes() == before


@pytest.mark.parametrize("attack", ATTACKS)
@pytest.mark.parametrize("field", ["name", "subtitle", "label"])
def test_result_html_roundtrips_frame_text_without_creating_markup(attack, field):
    layer = {"label": attack if field == "label" else "Bottom", "color": "#264653", "positions": [[.1, .2, .3]]}
    frame = {"name": attack if field == "name" else "candidate 1",
             "subtitle": attack if field == "subtitle" else "ordinary<br>scientific subtitle",
             "lattice": np.eye(3).tolist(), "layers": [layer]}
    text = results_html._build_html([frame], "ordinary title")
    document = _Document()
    document.feed(text)
    assert [tag for tag, _ in document.tags].count("script") == 2
    assert not any(tag in {"img", "svg"} for tag, _ in document.tags)
    loaded = _constant(text, "frames")[0]
    assert loaded["name"] == frame["name"]
    assert loaded["layout"]["title"]["text"] == frame["subtitle"]
    assert loaded["data"][0]["name"] == layer["label"]
    assert loaded["data"][0]["x"] == [.1]
    assert _constant(text, "sliderSteps")[0]["label"] == frame["name"]


def test_structure_sizes_are_looked_up_once_per_distinct_expanded_label(tmp_path, monkeypatch):
    calls = Counter()
    original = structure_html._atomic_radius

    def observed(label):
        calls[label] += 1
        return original(label)

    monkeypatch.setattr(structure_html, "_atomic_radius", observed)
    path = structure_html.write_structure_html(_record(), output_path=tmp_path / "sizes.html")
    payload = _constant(path.read_text(encoding="utf-8"), "payload")
    assert calls == {"Fe_pv": 1, "C": 1}
    expected = [max(7., min(22., 9. * original(label))) for label in payload["atoms"]["text"]]
    assert payload["atoms"]["size"] == expected


def _payload(count=3):
    candidate = dict(angle_deg=1., strain=[.001, -.002], top_layer_strain=[.001, -.001],
        bottom_layer_strain=[.001, -.001], moire_a=3., moire_b=4., moire_gamma_deg=90.,
        coincidence_index=1, top_atom_count=2, bottom_atom_count=2, atom_count=4,
        pareto_optimal=True, loewner_certified=True, loewner_borderline=False,
        top_matrix=[[1, 0], [0, 1]], bottom_matrix=[[1, 0], [0, 1]], shared_lattice=[[3., 0.], [0., 4.]])
    return {"candidates": [dict(candidate, index=i, rank=i) for i in range(1, count + 1)],
            "metadata": {"engine": "gram", "symmetric_used": False, "symmetric_fallback": None},
            "search": {"max_length": 5., "top_strain": .01, "bottom_strain": .01}}


def test_selected_summary_does_not_copy_every_unselected_row(monkeypatch):
    payload = _payload(10000)
    monkeypatch.setattr(static, "read_results", lambda path: payload)
    tracemalloc.start()
    try:
        kind, rows, returned = static._read_moire_summary("validated.npz", [9999])
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert kind == "bilayer" and returned is payload
    assert [row["index"] for row in rows] == [9999]
    assert peak < 1024**2, f"one selected row allocated {peak} bytes"


@pytest.mark.parametrize("indices,expected", [(None, [1, 2, 3]), ([3, 1, 3], [1, 3]), ([99, 2], [2])])
def test_summary_filter_keeps_source_order_and_row_values(monkeypatch, indices, expected):
    payload = _payload()
    before = json.dumps(payload)
    monkeypatch.setattr(static, "read_results", lambda path: payload)
    _, rows, returned = static._read_moire_summary("validated.json", indices)
    assert [row["index"] for row in rows] == expected
    assert all(row["relative_principal_strain"] == [.001, -.002] for row in rows)
    assert all(row is not payload["candidates"][row["index"] - 1] for row in rows)
    assert returned is payload and json.dumps(payload) == before


@pytest.mark.parametrize("indices", [[], [99]])
def test_empty_summary_selection_remains_an_error(monkeypatch, indices):
    monkeypatch.setattr(static, "read_results", lambda path: _payload())
    with pytest.raises(ValueError, match="no candidates were selected"):
        static._read_moire_summary("validated.npz", indices)


@pytest.mark.parametrize("show", [False, True])
def test_backend_selection_keeps_configured_backend_for_show(monkeypatch, show):
    matplotlib = pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt
    calls = []
    monkeypatch.setattr(matplotlib, "use", lambda backend, **kwargs: calls.append((backend, kwargs)))
    assert static._pyplot(show=show) is plt
    assert calls == ([] if show else [("Agg", {"force": True})])


@pytest.mark.parametrize("kind", ["structure", "summary"])
@pytest.mark.parametrize("show", [False, True])
def test_renderer_forwards_the_requested_show_policy(tmp_path, monkeypatch, kind, show):
    pytest.importorskip("matplotlib")
    from matplotlib.figure import Figure
    plt = static._pyplot()
    requests = []

    def pyplot(*, show=False):
        requests.append(show)
        return plt

    monkeypatch.setattr(static, "_pyplot", pyplot)
    monkeypatch.setattr(plt, "show", lambda: None)
    monkeypatch.setattr(Figure, "savefig", lambda *args, **kwargs: None)
    monkeypatch.setattr(static, "read_results", lambda path: _payload())
    if kind == "structure":
        static.plot_structure_multiview(_record(), output_path=tmp_path / "show.png", show=show)
    else:
        static.plot_moire_summary("validated.npz", output_path=tmp_path / "show.png", show=show)
    assert requests == [show]


@pytest.mark.parametrize("kind", ["structure", "summary"])
@pytest.mark.parametrize("phase", ["construct", "save", "show"])
def test_failed_export_closes_only_its_own_figure(tmp_path, monkeypatch, kind, phase):
    pytest.importorskip("matplotlib")
    from matplotlib.figure import Figure
    plt = static._pyplot()
    caller_figure = plt.figure()
    existing = set(plt.get_fignums())
    failure = OSError("owned export failure")

    def fail(*args, **kwargs):
        raise failure

    if phase == "construct":
        monkeypatch.setattr(Figure, "add_subplot" if kind == "structure" else "subplots", fail)
    monkeypatch.setattr(Figure, "savefig", fail if phase == "save" else lambda *args, **kwargs: None)
    monkeypatch.setattr(plt, "show", fail)
    monkeypatch.setattr(static, "read_results", lambda path: _payload())
    try:
        with pytest.raises(OSError) as raised:
            if kind == "structure":
                static.plot_structure_multiview(_record(), output_path=tmp_path / "failure.png", show=phase == "show")
            else:
                static.plot_moire_summary("validated.npz", output_path=tmp_path / "failure.png", show=phase == "show")
        assert raised.value is failure
        assert set(plt.get_fignums()) == existing
    finally:
        for number in set(plt.get_fignums()) - existing:
            plt.close(number)
        plt.close(caller_figure)
