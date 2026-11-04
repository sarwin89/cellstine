"""Opt-in archives work through CLI, manifests and scientific consumers."""

from pathlib import Path

import numpy as np
import pytest

from cellstine.cli.parsers import build_parser
from cellstine.core.manifests import RunManifest
from cellstine.core.previews import _preview_moire_results_with_count
from cellstine.io.native import read_poscar
from cellstine.moire.moire import Moire
from cellstine.moire.search.find import run_find
from cellstine.moire.search.results import read_results, write_results


@pytest.fixture
def layer(tmp_path):
    path = tmp_path / "layer.vasp"
    path.write_text("carbon\n1.0\n2 0 0\n0 2 0\n0 0 12\nC\n1\nDirect\n0 0 0.5\n", encoding="utf-8")
    return path


@pytest.mark.parametrize("kind", ["json", "npz", "npz-compressed"])
def test_cli_dispatch_forwards_result_format(monkeypatch, kind):
    import importlib
    main = importlib.import_module("cellstine.cli.main")
    calls = []

    class Workflow:
        def find(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr(main, "_workflow", lambda _: Workflow)
    base = ["moire", "search", "top.vasp", "bottom.vasp", "--length", "20", "--rigid"]
    assert build_parser().parse_args(base).results_format == "json"
    main.execute_namespace(build_parser().parse_args(base + ["--results-format", kind]))
    assert calls[0]["results_format"] == kind


@pytest.mark.parametrize("kind", ["npz", "npz-compressed"])
def test_manifest_build_and_preview_use_archive(tmp_path, layer, kind):
    workflow = Moire(backend="native", runs_root=tmp_path / "runs", output_root=tmp_path / "output")
    found = workflow.find(top_poscar=str(layer), bottom_poscar=str(layer), max_length=2.1,
                          top_strain=0., bottom_strain=0., reduce_layers=False, results_format=kind)
    assert "results_json" not in found.artifacts
    archive = Path(found.artifacts["results_npz"])
    assert archive.suffix == ".npz" and archive.is_file()
    manifest = RunManifest.load(found.manifest_path)
    assert manifest.parameters["results_format"] == kind
    preview, count = _preview_moire_results_with_count(found.manifest_path)
    document = read_results(archive)
    assert count == len(document["candidates"]) > 0 and "Saved Gram candidates" in preview
    baseline = write_results(tmp_path / "results.json", document)
    archive_build = workflow.make(results_file=str(found.manifest_path), indexes=[1], interlayer_distance=3.35,
                                  output_dir=str(tmp_path / "archive-build"))
    json_build = workflow.make(results_file=str(baseline), indexes=[1], interlayer_distance=3.35,
                               output_dir=str(tmp_path / "json-build"))
    before = read_poscar(json_build.artifacts["structures"][0])
    after = read_poscar(archive_build.artifacts["structures"][0])
    assert before.species == after.species and before.counts == after.counts
    assert np.array_equal(before.lattice, after.lattice)
    assert np.array_equal(before.positions_direct, after.positions_direct)
    assert np.array_equal(before.positions_cartesian, after.positions_cartesian)


def test_invalid_format_fails_before_run_directory_creation(tmp_path, layer):
    workflow = Moire(backend="native", runs_root=tmp_path / "runs", output_root=tmp_path / "output")
    with pytest.raises(ValueError, match="results_format"):
        workflow.find(top_poscar=str(layer), bottom_poscar=str(layer), max_length=2.1,
                      top_strain=0., bottom_strain=0., results_format="pickle")
    assert not (tmp_path / "runs").exists() and not (tmp_path / "output").exists()


def test_direct_search_progress_names_actual_archive(tmp_path, layer):
    progress = []
    run = run_find(top_poscar=str(layer), bottom_poscar=str(layer), max_length=2.1,
                   top_strain=0., bottom_strain=0., reduce_layers=False, output_root=str(tmp_path / "direct"),
                   results_format="npz", progress_callback=lambda stage, text: progress.append((stage, text)))
    assert run.result_path.name == "results.npz"
    writes = [text for stage, text in progress if stage == "write"]
    assert writes and all("results.npz" in text and "results.json" not in text for text in writes)


def test_guided_advanced_options_choose_archive(monkeypatch, layer):
    from cellstine.cli.interactive import build_moire

    monkeypatch.setattr(build_moire, "_prompt_path", lambda *a, **k: str(layer))
    answers = iter(["2.1", "y", "0", "y", "n", "n", "n", "n", "10", "npz"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    namespace = build_parser().parse_args(build_moire._build_moire_find())
    assert namespace.results_format == "npz" and namespace.max_atoms is None
    with pytest.raises(StopIteration):
        next(answers)


@pytest.mark.parametrize("plotly", [False, True])
def test_archive_manifest_drives_real_plot(tmp_path, layer, monkeypatch, plotly):
    monkeypatch.setenv("MPLBACKEND", "Agg")
    pytest.importorskip("plotly" if plotly else "matplotlib")
    from cellstine.visualize.visualize import Visualize

    workflow = Moire(backend="native", runs_root=tmp_path / "runs", output_root=tmp_path / "output")
    found = workflow.find(top_poscar=str(layer), bottom_poscar=str(layer), max_length=2.1,
                          top_strain=0., bottom_strain=0., reduce_layers=False, results_format="npz")
    viewer = Visualize(backend="native", runs_root=tmp_path / "plots", output_root=tmp_path / "output")
    extension = "html" if plotly else "png"
    rendered = viewer.moire_results(results_file=str(found.manifest_path), indices=[1], plotly=plotly,
                                    output_path=str(tmp_path / ("view."+extension)))
    output = Path(rendered.artifacts[extension])
    assert output.is_file() and output.stat().st_size > 0
    if not plotly:
        assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    manifest = RunManifest.load(rendered.manifest_path)
    assert manifest.inputs["results_file"] == str(Path(found.artifacts["results_npz"]).resolve())


@pytest.mark.parametrize("stage", ["build", "view"])
def test_guided_file_discovery_finds_archive(tmp_path, layer, monkeypatch, stage):
    from cellstine.cli.interactive import build_moire

    monkeypatch.chdir(tmp_path)
    workflow = Moire(backend="native", runs_root=tmp_path / "runs", output_root=tmp_path / "output")
    found = workflow.find(top_poscar=str(layer), bottom_poscar=str(layer), max_length=2.1,
                          top_strain=0., bottom_strain=0., reduce_layers=False, results_format="npz")
    script = ["1", "1", "3.35", "n", "n"] if stage == "build" else ["1", "n", "n"]
    answers = iter(script)
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    argv = build_moire._build_moire_make() if stage == "build" else build_moire._build_moire_visualize()
    namespace = build_parser().parse_args(argv)
    assert namespace.results_file == found.artifacts["results_npz"]
    with pytest.raises(StopIteration):
        next(answers)
